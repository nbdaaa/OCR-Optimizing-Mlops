"""
Auto-scaler for vLLM serving instances.

Polls Prometheus for P95 latency every poll_interval seconds, then scales
Vast.ai GPU instances up or down and rewrites the nginx upstream config.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field

import requests


_VAST_BASE = "https://console.vast.ai/api/v0"

# How long to poll for a new Vast.ai instance to become "running"
_INSTANCE_READY_TIMEOUT_S = 300
_INSTANCE_POLL_INTERVAL_S = 10


# ── Config & State ────────────────────────────────────────────────────────────

@dataclass
class ScalerConfig:
    vast_api_key: str
    gpu_template_id: str
    nginx_upstream_conf: str
    state_file: str
    prometheus_targets_file: str = "infra/prometheus/targets.json"
    nginx_container_name: str = "infra-nginx-1"
    scale_up_threshold: float = 5.0
    scale_down_threshold: float = 1.5
    min_instances: int = 1
    max_instances: int = 5
    cooldown_seconds: int = 180
    poll_interval: int = 30


@dataclass
class ScalerState:
    instances: list[dict] = field(default_factory=list)
    last_scale_time: float = 0.0


# ── AutoScaler ────────────────────────────────────────────────────────────────

class AutoScaler:
    def __init__(self, config: ScalerConfig) -> None:
        self.config = config
        self.state = ScalerState()

    # ── Decision helpers ──────────────────────────────────────────────────────

    def should_scale_up(self, p95_latency: float) -> bool:
        """Return True if P95 latency exceeds the scale-up threshold."""
        return bool(p95_latency > self.config.scale_up_threshold)

    def should_scale_down(self, p95_latency: float) -> bool:
        """Return True if P95 latency is below the scale-down threshold."""
        return bool(p95_latency < self.config.scale_down_threshold)

    def is_cooldown_active(self) -> bool:
        """Return True if a scale event happened within cooldown_seconds."""
        if self.state.last_scale_time == 0.0:
            return False
        return bool((time.time() - self.state.last_scale_time) < self.config.cooldown_seconds)

    def can_scale_up(self) -> bool:
        """Return True if current instance count is below max_instances."""
        return bool(len(self.state.instances) < self.config.max_instances)

    def can_scale_down(self) -> bool:
        """Return True if current instance count is above min_instances."""
        return bool(len(self.state.instances) > self.config.min_instances)

    # ── Scale actions ─────────────────────────────────────────────────────────

    def scale_up(self) -> None:
        """
        Provision a new Vast.ai instance, add it to nginx upstream, persist state.
        Updates last_scale_time after a successful scale event.
        """
        new_instance = self._create_vast_instance()
        self.state.instances.append(new_instance)
        self.state.last_scale_time = time.time()
        self._write_nginx_upstream()
        self._write_prometheus_targets()
        self.save_state()

    def scale_down(self) -> None:
        """
        Destroy one Vast.ai instance (last in list), remove from nginx upstream,
        persist state. Updates last_scale_time after a successful scale event.
        """
        instance = self.state.instances.pop()
        self.state.last_scale_time = time.time()
        self._destroy_vast_instance(instance["id"])
        self._write_nginx_upstream()
        self._write_prometheus_targets()
        self.save_state()

    # ── Nginx ─────────────────────────────────────────────────────────────────

    def _write_nginx_upstream(self) -> None:
        """
        Rewrite nginx_upstream_conf with all current instance addresses,
        then call _reload_nginx().
        """
        server_lines = "\n".join(
            f"    server {inst['address']};" for inst in self.state.instances
        )
        content = f"upstream vllm_backend {{\n    least_conn;\n{server_lines}\n}}\n"
        with open(self.config.nginx_upstream_conf, "w") as f:
            f.write(content)
        self._reload_nginx()

    def _reload_nginx(self) -> None:
        """Send `nginx -s reload` inside the nginx container to apply the updated upstream config."""
        subprocess.run(
            ["docker", "exec", self.config.nginx_container_name, "nginx", "-s", "reload"],
            check=True,
        )

    def _write_prometheus_targets(self) -> None:
        """
        Write Prometheus file_sd targets.json with current instance addresses.
        Prometheus watches this file (refresh_interval: 15s) and automatically
        updates scrape targets when instances are added or removed.
        Format: [{"targets": ["<host>:8000", ...], "labels": {"job": "vllm"}}]
        """
        targets = [inst["address"] for inst in self.state.instances]
        content = [{"targets": targets, "labels": {"job": "vllm"}}]
        with open(self.config.prometheus_targets_file, "w") as f:
            json.dump(content, f, indent=2)

    # ── State persistence ─────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Serialise ScalerState to state_file as JSON."""
        data = {
            "instances": self.state.instances,
            "last_scale_time": self.state.last_scale_time,
        }
        with open(self.config.state_file, "w") as f:
            json.dump(data, f)

    def load_state(self) -> None:
        """
        Load ScalerState from state_file.
        Initialises empty state if the file does not exist.
        """
        if not os.path.exists(self.config.state_file):
            self.state = ScalerState()
            return
        with open(self.config.state_file) as f:
            data = json.load(f)
        self.state = ScalerState(
            instances=data.get("instances", []),
            last_scale_time=data.get("last_scale_time", 0.0),
        )

    # ── Vast.ai API ───────────────────────────────────────────────────────────

    # GPUs we accept — all >= 24GB VRAM, common on Vast.ai
    _ALLOWED_GPUS = ["RTX 3090", "RTX 4090", "RTX 5090"]

    def _find_best_offer(self) -> str:
        """
        Search Vast.ai marketplace for a rentable offer on one of _ALLOWED_GPUS
        and return its id.

        Strategy: filter offers that meet the minimum requirements (allowed GPU,
        disk, network up/down speed), then pick the CHEAPEST among them. Network
        speed is a hard floor (configurable), not the sort key — the fastest
        network usually costs more; we want the cheapest box with fast-enough
        up/down for pulling the base model + pushing the adapter.

        Tunable via env:
            VAST_MIN_INET_MBPS  (default 100)  — floor for both up and down
            VAST_MIN_DISK_GB    (default 40)
            VAST_MAX_PRICE      (default none) — $/hr cap, optional

        Returns the offer id as a string. Raises RuntimeError if none match.
        """
        min_inet  = float(os.environ.get("VAST_MIN_INET_MBPS", "100"))
        min_disk  = int(os.environ.get("VAST_MIN_DISK_GB", "40"))
        max_price = os.environ.get("VAST_MAX_PRICE")
        # Exclude countries with throttled/blocked international links (e.g. CN
        # behind the Great Firewall → slow transfers to our GCP VM + no GitHub).
        exclude = {
            c.strip().upper()
            for c in os.environ.get("VAST_EXCLUDE_COUNTRIES", "CN").split(",")
            if c.strip()
        }

        query: dict = {
            "rentable":   {"eq": True},
            "num_gpus":   {"eq": 1},
            "gpu_name":   {"in": self._ALLOWED_GPUS},
            "disk_space": {"gte": min_disk},
            "inet_down":  {"gte": min_inet},
            "inet_up":    {"gte": min_inet},
            "type":       "on-demand",
            "order":      [["dph_total", "asc"]],
        }

        resp = requests.put(
            f"{_VAST_BASE}/search/asks/",
            params={"api_key": self.config.vast_api_key},
            json={"q": query},
            timeout=30,
        )
        resp.raise_for_status()
        offers = resp.json().get("offers", [])

        # Re-filter client-side (API query is best-effort) and sort by price
        candidates = [
            o for o in offers
            if o.get("gpu_name") in self._ALLOWED_GPUS
            and o.get("disk_space", 0) >= min_disk
            and o.get("inet_down", 0) >= min_inet
            and o.get("inet_up", 0) >= min_inet
            and (max_price is None or o.get("dph_total", 1e9) <= float(max_price))
            and (o.get("geolocation") or "").upper()[-2:] not in exclude
        ]
        if not candidates:
            raise RuntimeError(
                "No Vast.ai offer matched: "
                f"GPU in {self._ALLOWED_GPUS}, >= {min_inet}Mbps up/down, "
                f"excluding {sorted(exclude)}. Loosen VAST_* env vars or try later."
            )

        best = min(candidates, key=lambda o: o["dph_total"])
        print(
            f"[scaler] picked offer {best['id']}: {best.get('gpu_name')} "
            f"${best['dph_total']:.3f}/hr  "
            f"down={best.get('inet_down')}Mbps up={best.get('inet_up')}Mbps",
            flush=True,
        )
        return str(best["id"])

    def _create_vast_instance(
        self, image: str | None = None, onstart: str | None = None
    ) -> dict:
        """
        Call Vast.ai REST API to launch a new GPU instance.
        Uses gpu_template_id if set, otherwise auto-selects the cheapest offer
        meeting the requirements via _find_best_offer().
        Polls until the instance is running, then returns
        {"id": <instance_id>, "address": "<host>:<mapped_port>", "ssh_port": <port>}.
        Internal port 8000 is exposed at rent time via `-p 8000:8000`; Vast maps
        it to a dynamic external port, read back from the instance's `ports` info.

        Args:
            image:   Docker image for the instance. Serving uses the vLLM image
                     (default); training passes its own PyTorch image.
            onstart: Bootstrap script the instance runs on boot (clone + install
                     + run). When set, no SSH is needed — the instance self-runs.
        """
        img = image or os.environ.get("VLLM_DOCKER_IMAGE", "vllm/vllm-openai:latest")
        offer_id = self.config.gpu_template_id or self._find_best_offer()
        url = f"{_VAST_BASE}/asks/{offer_id}/"
        payload = {
            "client_id": "me",
            "image": img,
            "runtype": "ssh",
            "disk": 40,
            # Expose the vLLM serving port; Vast assigns a dynamic external port.
            # env must be a dict — port mapping is a key with a dummy "1" value.
            "env": {"-p 8000:8000": "1"},
        }
        if onstart:
            payload["onstart"] = onstart
        resp = requests.put(
            url,
            params={"api_key": self.config.vast_api_key},
            json=payload,
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Vast create failed {resp.status_code} for offer {offer_id}: "
                f"{resp.text} | payload={payload}"
            )
        result = resp.json()
        if not result.get("success"):
            raise RuntimeError(f"Vast.ai launch failed: {result}")
        instance_id = str(result["new_contract"])

        # Poll until the instance reaches "running" state
        deadline = time.time() + _INSTANCE_READY_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(_INSTANCE_POLL_INTERVAL_S)
            status_resp = requests.get(
                f"{_VAST_BASE}/instances/{instance_id}/",
                params={"api_key": self.config.vast_api_key},
                timeout=10,
            )
            status_resp.raise_for_status()
            # GET /instances/{id}/ returns instances as a single dict;
            # GET /instances/ returns a list. Handle both.
            data = status_resp.json().get("instances")
            if isinstance(data, list):
                inst = data[0] if data else {}
            else:
                inst = data or {}
            if inst.get("actual_status") == "running":
                host = inst["public_ipaddr"]
                ssh_port = str(inst.get("ssh_port", 22))
                # Vast maps internal 8000 → a dynamic external port (in `ports`).
                ports = inst.get("ports") or {}
                mapping = ports.get("8000/tcp") or [{}]
                ext_port = mapping[0].get("HostPort", "8000")
                return {
                    "id": instance_id,
                    "address": f"{host}:{ext_port}",
                    "ssh_port": ssh_port,
                }

        raise TimeoutError(
            f"Instance {instance_id} did not reach 'running' within "
            f"{_INSTANCE_READY_TIMEOUT_S}s"
        )

    def _destroy_vast_instance(self, instance_id: str) -> None:
        """Call Vast.ai REST API to destroy the instance with the given id."""
        url = f"{_VAST_BASE}/instances/{instance_id}/"
        resp = requests.delete(
            url,
            params={"api_key": self.config.vast_api_key},
            timeout=30,
        )
        resp.raise_for_status()

    # ── Prometheus ────────────────────────────────────────────────────────────

    def _fetch_p95_latency(self) -> float:
        """
        Query Prometheus for the P95 request latency across all vLLM instances.
        Returns the latency in seconds.
        """
        prometheus_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
        query = (
            "histogram_quantile(0.95, "
            "sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))"
        )
        resp = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json()["data"]["result"]
        if not results:
            return 0.0
        return float(results[0]["value"][1])

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Main polling loop. Every poll_interval seconds:
          1. Fetch P95 latency from Prometheus.
          2. If above scale_up_threshold and cooldown expired → scale_up().
          3. If below scale_down_threshold and cooldown expired → scale_down().
        Runs indefinitely until interrupted.
        """
        self.load_state()
        while True:
            try:
                p95 = self._fetch_p95_latency()
                if not self.is_cooldown_active():
                    if self.should_scale_up(p95) and self.can_scale_up():
                        self.scale_up()
                    elif self.should_scale_down(p95) and self.can_scale_down():
                        self.scale_down()
            except Exception as exc:
                print(f"[scaler] error during poll: {exc}")
            time.sleep(self.config.poll_interval)
