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
        """Send `nginx -s reload` to apply the updated upstream config."""
        subprocess.run(["nginx", "-s", "reload"], check=True)

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

    def _create_vast_instance(self) -> dict:
        """
        Call Vast.ai REST API to launch a new GPU instance from gpu_template_id.
        Polls until the instance is running, then returns
        {"id": <instance_id>, "address": "<host>:8000"}.
        """
        url = f"{_VAST_BASE}/asks/{self.config.gpu_template_id}/"
        payload = {
            "client_id": "me",
            "image": os.environ.get("VLLM_DOCKER_IMAGE", "vllm/vllm-openai:latest"),
            "runtype": "ssh_direc tcp",
            "disk": 40,
        }
        resp = requests.put(
            url,
            params={"api_key": self.config.vast_api_key},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
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
            inst = status_resp.json().get("instances", [{}])[0]
            if inst.get("actual_status") == "running":
                host = inst["public_ipaddr"]
                return {"id": instance_id, "address": f"{host}:8000"}

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
