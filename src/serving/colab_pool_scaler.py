"""
Colab serving-pool autoscaler — runs on the HOST VM (needs the `colab` CLI on
PATH + ADC creds; NOT inside the fastapi container).

Manages 0..MAX vLLM-on-Colab instances, each exposed via its own Cloudflare
quick tunnel, load-balanced by nginx (split_clients, one Host/SNI per backend).

Driven by a shared state file that FastAPI /deploy writes:
    desired_floor = 1  → deploy (bring up at least 1, autoscale 1<->2 by load)
    desired_floor = 0  → teardown / sleep (stop all)

Auto behaviour (all configurable via env):
    queue high      → scale up   (up to POOL_MAX)
    idle 5 min      → scale 2->1
    idle 10 min     → auto-sleep 1->0 (also sets desired_floor=0)
                      set AUTO_SLEEP_IDLE_S=0 to DISABLE → pool stays at floor 1
                      until an explicit /deploy teardown.
    COOLDOWN_S between scale actions to avoid flapping.

Run:  HF_TOKEN=... python -m src.serving.colab_pool_scaler
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass

# ── Config (env-overridable) ──────────────────────────────────────────────────
LAUNCH            = os.environ.get("COLAB_LAUNCH_SCRIPT", "launch_vllm.py")
HF_TOKEN          = os.environ["HF_TOKEN"]
GPU               = os.environ.get("COLAB_GPU", "L4")
POOL_MAX          = int(os.environ.get("POOL_MAX", "2"))
SCALE_UP_QUEUE    = float(os.environ.get("SCALE_UP_QUEUE", "4"))      # avg (waiting+running)/inst
SCALE_DOWN_IDLE_S = int(os.environ.get("SCALE_DOWN_IDLE_S", "300"))   # 2->1
AUTO_SLEEP_IDLE_S = int(os.environ.get("AUTO_SLEEP_IDLE_S", "600"))   # 1->0; 0 = never auto-sleep
COOLDOWN_S        = int(os.environ.get("COOLDOWN_S", "240"))
POLL_S            = int(os.environ.get("POLL_INTERVAL_S", "15"))
SESSION_TIMEOUT   = os.environ.get("COLAB_SESSION_TIMEOUT", "21600")
STATE_FILE        = os.environ.get("SERVE_STATE_FILE", "/opt/ocr/serve_state.json")
NGINX_CONF        = os.environ.get("NGINX_LB_CONF", "/opt/ocr/nginx/ocr_lb.conf")
NGINX_CONTAINER   = os.environ.get("NGINX_CONTAINER", "infra-nginx-1")
PROM_TARGETS      = os.environ.get("PROM_TARGETS_FILE", "/opt/ocr/prometheus/targets.json")
COLAB             = os.environ.get("COLAB_CLI", "colab")
LOG_DIR           = os.environ.get("POOL_LOG_DIR", "/opt/ocr/logs")
READY_GIVEUP_S    = int(os.environ.get("READY_GIVEUP_S", "600"))      # kill instance if no tunnel in time
HEALTH_FAIL_CHECKS = int(os.environ.get("HEALTH_FAIL_CHECKS", "3"))   # consecutive /metrics fails → ready instance is dead

_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


@dataclass
class Instance:
    name: str
    proc: subprocess.Popen
    log_path: str
    started_at: float
    tunnel_url: str | None = None
    adapter: str | None = None      # which adapter version this instance loaded
    health_fails: int = 0           # consecutive /metrics-unreachable polls (post-ready)


# ── Colab session control ─────────────────────────────────────────────────────

def _launch(name: str) -> Instance:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/{name}.log"
    # Pass MLflow/MinIO config so launch_vllm.py pulls the Production adapter.
    # Empty values → launch falls back to its pinned HF adapter.
    extra = [
        os.environ.get("PUBLIC_MLFLOW_TRACKING_URI", ""),
        os.environ.get("PUBLIC_MINIO_ENDPOINT", ""),
        os.environ.get("MINIO_ACCESS_KEY", ""),
        os.environ.get("MINIO_SECRET_KEY", ""),
    ]
    cmd = [COLAB, "--auth", "adc", "run", "--gpu", GPU, "--keep",
           "-s", name, "--timeout", SESSION_TIMEOUT, LAUNCH, HF_TOKEN, *extra]
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    print(f"[pool] launching {name} (gpu={GPU})", flush=True)
    return Instance(name=name, proc=proc, log_path=log_path, started_at=time.time())


def _resolve_tunnel(inst: Instance) -> None:
    try:
        log = open(inst.log_path).read()
    except FileNotFoundError:
        return
    if inst.adapter is None:
        a = re.search(r"ADAPTER=(.+)", log)
        if a:
            inst.adapter = a.group(1).strip()
    if not inst.tunnel_url:
        m = _TUNNEL_RE.search(log)
        if m:
            inst.tunnel_url = m.group(0)
            print(f"[pool] {inst.name} ready → {inst.tunnel_url} "
                  f"(adapter: {inst.adapter})", flush=True)


def _stop(inst: Instance) -> None:
    print(f"[pool] stopping {inst.name}", flush=True)
    subprocess.run([COLAB, "--auth", "adc", "stop", "-s", inst.name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        inst.proc.terminate()
    except Exception:
        pass


# ── Metrics (vLLM /metrics via the instance's tunnel) ─────────────────────────

def _queue(url: str) -> float | None:
    """Return waiting+running requests, or None if the instance is unreachable."""
    try:
        txt = urllib.request.urlopen(url + "/metrics", timeout=5).read().decode()
    except Exception:
        return None

    def gauge(name: str) -> float:
        m = re.search(rf"{re.escape(name)}[^\s]* ([0-9.eE+]+)", txt)
        return float(m.group(1)) if m else 0.0

    return gauge("vllm:num_requests_waiting") + gauge("vllm:num_requests_running")


# ── State + nginx ─────────────────────────────────────────────────────────────

def _read_state() -> dict:
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}


def _read_floor() -> int:
    try:
        return int(_read_state().get("desired_floor", 0))
    except Exception:
        return 0


def _set_floor(n: int) -> None:
    """Persist desired_floor (used by auto-sleep; FastAPI /deploy owns it otherwise)."""
    st = _read_state()
    st["desired_floor"] = n
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    json.dump(st, open(STATE_FILE, "w"), indent=2)


def _write_status(instances: list[Instance], status: str) -> None:
    """Write pool status WITHOUT clobbering desired_floor (re-read + preserve)."""
    st = _read_state()
    st["desired_floor"] = int(st.get("desired_floor", 0))
    st["status"] = status
    st["lb_url"] = os.environ.get("PUBLIC_LB_URL", "")
    st["instances"] = [
        {"name": i.name, "tunnel_url": i.tunnel_url, "adapter": i.adapter,
         "ready": bool(i.tunnel_url), "uptime_s": int(time.time() - i.started_at)}
        for i in instances
    ]
    st["updated_at"] = int(time.time())
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    json.dump(st, open(STATE_FILE, "w"), indent=2)


def _write_nginx(ready_urls: list[str]) -> None:
    hosts = [u.replace("https://", "").rstrip("/") for u in ready_urls]
    if not hosts:
        conf = (
            'server {\n'
            '  listen 80;\n'
            '  location = /lb-health { return 200 "ok\\n"; }\n'
            '  location / { return 503 "no serving instances\\n"; }\n'
            '}\n'
        )
    else:
        if len(hosts) == 1:
            split = f"    * {hosts[0]};\n"
        else:
            pct = 100 // len(hosts)
            split = "".join(f"    {pct}% {h};\n" for h in hosts[:-1])
            split += f"    * {hosts[-1]};\n"
        conf = (
            f'split_clients "$remote_addr$request_id" $ocr_backend {{\n{split}}}\n'
            'server {\n'
            '  listen 80;\n'
            '  client_max_body_size 50m;\n'   # base64 document images exceed nginx default 1m
            '  resolver 1.1.1.1 ipv6=off valid=30s;\n'
            '  location = /lb-health { return 200 "ok\\n"; }\n'
            '  location / {\n'
            '    proxy_pass https://$ocr_backend;\n'
            '    proxy_ssl_server_name on;\n'
            '    proxy_ssl_name $ocr_backend;\n'
            '    proxy_set_header Host $ocr_backend;\n'
            '    proxy_http_version 1.1;\n'
            '    proxy_read_timeout 300s;\n'
            '  }\n'
            '}\n'
        )
    os.makedirs(os.path.dirname(NGINX_CONF), exist_ok=True)
    open(NGINX_CONF, "w").write(conf)
    subprocess.run(["docker", "exec", NGINX_CONTAINER, "nginx", "-s", "reload"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _write_prometheus_targets(ready: list[Instance]) -> None:
    """Write Prometheus file_sd targets (observability only — NOT the scale loop).
    Each ready instance's CF-tunnel host:443 with its pool name as a label.
    Prometheus picks this up via refresh_interval (no reload needed)."""
    targets = [
        {"targets": [i.tunnel_url.replace("https://", "").rstrip("/") + ":443"],
         "labels": {"instance": i.name, "adapter": i.adapter or "unknown"}}
        for i in ready if i.tunnel_url
    ]
    os.makedirs(os.path.dirname(PROM_TARGETS), exist_ok=True)
    json.dump(targets, open(PROM_TARGETS, "w"), indent=2)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    instances: list[Instance] = []
    last_scale = 0.0
    last_activity = time.time()
    fail_count = 0          # consecutive launch failures
    blocked_until = 0.0     # backoff: don't launch before this time
    print(f"[pool] scaler started · max={POOL_MAX} up_q={SCALE_UP_QUEUE} "
          f"down_idle={SCALE_DOWN_IDLE_S}s sleep_idle={AUTO_SLEEP_IDLE_S}s "
          f"cooldown={COOLDOWN_S}s", flush=True)

    while True:
        try:
            floor = _read_floor()

            # reconcile: drop instances that died (startup OR post-ready), so the
            # loop relaunches a fresh one instead of routing to a stale tunnel (530).
            for inst in list(instances):
                _resolve_tunnel(inst)
                # the `colab run` subprocess exited → session/vLLM gone (any phase)
                proc_dead = inst.proc.poll() is not None
                # never produced a tunnel within the giveup window
                stuck = (not inst.tunnel_url
                         and time.time() - inst.started_at > READY_GIVEUP_S)
                # was ready, but origin stopped answering /metrics (Colab reclaim,
                # cloudflared drop, vLLM OOM) for HEALTH_FAIL_CHECKS consecutive polls
                unhealthy = False
                if inst.tunnel_url and not proc_dead:
                    if _queue(inst.tunnel_url) is None:
                        inst.health_fails += 1
                        unhealthy = inst.health_fails >= HEALTH_FAIL_CHECKS
                    else:
                        inst.health_fails = 0
                if proc_dead or stuck or unhealthy:
                    reason = "dead" if proc_dead else ("unhealthy" if unhealthy else "stuck")
                    print(f"[pool] {inst.name} removed ({reason}) → relaunching", flush=True)
                    _stop(inst)
                    instances.remove(inst)
                    # Back off only on STARTUP failures (never became ready). A
                    # healthy instance that later died should relaunch promptly.
                    if not inst.tunnel_url:
                        fail_count += 1
                        blocked_until = time.time() + min(30 * fail_count, 300)

            # any healthy instance → reset the failure backoff
            if any(i.tunnel_url for i in instances):
                fail_count = 0

            if floor == 0:
                if instances:
                    for inst in instances:
                        _stop(inst)
                    instances = []
                    _write_nginx([])
                    _write_prometheus_targets([])
                    print("[pool] floor=0 → all stopped", flush=True)
                _write_status(instances, "sleeping")
                time.sleep(POLL_S)
                continue

            # floor >= 1 but launches keep failing → give up, go to sleep
            if fail_count >= 5:
                print("[pool] too many launch failures → floor=0 "
                      "(check colab CLI / launch_vllm.py path / ADC)", flush=True)
                for inst in instances:
                    _stop(inst)
                instances = []
                _write_nginx([])
                _write_prometheus_targets([])
                _set_floor(0)
                _write_status(instances, "launch_failed")
                time.sleep(POLL_S)
                continue

            # in backoff window after a failure → wait
            if time.time() < blocked_until:
                _write_status(instances, "backoff")
                time.sleep(POLL_S)
                continue

            # floor >= 1: ensure at least one
            if not instances:
                instances.append(_launch("pool-1"))
                last_scale = time.time()
                last_activity = time.time()
                _write_status(instances, "starting")
                time.sleep(POLL_S)
                continue

            ready = [i for i in instances if i.tunnel_url]

            # load + activity (queue>0 counts as activity → resets idle timer)
            queues = [q for i in ready if (q := _queue(i.tunnel_url)) is not None]
            avg_q = (sum(queues) / len(queues)) if queues else 0.0
            if avg_q > 0:
                last_activity = time.time()
            idle = time.time() - last_activity
            cooled = time.time() - last_scale > COOLDOWN_S
            all_ready = len(ready) == len(instances)

            if (len(instances) < POOL_MAX and avg_q > SCALE_UP_QUEUE
                    and cooled and all_ready):
                instances.append(_launch(f"pool-{len(instances) + 1}"))
                last_scale = time.time()
                print(f"[pool] SCALE UP → {len(instances)} (avg_q={avg_q:.1f})", flush=True)

            elif len(instances) > 1 and idle > SCALE_DOWN_IDLE_S and cooled:
                _stop(instances.pop())
                last_scale = time.time()
                print(f"[pool] SCALE DOWN → {len(instances)} (idle {int(idle)}s)", flush=True)

            elif (AUTO_SLEEP_IDLE_S > 0 and len(instances) == 1
                    and idle > AUTO_SLEEP_IDLE_S):
                # AUTO_SLEEP_IDLE_S=0 disables this → pool stays at floor 1 until
                # an explicit /deploy teardown (desired_floor=0).
                _stop(instances.pop())
                _set_floor(0)   # persist auto-sleep so it stays down until next Deploy
                floor = 0
                print(f"[pool] AUTO-SLEEP → 0 (idle {int(idle)}s)", flush=True)

            ready = [i for i in instances if i.tunnel_url]
            _write_nginx([i.tunnel_url for i in ready])
            _write_prometheus_targets(ready)
            status = "active" if ready else ("sleeping" if floor == 0 else "starting")
            _write_status(instances, status)

        except Exception as exc:  # noqa: BLE001 — daemon must not die
            print(f"[pool] loop error: {exc}", flush=True)

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
