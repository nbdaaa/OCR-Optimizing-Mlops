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
AUTO_SLEEP_IDLE_S = int(os.environ.get("AUTO_SLEEP_IDLE_S", "600"))   # 1->0
COOLDOWN_S        = int(os.environ.get("COOLDOWN_S", "240"))
POLL_S            = int(os.environ.get("POLL_INTERVAL_S", "15"))
SESSION_TIMEOUT   = os.environ.get("COLAB_SESSION_TIMEOUT", "21600")
STATE_FILE        = os.environ.get("SERVE_STATE_FILE", "/opt/ocr/serve_state.json")
NGINX_CONF        = os.environ.get("NGINX_LB_CONF", "/opt/ocr/nginx/ocr_lb.conf")
NGINX_CONTAINER   = os.environ.get("NGINX_CONTAINER", "infra-nginx-1")
COLAB             = os.environ.get("COLAB_CLI", "colab")
LOG_DIR           = os.environ.get("POOL_LOG_DIR", "/opt/ocr/logs")
READY_GIVEUP_S    = int(os.environ.get("READY_GIVEUP_S", "600"))      # kill instance if no tunnel in time

_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


@dataclass
class Instance:
    name: str
    proc: subprocess.Popen
    log_path: str
    started_at: float
    tunnel_url: str | None = None


# ── Colab session control ─────────────────────────────────────────────────────

def _launch(name: str) -> Instance:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/{name}.log"
    cmd = [COLAB, "--auth", "adc", "run", "--gpu", GPU, "--keep",
           "-s", name, "--timeout", SESSION_TIMEOUT, LAUNCH, HF_TOKEN]
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    print(f"[pool] launching {name} (gpu={GPU})", flush=True)
    return Instance(name=name, proc=proc, log_path=log_path, started_at=time.time())


def _resolve_tunnel(inst: Instance) -> None:
    if inst.tunnel_url:
        return
    try:
        m = _TUNNEL_RE.search(open(inst.log_path).read())
        if m:
            inst.tunnel_url = m.group(0)
            print(f"[pool] {inst.name} ready → {inst.tunnel_url}", flush=True)
    except FileNotFoundError:
        pass


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

def _read_floor() -> int:
    try:
        return int(json.load(open(STATE_FILE)).get("desired_floor", 0))
    except Exception:
        return 0


def _write_state(instances: list[Instance], floor: int, status: str) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    json.dump({
        "desired_floor": floor,
        "status": status,
        "lb_url": os.environ.get("PUBLIC_LB_URL", ""),
        "instances": [
            {"name": i.name, "tunnel_url": i.tunnel_url,
             "ready": bool(i.tunnel_url), "age_s": int(time.time() - i.started_at)}
            for i in instances
        ],
        "updated_at": int(time.time()),
    }, open(STATE_FILE, "w"), indent=2)


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

            # reconcile: drop instances whose process died, or that never became
            # ready within the giveup window
            for inst in list(instances):
                _resolve_tunnel(inst)
                dead = inst.proc.poll() is not None and not inst.tunnel_url
                stuck = (not inst.tunnel_url
                         and time.time() - inst.started_at > READY_GIVEUP_S)
                if dead or stuck:
                    print(f"[pool] {inst.name} failed to start ({'dead' if dead else 'stuck'})",
                          flush=True)
                    _stop(inst)
                    instances.remove(inst)
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
                    print("[pool] floor=0 → all stopped", flush=True)
                _write_state(instances, 0, "sleeping")
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
                _write_state(instances, 0, "launch_failed")
                time.sleep(POLL_S)
                continue

            # in backoff window after a failure → wait
            if time.time() < blocked_until:
                _write_state(instances, floor, "backoff")
                time.sleep(POLL_S)
                continue

            # floor >= 1: ensure at least one
            if not instances:
                instances.append(_launch("pool-1"))
                last_scale = time.time()
                last_activity = time.time()
                _write_state(instances, floor, "starting")
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

            elif len(instances) == 1 and idle > AUTO_SLEEP_IDLE_S:
                _stop(instances.pop())
                floor = 0   # persist auto-sleep so it stays down until next Deploy
                print(f"[pool] AUTO-SLEEP → 0 (idle {int(idle)}s)", flush=True)

            ready = [i for i in instances if i.tunnel_url]
            _write_nginx([i.tunnel_url for i in ready])
            status = "active" if ready else ("sleeping" if floor == 0 else "starting")
            _write_state(instances, floor, status)

        except Exception as exc:  # noqa: BLE001 — daemon must not die
            print(f"[pool] loop error: {exc}", flush=True)

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
