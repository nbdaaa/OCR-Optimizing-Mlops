"""
Training watchdog — auto-recovery for dead training instances (Phase 2 / B).

A long-lived daemon (separate from the API, like the autoscaler) that:
  1. Lists MLflow runs still in RUNNING state across the ocr-training (phase 1)
     and post-training (phase 2 bbox) experiments.
  2. For each run that has a `vast_instance_id` tag (set ONLY after the instance
     first reached "running" — so a run still provisioning its FIRST boot has no
     tag yet and is skipped, never mistaken for dead), polls the Vast.ai instance.
  3. If the instance is gone / not "running" for GRACE consecutive checks, it is
     considered dead and the watchdog calls POST /training/{run_id}/recover. The
     server-side resolver decides the stage (REGISTER_ONLY / FINALIZE / RESUME).
  4. Per-run retry cap stops runaway re-provisioning of a chronically failing job.

Run:  python -m src.serving.training_watchdog
"""
from __future__ import annotations

import os
import time

import requests

try:
    import mlflow
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mlflow is required for the watchdog") from exc


from src.training.config import EXPERIMENT_POST, EXPERIMENT_TRAIN

_VAST_BASE = "https://console.vast.ai/api/v0"
_EXPERIMENTS = (EXPERIMENT_TRAIN, EXPERIMENT_POST)

# Tunables (env-overridable)
_POLL_INTERVAL_S = int(os.environ.get("WATCHDOG_POLL_INTERVAL_S", "60"))
_GRACE_CHECKS = int(os.environ.get("WATCHDOG_GRACE_CHECKS", "3"))   # consecutive
_MAX_RETRIES = int(os.environ.get("WATCHDOG_MAX_RETRIES", "3"))     # per run
# After triggering recovery we wait for the run's vast_instance_id tag to change
# to the NEW instance before health-checking again — so we monitor the new box,
# never the old dead one. If provisioning never completes within this window
# (e.g. no offer found, recover_error), we stop waiting and re-evaluate (which
# may re-trigger, consuming a retry, or eventually hit the retry cap).
_READY_TIMEOUT_S = int(os.environ.get("VAST_READY_TIMEOUT_S", "900"))
_PROVISION_TIMEOUT_S = _READY_TIMEOUT_S + 300

_API_BASE = os.environ.get("WATCHDOG_API_BASE", "http://fastapi:8000")
_VAST_API_KEY = os.environ.get("VAST_API_KEY", "")


def _instance_state(instance_id: str) -> str:
    """
    Return the Vast instance state: "running", "dead" (gone / not running), or
    "unknown" (transient API/network error — don't count toward the grace limit).
    """
    try:
        resp = requests.get(
            f"{_VAST_BASE}/instances/{instance_id}/",
            params={"api_key": _VAST_API_KEY}, timeout=10,
        )
    except requests.RequestException:
        return "unknown"
    if resp.status_code != 200:
        # 404 → instance truly gone (definitive dead); other codes → inconclusive
        return "dead" if resp.status_code == 404 else "unknown"
    data = resp.json().get("instances")
    inst = (data[0] if data else {}) if isinstance(data, list) else (data or {})
    if not inst:
        return "dead"
    return "running" if inst.get("actual_status") == "running" else "dead"


def _trigger_recover(run_id: str) -> tuple[bool, str]:
    """Call the API's recover endpoint (reuses the server-side stage resolver)."""
    try:
        r = requests.post(f"{_API_BASE}/training/{run_id}/recover", json={}, timeout=30)
        if r.status_code == 200:
            body = r.json()
            return True, f"action={body.get('action')} provisioned={body.get('provisioned')}"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.RequestException as exc:
        return False, str(exc)


def run_forever() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.MlflowClient()
    print(f"[watchdog] started · poll={_POLL_INTERVAL_S}s grace={_GRACE_CHECKS} "
          f"max_retries={_MAX_RETRIES} api={_API_BASE}", flush=True)

    fail_count: dict[str, int] = {}
    retry_count: dict[str, int] = {}
    # While a recovery provisions, remember the dead instance we replaced and
    # when — we wait for the tag to point at a NEW instance, then monitor that.
    awaiting: dict[str, tuple[str, float]] = {}  # rid -> (old_instance_id, since)

    while True:
        try:
            exp_ids = [
                e.experiment_id
                for name in _EXPERIMENTS
                if (e := client.get_experiment_by_name(name)) is not None
            ]
            runs = client.search_runs(
                exp_ids, "attributes.status = 'RUNNING'", max_results=100,
            ) if exp_ids else []

            active = set()
            for run in runs:
                rid = run.info.run_id
                active.add(rid)

                iid = run.data.tags.get("vast_instance_id")
                if not iid:
                    continue  # still on its first boot — no tag yet, skip

                # A recovery for this run is in flight: wait for the new instance.
                pend = awaiting.get(rid)
                if pend is not None:
                    old_iid, since = pend
                    if iid != old_iid:
                        # tag now points at the freshly provisioned instance →
                        # resume monitoring IT (recover again if it dies too)
                        print(f"[watchdog] {rid[:8]} new instance {iid} up → resuming "
                              f"monitoring", flush=True)
                        awaiting.pop(rid)
                        fail_count[rid] = 0
                    elif time.time() - since > _PROVISION_TIMEOUT_S:
                        print(f"[watchdog] {rid[:8]} provisioning stuck "
                              f"(>{_PROVISION_TIMEOUT_S}s, still {old_iid}) → re-evaluating",
                              flush=True)
                        awaiting.pop(rid)  # fall through, may re-trigger / hit cap
                    else:
                        continue  # still provisioning the replacement — keep waiting

                state = _instance_state(iid)
                if state == "running":
                    fail_count[rid] = 0
                    continue
                if state == "unknown":
                    continue  # inconclusive — don't penalize

                # state == "dead"
                fail_count[rid] = fail_count.get(rid, 0) + 1
                if fail_count[rid] < _GRACE_CHECKS:
                    print(f"[watchdog] {rid[:8]} instance {iid} not running "
                          f"({fail_count[rid]}/{_GRACE_CHECKS})", flush=True)
                    continue

                # Confirmed dead
                if retry_count.get(rid, 0) >= _MAX_RETRIES:
                    print(f"[watchdog] {rid[:8]} exceeded {_MAX_RETRIES} retries "
                          f"→ marking FAILED", flush=True)
                    client.set_tag(rid, "watchdog", "max_retries_exceeded")
                    client.set_terminated(rid, status="FAILED")
                    fail_count[rid] = 0
                    continue

                print(f"[watchdog] {rid[:8]} DEAD → recovering "
                      f"(attempt {retry_count.get(rid, 0) + 1})", flush=True)
                ok, msg = _trigger_recover(rid)
                print(f"[watchdog] {rid[:8]} recover → {ok}: {msg}", flush=True)
                if ok:
                    retry_count[rid] = retry_count.get(rid, 0) + 1
                    fail_count[rid] = 0
                    # Wait for the tag to change to the new instance before the
                    # next health check (so we follow the new box, not the old one).
                    awaiting[rid] = (iid, time.time())

            # Drop bookkeeping for runs no longer RUNNING
            for d in (fail_count, retry_count, awaiting):
                for rid in list(d):
                    if rid not in active:
                        d.pop(rid, None)

        except Exception as exc:  # noqa: BLE001 — daemon must never die
            print(f"[watchdog] loop error: {exc}", flush=True)

        time.sleep(_POLL_INTERVAL_S)


if __name__ == "__main__":
    run_forever()
