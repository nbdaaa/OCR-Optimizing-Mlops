"""Training — trigger jobs, track status, view logs."""
import streamlit as st

from api import api_get, api_post

st.title("🏋️ Training")

st.session_state.setdefault("training_jobs", [])  # job_ids triggered this session

# ── Trigger ────────────────────────────────────────────────────────────────
ok_d, data = api_get("/data/versions")
ok_m, models = api_get("/models/versions")
data_versions = data.get("versions", []) if ok_d else []
model_versions = sorted(
    (str(v["version"]) for v in models.get("versions", [])), key=int, reverse=True
) if ok_m else []

with st.form("trigger_training"):
    st.subheader("Trigger a training job")
    dv = st.selectbox("Data version", data_versions) if data_versions else st.text_input("Data version")
    init_v = st.selectbox(
        "Warm-start from model version", ["latest (default)"] + model_versions
    )
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("num_epochs", min_value=1, value=None, step=1)
    batch = c2.number_input("batch_size", min_value=1, value=None, step=1)
    grad = c3.number_input("grad_accum", min_value=1, value=None, step=1)
    lr = c4.number_input("learning_rate", min_value=0.0, value=None, format="%.6f")
    go = st.form_submit_button("🚀 Trigger")

if go:
    body = {"data_version": dv}
    if init_v and not init_v.startswith("latest"):
        body["init_adapter_version"] = init_v
    if epochs is not None: body["num_epochs"] = int(epochs)
    if batch is not None:  body["batch_size"] = int(batch)
    if grad is not None:   body["grad_accum"] = int(grad)
    if lr:                 body["learning_rate"] = float(lr)
    ok, res = api_post("/training/trigger", json=body)
    if ok:
        job_id = res["job_id"]
        if job_id not in st.session_state.training_jobs:
            st.session_state.training_jobs.insert(0, job_id)
        st.success(f"Triggered · job_id = {job_id}")
    else:
        st.error(res)

st.divider()

# ── Track ─────────────────────────────────────────────────────────────────
st.subheader("Track a job")
job_id = st.selectbox(
    "Job id", st.session_state.training_jobs, index=0 if st.session_state.training_jobs else None,
    placeholder="select a triggered job",
) or st.text_input("…or paste a job_id")

if job_id:
    cols = st.columns([1, 1, 4])
    if cols[0].button("🔄 Refresh"):
        st.rerun()

    ok, status = api_get(f"/training/{job_id}/status")
    if not ok:
        st.error(status)
    else:
        s = status.get("status")
        badge = {"running": "🟡", "completed": "🟢", "failed": "🔴"}.get(s, "⚪")
        st.write(f"{badge} **{s}** · instance: {status.get('vast_instance_id') or '—'}")
        links = []
        if status.get("mlflow_run_url"): links.append(f"[MLflow]({status['mlflow_run_url']})")
        if status.get("wandb_run_url"):  links.append(f"[W&B]({status['wandb_run_url']})")
        if links:
            st.markdown(" · ".join(links))
        if status.get("metrics"):
            st.json(status["metrics"])

    st.subheader("Logs")
    st.caption("Raw onstart stdout via Vast (≈1 min lag). Use W&B for live metrics.")
    tail = st.slider("tail lines", 50, 1000, 200, step=50)
    if st.button("Fetch logs"):
        with st.spinner("Fetching logs…"):
            ok, logs = api_get(f"/training/{job_id}/logs", params={"tail": tail}, timeout=60)
        if ok:
            st.code(logs.get("logs", "") or "(empty)")
        else:
            st.error(logs)
