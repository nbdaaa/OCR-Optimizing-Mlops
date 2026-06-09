"""
Pipeline — the actions no existing tool provides (trigger data versioning,
training, CI gate). Viewing/inspection lives in MLflow / MinIO / Grafana / W&B
(see the Links page).
"""
import streamlit as st

from api import api_get, api_post

st.title("🎛️ Pipeline Actions")

tab_data, tab_train, tab_gate = st.tabs(["🗂️ Data", "🏋️ Training", "✅ CI/CD Gate"])

# ── Data: create a version ────────────────────────────────────────────────────
with tab_data:
    st.subheader("Create a data version")
    st.caption("Browse existing versions in the MinIO console (see Links).")
    with st.form("create_version"):
        version = st.text_input("Version name", placeholder="v1")
        max_samples = st.number_input("Max samples (empty = all)", min_value=1, value=None, step=100)
        submitted = st.form_submit_button("Create")
    if submitted:
        if not version:
            st.error("Version name is required.")
        else:
            body = {"version": version}
            if max_samples is not None:
                body["max_samples"] = int(max_samples)
            with st.spinner("Creating version (download → filter → dedup → upload)…"):
                ok, res = api_post("/data/versions/create", json=body)
            if ok:
                st.success(f"Created {res['version']} · run {res['mlflow_run_id'][:8]}")
            else:
                st.error(res)

# ── Training: trigger + track ────────────────────────────────────────────────
with tab_train:
    st.session_state.setdefault("training_jobs", [])

    ok_d, data = api_get("/data/versions")
    ok_m, models = api_get("/models/versions")
    data_versions = data.get("versions", []) if ok_d else []
    model_versions = sorted(
        (str(v["version"]) for v in models.get("versions", [])), key=int, reverse=True
    ) if ok_m else []

    with st.form("trigger_training"):
        st.subheader("Trigger a training job")
        dv = st.selectbox("Data version", data_versions) if data_versions else st.text_input("Data version")
        init_v = st.selectbox("Warm-start from model version", ["latest (default)"] + model_versions)
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
    st.subheader("Track a job")
    job_id = st.selectbox(
        "Job id", st.session_state.training_jobs,
        index=0 if st.session_state.training_jobs else None,
        placeholder="select a triggered job",
    ) or st.text_input("…or paste a job_id")

    if job_id:
        if st.button("🔄 Refresh"):
            st.rerun()
        ok, status = api_get(f"/training/{job_id}/status")
        if not ok:
            st.error(status)
        else:
            badge = {"running": "🟡", "completed": "🟢", "failed": "🔴"}.get(status.get("status"), "⚪")
            st.write(f"{badge} **{status.get('status')}** · instance: {status.get('vast_instance_id') or '—'}")
            links = []
            if status.get("mlflow_run_url"): links.append(f"[MLflow]({status['mlflow_run_url']})")
            if status.get("wandb_run_url"):  links.append(f"[W&B]({status['wandb_run_url']})")
            if links:
                st.markdown(" · ".join(links))
            if status.get("metrics"):
                st.json(status["metrics"])

        st.caption("Raw onstart log via Vast (≈1 min lag). Use W&B for live metrics.")
        tail = st.slider("tail lines", 50, 1000, 200, step=50)
        if st.button("Fetch logs"):
            with st.spinner("Fetching logs…"):
                ok, logs = api_get(f"/training/{job_id}/logs", params={"tail": tail}, timeout=60)
            if ok:
                st.code(logs.get("logs", "") or "(empty)")
            else:
                st.error(logs)

# ── CI/CD gate ────────────────────────────────────────────────────────────────
with tab_gate:
    st.subheader("Run CI/CD gate")
    st.caption("Reads Staging CER, compares to threshold + Production, transitions stage.")
    with st.form("run_gate"):
        c1, c2 = st.columns(2)
        threshold = c1.number_input("cer_threshold", min_value=0.0, value=0.15, format="%.4f")
        tolerance = c2.number_input("regression_tolerance", min_value=1.0, value=1.05, format="%.2f")
        run = st.form_submit_button("Run gate")
    if run:
        ok, res = api_post("/cicd/gate", json={
            "cer_threshold": float(threshold), "regression_tolerance": float(tolerance),
        })
        if not ok:
            st.error(res)
        else:
            result = res.get("result")
            msg = f"{result.upper()} → {res.get('new_stage')}"
            if result == "pass":
                st.success(msg, icon="🟢")
            else:
                st.error(msg, icon="🔴")
            c1, c2, c3 = st.columns(3)
            c1.metric("Staging version", res.get("staging_version"))
            c2.metric("Staging CER", f"{res.get('staging_cer'):.4f}")
            prod = res.get("production_cer")
            c3.metric("Production CER", f"{prod:.4f}" if prod is not None else "—")
