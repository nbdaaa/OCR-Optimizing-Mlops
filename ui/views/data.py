"""Data versions — list, inspect, create."""
import streamlit as st

from api import api_get, api_post

st.title("🗂️ Data versions")

# ── Create ────────────────────────────────────────────────────────────────────
with st.expander("➕ Create a new version", expanded=False):
    with st.form("create_version"):
        version = st.text_input("Version name", placeholder="v1")
        max_samples = st.number_input(
            "Max samples (empty = all)", min_value=1, value=None, step=100
        )
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

st.divider()

# ── List + detail ───────────────────────────────────────────────────────────
ok, data = api_get("/data/versions")
if not ok:
    st.error(data)
    st.stop()

versions = data.get("versions", [])
if not versions:
    st.caption("No data versions yet.")
    st.stop()

st.write(f"**{len(versions)}** versions: {', '.join(versions)}")

selected = st.selectbox("Inspect version", versions)
if selected:
    ok, meta = api_get(f"/data/versions/{selected}")
    if not ok:
        st.error(meta)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Count", meta.get("count"))
        c2.metric("Offset", meta.get("offset"))
        c3.metric("Split", meta.get("split"))
        st.caption(f"hf_repo: {meta.get('hf_repo')} · created: {meta.get('created_at')}")
        st.subheader("Filter stats")
        st.json(meta.get("filter_stats", {}))
