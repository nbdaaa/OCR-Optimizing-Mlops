"""Model registry — versions, stages, CER, params."""
import pandas as pd
import streamlit as st

from api import api_get

st.title("📦 Model registry")

ok, data = api_get("/models/versions")
if not ok:
    st.error(data)
    st.stop()

versions = data.get("versions", [])
if not versions:
    st.caption("No model versions yet.")
    st.stop()

_STAGE_BADGE = {"Production": "🟢", "Staging": "🟡", "Archived": "⚫", "None": "⚪"}

rows = []
for v in sorted(versions, key=lambda v: v["version"], reverse=True):
    m = v.get("metrics") or {}
    p = v.get("params") or {}
    rows.append({
        "version": v["version"],
        "stage": f"{_STAGE_BADGE.get(v.get('stage'), '')} {v.get('stage')}",
        "cer": m.get("cer"),
        "data_version": p.get("data_version"),
        "created_at": v.get("created_at"),
        "run_id": v.get("run_id", "")[:8],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
sel = st.selectbox("Inspect version", [v["version"] for v in
                                       sorted(versions, key=lambda v: v["version"], reverse=True)])
if sel is not None:
    ok, info = api_get(f"/models/versions/{sel}")
    if not ok:
        st.error(info)
    else:
        st.write(f"**Version {info['version']}** · stage **{info.get('stage')}** · run `{info.get('run_id')}`")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Metrics")
            st.json(info.get("metrics") or {})
        with c2:
            st.subheader("Params")
            st.json(info.get("params") or {})
