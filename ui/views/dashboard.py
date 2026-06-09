"""Dashboard — pipeline state at a glance."""
import pandas as pd
import streamlit as st

from api import api_get

st.title("📊 Dashboard")

ok_d, data = api_get("/data/versions")
ok_m, models = api_get("/models/versions")
ok_dep, deploy = api_get("/deploy/status")

data_versions = data.get("versions", []) if ok_d else []
model_versions = models.get("versions", []) if ok_m else []
instances = deploy.get("instances", []) if ok_dep else []

# Latest / production model version
latest = max(model_versions, key=lambda v: v["version"]) if model_versions else None
production = next((v for v in model_versions if v.get("stage") == "Production"), None)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Data versions", len(data_versions))
c2.metric("Model versions", len(model_versions))
c3.metric("Latest version", latest["version"] if latest else "—")
c4.metric("Serving instances", len(instances))

if production:
    prod_cer = (production.get("metrics") or {}).get("cer")
    st.info(f"**Production**: version {production['version']}"
            + (f" · CER {prod_cer:.4f}" if prod_cer is not None else ""))
else:
    st.warning("No Production version yet.")

st.divider()

if model_versions:
    rows = []
    for v in sorted(model_versions, key=lambda v: v["version"]):
        rows.append({
            "version": v["version"],
            "stage": v.get("stage"),
            "cer": (v.get("metrics") or {}).get("cer"),
            "run_id": v.get("run_id", "")[:8],
        })
    df = pd.DataFrame(rows)

    st.subheader("CER by version")
    cer_df = df.dropna(subset=["cer"]).set_index("version")[["cer"]]
    if not cer_df.empty:
        st.line_chart(cer_df)
    else:
        st.caption("No CER metrics logged yet.")

    st.subheader("Model versions")
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("No model versions found. Check the API connection in the sidebar.")
