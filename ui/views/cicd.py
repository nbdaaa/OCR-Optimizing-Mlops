"""CI/CD gate — evaluate Staging vs threshold/Production, transition stage."""
import streamlit as st

from api import api_post

st.title("✅ CI/CD Gate")

st.caption(
    "Reads the Staging version's CER, compares to the threshold and (if present) "
    "the Production version, then transitions it to Production or Archived."
)

with st.form("run_gate"):
    c1, c2 = st.columns(2)
    threshold = c1.number_input("cer_threshold", min_value=0.0, value=0.15, format="%.4f")
    tolerance = c2.number_input("regression_tolerance", min_value=1.0, value=1.05, format="%.2f")
    run = st.form_submit_button("Run gate")

if run:
    ok, res = api_post("/cicd/gate", json={
        "cer_threshold": float(threshold),
        "regression_tolerance": float(tolerance),
    })
    if not ok:
        st.error(res)
    else:
        result = res.get("result")
        if result == "pass":
            st.success(f"PASS → {res.get('new_stage')}", icon="🟢")
        else:
            st.error(f"{result.upper()} → {res.get('new_stage')}", icon="🔴")
        c1, c2, c3 = st.columns(3)
        c1.metric("Staging version", res.get("staging_version"))
        c2.metric("Staging CER", f"{res.get('staging_cer'):.4f}")
        prod = res.get("production_cer")
        c3.metric("Production CER", f"{prod:.4f}" if prod is not None else "—")
