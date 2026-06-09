"""Monitor — embed the existing Grafana dashboards."""
import streamlit as st
import streamlit.components.v1 as components

from common import get_grafana_url

st.title("📈 Monitor")

grafana = get_grafana_url().rstrip("/")

# Link to the prebuilt OCR Inference dashboard (uid: ocr-inference)
dash_url = f"{grafana}/d/ocr-inference/ocr-inference?kiosk"

st.markdown(f"[Open Grafana in a new tab]({grafana}) · "
            f"[OCR Inference dashboard]({dash_url})")
st.caption(
    "If the embed below is blank, enable GF_SECURITY_ALLOW_EMBEDDING=true on the "
    "grafana service and restart it (Grafana blocks iframe embedding by default)."
)

components.iframe(dash_url, height=900, scrolling=True)
