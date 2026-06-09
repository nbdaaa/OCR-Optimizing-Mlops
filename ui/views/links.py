"""
Links — open the existing tool UIs for viewing/inspection. We don't rebuild
what MLflow / MinIO / Grafana / W&B already display well; this app only owns
the pipeline *actions* (see Pipeline page).
"""
import streamlit as st

from common import service_links

st.title("🔗 Links")
st.caption("Xem & inspect ở các tool sẵn có — UI này chỉ lo phần *trigger*.")

for name, (url, desc) in service_links().items():
    st.markdown(f"### [{name}]({url})")
    st.caption(desc)
    st.code(url, language=None)
    st.divider()
