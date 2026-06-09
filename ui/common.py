"""
Shared sidebar config + helpers. The VM IP is dynamic, so API base and Grafana
URL are editable in the sidebar and persisted in session_state.
"""
from __future__ import annotations

import os

import streamlit as st

from api import api_get

DEFAULT_API = os.environ.get("OCR_API_BASE", "http://34.142.198.19:8000")
DEFAULT_GRAFANA = os.environ.get("OCR_GRAFANA_URL", "http://34.142.198.19:3000")


def render_sidebar_config() -> None:
    """Render connection settings in the sidebar (shown on every page)."""
    st.session_state.setdefault("api_base", DEFAULT_API)
    st.session_state.setdefault("grafana_url", DEFAULT_GRAFANA)

    with st.sidebar:
        st.subheader("⚙️ Connection")
        st.session_state.api_base = st.text_input("API base", st.session_state.api_base)
        st.session_state.grafana_url = st.text_input("Grafana URL", st.session_state.grafana_url)

        ok, _ = api_get("/health", timeout=5)
        if ok:
            st.success("API reachable", icon="✅")
        else:
            st.error("API unreachable", icon="🚫")


def get_grafana_url() -> str:
    return st.session_state.get("grafana_url", DEFAULT_GRAFANA)
