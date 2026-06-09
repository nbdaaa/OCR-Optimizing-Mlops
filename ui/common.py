"""
Shared sidebar config + helpers. The VM IP is dynamic, so API base and Grafana
URL are editable in the sidebar and persisted in session_state.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import streamlit as st

from api import api_get

DEFAULT_API = os.environ.get("OCR_API_BASE", "http://34.142.198.19:8000")
DEFAULT_GRAFANA = os.environ.get("OCR_GRAFANA_URL", "http://34.142.198.19:3000")
DEFAULT_WANDB = os.environ.get(
    "OCR_WANDB_URL",
    "https://wandb.ai/ducanhcttp-hanoi-university-of-science-and-technology/chandra-ocr",
)


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


def _api_host() -> str:
    return urlparse(st.session_state.get("api_base", DEFAULT_API)).hostname or "localhost"


def service_links() -> dict[str, tuple[str, str]]:
    """Existing tool UIs to view/inspect (derived from the API host). {name: (url, desc)}."""
    h = _api_host()
    return {
        "MLflow": (f"http://{h}:5000",
                   "Experiments, runs, metrics/params, Model Registry (versions, stages, lineage)"),
        "MinIO console": (f"http://{h}:9001",
                          "Browse buckets: ocr-data (data versions) + mlflow-artifacts (adapters)"),
        "Grafana": (get_grafana_url(),
                    "Serving dashboards: latency, throughput, GPU, request rate"),
        "Prometheus": (f"http://{h}:9090", "Raw metrics + query"),
        "Weights & Biases": (st.session_state.get("wandb_url", DEFAULT_WANDB),
                             "Live training curves (loss, CER) per run"),
    }
