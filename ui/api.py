"""
Thin HTTP client around the OCR-MLOps FastAPI.

Every call returns (ok: bool, data_or_error) so pages can render errors gracefully
instead of crashing on exceptions. The API base URL is read from session_state
(set by common.render_sidebar_config) so it tracks the VM's dynamic IP.
"""
from __future__ import annotations

import requests
import streamlit as st

_DEFAULT_TIMEOUT = 30


def _base() -> str:
    return st.session_state.get("api_base", "").rstrip("/")


def _error_text(exc: Exception) -> str:
    # Surface FastAPI's {"detail": ...} body when present
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            return f"{resp.status_code}: {resp.json().get('detail', resp.text)}"
        except Exception:
            return f"{resp.status_code}: {resp.text}"
    return str(exc)


def api_get(path: str, params: dict | None = None, timeout: int = _DEFAULT_TIMEOUT):
    try:
        r = requests.get(f"{_base()}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as exc:  # noqa: BLE001
        return False, _error_text(exc)


def api_post(path: str, json: dict | None = None, timeout: int = _DEFAULT_TIMEOUT):
    try:
        r = requests.post(f"{_base()}{path}", json=json, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as exc:  # noqa: BLE001
        return False, _error_text(exc)
