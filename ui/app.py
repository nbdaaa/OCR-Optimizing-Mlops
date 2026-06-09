"""
OCR-MLOps Control UI — Streamlit entry point.

Run locally:  streamlit run ui/app.py
Talks to the FastAPI on the VM (configurable in the sidebar).

Uses st.navigation so the `views/` folder is NOT auto-discovered as pages
(avoids Streamlit's magic pages/ behaviour). Sidebar config is rendered here,
before pg.run(), so it appears on every page and populates session_state first.
"""
import streamlit as st

from common import render_sidebar_config

st.set_page_config(page_title="OCR-MLOps", page_icon="🧩", layout="wide")

render_sidebar_config()

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon="📊"),
    st.Page("views/data.py",      title="Data",      icon="🗂️"),
    st.Page("views/training.py",  title="Training",  icon="🏋️"),
    st.Page("views/models.py",    title="Models",    icon="📦"),
    st.Page("views/cicd.py",      title="CI/CD Gate", icon="✅"),
    st.Page("views/monitor.py",   title="Monitor",   icon="📈"),
]

st.navigation(pages).run()
