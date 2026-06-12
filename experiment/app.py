"""
OCR Experiment UI (Gradio) — separate from the Streamlit pipeline-control app.

Runs locally on its own port; talks to the same VM:
  - inference  → the serving LB  (OpenAI-compatible /v1/chat/completions)
  - registry   → the FastAPI     (/models/versions, /deploy/status)

Tabs:
  1. Playground       — upload image → DocTags + rendered Markdown/HTML + latency/tokens
  2. Versions         — CER & benchmark_loss across model versions (continual curve)
  3. Base vs Adapter  — same image through base model vs LoRA adapter, side by side

Run:  python experiment/app.py
Env:  LB_URL (default http://34.142.198.19), API_BASE (default :8000)
"""
from __future__ import annotations

import base64
import io
import os
import time

import gradio as gr
import pandas as pd
import requests
from PIL import Image

LB_URL   = os.environ.get("LB_URL", "http://34.142.198.19").rstrip("/")
API_BASE = os.environ.get("API_BASE", "http://34.142.198.19:8000").rstrip("/")
BASE_MODEL  = "ibm-granite/granite-docling-258M"
ADAPTER     = "ocr-adapter"
PROMPT      = "Convert this page to docling format."


# ── inference ─────────────────────────────────────────────────────────────────

def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _infer(img: Image.Image, model: str, max_tokens: int):
    """Call the LB. Returns (doctags, latency_s, completion_tokens) or raises."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_uri(img)}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": int(max_tokens), "temperature": 0.0,
        "skip_special_tokens": False,   # keep DocTags element tags
    }
    t0 = time.time()
    r = requests.post(f"{LB_URL}/v1/chat/completions", json=body, timeout=300)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"]
    toks = (j.get("usage") or {}).get("completion_tokens", 0)
    return text, dt, toks


def _render(doctags: str, img: Image.Image) -> str:
    """DocTags → HTML via docling-core; fall back to a <pre> dump."""
    try:
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
        dt = DocTagsDocument.from_doctags_and_image_pairs([doctags], [img.convert("RGB")])
        doc = DoclingDocument.load_from_doctags(dt, document_name="page")
        return doc.export_to_html()
    except Exception as exc:  # noqa: BLE001
        return f"<p style='color:#b00'>render lỗi ({exc}); hiển thị raw:</p><pre>{doctags}</pre>"


# ── Tab 1: Playground ─────────────────────────────────────────────────────────

def playground(img, max_tokens):
    if img is None:
        return "Hãy upload ảnh.", "", ""
    try:
        doctags, dt, toks = _infer(img, ADAPTER, max_tokens)
    except Exception as exc:  # noqa: BLE001
        return f"Lỗi gọi serving: {exc}", "", ""
    tps = toks / dt if dt else 0
    meta = f"⏱️ {dt:.1f}s · {toks} tokens · {tps:.1f} tok/s · model={ADAPTER}"
    return meta, doctags, _render(doctags, img)


# ── Tab 2: Versions ───────────────────────────────────────────────────────────

def load_versions():
    try:
        r = requests.get(f"{API_BASE}/models/versions", timeout=30)
        r.raise_for_status()
        vs = r.json().get("versions", [])
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame([{"error": str(exc)}]), pd.DataFrame()
    rows = []
    for v in vs:
        m = v.get("metrics") or {}
        rows.append({"version": v.get("version"), "stage": v.get("stage"),
                     "cer": m.get("cer"), "benchmark_loss": m.get("benchmark_loss"),
                     "data_version": (v.get("params") or {}).get("data_version")})
    df = pd.DataFrame(rows).sort_values("version")
    chart = df[["version", "cer"]].dropna()
    return df, chart


# ── Tab 3: Base vs Adapter ────────────────────────────────────────────────────

def compare(img, max_tokens):
    if img is None:
        return "Hãy upload ảnh.", "", "", ""
    out = {}
    for label, model in (("base", BASE_MODEL), ("adapter", ADAPTER)):
        try:
            dtags, dt, toks = _infer(img, model, max_tokens)
            out[label] = (f"⏱️ {dt:.1f}s · {toks} tok · {toks/dt if dt else 0:.1f} tok/s",
                          _render(dtags, img))
        except Exception as exc:  # noqa: BLE001
            out[label] = (f"lỗi: {exc}", "")
    return out["base"][0], out["base"][1], out["adapter"][0], out["adapter"][1]


# ── UI ──────────────────────────────────────────────────────────────────────--

with gr.Blocks(title="OCR Experiment") as demo:
    gr.Markdown(f"# 🔬 OCR Experiment\nServing LB: `{LB_URL}` · API: `{API_BASE}`")

    with gr.Tab("Playground"):
        with gr.Row():
            with gr.Column():
                pg_img = gr.Image(type="pil", label="Ảnh tài liệu")
                pg_tok = gr.Slider(256, 4096, value=2048, step=128, label="max_tokens")
                pg_btn = gr.Button("Convert → DocTags", variant="primary")
            with gr.Column():
                pg_meta = gr.Markdown()
                pg_html = gr.HTML(label="Rendered")
        with gr.Accordion("DocTags (raw)", open=False):
            pg_raw = gr.Code(label="raw")
        pg_btn.click(playground, [pg_img, pg_tok], [pg_meta, pg_raw, pg_html])

    with gr.Tab("Versions"):
        v_btn = gr.Button("Tải bảng version + CER", variant="primary")
        v_tbl = gr.Dataframe(label="Model versions")
        v_plot = gr.LinePlot(x="version", y="cer", label="CER theo version (thấp hơn = tốt hơn)")
        v_btn.click(load_versions, None, [v_tbl, v_plot])

    with gr.Tab("Base vs Adapter"):
        with gr.Row():
            cmp_img = gr.Image(type="pil", label="Ảnh")
            cmp_tok = gr.Slider(256, 4096, value=2048, step=128, label="max_tokens")
        cmp_btn = gr.Button("So sánh", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base (chưa fine-tune)")
                cmp_bmeta = gr.Markdown(); cmp_bhtml = gr.HTML()
            with gr.Column():
                gr.Markdown("### Adapter (LoRA)")
                cmp_ameta = gr.Markdown(); cmp_ahtml = gr.HTML()
        cmp_btn.click(compare, [cmp_img, cmp_tok],
                      [cmp_bmeta, cmp_bhtml, cmp_ameta, cmp_ahtml])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("EXPERIMENT_PORT", "7860")))
