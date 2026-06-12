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
import re
import time
import unicodedata

import gradio as gr
import pandas as pd
import requests
from PIL import Image, ImageDraw

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
    # Vietnamese diacritics often come back decomposed (NFD: â + ◌́) → compose to
    # NFC so "suâ´t" renders as "suất".
    text = unicodedata.normalize("NFC", text)
    toks = (j.get("usage") or {}).get("completion_tokens", 0)
    return text, dt, toks


_RENDER_CSS = """
<style>
.doc-render{max-width:820px;margin:0 auto;padding:32px 40px;background:#fff;color:#1a1a1a;
  font-family:Georgia,'Times New Roman',serif;line-height:1.55;box-shadow:0 1px 6px rgba(0,0,0,.15);}
.doc-render h1{font-size:1.5em;font-weight:700;margin:.2em 0 .6em;}
.doc-render h2,.doc-render h3{font-weight:700;margin:1em 0 .4em;}
.doc-render p{margin:.5em 0;text-align:justify;}
.doc-render ul,.doc-render ol{margin:.5em 0 .5em 1.4em;}
.doc-render table{border-collapse:collapse;margin:.8em 0;width:100%;}
.doc-render th,.doc-render td{border:1px solid #bbb;padding:4px 8px;font-size:.9em;}
</style>
"""

def _render(doctags: str, img: Image.Image) -> str:
    """DocTags → styled HTML via docling-core; fall back to a <pre> dump."""
    try:
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
        dt = DocTagsDocument.from_doctags_and_image_pairs([doctags], [img.convert("RGB")])
        doc = DoclingDocument.load_from_doctags(dt, document_name="page")
        return _RENDER_CSS + f'<div class="doc-render">{doc.export_to_html()}</div>'
    except Exception as exc:  # noqa: BLE001
        return f"<p style='color:#b00'>render lỗi ({exc}); raw:</p><pre>{doctags}</pre>"


# DocTags loc tokens are on a 0-500 normalized grid. Draw a box per element,
# coloured by type, to mirror docling's layout visualisation (left panel).
_BOX_RE = re.compile(r"<([a-z_0-9]+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
_LOC_SCALE = 500.0
_TYPE_COLORS = {
    "section_header": (220, 40, 40), "title": (220, 40, 40),
    "text": (40, 110, 215), "list_item": (40, 160, 60),
    "caption": (210, 130, 20), "picture": (160, 40, 200),
    "otsl": (20, 170, 170), "page_footer": (130, 130, 130),
    "page_header": (130, 130, 130), "formula": (170, 90, 30),
}

def _color(tag: str):
    for key, col in _TYPE_COLORS.items():
        if tag.startswith(key):
            return col
    return (90, 90, 90)

def _annotate(img: Image.Image, doctags: str) -> Image.Image:
    """Overlay element bounding boxes (coloured by type) on the original image."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    for m in _BOX_RE.finditer(doctags):
        tag = m.group(1)
        x1, y1, x2, y2 = (int(v) for v in m.groups()[1:])
        box = [x1 / _LOC_SCALE * W, y1 / _LOC_SCALE * H,
               x2 / _LOC_SCALE * W, y2 / _LOC_SCALE * H]
        d.rectangle(box, outline=_color(tag), width=2)
    return im


# ── Tab 1: Playground ─────────────────────────────────────────────────────────

def playground(img, max_tokens):
    if img is None:
        return None, "Hãy upload ảnh.", "", ""
    try:
        doctags, dt, toks = _infer(img, ADAPTER, max_tokens)
    except Exception as exc:  # noqa: BLE001
        return None, f"Lỗi gọi serving: {exc}", "", ""
    tps = toks / dt if dt else 0
    meta = f"⏱️ {dt:.1f}s · {toks} tokens · {tps:.1f} tok/s · model={ADAPTER}"
    return _annotate(img, doctags), meta, doctags, _render(doctags, img)


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
            pg_img = gr.Image(type="pil", label="Ảnh tài liệu", height=300)
            pg_tok = gr.Slider(256, 4096, value=2048, step=128, label="max_tokens")
        pg_btn = gr.Button("Convert → DocTags", variant="primary")
        pg_meta = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Layout (bounding boxes theo loại)")
                pg_boxes = gr.Image(label="Annotated", height=720)
            with gr.Column():
                gr.Markdown("#### Rendered")
                pg_html = gr.HTML()
        with gr.Accordion("DocTags (raw)", open=False):
            pg_raw = gr.Code(label="raw")
        pg_btn.click(playground, [pg_img, pg_tok],
                     [pg_boxes, pg_meta, pg_raw, pg_html])

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
