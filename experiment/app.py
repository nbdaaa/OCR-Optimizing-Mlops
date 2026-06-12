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
    text = _fix_vn(j["choices"][0]["message"]["content"])
    toks = (j.get("usage") or {}).get("completion_tokens", 0)
    return text, dt, toks


# This model emits Vietnamese tone marks as SPACING characters (´ U+00B4, ` U+0060,
# …) instead of combining marks, so plain NFC can't compose them ("PHỐ´", "vê`").
# Map the spacing marks → their combining equivalents, then NFC composes them
# ("Ô"+◌́ → "Ố", "ê"+◌̀ → "ề").
_SPACING_TONE = str.maketrans({
    "´": "́", "ˊ": "́",   # acute  (sắc)
    "`": "̀", "ˋ": "̀",   # grave  (huyền)
    "˜": "̃",                        # tilde  (ngã)
    "ˇ": "̉",                        # hook above (hỏi) — best-effort
})

def _fix_vn(text: str) -> str:
    return unicodedata.normalize("NFC", text.translate(_SPACING_TONE))


_RENDER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif:ital,wght@0,400;0,700;1,400&display=swap');
/* Noto Serif / Times New Roman have FULL Vietnamese glyphs; Georgia does NOT
   (it renders Ố/Ế/Ầ as base+detached mark) — that was the "PHỐ´" bug. */
.doc-render{max-width:820px;margin:0 auto;padding:32px 40px;background:#fff;color:#1a1a1a;
  font-family:'Noto Serif','Times New Roman',serif;line-height:1.55;box-shadow:0 1px 6px rgba(0,0,0,.15);}
.doc-render h1{font-size:1.5em;font-weight:700;margin:.2em 0 .6em;}
.doc-render h2,.doc-render h3{font-weight:700;margin:1em 0 .4em;}
.doc-render p{margin:.5em 0;text-align:justify;}
.doc-render ul,.doc-render ol{margin:.5em 0 .5em 1.4em;}
.doc-render table{border-collapse:collapse;margin:.8em 0;width:100%;}
.doc-render th,.doc-render td{border:1px solid #bbb;padding:4px 8px;font-size:.9em;}
</style>
"""

def _render(doctags: str, img: Image.Image | None = None) -> str:
    """
    Render DocTags → styled HTML with a custom converter.

    This model emits element tags (text/section_header/list_item/caption/...) with
    <loc> coords, and TABLES as literal HTML (<table border="1">...), NOT OTSL — so
    docling-core can't parse them. We convert element tags to HTML and keep table
    HTML as-is (browsers render it natively).
    """
    s = doctags
    s = re.sub(r"</?doctag>", "", s)
    s = re.sub(r"<loc_\d+>", "", s)                      # drop coordinate tokens
    # collapse the <table> doctag wrapper around the inner HTML table
    s = re.sub(r"<table>\s*(<table)", r"\1", s)
    s = re.sub(r"(</table>)\s*</table>", r"\1", s)
    # element tags → HTML
    s = re.sub(r"<section_header_level_\d+>(.*?)</section_header_level_\d+>",
               r"<h3>\1</h3>", s, flags=re.S)
    s = re.sub(r"<title>(.*?)</title>", r"<h1>\1</h1>", s, flags=re.S)
    s = re.sub(r"<caption>(.*?)</caption>", r'<p class="cap"><b>\1</b></p>', s, flags=re.S)
    s = re.sub(r"<text>(.*?)</text>", r"<p>\1</p>", s, flags=re.S)
    s = re.sub(r"<(page_header|page_footer)>(.*?)</\1>",
               r'<div class="meta">\2</div>', s, flags=re.S)
    s = re.sub(r"<unordered_list>(.*?)</unordered_list>", r"<ul>\1</ul>", s, flags=re.S)
    s = re.sub(r"<ordered_list>(.*?)</ordered_list>", r"<ol>\1</ol>", s, flags=re.S)
    s = re.sub(r"<list_item>(.*?)</list_item>", r"<li>\1</li>", s, flags=re.S)
    s = re.sub(r"<picture>(.*?)</picture>", r'<div class="fig">🖼 \1</div>', s, flags=re.S)
    s = re.sub(r"</?formula>", "", s)
    return _RENDER_CSS + f'<div class="doc-render">{s}</div>'


# DocTags loc tokens are on a 0-500 normalized grid. Draw a box per element,
# coloured by type, to mirror docling's layout visualisation (left panel).
_BOX_RE = re.compile(r"<([a-z_0-9]+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
_LOC_SCALE = 500.0
_TYPE_COLORS = {
    "section_header": (220, 40, 40), "title": (220, 40, 40),
    "text": (40, 110, 215), "list_item": (40, 160, 60),
    "caption": (210, 130, 20), "picture": (160, 40, 200),
    "table": (20, 170, 170), "otsl": (20, 170, 170),
    "page_footer": (130, 130, 130), "page_header": (130, 130, 130),
    "formula": (170, 90, 30),
}

def _color(tag: str):
    for key, col in _TYPE_COLORS.items():
        if tag.startswith(key):
            return col
    return (90, 90, 90)

# loc → pixel mapping uses per-axis grids (NOT 500). Calibrated on ground-truth
# training samples: this dataset's loc are normalized with different effective
# grids per axis. Tunable live in the UI.
GRID_X_DEFAULT = 345.0
GRID_Y_DEFAULT = 293.0

def _annotate(img: Image.Image, doctags: str,
              grid_x: float = GRID_X_DEFAULT, grid_y: float = GRID_Y_DEFAULT) -> Image.Image:
    """Overlay element bounding boxes (coloured by type). x_px=loc_x/grid_x*W."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    seen = set()
    for m in _BOX_RE.finditer(doctags):
        tag = m.group(1)
        x1, y1, x2, y2 = (int(v) for v in m.groups()[1:])
        key = (x1, y1, x2, y2)
        if key in seen:           # collapsed list_item boxes → draw the region once
            continue
        seen.add(key)
        box = [x1 / grid_x * W, y1 / grid_y * H, x2 / grid_x * W, y2 / grid_y * H]
        d.rectangle(box, outline=_color(tag), width=2)
    return im


# ── Tab 1: Playground ─────────────────────────────────────────────────────────

def playground(img, max_tokens, show_boxes, grid_x, grid_y):
    if img is None:
        return None, "Hãy upload ảnh.", "", ""
    try:
        doctags, dt, toks = _infer(img, ADAPTER, max_tokens)
    except Exception as exc:  # noqa: BLE001
        return None, f"Lỗi gọi serving: {exc}", "", ""
    tps = toks / dt if dt else 0
    W, H = img.size
    meta = (f"⏱️ {dt:.1f}s · {toks} tokens · {tps:.1f} tok/s · model={ADAPTER} · "
            f"ảnh {W}×{H}px · grid {grid_x:g}/{grid_y:g}")
    left = _annotate(img, doctags, grid_x, grid_y) if show_boxes else img
    return left, meta, doctags, _render(doctags, img)


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
        with gr.Row():
            pg_box = gr.Checkbox(value=True, label="Hiện bounding box")
            pg_gx = gr.Number(value=GRID_X_DEFAULT, label="grid_x (giảm → box rộng hơn)")
            pg_gy = gr.Number(value=GRID_Y_DEFAULT, label="grid_y (giảm → box cao hơn)")
        pg_btn = gr.Button("Convert → DocTags", variant="primary")
        pg_meta = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Layout")
                pg_boxes = gr.Image(label="Ảnh", height=720)
            with gr.Column():
                gr.Markdown("#### Rendered")
                pg_html = gr.HTML()
        with gr.Accordion("DocTags (raw)", open=False):
            pg_raw = gr.Code(label="raw")
        pg_btn.click(playground, [pg_img, pg_tok, pg_box, pg_gx, pg_gy],
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
