"""
OCR Experiment UI (Gradio) — separate from the Streamlit pipeline-control app.

Runs locally on its own port; talks to the same VM:
  - inference  → the serving LB  (OpenAI-compatible /v1/chat/completions)
  - registry   → the FastAPI     (/models/versions, /deploy/status)

Tabs:
  1. Playground       — upload an image OR a PDF (split client-side, inferred page
                        by page) → per-page DocTags + rendered HTML + latency/tokens,
                        browsable with a page slider
  2. Base vs Adapter  — same image through base model vs LoRA adapter, side by side

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
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import requests
from PIL import Image, ImageDraw

LB_URL   = os.environ.get("LB_URL", "http://34.142.198.19").rstrip("/")
API_BASE = os.environ.get("API_BASE", "http://34.142.198.19:8000").rstrip("/")
BASE_MODEL  = "ibm-granite/granite-docling-258M"
ADAPTER     = "ocr-adapter"
PROMPT      = "Convert this page to docling format."
PDF_CONCURRENCY = 10   # pages inferred in parallel for a multi-page PDF
TEACHER_URL   = os.environ.get("TEACHER_URL", "").rstrip("/")   # base URL of the teacher (OpenAI-compatible)
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "chandra")


# ── inference ─────────────────────────────────────────────────────────────────

def _normalize(img: Image.Image) -> Image.Image:
    """Resize to _NORM_W width (keep aspect ratio) — matches training DPI distribution."""
    w, h = img.size
    if w == _NORM_W:
        return img.convert("RGB")
    return img.convert("RGB").resize((_NORM_W, round(h * _NORM_W / w)), Image.LANCZOS)

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

def _infer_at(img: Image.Image, base_url: str, model: str, max_tokens: int):
    """Like _infer but against an arbitrary OpenAI-compatible base URL (e.g. the teacher)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_uri(img)}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": int(max_tokens), "temperature": 0.0,
        "skip_special_tokens": False,
    }
    t0 = time.time()
    r = requests.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=body, timeout=300)
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


# DocTags loc tokens are on a 0-500 grid. Draw a box per element, coloured by type.
_BOX_RE = re.compile(r"<([a-z_0-9]+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
_NORM_W = 1240      # normalize input to this width before inference
# bbox mapping: pixel = loc / (K/dim) * dim. The ÷500 convention matches the
# ground-truth LABELS exactly, but MODEL INFERENCE output is on a different
# effective scale (granite-docling resizes internally) — empirically the K-grid
# tracks the predicted boxes better, so the playground uses it for visualisation.
_GRID_K = 500_000
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

def _annotate(img: Image.Image, doctags: str) -> Image.Image:
    """Overlay bounding boxes using the K-grid (pixel = loc / (K/dim) * dim)."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    gx, gy = _GRID_K / W, _GRID_K / H
    seen: set = set()
    for m in _BOX_RE.finditer(doctags):
        tag = m.group(1)
        x1, y1, x2, y2 = (int(v) for v in m.groups()[1:])
        key = (x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        d.rectangle([x1/gx*W, y1/gy*H, x2/gx*W, y2/gy*H],
                    outline=_color(tag), width=2)
    return im


# ── Tab 1: Playground (single image OR multi-page PDF) ────────────────────────

def _pdf_to_images(pdf_path: str, dpi: int = 200):
    """Client-side PDF split: yield one RGB page image per page (PyMuPDF)."""
    import fitz  # PyMuPDF — lazy import so the app loads without it
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def _page_view(page: dict, show_boxes: bool):
    """One page entry → (left image, meta, raw DocTags, rendered HTML)."""
    img_orig = page["img_orig"]
    doctags = page.get("doctags") or ""
    if show_boxes and doctags:
        left = _annotate(page["img_norm"], doctags).resize(img_orig.size, Image.LANCZOS)
    else:
        left = img_orig
    return left, page["meta"], doctags, _render(doctags, img_orig)


def _infer_page(i: int, src: Image.Image, n: int, max_tokens: int) -> dict:
    """Infer one page → page entry. Safe to call from a worker thread."""
    w, h = src.size
    norm = _normalize(src)
    try:
        doctags, dt, toks = _infer(norm, ADAPTER, max_tokens)
        tps = toks / dt if dt else 0
        meta = (f"Trang {i+1}/{n} · ⏱️ {dt:.1f}s · {toks} tok · {tps:.1f} tok/s · "
                f"ảnh gốc {w}×{h}px")
    except Exception as exc:  # noqa: BLE001 — skip a bad page, keep the rest
        doctags, meta = "", f"Trang {i+1}/{n} · Lỗi inference: {exc}"
    return {"img_orig": src, "img_norm": norm, "doctags": doctags, "meta": meta}


def run_playground(file, max_tokens, show_boxes, progress=gr.Progress()):
    """Infer one uploaded file — auto-detected as a single image or a PDF. A PDF is
    split on the client and its pages are inferred with up to PDF_CONCURRENCY
    requests in flight at once (ThreadPoolExecutor); results are stored in page
    order regardless of completion order. The concurrent load is also what drives
    the queue-depth-based serving autoscaler to add instances.
    Returns (left, meta, raw, html, page-slider update, state)."""
    if not file:
        return None, "Hãy upload một ảnh hoặc một file PDF.", "", "", gr.update(visible=False), None
    ext = os.path.splitext(file)[1].lower()
    try:
        if ext == ".pdf":
            srcs = list(_pdf_to_images(file))
        else:
            srcs = [Image.open(file).convert("RGB")]   # any image format
    except Exception as exc:  # noqa: BLE001
        return None, f"Lỗi đọc file: {exc}", "", "", gr.update(visible=False), None
    if not srcs:
        return None, "File không có trang/ảnh nào.", "", "", gr.update(visible=False), None

    n = len(srcs)
    c = max(1, min(PDF_CONCURRENCY, n))   # don't spawn more workers than pages
    pages: list = [None] * n
    with ThreadPoolExecutor(max_workers=c) as ex:
        futs = {ex.submit(_infer_page, i, src, n, max_tokens): i for i, src in enumerate(srcs)}
        for done, fut in enumerate(as_completed(futs), 1):
            pages[futs[fut]] = fut.result()        # placed by index → page order kept
            progress(done / n, desc=f"Inference {n} trang (×{c} song song)")

    state = {"pages": pages}
    left, meta0, raw0, html0 = _page_view(pages[0], show_boxes)
    slider = gr.update(minimum=1, maximum=max(2, n), value=1, step=1, visible=(n > 1))
    return left, meta0, raw0, html0, slider, state


def view_page(page_no, show_boxes, state):
    """Show page `page_no` (1-based) of the current result; also used to toggle
    boxes on/off without re-running the model."""
    if not state or not state.get("pages"):
        return None, "", "", ""
    pages = state["pages"]
    idx = max(0, min(int(page_no) - 1, len(pages) - 1))
    return _page_view(pages[idx], show_boxes)


def _step_page(page_no, state, delta):
    n = len(state["pages"]) if state and state.get("pages") else 1
    return gr.update(value=max(1, min(n, int(page_no) + delta)))


# ── Tab 2: Base vs Adapter ────────────────────────────────────────────────────

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


# ── Tab 3: Proposed (student via LB) vs Teacher (via its own URL) ─────────────--

def compare_teacher(img, max_tokens, teacher_url, teacher_model):
    """Run the SAME image through the proposed distilled student (via the serving LB)
    and the base teacher model (via the URL provided). Returns, for each side,
    (meta, rendered, raw)."""
    if img is None:
        return "Hãy upload ảnh.", "", "", "Hãy upload ảnh.", "", ""

    # ── proposed student (served Production adapter through the LB) ──
    try:
        s_text, s_dt, s_tok = _infer(_normalize(img), ADAPTER, max_tokens)
        s_meta = (f"⏱️ {s_dt:.1f}s · {s_tok} tok · "
                  f"{s_tok/s_dt if s_dt else 0:.1f} tok/s · model=`{ADAPTER}`")
        s_html = _render(s_text, img)
    except Exception as exc:  # noqa: BLE001
        s_text, s_meta, s_html = "", f"Lỗi proposed: {exc}", ""

    # ── teacher (arbitrary OpenAI-compatible endpoint) ──
    url = (teacher_url or "").strip()
    if not url:
        return s_meta, s_html, s_text, "Chưa nhập Teacher URL.", "", ""
    try:
        t_text, t_dt, t_tok = _infer_at(img, url, (teacher_model or "chandra").strip(), max_tokens)
        t_meta = (f"⏱️ {t_dt:.1f}s · {t_tok} tok · "
                  f"{t_tok/t_dt if t_dt else 0:.1f} tok/s · model=`{teacher_model}`")
    except Exception as exc:  # noqa: BLE001
        t_text, t_meta = "", f"Lỗi teacher: {exc}"
    # teacher emits Markdown/HTML, not DocTags → render via Markdown component (raw text)
    return s_meta, s_html, s_text, t_meta, t_text, t_text


# ── UI ──────────────────────────────────────────────────────────────────────--

with gr.Blocks(title="OCR Experiment") as demo:
    gr.Markdown(f"# 🔬 OCR Experiment\nServing LB: `{LB_URL}` · API: `{API_BASE}`")

    with gr.Tab("Playground"):
        pg_state = gr.State(None)
        pg_file = gr.File(label="Ảnh hoặc PDF (tự nhận diện)",
                          file_types=["image", ".pdf"], type="filepath")
        with gr.Row():
            pg_tok = gr.Slider(256, 4096, value=2048, step=128, label="max_tokens")
            pg_box = gr.Checkbox(value=True, label="Hiện bounding box")
        pg_btn = gr.Button("Convert → DocTags", variant="primary")
        pg_meta = gr.Markdown()
        with gr.Row():
            pg_prev = gr.Button("◀ Trang trước")
            pg_page = gr.Slider(1, 2, value=1, step=1, label="Trang", visible=False)
            pg_next = gr.Button("Trang sau ▶")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Layout (ảnh gốc)")
                pg_boxes = gr.Image(label="Ảnh", height=720)
            with gr.Column():
                gr.Markdown("#### Rendered")
                pg_html = gr.HTML()
        with gr.Accordion("DocTags (raw)", open=False):
            pg_raw = gr.Code(label="raw")

        _pg_view_outs = [pg_boxes, pg_meta, pg_raw, pg_html]
        pg_btn.click(run_playground, [pg_file, pg_tok, pg_box],
                     [pg_boxes, pg_meta, pg_raw, pg_html, pg_page, pg_state])
        pg_page.change(view_page, [pg_page, pg_box, pg_state], _pg_view_outs)
        pg_box.change(view_page, [pg_page, pg_box, pg_state], _pg_view_outs)
        pg_prev.click(lambda p, s: _step_page(p, s, -1), [pg_page, pg_state], [pg_page])
        pg_next.click(lambda p, s: _step_page(p, s, +1), [pg_page, pg_state], [pg_page])

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

    with gr.Tab("Proposed vs Teacher"):
        with gr.Row():
            pt_img = gr.Image(type="pil", label="Ảnh")
            pt_tok = gr.Slider(256, 4096, value=2048, step=128, label="max_tokens")
        with gr.Row():
            pt_url = gr.Textbox(value=TEACHER_URL, label="Teacher base URL (OpenAI-compatible)",
                                placeholder="https://<teacher-host>")
            pt_model = gr.Textbox(value=TEACHER_MODEL, label="Teacher model name")
        pt_btn = gr.Button("So sánh", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Proposed model (distilled 258M)")
                pt_pmeta = gr.Markdown()
                pt_phtml = gr.HTML()
                with gr.Accordion("DocTags (raw)", open=False):
                    pt_praw = gr.Code(label="raw")
            with gr.Column():
                gr.Markdown("### Base Teacher Model")
                pt_tmeta = gr.Markdown()
                pt_tmd = gr.Markdown()
                with gr.Accordion("Teacher output (raw)", open=False):
                    pt_traw = gr.Code(label="raw")
        pt_btn.click(compare_teacher, [pt_img, pt_tok, pt_url, pt_model],
                     [pt_pmeta, pt_phtml, pt_praw, pt_tmeta, pt_tmd, pt_traw])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("EXPERIMENT_PORT", "7860")))

