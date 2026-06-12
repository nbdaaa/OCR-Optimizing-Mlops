"""
Calibrate the DocTags loc → pixel mapping using a GROUND-TRUTH training sample.

Formula:
    x_px = margin_x + loc_x / grid_x * (W - 2*margin_x)
    y_px = margin_y + loc_y / grid_y * (H - 2*margin_y)

  grid_x/y  : effective normalisation range (start with 355/300, tune ±20)
  margin_x/y: page border in pixels that loc_x=0 / loc_y=0 maps to (start with 0)

Usage:
    python calibrate_boxes.py --row 0 --grid-x 355 --grid-y 300
    python calibrate_boxes.py --row 0 --grid-x 370 --grid-y 310 --margin-x 20 --margin-y 15
    python calibrate_boxes.py --scan-rows 20   # find row with widest loc range
"""
from __future__ import annotations

import argparse
import io
import os
import re

import pandas as pd
from PIL import Image, ImageDraw

_BOX_RE = re.compile(r"<([a-z_0-9]+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
_COLORS = {"section_header": (220, 40, 40), "title": (220, 40, 40),
           "text": (40, 110, 215), "list_item": (40, 160, 60),
           "caption": (210, 130, 20), "picture": (160, 40, 200),
           "table": (20, 170, 170), "page_footer": (130, 130, 130),
           "page_header": (130, 130, 130)}

def _color(tag):
    for k, c in _COLORS.items():
        if tag.startswith(k):
            return c
    return (90, 90, 90)

def _decode(field):
    b = field.get("bytes") if isinstance(field, dict) else field
    return Image.open(io.BytesIO(b)).convert("RGB")

def _loc_to_px(lx, ly, W, H, grid_x, grid_y, margin_x, margin_y):
    """x_px = margin + loc/grid * (dim - 2*margin)"""
    cw = W - 2 * margin_x
    ch = H - 2 * margin_y
    return margin_x + lx / grid_x * cw, margin_y + ly / grid_y * ch

def _draw(img, doctags, grid_x, grid_y, margin_x, margin_y, padding=0.0):
    im = img.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    seen = set()
    for m in _BOX_RE.finditer(doctags):
        tag = m.group(1)
        x1, y1, x2, y2 = (int(v) for v in m.groups()[1:])
        key = (x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        px1, py1 = _loc_to_px(x1, y1, W, H, grid_x, grid_y, margin_x, margin_y)
        px2, py2 = _loc_to_px(x2, y2, W, H, grid_x, grid_y, margin_x, margin_y)
        d.rectangle([px1 - padding, py1 - padding, px2 + padding, py2 + padding],
                    outline=_color(tag), width=3)
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",     default="nbdaaa/all-ocr-data")
    ap.add_argument("--file",     default="data/train-00000-of-00064.parquet")
    ap.add_argument("--row",      type=int,   default=0)
    ap.add_argument("--grid-x",   type=float, default=355.0,
                    help="Effective X normalisation range. Increase → boxes shift left / shrink width.")
    ap.add_argument("--grid-y",   type=float, default=300.0,
                    help="Effective Y normalisation range. Increase → boxes shift up / shrink height.")
    ap.add_argument("--margin-x", type=float, default=0.0,
                    help="Pixel offset at left/right page border (loc_x=0 → x_px=margin_x).")
    ap.add_argument("--margin-y", type=float, default=0.0,
                    help="Pixel offset at top/bottom page border.")
    ap.add_argument("--padding", type=float, default=0.0,
                    help="Extra pixels added to each side of every box (purely visual).")
    ap.add_argument("--scan-rows", type=int, default=0,
                    help="If >0: scan this many rows, print which has widest loc range (best calibration anchor).")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    p = hf_hub_download(args.repo, args.file, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"))
    df = pd.read_parquet(p)

    # ── scan mode: find the row whose elements cover the most of the loc space ──
    if args.scan_rows > 0:
        print(f"Scanning rows 0-{args.scan_rows - 1} for widest loc coverage …")
        best = None
        for i in range(min(args.scan_rows, len(df))):
            dt = df.iloc[i]["output_text"]
            ms = list(_BOX_RE.finditer(dt))
            if not ms:
                continue
            xv = [int(m.group(2)) for m in ms] + [int(m.group(4)) for m in ms]
            yv = [int(m.group(3)) for m in ms] + [int(m.group(5)) for m in ms]
            span = (max(xv) - min(xv)) + (max(yv) - min(yv))
            print(f"  row {i:3d}: X {min(xv):3d}..{max(xv):3d}  Y {min(yv):3d}..{max(yv):3d}  span={span}")
            if best is None or span > best[0]:
                best = (span, i, max(xv), max(yv))
        if best:
            print(f"\n→ Widest row: {best[1]}  (max_x={best[2]}, max_y={best[3]})")
            print(f"  Derived grid estimate: grid_x ≈ {best[2]/0.88:.0f}  grid_y ≈ {best[3]/0.88:.0f}  (assumes content fills ~88% of page)")
        return

    # ── single-row calibration ────────────────────────────────────────────────
    row     = df.iloc[args.row]
    img     = _decode(row["image"])
    doctags = row["output_text"]
    W, H    = img.size

    elems = list(_BOX_RE.finditer(doctags))
    xvals = [int(m.group(2)) for m in elems] + [int(m.group(4)) for m in elems]
    yvals = [int(m.group(3)) for m in elems] + [int(m.group(5)) for m in elems]
    print(f"row {args.row} · image {W}×{H}px · {len(elems)} elems")
    print(f"  loc X {min(xvals)}..{max(xvals)}   loc Y {min(yvals)}..{max(yvals)}")
    print(f"  grid_x={args.grid_x}  grid_y={args.grid_y}  "
          f"margin_x={args.margin_x}  margin_y={args.margin_y}")

    # Derived pixel extents (for sanity check)
    cw = W - 2 * args.margin_x
    ch = H - 2 * args.margin_y
    print(f"  → content px X: {args.margin_x:.0f}..{args.margin_x + max(xvals)/args.grid_x*cw:.0f}/{W}  "
          f"Y: {args.margin_y:.0f}..{args.margin_y + max(yvals)/args.grid_y*ch:.0f}/{H}")

    print("  per-element (tag | loc x1,y1–x2,y2 | text):")
    for m in elems[:12]:
        tag = m.group(1)
        x1, y1, x2, y2 = (int(v) for v in m.groups()[1:])
        after = doctags[m.end():m.end() + 60].split("<")[0].strip()
        px1, py1 = _loc_to_px(x1, y1, W, H, args.grid_x, args.grid_y, args.margin_x, args.margin_y)
        px2, py2 = _loc_to_px(x2, y2, W, H, args.grid_x, args.grid_y, args.margin_x, args.margin_y)
        print(f"    {tag:24s} loc({x1},{y1}→{x2},{y2})  px({px1:.0f},{py1:.0f}→{px2:.0f},{py2:.0f})  | {after[:35]}")

    out = (f"box_gx{int(args.grid_x)}_gy{int(args.grid_y)}"
           f"_mx{int(args.margin_x)}_my{int(args.margin_y)}"
           f"_pad{int(args.padding)}_r{args.row}.png")
    _draw(img, doctags, args.grid_x, args.grid_y,
          args.margin_x, args.margin_y, args.padding).save(out)
    print(f"  saved {out}")

if __name__ == "__main__":
    main()
