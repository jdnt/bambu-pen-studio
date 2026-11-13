#!/usr/bin/env python3
"""Normalize handwritten glyph PNGs: crop content, scale up, center.

Used before building the training dataset so small handwriting fills the 256
canvas consistently without touching the four edges.

Usage:
  python scripts/normalize_glyphs.py \
    --src data/user_glyphs --dst data/user_glyphs_norm \
    --canvas 256 --fill 0.86
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

PUNCTUATION = set("，。！？；：、（）《》“”‘’—…·「」『』【】〔〕,.!?;:()[]{}<>\"'")


def content_bbox(im: Image.Image, threshold: int = 240):
    rgb = np.asarray(im.convert("RGB"))
    mask = rgb.min(axis=2) < threshold
    if mask.any():
        ys, xs = np.where(mask)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--canvas", type=int, default=256)
    ap.add_argument("--fill", type=float, default=0.86,
                    help="max content side as fraction of canvas; keeps a small safety margin")
    ap.add_argument("--threshold", type=int, default=240)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chars-file", default=None,
                    help="optional text file of allowed characters; filters the source dir")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    allowed = None
    if args.chars_file:
        allowed = set(Path(args.chars_file).read_text(encoding="utf-8"))
    files = sorted(p for p in src.iterdir()
                   if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                   and p.stem not in PUNCTUATION
                   and (allowed is None or p.stem in allowed))
    if args.limit:
        files = files[: args.limit]

    # Pass 1: load + crop everything; Pass 2: apply ONE global scale.
    # Never fit each glyph to the canvas individually: that would destroy the
    # relative size/proportion of continuous handwriting.
    items = []
    max_side = 0
    n_empty = 0
    for p in files:
        im = Image.open(p)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        box = content_bbox(im, args.threshold)
        if box is None:
            items.append((p, None))
            n_empty += 1
            continue
        x0, y0, x1, y1 = box
        crop = im.crop((x0, y0, x1, y1))
        max_side = max(max_side, crop.width, crop.height)
        items.append((p, crop))

    if max_side <= 0:
        raise SystemExit("no glyph content found")
    scale = float(args.canvas) * args.fill / max_side
    print(f"global scale={scale:.4f} (max content side={max_side}px)")

    n_done = 0
    for p, crop in items:
        if crop is None:
            Image.new("RGB", (args.canvas, args.canvas), "white").save(dst / p.name)
            continue
        nw = max(1, round(crop.width * scale))
        nh = max(1, round(crop.height * scale))
        nw, nh = min(nw, args.canvas), min(nh, args.canvas)
        crop = crop.resize((nw, nh), Image.LANCZOS)
        out = Image.new("RGB", (args.canvas, args.canvas), "white")
        out.paste(crop, ((args.canvas - nw) // 2, (args.canvas - nh) // 2))
        out.save(dst / p.name)
        n_done += 1

    print(f"normalized {n_done} glyphs, empty {n_empty}, -> {dst}")


if __name__ == "__main__":
    main()
