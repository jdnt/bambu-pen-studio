#!/usr/bin/env python3
"""Typeset text with an Inkscape "Custom Stroke Font" (SVG font) and emit a
single-line SVG for Bambu Suite (线条绘画 / Drawing Line).

Why: the Inkscape stroke-font extension can create a real single-line font
from your handwriting (one SVG path per glyph), but its Render Text dialog is
GUI-only. This script is a tiny CLI middleware:

    stroke font .svg  +  text  ->  cropped mm SVG of polylines

The font file is the SVG font written by the extension into
`strokefontdata/<FontName>.svg` (standard SVG <font>/<glyph> elements).

Extra features that avoid writing every character:
- `--fallback FONT.svg` (repeatable): use another stroke font for characters
  missing from your own font (a hybrid font: your glyphs where you have them).
- `--slant`, `--x-scale`, `--y-scale`, `--jitter`: cheap style variants of the
  same font (italic / condensed / casual), without redrawing anything.

Usage:
    python stroke_font_to_svg.py "你好，世界" --font myfont.svg --size 20 --out out.svg
    python stroke_font_to_svg.py "你好吗" --font myfont.svg --fallback base.svg
    python stroke_font_to_svg.py "Hello" --font myfont.svg --slant 12 --jitter 0.08

Notes:
- `--size` is the em size in mm (roughly the character height).
- Glyphs are emitted as <path> with a translate/scale transform, so curves are
  preserved exactly; no path parsing is needed.
- Fallback fonts are assumed to use their own units-per-em; each glyph is
  scaled by its own font's em, so mixing fonts is safe.
"""

import argparse
import os
import random
import sys
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"


def parse_font(path):
    """Return (upm, ascent, descent, glyphs, missing).

    glyphs maps a character to (path_d, advance_units, units_per_em).
    """
    tree = ET.parse(path)
    root = tree.getroot()

    font = root.find(f".//{{{SVG_NS}}}font")
    if font is None:
        font = root.find(".//font")  # some files omit the namespace
    if font is None:
        raise ValueError(f"{path}: no <font> element; is this an SVG stroke font?")

    def find(parent, tag):
        el = parent.find(f"{{{SVG_NS}}}{tag}")
        if el is None:
            el = parent.find(tag)
        return el

    face = find(font, "font-face")
    if face is None:
        raise ValueError(f"{path}: no <font-face> element")

    upm = float(face.get("units-per-em", "1000"))
    ascent = float(face.get("ascent", str(upm * 0.8)))
    descent = float(face.get("descent", str(-upm * 0.2)))
    default_adv = float(font.get("horiz-adv-x", "0") or 0)

    glyphs = {}
    for g in list(font.findall(f"{{{SVG_NS}}}glyph")) + list(font.findall("glyph")):
        ch = g.get("unicode")
        if ch is None:
            continue
        adv = g.get("horiz-adv-x")
        adv = float(adv) if adv is not None else default_adv
        glyphs[ch] = (g.get("d") or "", adv, upm)

    miss_el = find(font, "missing-glyph")
    if miss_el is not None:
        adv = miss_el.get("horiz-adv-x")
        missing = (miss_el.get("d") or "",
                   float(adv) if adv is not None else default_adv, upm)
    else:
        missing = ("", default_adv, upm)

    if not glyphs:
        raise ValueError(f"{path}: font contains no <glyph> elements")
    return upm, ascent, descent, glyphs, missing


def wrap_text(text, glyphs, missing, size, max_width):
    """Split text into lines of (char, path_d, advance_mm)."""
    lines = []
    cur = []
    cur_w = 0.0
    for ch in text:
        if ch == "\n":
            lines.append((cur, cur_w))
            cur, cur_w = [], 0.0
            continue
        d, adv_units, upm = glyphs.get(ch, missing)
        adv = adv_units * size / upm
        if cur and cur_w + adv > max_width:
            lines.append((cur, cur_w))
            cur, cur_w = [], 0.0
        cur.append((ch, d, adv, upm))
        cur_w += adv
    if cur or not lines:
        lines.append((cur, cur_w))
    return lines


def build_svg(lines, ascent, descent, size, upm, margin, line_gap, align, stroke,
              max_width, slant, x_scale, y_scale, jitter, seed):
    if not lines:
        raise ValueError("nothing to render")

    line_h = size * (ascent - descent) / upm * line_gap
    content_w = max(w for _, w in lines)
    width = max(content_w, 1.0) + 2 * margin
    height = margin * 2 + (len(lines) - 1) * line_h + size

    rng = random.Random(seed)
    parts = []
    for i, (chars, cur_w) in enumerate(lines):
        baseline = margin + size * (ascent / upm) + i * line_h
        if align == "center":
            x = margin + (content_w - cur_w) / 2.0
        elif align == "right":
            x = margin + (content_w - cur_w)
        else:
            x = margin
        for ch, d, adv, upm in chars:
            if d.strip():
                s = size / upm
                sx = s * x_scale
                sy = -s * y_scale
                dx = dy = 0.0
                rot = 0.0
                if jitter > 0:
                    dx = rng.uniform(-jitter, jitter) * size
                    dy = rng.uniform(-jitter, jitter) * size
                    rot = rng.uniform(-jitter, jitter) * 12.0
                transform = (
                    f"translate({x + dx:.3f},{baseline + dy:.3f}) "
                    f"rotate({rot:.3f}) "
                    f"scale({sx:.6f},{sy:.6f}) "
                    f"skewX({slant:.3f})"
                )
                parts.append(
                    f'<path d="{d}" fill="none" stroke="#000000" '
                    f'stroke-width="{stroke:g}" stroke-linecap="round" '
                    f'stroke-linejoin="round" transform="{transform}"/>'
                )
            x += adv

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}mm" '
        f'height="{height:.2f}mm" viewBox="0 0 {width:.2f} {height:.2f}">\n'
        '  <rect width="100%" height="100%" fill="#ffffff"/>\n  '
        + "\n  ".join(parts)
        + "\n</svg>\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="?", help="text to typeset")
    ap.add_argument("--text-file", help="read text from this UTF-8 file instead")
    ap.add_argument("--font", required=True, help="SVG stroke font file from Inkscape")
    ap.add_argument("--fallback", action="append", default=[],
                    help="second stroke font for missing characters (repeatable)")
    ap.add_argument("--out", default="text_singleline.svg")
    ap.add_argument("--size", type=float, default=20.0, help="em size in mm (character height)")
    ap.add_argument("--width", type=float, default=180.0, help="max line width in mm")
    ap.add_argument("--margin", type=float, default=5.0, help="margin in mm")
    ap.add_argument("--line-gap", type=float, default=1.7, help="line spacing multiplier")
    ap.add_argument("--align", choices=["left", "center", "right"], default="left")
    ap.add_argument("--stroke", type=float, default=0.35, help="SVG stroke width in mm")
    # cheap style variants
    ap.add_argument("--slant", type=float, default=0.0, help="italic slant in degrees")
    ap.add_argument("--x-scale", type=float, default=1.0, help="horizontal scale")
    ap.add_argument("--y-scale", type=float, default=1.0, help="vertical scale")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="0..0.3 random per-glyph offset/rotation for a casual look")
    ap.add_argument("--seed", type=int, default=1, help="jitter seed")
    args = ap.parse_args()

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as fh:
            text = fh.read().rstrip("\n")
    elif args.text:
        text = args.text
    else:
        ap.error("provide text or --text-file")

    if not os.path.isfile(args.font):
        ap.error(f"font not found: {args.font}")

    upm, ascent, descent, glyphs, missing = parse_font(args.font)
    for fb in args.fallback:
        if not os.path.isfile(fb):
            ap.error(f"fallback font not found: {fb}")
        _, _, _, fb_glyphs, _ = parse_font(fb)
        for ch, val in fb_glyphs.items():
            glyphs.setdefault(ch, val)

    max_width = max(1.0, args.width - 2 * args.margin)
    lines = wrap_text(text, glyphs, missing, args.size, max_width)

    missing_chars = sorted({ch for ch in text if ch not in glyphs and ch not in "\n"})
    if missing_chars:
        print("warning: font is missing: " + "".join(missing_chars), file=sys.stderr)

    svg = build_svg(lines, ascent, descent, args.size, upm, args.margin,
                    args.line_gap, args.align, args.stroke, max_width,
                    args.slant, args.x_scale, args.y_scale, args.jitter, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"wrote {args.out}  ({len(lines)} line(s), {len(svg)} bytes)")


if __name__ == "__main__":
    main()
