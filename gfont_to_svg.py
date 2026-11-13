#!/usr/bin/env python3
"""Render text with a .gfont / .tfont single-line (stroke) font to an SVG
that Bambu Suite can import as 线条绘画 (Drawing Line).

The Taobao writing-machine ecosystem uses stroke fonts (KenjoyDraw `.gfont`),
not outline TTFs: each glyph is a list of pen strokes, so the pen draws every
stroke once. This script is a thin bridge:

    .gfont / .tfont  ->  CNCDraw layout (mm, A4)  ->  cropped SVG (mm)

Requires CNCDraw (tested with https://github.com/jiangmuran/CNCDraw):

    pip install "git+https://github.com/jiangmuran/CNCDraw"

NOTE: CNCDraw has no LICENSE file (it is a reverse-engineered implementation
of KenjoyDraw) and the bundled .gfont fonts are third-party. Use it for your
own experiments; check the rights before redistributing fonts or output.

Usage:
    python gfont_to_svg.py "你好，世界" --font my_handwriting.gfont --out out.svg
    python gfont_to_svg.py "Hello" --font font.tfont --size 120 --margin 8
"""

import argparse
import os
import sys


def fail(msg: str) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def load_layout():
    try:
        from cncdraw.layout.text_layout import PageInfo, TextLayout
    except Exception as exc:  # pragma: no cover - depends on user env
        fail(
            "CNCDraw is not importable. Install it first:\n"
            '  pip install "git+https://github.com/jiangmuran/CNCDraw"\n'
            f"  (import error: {exc})"
        )
    return PageInfo, TextLayout


def build_svg(paths, margin: float, stroke: float, title: str) -> str:
    points = [(x, y) for p in paths for x, y in p.points]
    if not points:
        fail("layout produced no paths (empty text or missing glyphs)")

    minx = min(x for x, y in points)
    maxx = max(x for x, y in points)
    miny = min(y for x, y in points)
    maxy = max(y for x, y in points)

    width = maxx - minx + 2 * margin
    height = maxy - miny + 2 * margin

    parts = []
    for p in paths:
        if not p.points:
            continue
        d = []
        for (x, y), cmd in zip(p.points, p.commands):
            X = x - minx + margin
            Y = y - miny + margin
            d.append(("M " if cmd == 0 else "L ") + f"{X:.3f},{Y:.3f}")
        parts.append(
            '<path d="' + " ".join(d) + '" fill="none" stroke="#000000" '
            f'stroke-width="{stroke:g}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}mm" '
        f'height="{height:.2f}mm" viewBox="0 0 {width:.2f} {height:.2f}">\n'
        f"  <title>{title}</title>\n"
        f'  <rect width="100%" height="100%" fill="#ffffff"/>\n  '
        + "\n  ".join(parts)
        + "\n</svg>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", help="text to typeset")
    ap.add_argument("--font", required=True, help="path to a .gfont or .tfont file")
    ap.add_argument("--out", default="text_singleline.svg", help="output SVG path")
    ap.add_argument("--size", type=float, default=150.0, help="font size percent (100 = normal)")
    ap.add_argument("--line-num", type=int, default=10, help="lines per page (controls line height)")
    ap.add_argument("--margin", type=float, default=10.0, help="margin in mm around the text")
    ap.add_argument("--stroke", type=float, default=0.35, help="SVG stroke width in mm (display only)")
    ap.add_argument("--handwriting", action="store_true", help="add slight random jitter per glyph")
    ap.add_argument("--page", default="A4", help="layout page: A4 / A3 / 16K")
    args = ap.parse_args()

    font = os.path.abspath(os.path.expanduser(args.font))
    if not os.path.isfile(font):
        fail(f"font not found: {font}")
    if not font.lower().endswith((".gfont", ".tfont")):
        fail("font must be .gfont or .tfont (TTF is an outline font, not a stroke font)")

    PageInfo, TextLayout = load_layout()

    presets = {
        "A4": PageInfo.default_a4,
        "A3": PageInfo.default_a3,
        "16K": PageInfo.default_16k,
    }
    page = presets.get(args.page.upper(), PageInfo.default_a4)()
    page.font_paths = [font]
    page.font_size = args.size
    page.line_num = args.line_num

    with TextLayout(page) as lay:
        lay.load_fonts()
        missing = lay.get_missing_chars(args.text)
        if args.handwriting:
            paths = lay.layout_text(
                args.text,
                random_font_size=(0.95, 1.05),
                random_rotate=(-3, 3),
                random_x_scale=(0.97, 1.03),
                random_y_scale=(0.97, 1.03),
            )
        else:
            paths = lay.layout_text(args.text)

    if missing:
        print("warning: font is missing characters: " + "".join(missing), file=sys.stderr)

    svg = build_svg(paths, args.margin, args.stroke, os.path.basename(font))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"wrote {args.out}  ({len(paths)} stroke paths, {len(svg)} bytes)")


if __name__ == "__main__":
    main()
