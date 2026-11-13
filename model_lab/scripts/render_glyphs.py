#!/usr/bin/env python3
"""Render handwritten glyphs from webapp strokes (256 canvas, width 9) to PNG."""
import json, sys
from PIL import Image, ImageDraw

INNER = 256
W = 9

def render(strokes):
    img = Image.new("L", (INNER, INNER), 255)
    d = ImageDraw.Draw(img)
    for s in strokes:
        if not s:
            continue
        if len(s) == 1:
            x, y = s[0]["x"], s[0]["y"]
            d.ellipse([x - W/2, y - W/2, x + W/2, y + W/2], fill=17)
            continue
        pts = [(p["x"], p["y"]) for p in s]
        d.line(pts, fill=17, width=W, joint="curve")
        r = W / 2
        for (x, y) in (pts[0], pts[-1]):
            d.ellipse([x - r, y - r, x + r, y + r], fill=17)
    return img

def main(src, chars, outdir):
    data = json.load(open(src))
    by = {c["char"]: c["strokes"] for c in data["characters"]}
    for ch in chars:
        if ch not in by:
            print("MISSING", ch); continue
        img = render(by[ch])
        img.convert("RGB").save(f"{outdir}/{ch}.png")
        print("OK", ch, len(by[ch]))
if __name__ == "__main__":
    main(sys.argv[1], list(sys.argv[2]), sys.argv[3])
