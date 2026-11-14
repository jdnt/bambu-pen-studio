#!/usr/bin/env python3
"""Build a TTF or single-line SVG pack from fontgen glyph records.

Input JSON:
{ "style":"hand"|"brush", "charset":"s"|"t", "format":"ttf"|"single",
  "out": "/path/font.ttf", "glyphs":[{"char":"你","strokes":[[[x,y],...]],"image":"/path.png"}] }
"""
import json, sys, zipfile, io, math
from pathlib import Path
import numpy as np, cv2
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

UPM = 1000
PAD = 40
PUNCT = set("，。！？；：、（）《》“”‘’—…·「」『』【】〔〕,.!?;:()[]{}<>\"'")

def glyph_name(ch):
    cp = ord(ch)
    return f"uni{cp:04X}" if cp <= 0xFFFF else f"u{cp:05X}"

def raster_hand(strokes, size=256):
    img = np.zeros((size, size), np.uint8)
    for st in strokes:
        pts = np.array([[int(round(p[0] if isinstance(p, (list, tuple)) else p["x"])), int(round(p[1] if isinstance(p, (list, tuple)) else p["y"]))] for p in st], np.int32)
        if len(pts) == 1:
            cv2.circle(img, tuple(pts[0]), 3, 255, -1)
        elif len(pts) >= 2:
            cv2.polylines(img, [pts], False, 255, 9, cv2.LINE_AA)
    return img

def contours_from_mask(mask):
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(cnts):
        if len(c) < 3:
            continue
        # outer contours and holes; TTGlyphPen closes each contour
        pts = []
        for p in c[:, 0, :]:
            x, y = int(p[0]), int(p[1])
            x = PAD + x * (UPM - 2 * PAD) // 256
            y = PAD + (255 - y) * (UPM - 2 * PAD) // 256
            pts.append((x, y))
        # drop consecutive duplicates
        clean = []
        for p in pts:
            if not clean or p != clean[-1]:
                clean.append(p)
        if len(clean) >= 3:
            out.append((clean, hier[0, i, 3] != -1))
    return out

def draw_tt_glyph(contours, advance=UPM):
    pen = TTGlyphPen(None)
    for pts, hole in contours:
        pen.moveTo(pts[0])
        for p in pts[1:]:
            pen.lineTo(p)
        pen.closePath()
    return pen.glyph()

def build_ttf(glyphs, out, style="hand"):
    glyph_order = [".notdef"]
    cmap = {}
    glyf = {".notdef": TTGlyphPen(None).glyph()}
    hmtx = {".notdef": (UPM, 0)}
    seen = set()
    built = []  # (char, contours, advance)
    for g in glyphs:
        ch = g["char"]
        if ch in seen or len(ch) != 1:
            continue
        seen.add(ch)
        if g.get("image"):
            img = cv2.imread(g["image"], cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3]
                mask = (alpha > 100).astype(np.uint8) * 255
            else:
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = ((img < 127) if float(img.mean()) > 127 else (img > 127)).astype(np.uint8) * 255
        else:
            mask = raster_hand(g.get("strokes") or [])
        contours = contours_from_mask(mask)
        if not contours:
            continue
        adv = 500 if ch in PUNCT else UPM
        built.append((ch, contours, adv))

    # Global (not per-glyph) scale: largest glyph reaches 95% of the em.
    if built:
        xs = [pt[0] for _ch, cs, _adv in built for pts, _hole in cs for pt in pts]
        ys = [pt[1] for _ch, cs, _adv in built for pts, _hole in cs for pt in pts]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        span = max(maxx - minx, maxy - miny, 1)
        scale = (UPM * 0.95) / span
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        for ch, contours, adv in built:
            scaled = []
            for pts, hole in contours:
                sp = [(500.0 + (float(x) - cx) * scale, 500.0 + (float(y) - cy) * scale) for x, y in pts]
                scaled.append((sp, hole))
            name = glyph_name(ch)
            glyf[name] = draw_tt_glyph(scaled)
            xmin = min(pt[0] for pts, _hole in scaled for pt in pts)
            hmtx[name] = (adv, int(round(xmin)))
            cmap[ord(ch)] = name
            glyph_order.append(name)

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(hmtx)
    fb.setupHorizontalHeader(ascent=880, descent=-120, lineGap=0)
    family = "MyBrush" if style == "brush" else "MyHand"
    fb.setupNameTable({"familyName": family, "styleName": "Regular", "uniqueFontIdentifier": f"{family} Regular", "fullName": f"{family} Regular", "psName": f"{family}-Regular", "version": "Version 1.0"})
    fb.setupOS2(sTypoAscender=880, sTypoDescender=-120, usWinAscent=880, usWinDescent=120)
    fb.setupPost()
    fb.save(out)
    return len(glyph_order) - 1

def build_single(glyphs, out):
    z = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    manifest = {"format": "single-line-svg", "upm": UPM, "glyphs": {}}
    for g in glyphs:
        ch = g["char"]
        if len(ch) != 1 or not g.get("strokes"):
            continue
        parts = []
        for st in g["strokes"]:
            pts = []
            for p in st:
                px = p[0] if isinstance(p, (list, tuple)) else p.get("x")
                py = p[1] if isinstance(p, (list, tuple)) else p.get("y")
                if px is None or py is None: continue
                x = PAD + float(px) * (UPM - 2 * PAD) / 256
                y = PAD + (255 - float(py)) * (UPM - 2 * PAD) / 256
                pts.append(f"{x:.2f},{y:.2f}")
            if len(pts) == 1:
                pts.append(pts[0])
            if len(pts) >= 2:
                parts.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="#000" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>')
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{UPM}" height="{UPM}" viewBox="0 0 {UPM} {UPM}"><rect width="{UPM}" height="{UPM}" fill="#fff"/>{"".join(parts)}</svg>'
        name = f"glyphs/{ord(ch):04X}.svg"
        z.writestr(name, svg)
        manifest["glyphs"][ch] = name
    z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    z.close()
    return len(manifest["glyphs"])

if __name__ == "__main__":
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    out = a["out"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if a.get("format") == "single":
        n = build_single(a["glyphs"], out)
    else:
        n = build_ttf(a["glyphs"], out, a.get("style", "hand"))
    print(json.dumps({"ok": True, "glyphs": n, "out": out}, ensure_ascii=False))
