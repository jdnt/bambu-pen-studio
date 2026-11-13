#!/usr/bin/env python3
"""Split a scanned calligraphy sheet into per-character PNGs and OCR labels.

Used by server.py for font 1 (brush). Keeps raw crops (no per-glyph resize);
global uniform scaling happens later at training-set build time.
"""
import argparse, json, os, re, sys, uuid
from pathlib import Path

import cv2
import numpy as np


def _components(binary):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    h, w = binary.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < max(40, 0.00025 * h * w):
            continue
        if bw < 8 or bh < 8:
            continue
        # very long thin grid lines
        if bw > 0.85 * w and bh < 12:
            continue
        if bh > 0.85 * h and bw < 12:
            continue
        out.append([int(x), int(y), int(x + bw), int(y + bh)])
    return out


def _merge(boxes, gap_ratio=0.16, overlap_ratio=0.45):
    if not boxes:
        return []
    hs = [b[3] - b[1] for b in boxes]
    med = float(np.median(hs)) if hs else 20
    gap = max(6.0, med * gap_ratio)
    parent = list(range(len(boxes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(boxes)):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            # vertical overlap
            ov = max(0, min(ay1, by1) - max(ay0, by0))
            minh = max(1, min(ay1 - ay0, by1 - by0))
            if ov / minh < overlap_ratio:
                continue
            dx = max(0, max(ax0, bx0) - min(ax1, bx1))
            if dx <= gap:
                union(i, j)
    groups = {}
    for i, b in enumerate(boxes):
        r = find(i)
        if r not in groups:
            groups[r] = list(b)
        else:
            g = groups[r]
            g[0] = min(g[0], b[0]); g[1] = min(g[1], b[1])
            g[2] = max(g[2], b[2]); g[3] = max(g[3], b[3])
    return [groups[k] for k in sorted(groups)]


def _split_wide(box, gray, ratio=1.75):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if h <= 0 or w / h <= ratio:
        return [box]
    crop = gray[y0:y1, x0:x1]
    ink = (crop < 160).astype(np.uint8)
    col = ink.sum(axis=0)
    # find valleys near expected boundaries
    n = max(2, int(round(w / h)))
    cuts = []
    for k in range(1, n):
        center = int(w * k / n)
        span = max(3, w // (n * 6))
        lo, hi = max(1, center - span), min(w - 1, center + span)
        if hi <= lo:
            continue
        idx = lo + int(np.argmin(col[lo:hi]))
        cuts.append(idx)
    if not cuts:
        return [box]
    parts = []
    last = 0
    for c in cuts + [w]:
        if c - last >= max(8, h * 0.2):
            parts.append((x0 + last, y0, x0 + c, y1))
            last = c
    return parts or [box]


def _label(crop):
    # No OCR by design: every cut is unknown until manually labelled.
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit("cannot read image")
    h, w = img.shape
    # trim 1% border that may contain scan edges
    mx, my = max(1, w // 100), max(1, h // 100)
    roi = img[my:h - my, mx:w - mx]
    gray = cv2.GaussianBlur(roi, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # close gaps between strokes / radicals
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    # remove isolated specks
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    boxes = _components(closed)
    boxes = _merge(boxes)
    # split obviously wide merged boxes, then re-merge only small pieces
    split = []
    for b in boxes:
        split.extend(_split_wide(b, gray))
    # sort top-to-bottom, left-to-right (row tolerance)
    split.sort(key=lambda b: (round(b[1] / max(1, np.median([x[3]-x[1] for x in split]))), b[0]))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for b in split:
        x0, y0, x1, y1 = b
        # map back to original coords
        x0 += mx; x1 += mx; y0 += my; y1 += my
        pad_x = max(4, int((x1 - x0) * 0.08)); pad_y = max(4, int((y1 - y0) * 0.08))
        x0 = max(0, x0 - pad_x); x1 = min(w, x1 + pad_x)
        y0 = max(0, y0 - pad_y); y1 = min(h, y1 + pad_y)
        crop = img[y0:y1, x0:x1]
        # pad to a stable white canvas preserving raw aspect; training adds global margin
        ch = max(crop.shape[0], crop.shape[1])
        canvas = np.full((ch, ch), 255, dtype=np.uint8)
        oy, ox = (ch - crop.shape[0]) // 2, (ch - crop.shape[1]) // 2
        canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        vid = uuid.uuid4().hex[:12]
        path = out_dir / f"{vid}.png"
        cv2.imwrite(str(path), canvas)
        label = _label(canvas)
        results.append({"id": vid, "file": str(path), "char": label,
                        "bbox": [x0, y0, x1, y1], "size": [int(x1-x0), int(y1-y0)]})

    Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "count": len(results), "variants": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
