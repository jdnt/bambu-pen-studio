#!/usr/bin/env python3
"""Measure rendered glyph bounding boxes for the fontgen UI.

Input JSON on stdin:
{"items":[{"char":"国","image":"/path.png"} | {"char":"国","strokes":[...]}]}
Output: {"ok":true,"items":[{"char":..., "w":..., "h":...}, ...]}
"""
import json, sys
import cv2
import numpy as np


def measure(item):
    if item.get("image"):
        img = cv2.imread(item["image"], cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            mask = (img[:, :, 3] > 20).astype(np.uint8)
        else:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = ((img < 245) if float(img.mean()) > 127 else (img > 10)).astype(np.uint8)
        ys, xs = np.where(mask > 0)
    else:
        pts = []
        for st in item.get("strokes") or []:
            for p in st:
                if isinstance(p, (list, tuple)):
                    pts.append((float(p[0]), float(p[1])))
                else:
                    pts.append((float(p.get("x", 0)), float(p.get("y", 0))))
        if not pts:
            return None
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {"char": item["char"], "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1)}


def main():
    data = json.load(sys.stdin)
    out = []
    for item in data.get("items", []):
        try:
            m = measure(item)
            if m:
                out.append(m)
        except Exception:
            pass
    out.sort(key=lambda x: max(x["w"], x["h"]), reverse=True)
    print(json.dumps({"ok": True, "items": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
