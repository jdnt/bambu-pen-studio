#!/usr/bin/env python3
"""Generate glyphs for arbitrary characters with the best user model.

This is intentionally a foreground CLI: server.py calls it per request.
It builds a tiny test npz (Noto Serif content + one user style reference),
runs zi2zi-JiT at cfg=4.5 / 50 steps, then skeletonizes the bitmap into
polyline strokes suitable for the layout page / Bambu Suite.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(os.environ.get("HW_AI_ROOT", "/mnt/qnap_models/bambu-font-ai"))
REPO = ROOT / "src" / "zi2zi-JiT"
FONT_PATH = ROOT / "data" / "source" / "NotoSerifCJKsc-Regular.otf"
CHECKPOINT = ROOT / "run" / "user726_all_scale" / "checkpoint-last.pth"
USER_DIR = ROOT / "data" / "user_glyphs_norm"
FONT1_REF = ROOT / "data" / "font1_training" / "style_ref.png"
CACHE_DIR = ROOT / "data" / "ai_cache"


def _render_content(ch: str, size: int = 256) -> Image.Image:
    font = ImageFont.truetype(str(FONT_PATH), int(size * 0.86))
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    b = d.textbbox((0, 0), ch, font=font)
    w, h = b[2] - b[0], b[3] - b[1]
    d.text(((size - w) / 2 - b[0], (size - h) / 2 - b[1]), ch, font=font, fill="black")
    return im


def _reference(font_id: int = 0) -> Image.Image:
    if font_id == 1:
        cand = FONT1_REF
        if not cand.exists():
            clean = ROOT / "data" / "font1_clean" / "你.png"
            cand = clean if clean.exists() else None
        if cand is None or not cand.exists():
            files = sorted(glob.glob(str(ROOT / "data" / "font1" / "crops" / "*.png")))
            if not files:
                raise RuntimeError("no font1 reference available")
            cand = Path(files[0])
    else:
        cand = USER_DIR / "你.png"
        if not cand.exists():
            files = sorted(glob.glob(str(USER_DIR / "*.png")))
            if not files:
                raise RuntimeError("no user glyphs available for style reference")
            cand = Path(files[0])
    return Image.open(cand).convert("RGB").resize((128, 128))


def build_arrays(chars, font_id: int = 0):
    ref = _reference(font_id)
    content, target, labels = [], [], []
    for ch in chars:
        arr = np.array(_render_content(ch)).transpose(2, 0, 1)
        content.append(arr)
        target.append(arr)
        labels.append(ord(ch))
    style = np.stack([np.array(ref).transpose(2, 0, 1)]).repeat(len(chars), axis=0)
    return {
        'font_labels': np.full(len(chars), int(font_id), dtype=np.int64),
        'char_labels': np.arange(len(chars), dtype=np.int64),
        'unicode_labels': np.array(labels, dtype=np.int64),
        'content_images': np.stack(content),
        'target_images': np.stack(target),
        'style_images': style,
    }


def _build_npz(chars, path: Path, font_id: int = 0):
    arrays = build_arrays(chars, font_id)
    np.savez_compressed(path, num_original_samples=np.int64(len(chars)), **arrays)


def _rdp(points, eps):
    if len(points) < 3:
        return points
    a = np.asarray(points[0], float)
    b = np.asarray(points[-1], float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0:
        d = np.linalg.norm(np.asarray(points, float) - a, axis=1)
    else:
        d = np.abs(np.cross(ab, np.asarray(points, float) - a)) / np.sqrt(denom)
    idx = int(np.argmax(d))
    if float(d[idx]) > eps:
        left = _rdp(points[:idx + 1], eps)
        right = _rdp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def _merge_skeleton_paths(sk):
    paths = []
    node_ends = defaultdict(list)
    for i in range(sk.n_paths):
        pix = sk.path(i)
        if len(pix) < 4:
            continue
        pts = [[float(x), float(y)] for y, x in sk.path_coordinates(i)]
        idx = len(paths)
        paths.append({"pts": pts, "used": False})
        node_ends[int(pix[0])].append((idx, 0))
        node_ends[int(pix[-1])].append((idx, 1))

    def tangent(i, end):
        pts = paths[i]["pts"]
        k = min(4, len(pts) - 1)
        if k <= 0:
            return np.array([0.0, 0.0])
        if end == 0:
            v = np.asarray(pts[k], float) - np.asarray(pts[0], float)
        else:
            v = np.asarray(pts[-1], float) - np.asarray(pts[-1 - k], float)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    # Only join at degree-2 nodes: both fragments are on the same
    # skeleton chain, so the join cannot invent a new stroke.
    pair = {}
    for node, ends in node_ends.items():
        if len(ends) != 2:
            continue
        a, b = ends
        if a[0] == b[0]:
            continue
        pair[a] = b
        pair[b] = a

    def trace(start):
        i, e = start
        chain = []
        while True:
            if paths[i]["used"]:
                break
            paths[i]["used"] = True
            pts = paths[i]["pts"] if e == 0 else paths[i]["pts"][::-1]
            chain.extend(pts if not chain else pts[1:])
            other = (i, 1 - e)
            nxt = pair.get(other)
            if not nxt or paths[nxt[0]]["used"]:
                break
            i, e = nxt
        return chain

    merged = []
    for i, p in enumerate(paths):
        if p["used"]:
            continue
        for e in (0, 1):
            if (i, e) not in pair:
                chain = trace((i, e))
                if len(chain) >= 2:
                    merged.append(chain)
                break
    for i, p in enumerate(paths):
        if p["used"]:
            continue
        chain = trace((i, 0))
        if len(chain) >= 2:
            merged.append(chain)

    out = []
    for pts in merged:
        if len(pts) > 2:
            pts = _rdp(pts, 0.9)
        if len(pts) >= 2:
            out.append([[round(x, 2), round(y, 2)] for x, y in pts])
    return out

def skeleton_array(arr):
    from skimage.morphology import skeletonize
    from skan import Skeleton

    mask = arr < 160
    skel = skeletonize(mask)
    if not skel.any():
        return []
    sk = Skeleton(skel)
    return _merge_skeleton_paths(sk)


def _skeleton_strokes(png: Path):
    return skeleton_array(np.array(Image.open(png).convert("L")))


def generate(chars):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    todo = []
    for ch in chars:
        cp = CACHE_DIR / f"{ord(ch):04X}.json"
        if cp.exists():
            try:
                result[ch] = json.loads(cp.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        todo.append(ch)

    if todo:
        with tempfile.TemporaryDirectory(prefix="hw_ai_") as td:
            td = Path(td)
            npz = td / "test.npz"
            out = td / "out"
            _build_npz(todo, npz)
            cmd = [
                sys.executable, "-u", "generate_chars.py",
                "--checkpoint", str(CHECKPOINT),
                "--test_npz", str(npz),
                "--output_dir", str(out),
                "--sampling_method", "ab2",
                "--cfg", "4.5",
                "--num_sampling_steps", "50",
                "--batch_size", str(max(1, len(todo))),
                "--lora_r", "32",
                "--lora_alpha", "32",
            ]
            env = dict(os.environ)
            env.setdefault("TORCH_COMPILE_DISABLE", "1")
            env.setdefault("TORCHDYNAMO_DISABLE", "1")
            p = subprocess.run(cmd, cwd=str(REPO), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=600)
            if p.returncode != 0:
                raise RuntimeError("generation failed:\n" + (p.stdout or "")[-2000:])
            files = glob.glob(str(out / "**" / "*.png"), recursive=True)
            by_cp = {}
            for f in files:
                name = Path(f).stem
                if "U+" not in name:
                    continue
                code = int(name.split("U+")[1].split("_")[0][:4], 16)
                by_cp[code] = f
            for ch in todo:
                f = by_cp.get(ord(ch))
                if not f:
                    result[ch] = []
                    continue
                strokes = _skeleton_strokes(Path(f))
                result[ch] = strokes
                (CACHE_DIR / f"{ord(ch):04X}.json").write_text(
                    json.dumps(strokes, ensure_ascii=False), encoding="utf-8"
                )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", required=True, help="characters to generate (one string)")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()
    chars = []
    for ch in args.chars:
        if ch.strip() and ch not in chars:
            chars.append(ch)
    chars = chars[:20]
    result = generate(chars)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
