#!/usr/bin/env python3
"""Build font-1 (brush) training data from selected, recognized variants.

- only variants with selected=true and char not in ('?','') are used
- one image per character (the selected variant)
- global uniform scale across all selected font-1 glyphs (same ratio, no per-glyph fill)
- writes data/font1_training/{raw,norm}, then a one-font dataset under data/font1_dataset
Run on the training server with the fontai venv.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path("/mnt/qnap_models/bambu-font-ai")
WEB = ROOT / "webapp"
STATE = ROOT / "data" / "font1" / "variants.json"
RAW = ROOT / "data" / "font1_training" / "raw"
CLEAN = ROOT / "data" / "font1_clean"
NORM = ROOT / "data" / "font1_training" / "norm"
DATASET = ROOT / "data" / "font1_dataset"
NORMALIZE = ROOT / "src" / "zi2zi-JiT" / "scripts" / "normalize_glyphs.py"
GEN = ROOT / "src" / "zi2zi-JiT" / "scripts" / "generate_glyph_dataset.py"
NOTO = ROOT / "data" / "source" / "NotoSerifCJKsc-Regular.otf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", type=float, default=0.86)
    ap.add_argument("--font-name", default="brush")
    args = ap.parse_args()
    st = json.loads(STATE.read_text(encoding="utf-8"))
    chosen = {}
    ref = None
    for v in st.get("variants", []):
        ch = v.get("char")
        if not v.get("selected") or not ch or ch == "?":
            continue
        if ch in chosen:
            continue  # one selected variant per char
        crop = ROOT / "data" / "font1" / v["crop"]
        # train raw selected crops; no background cleanup for now
        if crop.exists():
            chosen[ch] = crop
            if ref is None:
                ref = crop
    if not chosen:
        raise SystemExit("no selected+recognized font1 variants")
    RAW.mkdir(parents=True, exist_ok=True)
    NORM.mkdir(parents=True, exist_ok=True)
    for p in RAW.glob("*.png"):
        p.unlink()
    for ch, src in chosen.items():
        shutil.copyfile(src, RAW / f"{ch}.png")
    (ROOT / "data" / "font1_training" / "style_ref.png").write_bytes(ref.read_bytes())
    (ROOT / "data" / "font1_training" / "chars.txt").write_text("".join(chosen), encoding="utf-8")
    cmd = [sys.executable, str(NORMALIZE), "--src", str(RAW), "--dst", str(NORM),
           "--canvas", "256", "--fill", str(args.fill)]
    subprocess.run(cmd, check=True)
    # one font -> train/001_brush; rename to 002 so font_idx = int("002")-1 = 1
    if DATASET.exists():
        shutil.rmtree(DATASET)
    cmd = [sys.executable, str(GEN),
           "--source-font", str(NOTO),
           "--glyph-dir", str(NORM),
           "--output-dir", str(DATASET),
           "--train-count", str(len(chosen)),
           "--font-name", args.font_name,
           "--seed", "42", "--train-only"]
    subprocess.run(cmd, check=True)
    src = DATASET / "train" / f"001_{args.font_name}"
    dst = DATASET / "train" / f"002_{args.font_name}"
    if src.exists() and not dst.exists():
        src.rename(dst)
    print(json.dumps({"ok": True, "chars": len(chosen), "raw": str(RAW),
                      "norm": str(NORM), "dataset": str(DATASET),
                      "font_dir": str(dst)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
