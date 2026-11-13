#!/usr/bin/env python3
"""Persistent GPU FontDiffuser worker for layout comparison.

Protocol: one JSON object per stdin line {"chars": [...]}; one JSON line out.
Output strokes are 256x256 coordinates, same schema as ai_worker.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
REPO = "/root/model-lab/FontDiffuser"
TEST = "/root/model-lab/fontdiffuser-test"
CKPT = os.environ.get("HW_FD_CKPT", "/root/model-lab/FontDiffuser/ckpt_user926_thin")
TTF = "/mnt/qnap_models/bambu-font-ai/data/source/NotoSerifCJKsc-Regular.otf"
GLYPH_DIR = Path("/mnt/qnap_models/bambu-font-ai/data/user_glyphs")
sys.path.insert(0, REPO)
sys.path.insert(0, TEST)
sys.path.insert(0, "/mnt/qnap_models/bambu-font-ai/webapp")

REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import utils
from run_infer import get_args, load_pipe
import ai_generate as ag


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def crop_fill(im, fill=0.8, canvas=256, thr=128):
    a = np.array(im.convert("L"))
    ys, xs = np.where(a < thr)
    if len(xs) == 0:
        return im.resize((canvas, canvas))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = im.convert("RGB").crop((x0, y0, x1 + 1, y1 + 1))
    w, h = crop.size
    target = int(round(canvas * fill))
    sc = target / max(w, h)
    crop = crop.resize((max(1, int(round(w * sc))), max(1, int(round(h * sc)))), Image.LANCZOS)
    out = Image.new("RGB", (canvas, canvas), "white")
    out.paste(crop, ((canvas - crop.width) // 2, (canvas - crop.height) // 2))
    return out


def main():
    args = get_args()
    args.ckpt_dir = CKPT
    args.device = "cuda:0"
    args.num_inference_steps = 20
    args.guidance_scale = 7.5
    pipe = load_pipe(args)
    log("font worker model ready")
    ct = transforms.Compose([
        transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    st = transforms.Compose([
        transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    font = utils.load_ttf(TTF, fsize=512)
    style_path = Path("/root/model-lab/fontdiffuser-train/dataset/train/TargetImage/style0/style0+你.png")
    if not style_path.exists():
        style_path = GLYPH_DIR / "你.png"
    style_img = crop_fill(Image.open(style_path), 0.8)
    style = st(style_img)[None].to(args.device)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            chars = [str(c) for c in req.get("chars", []) if str(c).strip()][:20]
            out = {}
            variants = bool(req.get("variants"))
            for ch in chars:
                content_img = crop_fill(utils.ttf2im(font, ch, fsize=512), 0.8)
                content = ct(content_img)[None].to(args.device)
                torch.manual_seed(123)
                torch.cuda.manual_seed_all(123)
                with torch.no_grad():
                    gen = pipe.generate(content_images=content, style_images=style, batch_size=1, order=2,
                                        num_inference_step=20, content_encoder_downsample_size=3,
                                        t_start=None, t_end=None, dm_size=(96, 96),
                                        algorithm_type="dpmsolver++", skip_type="time_uniform",
                                        method="multistep", correcting_x0_fn=None)
                im = gen[0].resize((256, 256), Image.LANCZOS)
                gray = np.array(im.convert("L"))
                strokes = ag.skeleton_array(gray)
                if variants:
                    out.setdefault(ch, []).append(strokes)
                else:
                    out[ch] = strokes
            print(json.dumps({"ok": True, "variants" if variants else "strokes": out}, ensure_ascii=False), file=REAL_STDOUT, flush=True)
        except Exception as e:
            log("font request failed", repr(e))
            print(json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False), file=REAL_STDOUT, flush=True)


if __name__ == "__main__":
    main()
