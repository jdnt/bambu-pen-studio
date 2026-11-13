#!/usr/bin/env python3
"""Long-lived zi2zi-JiT worker for the layout page.

Protocol: one JSON object per stdin line {"chars": ["永", ...]};
one JSON object per stdout line {"ok": true, "strokes": {...}}.
Model is loaded once at startup; stderr is for logs.
"""
import base64
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("HW_AI_ROOT", "/mnt/qnap_models/bambu-font-ai"))
REPO = ROOT / "src" / "zi2zi-JiT"
ALL_CHECKPOINTS = {
    0: ROOT / "run" / "user2000_B_full" / "checkpoint-last.pth",
    1: ROOT / "run" / "user2000_brush1" / "checkpoint-last.pth",
}
_FONT_ENV = os.environ.get("HW_AI_FONT")
CHECKPOINTS = ({int(_FONT_ENV): ALL_CHECKPOINTS[int(_FONT_ENV)]}
               if _FONT_ENV not in (None, "") else ALL_CHECKPOINTS)
BASE_CHECKPOINT = ROOT / "models" / "zi2zi-JiT-models" / "zi2zi-JiT-B-16.pth"
LORA_CHECKPOINTS = {
    0: ROOT / "run" / "user2000_B_full" / "checkpoint-last.pth",
    1: ROOT / "run" / "user2000_brush1" / "checkpoint-last.pth",
}
USE_ORIGINAL_BASE = False
LABEL_FONT = int(os.environ.get("HW_AI_FONT_LABEL", "-1"))
CACHE_DIR = ROOT / "data" / "ai_cache"
BEST_N = max(1, min(4, int(os.environ.get("HW_AI_BEST_N", "1"))))
AI_STEPS = int(os.environ.get("HW_AI_STEPS", "50"))
WEB_DIR = Path(__file__).resolve().parent

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WEB_DIR))

import numpy as np
import torch

import ai_generate as ag
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image as PILImage
from denoiser import Denoiser
from util.lora_utils import inject_lora, _is_lora_state_dict
from generate_chars import resolve_device, patch_torch_for_device


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_model(checkpoint_path, prefer_model=False):
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    patch_torch_for_device(device)
    log("loading", checkpoint_path)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    model = Denoiser(ckpt_args)
    state = (ckpt.get("model") if prefer_model else (ckpt.get("model_ema1") or ckpt.get("model"))) or ckpt
    if _is_lora_state_dict(state):
        r = getattr(ckpt_args, "lora_r", 32)
        alpha = getattr(ckpt_args, "lora_alpha", 32)
        targets = [t.strip() for t in getattr(ckpt_args, "lora_targets", "qkv,proj,w12,w3").split(",") if t.strip()]
        inject_lora(model.net, targets, r=r, alpha=alpha, dropout=0.0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log("state_dict missing/unexpected", len(missing), len(unexpected))
    model.to(device)
    model.eval()
    model.cfg_scale = 2.6
    model.steps = AI_STEPS
    model.method = "ab2"
    model.cfg_interval = (0.0, 1.0)
    log("worker model ready on", device)
    return model, device


ADAPTER_MODE = os.environ.get("HW_AI_ADAPTER") == "1"
BRUSH_CKPT = ROOT / "run" / "user2000_brush1" / "checkpoint-last.pth"


def load_original_lora(font_id):
    """Original 2.2G JiT-B/16 base + hand/brush LoRA + font row."""
    model, device = load_model(BASE_CHECKPOINT, prefer_model=True)
    inject_lora(model.net, ["qkv", "proj", "w12", "w3"], r=32, alpha=32, dropout=0.0)
    full = torch.load(str(LORA_CHECKPOINTS[font_id]), map_location="cpu", weights_only=False)
    fstate = full.get("model_ema1") or full.get("model") or full
    lora = {k: v for k, v in fstate.items() if "lora_A" in k or "lora_B" in k}
    missing, unexpected = model.load_state_dict(lora, strict=False)
    log("original base lora", font_id, len(lora), "missing", len(missing), "unexpected", len(unexpected))
    fk = "net.y_embedder.font_embedding.weight"
    if fk in fstate:
        with torch.no_grad():
            model.net.y_embedder.font_embedding.weight[font_id].copy_(fstate[fk][font_id])
    for m in model.net.modules():
        if hasattr(m, "lora_A"):
            m.lora_A.data = m.lora_A.data.to(device)
            m.lora_B.data = m.lora_B.data.to(device)
            if hasattr(m, "scaling"):
                m.scaling = m.scaling.to(device)
    model.eval()
    log("original base lora ready", font_id)
    return model, device


def load_adapter_model():
    """Merged full2000 base + brush LoRA adapter + brush font row1.

    font0: adapter off, base merged full.
    font1: adapter on, brush LoRA; font row1 from brush.
    """
    base_path = CHECKPOINTS[0]
    model, device = load_model(base_path)
    r = 32
    alpha = 32
    targets = ["qkv", "proj", "w12", "w3"]
    inject_lora(model.net, targets, r=r, alpha=alpha, dropout=0.0)
    brush = torch.load(str(BRUSH_CKPT), map_location="cpu", weights_only=False)
    bstate = brush.get("model_ema1") or brush.get("model") or brush
    lora = {k: v for k, v in bstate.items() if "lora_A" in k or "lora_B" in k}
    missing, unexpected = model.load_state_dict(lora, strict=False)
    log("adapter lora loaded", len(lora), "missing", len(missing), "unexpected", len(unexpected))
    # font embedding: row 0 stays full2000; row 1 from brush
    fk = "net.y_embedder.font_embedding.weight"
    if fk in bstate:
        with torch.no_grad():
            model.net.y_embedder.font_embedding.weight[1].copy_(bstate[fk][1])
    for m in model.net.modules():
        if hasattr(m, "lora_A"):
            m.lora_A.data = m.lora_A.data.to(device)
            m.lora_B.data = m.lora_B.data.to(device)
            if hasattr(m, "scaling"):
                m.scaling = m.scaling.to(device)
    model.eval()
    log("adapter model ready on", device)
    return model, device


def set_font_adapter(model, font_id):
    for m in model.net.modules():
        if hasattr(m, "lora_A") and hasattr(m, "r"):
            m.r = 32 if font_id == 1 else 0


def _shape_score(gen_gray, content_gray):
    g = np.asarray(gen_gray) < 160
    c = np.asarray(content_gray) < 160
    if not g.any() or not c.any():
        return 1e9
    dg = edt(~g)
    dc = edt(~c)
    return float((dg[c].mean() + dc[g].mean()) / 2.0)


def _png_data_url(arr):
    # transparent background keeps the brush silhouette without a grey square
    g = PILImage.fromarray(arr).convert("L")
    a = 255 - np.asarray(g, dtype=np.uint8)
    rgba = PILImage.new("RGBA", g.size, (0, 0, 0, 0))
    rgba.putalpha(PILImage.fromarray(a))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def generate_best(model, device, chars, n=None, font_id=0):
    n = n or BEST_N
    if not chars:
        return {}
    expanded = []
    for ch in chars:
        expanded.extend([ch] * n)
    arrays = ag.build_arrays(expanded, font_id)
    style = torch.from_numpy(arrays["style_images"]).float().to(device)
    style = style / 255.0 * 2.0 - 1.0
    content = torch.from_numpy(arrays["content_images"]).float().to(device)
    content = content / 255.0 * 2.0 - 1.0
    font_labels = torch.from_numpy(arrays["font_labels"]).long().to(device)
    char_labels = torch.from_numpy(arrays["char_labels"]).long().to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            generated = model.generate((font_labels, char_labels, style, content))
    generated = ((generated + 1) / 2).clamp(0, 1)
    arrs = (generated.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    best = {}
    for i, ch in enumerate(chars):
        idxs = list(range(i * n, (i + 1) * n))
        content_gray = arrays["content_images"][i * n].mean(axis=0) * 255.0
        scores = []
        for j in idxs:
            gray = np.array(ag.Image.fromarray(arrs[j]).convert("L"))
            scores.append(_shape_score(gray, content_gray))
        pick = idxs[int(np.argmin(scores))]
        best[ch] = arrs[pick]
    return best


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if USE_ORIGINAL_BASE:
        fid = int(_FONT_ENV) if _FONT_ENV not in (None, "") else 0
        model, device = load_original_lora(fid)
        models = {fid: (model, device)}
    elif ADAPTER_MODE:
        model, device = load_adapter_model()
        models = {0: (model, device), 1: (model, device)}
    else:
        models = {fid: load_model(path) for fid, path in CHECKPOINTS.items()}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            raw = [str(ch) for ch in req.get("chars", []) if str(ch).strip()]
            font_id = 1 if int(req.get("font", 0) or 0) == 1 else 0
            if LABEL_FONT >= 0:
                font_id = LABEL_FONT
            if req.get("variants"):
                chars = raw[:20]
                if not chars:
                    print(json.dumps({"ok": True, "variants": {}}, ensure_ascii=False), flush=True)
                    continue
                model, device = models.get(font_id) or next(iter(models.values()))
                if ADAPTER_MODE:
                    set_font_adapter(model, font_id)
                best = generate_best(model, device, chars, font_id=font_id)
                out = {}
                images = {}
                for ch in chars:
                    gray = np.array(ag.Image.fromarray(best[ch]).convert("L"))
                    out.setdefault(ch, []).append(ag.skeleton_array(gray))
                    if font_id == 1:
                        images[ch] = _png_data_url(best[ch])
                print(json.dumps({"ok": True, "variants": out, "images": images}, ensure_ascii=False), flush=True)
                continue
            chars = []
            for ch in raw:
                if ch not in chars:
                    chars.append(ch)
            chars = chars[:20]
            out = {}
            todo = []
            for ch in chars:
                cp = CACHE_DIR / f"{font_id}_{ord(ch):04X}.json"
                if cp.exists():
                    try:
                        out[ch] = json.loads(cp.read_text(encoding="utf-8"))
                        continue
                    except Exception:
                        pass
                todo.append(ch)
            if todo:
                model, device = models.get(font_id) or next(iter(models.values()))
                if ADAPTER_MODE:
                    set_font_adapter(model, font_id)
                best = generate_best(model, device, todo, font_id=font_id)
                for ch in todo:
                    gray = np.array(ag.Image.fromarray(best[ch]).convert("L"))
                    strokes = ag.skeleton_array(gray)
                    out[ch] = strokes
                    (CACHE_DIR / f"{font_id}_{ord(ch):04X}.json").write_text(
                        json.dumps(strokes, ensure_ascii=False), encoding="utf-8"
                    )
            print(json.dumps({"ok": True, "strokes": out}, ensure_ascii=False), flush=True)
        except Exception as e:
            log("request failed:", repr(e))
            print(json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
