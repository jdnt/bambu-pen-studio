#!/usr/bin/env python3
"""Merge a LoRA-injected zi2zi checkpoint into a plain full checkpoint."""
import argparse, sys
from pathlib import Path
import torch

ROOT = Path("/mnt/qnap_models/bambu-font-ai")
sys.path.insert(0, str(ROOT / "src" / "zi2zi-JiT"))
from denoiser import Denoiser
from util.lora_utils import inject_lora, _is_lora_state_dict


def merge(src, dst):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    state = ckpt.get("model_ema1") or ckpt.get("model") or ckpt
    if not _is_lora_state_dict(state):
        print("not a LoRA state dict; copying as-is")
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, dst)
        return
    model = Denoiser(args)
    r = getattr(args, "lora_r", 32)
    alpha = getattr(args, "lora_alpha", 32)
    targets = [t.strip() for t in getattr(args, "lora_targets", "qkv,proj,w12,w3").split(",") if t.strip()]
    inject_lora(model.net, targets, r=r, alpha=alpha, dropout=0.0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("load", len(missing), len(unexpected))
    scale = float(alpha) / float(r)
    merged = 0
    for name, module in list(model.net.named_modules()):
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        with torch.no_grad():
            delta = (module.lora_B @ module.lora_A) * scale
            module.base.weight.add_(delta)
        merged += 1
    print("merged lora modules", merged)
    for name, module in list(model.net.named_modules()):
        if hasattr(module, "lora_A") and hasattr(module, "base"):
            parent_name, _, child = name.rpartition(".")
            parent = model.net.get_submodule(parent_name) if parent_name else model.net
            setattr(parent, child, module.base)
    plain = model.state_dict()
    assert not any(".base." in k or "lora_" in k for k in plain), "plain state still has LoRA keys"
    out_args = argparse.Namespace(**vars(args))
    out_args.lora_r = 0
    out_args.lora_alpha = 0
    out_args.lora_targets = ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": plain, "epoch": ckpt.get("epoch", -1), "args": out_args}, dst)
    print("saved", dst, "keys", len(plain))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    merge(Path(a.src), Path(a.dst))
