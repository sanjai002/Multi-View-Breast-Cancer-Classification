#!/usr/bin/env python3
"""Single-breast inference: give a CC and an MLO DICOM, get P(cancer).

Usage:
    python scripts/inference.py --checkpoint <best.pt> --cc <cc.dcm> --mlo <mlo.dcm>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils.seed import get_device
from src.models.fusion import build_model
from src.data.preprocessing import preprocess_image
from src.data.augmentation import eval_transforms
from src.data.dataset import _DictCfg


def load_view(path, pcfg, tfm, key):
    img = preprocess_image(path, _DictCfg(dict(pcfg)))
    img = np.repeat(img[..., None], 3, axis=-1)
    return tfm(image=img, mlo=img)[key].unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cc", required=True)
    ap.add_argument("--mlo", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    model = build_model(cfg.model).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(state)

    size = int(cfg.preprocess.img_size)
    tfm = eval_transforms(size)
    cc = load_view(args.cc, cfg.preprocess, tfm, "image").to(device)
    mlo = load_view(args.mlo, cfg.preprocess, tfm, "mlo").to(device)
    with torch.no_grad():
        prob = torch.softmax(model(cc, mlo), dim=1)[0, 1].item()
    print(f"P(cancer) = {prob:.4f}  ->  prediction: "
          f"{'CANCER' if prob >= 0.5 else 'NORMAL'}")


if __name__ == "__main__":
    main()
