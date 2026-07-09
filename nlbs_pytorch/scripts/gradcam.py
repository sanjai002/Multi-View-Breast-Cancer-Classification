#!/usr/bin/env python3
"""
Explainability for the dual-view model: Grad-CAM, Grad-CAM++, Score-CAM,
and Integrated Gradients, computed on the CC branch (MLO held fixed).

Usage:
    python scripts/gradcam.py --checkpoint <best.pt> [--n 6]
Outputs overlays to <output_root>/explain/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients

from src.config import load_config
from src.utils.seed import get_device
from src.models.fusion import build_model
from src.models.backbones import last_conv_module
from src.data.preprocessing import preprocess_image
from src.data.augmentation import eval_transforms
from src.data.dataset import _DictCfg
import pandas as pd


class CCWrapper(nn.Module):
    """Expose model(cc) with MLO fixed, so single-input CAM libraries work."""
    def __init__(self, model, mlo):
        super().__init__()
        self.model = model
        self.mlo = mlo

    def forward(self, cc):
        return self.model(cc, self.mlo)


def denorm(t):
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img = t.cpu().numpy().transpose(1, 2, 0) * std + mean
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    model = build_model(cfg.model).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"])

    df = pd.read_csv(cfg.paths.splits_csv)
    test = df[df.split == "test"]
    sample = pd.concat([test[test.label == 1].head(args.n // 2),
                        test[test.label == 0].head(args.n - args.n // 2)])
    size = int(cfg.preprocess.img_size)
    tfm = eval_transforms(size)
    out = Path(cfg.paths.output_root) / "explain"
    out.mkdir(parents=True, exist_ok=True)
    target_layer = (model.backbone if cfg.model.fusion == "early"
                    else model.enc_cc)
    target_layer = last_conv_module(cfg.model.backbone, target_layer)

    for i, (_, row) in enumerate(sample.iterrows()):
        cc_img = np.repeat(preprocess_image(row["cc_path"], _DictCfg(dict(cfg.preprocess)))[..., None], 3, -1)
        mlo_img = np.repeat(preprocess_image(row["mlo_path"], _DictCfg(dict(cfg.preprocess)))[..., None], 3, -1)
        t = tfm(image=cc_img, mlo=mlo_img)
        cc = t["image"].unsqueeze(0).to(device)
        mlo = t["mlo"].unsqueeze(0).to(device)
        wrapper = CCWrapper(model, mlo)
        rgb = denorm(t["image"])
        targets = [ClassifierOutputTarget(1)]

        panels = [(rgb * 255).astype(np.uint8)]
        for name, Cam in [("gradcam", GradCAM), ("gradcam++", GradCAMPlusPlus),
                          ("scorecam", ScoreCAM)]:
            with Cam(model=wrapper, target_layers=[target_layer]) as cam:
                g = cam(input_tensor=cc, targets=targets)[0]
            panels.append((show_cam_on_image(rgb, g, use_rgb=True)).astype(np.uint8))

        ig = IntegratedGradients(wrapper)
        attr = ig.attribute(cc, target=1, baselines=cc * 0, n_steps=16)
        amap = attr.squeeze(0).abs().sum(0).cpu().numpy()
        amap = (amap - amap.min()) / (amap.ptp() + 1e-8)
        panels.append(show_cam_on_image(rgb, amap, use_rgb=True).astype(np.uint8))

        strip = np.concatenate(panels, axis=1)
        labels = ["orig", "GradCAM", "GradCAM++", "ScoreCAM", "IntegratedGrad"]
        for j, lab in enumerate(labels):
            cv2.putText(strip, lab, (j * size + 5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cls = "cancer" if row["label"] == 1 else "normal"
        cv2.imwrite(str(out / f"explain_{i:02d}_{cls}_{row['patient_id']}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print(f"[gradcam] wrote explain_{i:02d}_{cls}")
    print(f"[gradcam] -> {out}")


if __name__ == "__main__":
    main()
