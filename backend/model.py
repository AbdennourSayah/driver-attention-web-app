"""DRNetRGB — RGB-only driver attention saliency model.

This module mirrors the architecture from the training script
(`pasted-1778045325259.md`) so checkpoints saved during training
(`rgb.pth`, `best_model.pth`, etc.) load 1:1 without remapping.

Architecture summary
--------------------
Backbone : torchvision.models.video.r3d_18 (no pretrained weights at
           module-construction time — we always load `rgb.pth`)
Input    : (B, 3, T, H, W) where T = 16, H = 112, W = 192
Decoder  : 4 stages of {Conv3x3 -> BN -> ReLU -> Dropout2d} with skip
           connections from temporally-pooled encoder features.
Head     : Conv3x3 -> ReLU -> Conv1x1 -> Sigmoid
Output   : (B, 1, SAL_H, SAL_W) saliency probability map in [0, 1]

The defaults below match the training config exactly so the same
checkpoint can be loaded without any flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models


# Image normalization stats — must match training (Kinetics R3D-18 stats).
RGB_MEAN: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645)
RGB_STD: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989)


@dataclass
class ModelConfig:
    """Subset of the training Config that affects the architecture."""

    CLIP_LEN: int = 16
    IMG_H: int = 112
    IMG_W: int = 192
    SAL_H: int = 112
    SAL_W: int = 192
    DROPOUT: float = 0.30


class DRNetRGB(nn.Module):
    """Driver attention saliency network — RGB-only baseline."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.out_size: Tuple[int, int] = (cfg.SAL_H, cfg.SAL_W)

        r3d = video_models.r3d_18(weights=None)
        self.stem = r3d.stem
        self.l1 = r3d.layer1
        self.l2 = r3d.layer2
        self.l3 = r3d.layer3
        self.l4 = r3d.layer4

        self.up4 = self._block(512, 256)
        self.up3 = self._block(256, 128)
        self.up2 = self._block(128, 64)
        self.up1 = self._block(64, 32)

        self.drop = nn.Dropout2d(p=cfg.DROPOUT)

        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _temporal_pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W)
        x = self.stem(x)
        s1 = self.l1(x)
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)

        p4 = self._temporal_pool(s4)
        p3 = self._temporal_pool(s3)
        p2 = self._temporal_pool(s2)
        p1 = self._temporal_pool(s1)

        d = self.up4(p4)
        d = self.drop(d)
        d = F.interpolate(d, size=p3.shape[-2:], mode="bilinear", align_corners=False) + p3
        d = self.up3(d)
        d = self.drop(d)
        d = F.interpolate(d, size=p2.shape[-2:], mode="bilinear", align_corners=False) + p2
        d = self.up2(d)
        d = self.drop(d)
        d = F.interpolate(d, size=p1.shape[-2:], mode="bilinear", align_corners=False) + p1
        d = self.up1(d)
        d = F.interpolate(d, size=self.out_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(d))


def _safe_torch_load(path: str, map_location: str | torch.device = "cpu"):
    """Load a checkpoint file regardless of PyTorch version quirks."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_rgb_checkpoint(
    model: DRNetRGB,
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> dict:
    """Load weights from a training-format checkpoint into ``model``.

    Accepts checkpoints saved as either a raw ``state_dict`` or a dict
    containing one of: ``state_dict``, ``model_state_dict``,
    ``model_state``, ``model``, ``net``. Strips ``module.`` prefixes
    (DataParallel) and shape-filters tensors so a partial-architecture
    match still loads cleanly.

    Returns a small dict with the keys ``loaded``, ``missing``,
    ``unexpected``, and ``skipped`` to help diagnose any mismatch.
    """
    ckpt = _safe_torch_load(checkpoint_path, map_location=device)

    state = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model_state", "model", "net"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                state = value
                break
        if state is None:
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    elif isinstance(ckpt, (list, tuple)):
        # Some custom training scripts save (state_dict, meta).
        state = next((c for c in ckpt if isinstance(c, dict)), None)
    if not state:
        raise RuntimeError(
            f"Could not extract a state_dict from checkpoint: {checkpoint_path}"
        )

    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    current = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in state.items():
        if k in current and hasattr(v, "shape") and v.shape == current[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "best_val_cc": ckpt.get("best_val_cc") if isinstance(ckpt, dict) else None,
    }


def build_rgb_model(
    checkpoint_path: str | None = None,
    device: str | torch.device = "cpu",
    cfg: ModelConfig | None = None,
) -> tuple[DRNetRGB, dict | None]:
    """Build :class:`DRNetRGB` and (optionally) load weights from disk."""
    model = DRNetRGB(cfg)
    info: dict | None = None
    if checkpoint_path:
        info = load_rgb_checkpoint(model, checkpoint_path, device)
    model = model.to(device).eval()
    return model, info
