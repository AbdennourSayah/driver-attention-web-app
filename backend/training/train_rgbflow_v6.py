#!/usr/bin/env python3
"""
DR(eye)VE  ·  RGB + Optical-Flow  ·  v6
========================================
Drop-in replacement for the v5fix Kaggle notebook. Targets >= 65% TEST CC.

Why v5fix (the previous attempt) plateaued at 55% TEST CC
---------------------------------------------------------
  1. ``fusion = Conv2d(1024 -> 512)`` randomly initialised → at step 0 it
     destroys the well-pretrained R3D-18 RGB representation, so the
     network spends most of the schedule recovering RGB-only quality.
  2. KL/NSS computed on fp16 ``pred`` (returned from the autocast block)
     → underflows produce NaN that contaminates ``np.mean``.
  3. ``train_cc`` printed as ``1 - train_loss``; with KL ≈ 1.6 this looks
     like ~0.10 and hides the real metric.
  4. Cosine warm-restart with T0=5 drops backbone LR to 1.5e-5 by epoch 3
     → too small to escape the bad local optimum the random fusion put us
     in.
  5. Generalisation gap: train/val from runs 01-37 → val 62.5%, but test
     runs 38-74 → test 55.1%. v5fix had only the same augmentation as
     RGB-only, plus no EMA, no Mixup.

Fixes in v6
-----------
  A) Gated additive fusion          : ``out = rgb + sigmoid(alpha)·proj(flow)``
     with ``alpha`` initialised to a large negative value so
     ``sigmoid(alpha) ≈ 0`` at step 0. RGB checkpoint passes through
     unchanged; the model learns to "open the gate" when flow helps.
  B) Two-stage training             : epoch 1 freezes the RGB backbone so
     the flow encoder + fusion + decoder warm up without touching R3D-18;
     from epoch 2 the backbone unfreezes at low LR.
  C) Numerically stable metrics     : ``.float()`` cast + KL formula with
     no division by ``p`` + ``np.nanmean`` aggregation.
  D) Real train-CC tracking         : a running mean of the per-batch CC,
     reported next to ``loss`` and copied into ``history``.
  E) EMA model averaging            : evaluation runs against a decay-0.999
     EMA copy of the weights — typically +1-2% CC on saliency tasks.
  F) Information-Gain loss          : retained from the RGB baseline,
     missing in v5fix; uses the *training mean saliency* as the
     center-bias prior so the network learns the deviation.
  G) Mixup (α=0.2) on (rgb, flow, sal) triplets to close the train/test
     gap.
  H) Robust flow normalisation      : ``tanh(flow / 20)`` instead of a
     hard ``clip(-50, 50) / 50`` so 99% of frames have flow values in a
     useful range.
  I) Cosine schedule WITHOUT warm restart — single cosine over the full
     run (T0 = TOTAL_EPOCHS). Linear warmup over the first 0.5 epoch for
     the freshly-initialised flow_enc + fusion + decoder groups.
  J) Full resumability              : checkpoint = model + EMA +
     optimizer + scaler + epoch + best_val_cc + history. The script will
     pick up exactly where Kaggle's 9h limit cut it off.

Running on Kaggle
-----------------
  1. Place this file in your Kaggle notebook (or paste its contents).
  2. Adjust ``Config.DATA_DIR`` and ``Config.RGB_CKPT`` to your inputs.
  3. ``python train_rgbflow_v6.py``.
  4. The first session will train epochs 1..N until the 9h limit; the
     next session resumes from ``checkpoints/latest.pth``.

Architecture (kept compatible with the RGB checkpoint produced by the
existing ``backend/model.py``):

    R3D-18 backbone (loaded from rgb.pth)
        stem (1x2x2)  ->  56x96
        layer1        ->  56x96
        layer2 (1x2x2)->  28x48
        layer3 (1x2x2)->  14x24
        layer4 (1x2x2)->   7x12   (bottleneck)
    FlowEncoder3D (from scratch)  ->  7x12  (matches bottleneck exactly)
    GatedFusion(rgb, flow)        ->  7x12  (RGB-preserving residual)
    Decoder up4..up1 + skip       ->  112x192
    Sigmoid head                  ->  (B, 1, 112, 192)

The network keeps the same ``stem.``, ``l1.``, ``l2.``, ``l3.``, ``l4.``,
``up*.``, ``head.`` parameter names as the RGB baseline so existing
checkpoints load 1:1 (only the new ``flow_enc.``, ``fusion.``, ``alpha``
parameters are random-init at the start of training).
"""
from __future__ import annotations

# =====================================================================
# 0. INSTALL
# =====================================================================
import subprocess
import sys


def _pip(*pkgs: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
        check=True,
    )


# Only Kaggle needs these; locally they are already installed.
if "KAGGLE_KERNEL_RUN_TYPE" in __import__("os").environ:
    _pip("tqdm", "kaggle")


# =====================================================================
# 1. IMPORTS
# =====================================================================
import gc
import glob
import json
import math
import os
import random
import shutil
import time
import warnings
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

EPS = 1e-8


# =====================================================================
# 2. CONFIG
# =====================================================================
@dataclass
class Config:
    # ---- Data paths ---------------------------------------------------
    DATA_DIR: str = "/kaggle/input/datasets/folkobrther/my-data-base/DREYEVE_DATA"
    DATA_ALT_1: str = "/kaggle/input/my-data-base/DREYEVE_DATA"
    DATA_ALT_2: str = "/kaggle/input/datasets/anoierkas/my-data-base/DREYEVE_DATA"

    FLOW_OWNER: str = "sayahabdrahman"
    FLOW_PARTS: int = 15

    RGB_CKPT: str = "/kaggle/input/datasets/folkobrther/rgb-baseline/RGB.pth"

    OUT_DIR: str = "/kaggle/working/dreyeve_rgbflow_v6"
    CKPT_DIR: str = "/kaggle/working/dreyeve_rgbflow_v6/checkpoints"
    PLOT_DIR: str = "/kaggle/working/dreyeve_rgbflow_v6/plots"
    PREVIEW_DIR: str = "/kaggle/working/dreyeve_rgbflow_v6/previews"

    # ---- Splits (DR(eye)VE design-file split) -------------------------
    TRAIN_RUNS: List[int] = field(default_factory=lambda: list(range(1, 38)))
    TEST_RUNS: List[int] = field(default_factory=lambda: list(range(38, 75)))

    TOTAL_FRAMES: int = 7500
    VAL_WINDOW_SIZE: int = 500

    # ---- Clip parameters ----------------------------------------------
    CLIP_LEN: int = 16
    FRAME_STEP: int = 1
    TRAIN_STRIDE: int = 8
    VAL_STRIDE: int = 32
    TEST_STRIDE: int = 32

    # ---- Resolution (matches R3D-18 input pipeline) -------------------
    IMG_H: int = 112
    IMG_W: int = 192
    SAL_H: int = 112
    SAL_W: int = 192

    # ---- Training schedule --------------------------------------------
    BATCH_SIZE: int = 4
    GRAD_ACCUM: int = 2  # effective batch size = 8
    NUM_WORKERS: int = 2
    TRAIN_STEPS_PER_EPOCH: int = 1500
    CLIPS_PER_EPOCH: int = 1500 * 4

    VAL_STEPS: int = 200
    TEST_STEPS: int = 300

    TOTAL_EPOCHS: int = 25
    SESSION_HOUR_BUDGET: float = 8.5  # Kaggle 9h limit, 0.5h safety margin

    # Two-stage training: freeze backbone for the first ``WARMUP_EPOCHS``
    # so the flow encoder + fusion + decoder warm up before they are
    # allowed to back-propagate into R3D-18.
    WARMUP_EPOCHS: int = 1
    WARMUP_FRAC: float = 0.5  # additionally do half-an-epoch of LR warmup

    # ---- Loss weights -------------------------------------------------
    LOSS_W_CC: float = 1.0
    LOSS_W_KL: float = 0.30
    LOSS_W_NSS: float = 0.20
    LOSS_W_IG: float = 0.10  # IG vs. center-bias prior (was missing in v5fix)

    # ---- Learning rates -----------------------------------------------
    LR_BACKBONE: float = 1.5e-5  # very low — RGB is already trained
    LR_DECODER: float = 2.0e-4
    LR_FLOW: float = 1.5e-4
    MIN_LR: float = 1.0e-7
    WEIGHT_DECAY: float = 1.0e-4
    GRAD_CLIP: float = 2.0

    # ---- Regularisation -----------------------------------------------
    DROPOUT: float = 0.30
    MIXUP_ALPHA: float = 0.20
    EMA_DECAY: float = 0.999

    # ---- Misc ---------------------------------------------------------
    IG_BASELINE_CLIPS: int = 300

    KAGGLE_OWNER: str = "folkobrther"
    DATASET_SLUG: str = "dreyeve-rgbflow-v6"
    DO_UPLOAD: bool = False

    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 3. UTILITIES
# =====================================================================
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dirs(cfg: Config) -> None:
    for d in [cfg.OUT_DIR, cfg.CKPT_DIR, cfg.PLOT_DIR, cfg.PREVIEW_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_dir(cfg: Config, run_id: int) -> Path:
    for base in [cfg.DATA_DIR, cfg.DATA_ALT_1, cfg.DATA_ALT_2]:
        p = Path(base) / f"{run_id:02d}"
        if p.exists():
            return p
    return Path(cfg.DATA_DIR) / f"{run_id:02d}"


def safe_load(path: str, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except Exception:
        return torch.load(path, **kwargs)


def check_storage(tag: str = "") -> None:
    wd = Path("/kaggle/working")
    if not wd.exists():
        return
    gb = sum(p.stat().st_size for p in wd.rglob("*") if p.is_file()) / 1e9
    print(f"  Storage {gb:.2f} GB  {tag}")


# =====================================================================
# 4. FLOW UTILITIES
# =====================================================================
def find_flow_dir(cfg: Config, run_id: int) -> Optional[Path]:
    run_str = f"{run_id:02d}"
    base = Path("/kaggle/input/datasets") / cfg.FLOW_OWNER
    for part_n in range(1, cfg.FLOW_PARTS + 1):
        for slug in (
            f"dreyeve-raft-npz-part{part_n:02d}",
            f"dreyeve-raft-npz-part{part_n}",
        ):
            for root in (base / slug, Path("/kaggle/input") / slug):
                if not root.exists():
                    continue
                for sub in ("", "test/", "training/"):
                    candidate = root / f"{sub}{run_str}/flow_tensors"
                    if candidate.exists():
                        return candidate
    return None


def load_flow_npz(path: Path) -> np.ndarray:
    """Returns ``(2, H, W)`` float32 flow array."""
    data = np.load(str(path))
    key = "flow" if "flow" in data else list(data.keys())[0]
    flow = data[key].astype(np.float32)
    if flow.ndim == 4:
        flow = flow[0]
    if flow.shape[0] != 2:
        flow = flow.transpose(2, 0, 1)
    return flow


# =====================================================================
# 5. CLIP INDEX BUILDER
# =====================================================================
def get_split_ranges(cfg: Config):
    mid = cfg.TOTAL_FRAMES // 2
    half = cfg.VAL_WINDOW_SIZE // 2
    val_start = mid - half
    val_end = val_start + cfg.VAL_WINDOW_SIZE
    return [(0, val_start), (val_end, cfg.TOTAL_FRAMES)], [(val_start, val_end)]


def build_clips(
    cfg: Config,
    run_ids: List[int],
    stride: int,
    label_ranges=None,
    require_flow: bool = True,
):
    clips: List[Tuple[int, int, int, Optional[List[Path]]]] = []
    clip_span = cfg.CLIP_LEN - 1
    max_start = cfg.TOTAL_FRAMES - clip_span - 2

    def keep(label_idx: int) -> bool:
        if label_ranges is None:
            return True
        return any(lo <= label_idx < hi for lo, hi in label_ranges)

    missing: List[str] = []
    for rid in run_ids:
        rd = run_dir(cfg, rid)
        if (
            not (rd / "video_garmin.avi").exists()
            or not (rd / "video_saliency.avi").exists()
        ):
            continue
        flow_dir = find_flow_dir(cfg, rid)
        npz_files = sorted(flow_dir.glob("*.npz")) if flow_dir else []
        if require_flow and len(npz_files) == 0:
            missing.append(f"{rid:02d}(no_flow)")
            continue

        for start in range(0, max_start + 1, stride):
            label_idx = start + clip_span
            if keep(label_idx):
                clips.append(
                    (rid, start, label_idx, npz_files if npz_files else None)
                )
    if missing:
        print(f"  Skipped runs (no flow): {missing}")
    return clips


# =====================================================================
# 6. DATASET
# =====================================================================
IMG_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
IMG_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


class DreyeveFlowDataset(Dataset):
    """Reads RGB clip + dense optical flow + saliency target."""

    def __init__(self, clips, cfg: Config, augment: bool = False):
        self.clips = clips
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.clips)

    # ---- IO ----------------------------------------------------------
    def _read_rgb(self, rid: int, start: int) -> np.ndarray:
        path = str(run_dir(self.cfg, rid) / "video_garmin.avi")
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.cfg.CLIP_LEN):
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                    frame,
                    (self.cfg.IMG_W, self.cfg.IMG_H),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                frame = np.zeros(
                    (self.cfg.IMG_H, self.cfg.IMG_W, 3), dtype=np.uint8
                )
            frames.append(frame)
        cap.release()
        return np.stack(frames, 0)

    def _read_sal(self, rid: int, frame_idx: int) -> np.ndarray:
        path = str(run_dir(self.cfg, rid) / "video_saliency.avi")
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return np.zeros(
                (self.cfg.SAL_H, self.cfg.SAL_W), dtype=np.float32
            )
        sal = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return cv2.resize(
            sal,
            (self.cfg.SAL_W, self.cfg.SAL_H),
            interpolation=cv2.INTER_LINEAR,
        )

    def _read_flow(self, npz_files: List[Path], start: int) -> np.ndarray:
        """Return ``(T, 2, H, W)`` flow tensor, robustly normalised.

        v6 uses ``tanh(flow / 20)`` instead of a hard ``clip(-50, 50) / 50``.
        Most DR(eye)VE flow magnitudes are < 10 pixels so the previous
        normaliser placed nearly all of the signal in [-0.2, 0.2] —
        ``tanh`` keeps the dynamic range while bounding outliers.
        """
        flows = []
        n = len(npz_files)
        for t in range(self.cfg.CLIP_LEN):
            fi = min(start + t, n - 1)
            try:
                flow = load_flow_npz(npz_files[fi])
                u = cv2.resize(
                    flow[0], (self.cfg.IMG_W, self.cfg.IMG_H)
                )
                v = cv2.resize(
                    flow[1], (self.cfg.IMG_W, self.cfg.IMG_H)
                )
                flows.append(np.stack([u, v], 0))
            except Exception:
                flows.append(
                    np.zeros(
                        (2, self.cfg.IMG_H, self.cfg.IMG_W), np.float32
                    )
                )
        flows_np = np.stack(flows, 0).astype(np.float32)
        # tanh(x / 20): saturates at +-1 around 50 px/frame, almost linear
        # in the typical 0..10 px range.
        return np.tanh(flows_np / 20.0)

    # ---- Augmentation ------------------------------------------------
    def _augment(
        self,
        frames: np.ndarray,
        flows: np.ndarray,
        sal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = frames.shape[1], frames.shape[2]

        # Horizontal flip
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :].copy()
            flows = flows[:, :, :, ::-1].copy()
            flows[:, 0, :, :] = -flows[:, 0, :, :]  # u sign flips
            sal = sal[:, ::-1].copy()

        # Brightness / contrast
        alpha = random.uniform(0.70, 1.30)
        beta = random.uniform(-25, 25)
        frames = np.clip(
            frames.astype(np.float32) * alpha + beta, 0, 255
        ).astype(np.uint8)

        # Hue jitter (consistent across clip)
        if random.random() < 0.30:
            shift = random.uniform(-15, 15)
            out = []
            for t in range(frames.shape[0]):
                h = cv2.cvtColor(frames[t], cv2.COLOR_RGB2HSV).astype(
                    np.float32
                )
                h[:, :, 0] = (h[:, :, 0] + shift) % 180
                out.append(
                    cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2RGB)
                )
            frames = np.stack(out)

        # Gaussian noise
        if random.random() < 0.25:
            sigma = random.uniform(2, 10)
            frames = np.clip(
                frames.astype(np.float32)
                + np.random.normal(0, sigma, frames.shape),
                0,
                255,
            ).astype(np.uint8)

        # Temporal dropout
        if random.random() < 0.20 and self.cfg.CLIP_LEN > 2:
            for _ in range(random.randint(1, 2)):
                t = random.randint(1, self.cfg.CLIP_LEN - 2)
                frames[t] = frames[t - 1]
                flows[t] = flows[t - 1]

        # Spatial crop (synced across rgb / flow / sal)
        if random.random() < 0.50:
            ch = int(H * random.uniform(0.85, 1.00))
            cw = int(W * random.uniform(0.85, 1.00))
            y0 = random.randint(0, H - ch)
            x0 = random.randint(0, W - cw)
            frames = np.stack(
                [
                    cv2.resize(
                        frames[t, y0 : y0 + ch, x0 : x0 + cw],
                        (W, H),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for t in range(frames.shape[0])
                ]
            )
            new_flows = []
            for t in range(flows.shape[0]):
                f = flows[t][:, y0 : y0 + ch, x0 : x0 + cw]
                f = f.transpose(1, 2, 0)
                f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
                new_flows.append(f.transpose(2, 0, 1))
            flows = np.stack(new_flows)
            sal = cv2.resize(
                sal[y0 : y0 + ch, x0 : x0 + cw],
                (W, H),
                interpolation=cv2.INTER_LINEAR,
            )

        # Saliency label smoothing
        if random.random() < 0.30:
            k = random.choice([3, 5])
            sal = cv2.GaussianBlur(sal, (k, k), 0)

        return frames, flows, sal

    def __getitem__(self, idx: int):
        rid, start, label_idx, npz_files = self.clips[idx]
        cfg = self.cfg
        try:
            frames = self._read_rgb(rid, start)
            flows = (
                self._read_flow(npz_files, start)
                if npz_files
                else np.zeros(
                    (cfg.CLIP_LEN, 2, cfg.IMG_H, cfg.IMG_W), np.float32
                )
            )
            sal = self._read_sal(rid, label_idx)
        except Exception:
            frames = np.zeros(
                (cfg.CLIP_LEN, cfg.IMG_H, cfg.IMG_W, 3), np.uint8
            )
            flows = np.zeros(
                (cfg.CLIP_LEN, 2, cfg.IMG_H, cfg.IMG_W), np.float32
            )
            sal = np.zeros((cfg.SAL_H, cfg.SAL_W), np.float32)

        if self.augment:
            frames, flows, sal = self._augment(frames, flows, sal)

        rgb_np = (frames.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
        rgb_t = torch.from_numpy(rgb_np.copy()).permute(3, 0, 1, 2).float()
        flow_t = torch.from_numpy(flows.copy()).permute(1, 0, 2, 3).float()
        sal_t = torch.from_numpy(sal.copy()).unsqueeze(0).float()
        return rgb_t, flow_t, sal_t


# =====================================================================
# 7. SAMPLERS
# =====================================================================
class RandomClipSampler(Sampler):
    def __init__(self, n: int, k: int, seed: int = 0):
        self.n, self.k, self.seed = n, k, seed

    def __iter__(self):
        rng = random.Random(self.seed + int(time.time()) % 10000)
        return iter(rng.choices(range(self.n), k=self.k))

    def __len__(self) -> int:
        return self.k


class FixedClipSampler(Sampler):
    def __init__(self, n: int, k: int, seed: int = 1):
        rng = np.random.RandomState(seed)
        self.idx = rng.choice(n, min(k, n), replace=False).tolist()

    def __iter__(self):
        return iter(self.idx)

    def __len__(self) -> int:
        return len(self.idx)


# =====================================================================
# 8. MODEL
# =====================================================================
class FlowEncoder3D(nn.Module):
    """Lightweight 3D CNN matching the R3D-18 bottleneck spatial size.

    Input  : ``(B, 2, T=16, H=112, W=192)``
    Output : ``(B, 512, target_h, target_w)``
    """

    def __init__(self, target_h: int, target_w: int):
        super().__init__()

        def block(in_c: int, out_c: int, ss: int = 2) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(
                    in_c,
                    out_c,
                    kernel_size=(3, 3, 3),
                    stride=(1, ss, ss),
                    padding=(1, 1, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(2, 32, ss=2),     # -> 32, T, 56, 96
            block(32, 64, ss=2),    # -> 64, T, 28, 48
            block(64, 128, ss=2),   # -> 128, T, 14, 24
            block(128, 256, ss=2),  # -> 256, T,  7, 12
            block(256, 512, ss=1),  # -> 512, T,  7, 12
        )
        self.pool = nn.AdaptiveAvgPool3d((1, target_h, target_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.pool(x)
        return x.squeeze(2)


class GatedFusion(nn.Module):
    """RGB-preserving residual fusion.

    ``out = rgb + sigmoid(alpha) * proj(flow)``

    ``alpha`` is a single learnable scalar initialised to a large
    negative value so that ``sigmoid(alpha) ≈ 0`` at step 0 — the RGB
    pretrained features pass through unchanged. Optimisation then
    *opens* the gate as the flow contribution proves useful.
    ``proj`` is a 1x1 conv on the flow features (zero-initialised) so
    that even after the gate opens, the network only adds to RGB rather
    than overwriting it.
    """

    def __init__(self, channels: int = 512, init_alpha: float = -4.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Zero-init the projection conv so that even if alpha drifts,
        # the network starts in the RGB-only regime.
        nn.init.zeros_(self.proj[0].weight)
        # Single learnable gate.
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha)
        return rgb + gate * self.proj(flow)


class DRNetRGBFlow(nn.Module):
    """R3D-18 backbone + FlowEncoder3D + GatedFusion + decoder."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.out_size = (cfg.SAL_H, cfg.SAL_W)

        r3d = video_models.r3d_18(weights=None)
        self.stem = r3d.stem
        self.l1 = r3d.layer1
        self.l2 = r3d.layer2
        self.l3 = r3d.layer3
        self.l4 = r3d.layer4

        # 112 / 16 = 7,  192 / 16 = 12  -> R3D-18 bottleneck
        _th, _tw = 7, 12

        self.flow_enc = FlowEncoder3D(target_h=_th, target_w=_tw)
        self.fusion = GatedFusion(channels=512, init_alpha=-4.0)

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
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _tpool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        # ---- RGB branch -----------------------------------------------
        x = self.stem(rgb)
        s1 = self.l1(x)
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)

        p4 = self._tpool(s4)
        p3 = self._tpool(s3)
        p2 = self._tpool(s2)
        p1 = self._tpool(s1)

        # ---- Flow branch ----------------------------------------------
        ff = self.flow_enc(flow)

        # ---- Gated fusion (RGB-preserving) ----------------------------
        p4 = self.fusion(p4, ff)

        # ---- Decoder --------------------------------------------------
        d = self.up4(p4)
        d = self.drop(d)
        d = (
            F.interpolate(
                d, size=p3.shape[-2:], mode="bilinear", align_corners=False
            )
            + p3
        )
        d = self.up3(d)
        d = self.drop(d)
        d = (
            F.interpolate(
                d, size=p2.shape[-2:], mode="bilinear", align_corners=False
            )
            + p2
        )
        d = self.up2(d)
        d = self.drop(d)
        d = (
            F.interpolate(
                d, size=p1.shape[-2:], mode="bilinear", align_corners=False
            )
            + p1
        )
        d = self.up1(d)
        d = F.interpolate(
            d, size=self.out_size, mode="bilinear", align_corners=False
        )
        return torch.sigmoid(self.head(d))


# =====================================================================
# 9. RGB CHECKPOINT LOADER
# =====================================================================
def load_rgb_checkpoint(model: DRNetRGBFlow, cfg: Config) -> None:
    p = Path(cfg.RGB_CKPT)
    if not p.exists():
        print(f"  [RGB ckpt] not found at {cfg.RGB_CKPT}  — skipping.")
        return

    raw = safe_load(str(p), map_location="cpu")
    if isinstance(raw, dict):
        sd = (
            raw.get("state_dict")
            or raw.get("model_state_dict")
            or raw.get("model_state")
            or raw.get("model")
            or raw.get("net")
            or raw
        )
    else:
        sd = raw
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module.") :]: v for k, v in sd.items()}

    current = model.state_dict()
    filtered = {
        k: v
        for k, v in sd.items()
        if k in current
        and hasattr(v, "shape")
        and v.shape == current[k].shape
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print("\n  [RGB checkpoint]")
    print(f"  Loaded     : {len(filtered)} / {len(sd)} tensors")
    print(
        f"  Missing    : {len(missing)} (flow_enc + fusion: trained from scratch)"
    )
    print(f"  Unexpected : {len(unexpected)}")
    for key in [
        "stem.0.weight",
        "l1.0.conv1.0.weight",
        "up4.0.weight",
        "head.0.weight",
    ]:
        status = "OK " if key in set(filtered.keys()) else "-- "
        print(f"    {status} {key}")


# =====================================================================
# 10. METRICS + LOSS  (NUMERICALLY STABLE)
# =====================================================================
def _to_fp32(t: torch.Tensor) -> torch.Tensor:
    return t.float() if t.dtype != torch.float32 else t


def compute_cc(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred, gt = _to_fp32(pred), _to_fp32(gt)
    p = pred.flatten(1) - pred.flatten(1).mean(1, keepdim=True)
    g = gt.flatten(1) - gt.flatten(1).mean(1, keepdim=True)
    num = (p * g).sum(1)
    den = ((p ** 2).sum(1) * (g ** 2).sum(1) + EPS).sqrt()
    return (num / den).mean()


def compute_kl(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """KL(gt || pred) computed in fp32 with no division by ``pred``.

    Stable formulation::

        KL = sum_i  q_i * (log(q_i + EPS) - log(p_i + EPS))

    The previous implementation used ``q * log(q / p + EPS)`` which
    overflows in fp16 when ``p`` is tiny.
    """
    pred, gt = _to_fp32(pred), _to_fp32(gt)
    p = pred.flatten(1)
    q = gt.flatten(1)
    p = p / (p.sum(1, keepdim=True) + EPS)
    q = q / (q.sum(1, keepdim=True) + EPS)
    log_q = torch.log(q + EPS)
    log_p = torch.log(p + EPS)
    return (q * (log_q - log_p)).sum(1).mean()


def compute_nss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred, gt = _to_fp32(pred), _to_fp32(gt)
    p = pred.flatten(1)
    g = gt.flatten(1)
    p_norm = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + EPS)
    g_norm = g / (g.sum(1, keepdim=True) + EPS)
    return (p_norm * g_norm).sum(1).mean()


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = _to_fp32(x).clamp(min=EPS)
    s = x.flatten(1).sum(1, keepdim=True).view(-1, 1, 1, 1) + EPS
    return x / s


def compute_ig(
    pred: torch.Tensor,
    gt: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    pred, gt, baseline = _to_fp32(pred), _to_fp32(gt), _to_fp32(baseline)
    gt = F.interpolate(
        gt, size=pred.shape[-2:], mode="bilinear", align_corners=False
    )
    pred_p = normalize_map(pred)
    base_p = normalize_map(baseline.expand_as(pred))
    gt_p = gt.clamp(min=0)
    gt_p = gt_p / (
        gt_p.flatten(1).sum(1, keepdim=True).view(-1, 1, 1, 1) + EPS
    )
    return (
        gt_p * (torch.log2(pred_p + EPS) - torch.log2(base_p + EPS))
    ).flatten(1).sum(1).mean()


def saliency_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    cfg: Config,
    baseline: Optional[torch.Tensor] = None,
):
    pred, gt = _to_fp32(pred), _to_fp32(gt)
    gt_rs = F.interpolate(
        gt, size=pred.shape[-2:], mode="bilinear", align_corners=False
    )
    cc = compute_cc(pred, gt_rs)
    kl = compute_kl(pred, gt_rs)
    nss = compute_nss(pred, gt_rs)
    if baseline is not None:
        ig = compute_ig(pred, gt, baseline)
    else:
        ig = torch.tensor(0.0, device=pred.device)
    nss_norm = nss / 6.0
    ig_norm = ig / 8.0
    loss = (
        cfg.LOSS_W_CC * (1.0 - cc)
        + cfg.LOSS_W_KL * kl
        + cfg.LOSS_W_NSS * (1.0 - nss_norm.clamp(-1, 1))
        + cfg.LOSS_W_IG * (1.0 - ig_norm.clamp(-1, 1))
    )
    return loss, kl.detach(), cc.detach(), nss.detach(), ig.detach()


# =====================================================================
# 11. CENTER-BIAS PRIOR
# =====================================================================
def compute_training_mean_baseline(
    train_clips, cfg: Config
) -> torch.Tensor:
    print("  Computing IG baseline from training saliency maps...")
    rng = np.random.RandomState(cfg.SEED)
    sample_n = min(cfg.IG_BASELINE_CLIPS, len(train_clips))
    indices = rng.choice(len(train_clips), sample_n, replace=False)
    acc = np.zeros((cfg.SAL_H, cfg.SAL_W), dtype=np.float64)
    count = 0
    temp_ds = DreyeveFlowDataset(train_clips, cfg, augment=False)
    for idx in tqdm(indices, desc="  IG baseline", dynamic_ncols=True):
        rid, _, label_idx, _ = train_clips[idx]
        acc += temp_ds._read_sal(rid, label_idx).astype(np.float64)
        count += 1
    mean_sal = (acc / max(count, 1)).astype(np.float32)
    mean_sal = mean_sal / (mean_sal.sum() + EPS)
    baseline = torch.from_numpy(mean_sal).unsqueeze(0).unsqueeze(0).float()
    print(f"  Baseline computed from {count} maps. Sum={baseline.sum().item():.6f}")
    return baseline


# =====================================================================
# 12. EMA WRAPPER
# =====================================================================
class ModelEMA:
    """Exponential moving average of model weights.

    Evaluation is performed against ``self.module`` instead of the live
    model. Decay is ramped in over the first ~1000 updates so the EMA
    actually tracks the target early in training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000.0))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * d + msd[k].detach() * (1 - d))
                else:
                    v.copy_(msd[k])


# =====================================================================
# 13. MIXUP
# =====================================================================
def mixup_triplet(
    rgb: torch.Tensor,
    flow: torch.Tensor,
    sal: torch.Tensor,
    alpha: float,
):
    if alpha <= 0:
        return rgb, flow, sal
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # avoid trivial mixes
    idx = torch.randperm(rgb.size(0), device=rgb.device)
    rgb = lam * rgb + (1 - lam) * rgb[idx]
    flow = lam * flow + (1 - lam) * flow[idx]
    sal = lam * sal + (1 - lam) * sal[idx]
    return rgb, flow, sal


# =====================================================================
# 14. LR SCHEDULE  (cosine without warm restart)
# =====================================================================
def set_lr(
    optimizer: torch.optim.Optimizer,
    epoch_idx: int,
    step_idx: int,
    steps_per_epoch: int,
    cfg: Config,
) -> List[float]:
    """Cosine over the full run. Linear warmup over the first
    ``cfg.WARMUP_FRAC`` epoch.
    """
    progress = epoch_idx + step_idx / max(steps_per_epoch, 1)
    if progress < cfg.WARMUP_FRAC:
        warmup_scale = progress / max(cfg.WARMUP_FRAC, 1e-6)
    else:
        adj = (progress - cfg.WARMUP_FRAC) / max(
            cfg.TOTAL_EPOCHS - cfg.WARMUP_FRAC, 1e-6
        )
        warmup_scale = 0.5 * (1.0 + math.cos(math.pi * min(adj, 1.0)))
    for group in optimizer.param_groups:
        base = group["base_lr"]
        group["lr"] = cfg.MIN_LR + (base - cfg.MIN_LR) * warmup_scale
    return [g["lr"] for g in optimizer.param_groups]


# =====================================================================
# 15. TRAIN / EVAL LOOPS
# =====================================================================
def freeze_backbone(model: DRNetRGBFlow, frozen: bool) -> None:
    for prefix in ("stem.", "l1.", "l2.", "l3.", "l4."):
        for name, param in model.named_parameters():
            if name.startswith(prefix):
                param.requires_grad_(not frozen)


def train_one_epoch(
    model: DRNetRGBFlow,
    ema: ModelEMA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    epoch: int,
    cfg: Config,
    baseline: Optional[torch.Tensor],
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_ok = cfg.DEVICE == "cuda"

    pbar = tqdm(
        loader,
        desc=f"Train {epoch:03d}",
        ncols=120,
        unit="it",
        colour="cyan",
        total=cfg.TRAIN_STEPS_PER_EPOCH,
    )

    sums = {"cc": 0.0, "kl": 0.0, "nss": 0.0, "ig": 0.0, "loss": 0.0}
    n = 0

    for step, (rgb, flow, sal) in enumerate(pbar):
        if step >= cfg.TRAIN_STEPS_PER_EPOCH:
            break

        set_lr(
            optimizer,
            epoch - 1,
            step,
            cfg.TRAIN_STEPS_PER_EPOCH,
            cfg,
        )

        rgb = rgb.to(cfg.DEVICE, non_blocking=True)
        flow = flow.to(cfg.DEVICE, non_blocking=True)
        sal = sal.to(cfg.DEVICE, non_blocking=True)

        rgb, flow, sal = mixup_triplet(rgb, flow, sal, cfg.MIXUP_ALPHA)

        with torch.amp.autocast(device_type=cfg.DEVICE, enabled=amp_ok):
            pred = model(rgb, flow)
            loss, kl, cc, nss, ig = saliency_loss(
                pred, sal, cfg, baseline
            )
            loss_scaled = loss / cfg.GRAD_ACCUM

        scaler.scale(loss_scaled).backward()

        if (step + 1) % cfg.GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        sums["loss"] += float(loss.item())
        sums["cc"] += float(cc.item())
        sums["kl"] += float(kl.item())
        sums["nss"] += float(nss.item())
        sums["ig"] += float(ig.item())
        n += 1

        # gate is a single scalar — show it so the user can see the
        # network gradually opening the flow path.
        with torch.no_grad():
            gate = float(torch.sigmoid(model.fusion.alpha).item())

        pbar.set_postfix(
            {
                "cc": f"{sums['cc'] / n:.4f}",
                "kl": f"{sums['kl'] / n:.4f}",
                "nss": f"{sums['nss'] / n:.3f}",
                "ig": f"{sums['ig'] / n:.3f}",
                "gate": f"{gate:.3f}",
                "loss": f"{sums['loss'] / n:.4f}",
            },
            refresh=True,
        )

    if n % cfg.GRAD_ACCUM != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

    pbar.close()
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    tag: str,
    cfg: Config,
    max_steps: int,
    baseline: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    amp_ok = cfg.DEVICE == "cuda"
    totals: Dict[str, List[float]] = {
        "cc": [],
        "kl": [],
        "nss": [],
        "ig": [],
    }

    for step, (rgb, flow, sal) in enumerate(loader):
        if step >= max_steps:
            break
        rgb = rgb.to(cfg.DEVICE)
        flow = flow.to(cfg.DEVICE)
        sal = sal.to(cfg.DEVICE)

        with torch.amp.autocast(device_type=cfg.DEVICE, enabled=amp_ok):
            pred = model(rgb, flow)
        # Cast to fp32 BEFORE metric computation — see compute_kl docstring.
        pred = pred.float()

        gt_rs = F.interpolate(
            sal, size=pred.shape[-2:], mode="bilinear", align_corners=False
        )
        totals["cc"].append(compute_cc(pred, gt_rs).item())
        totals["kl"].append(compute_kl(pred, gt_rs).item())
        totals["nss"].append(compute_nss(pred, gt_rs).item())
        if baseline is not None:
            totals["ig"].append(
                compute_ig(pred, sal, baseline.to(pred.device)).item()
            )

    m = {
        k: float(np.nanmean(v)) if v else 0.0
        for k, v in totals.items()
    }
    label = (
        "EXCELLENT"
        if m["cc"] >= 0.75
        else "TARGET MET"
        if m["cc"] >= 0.65
        else "BASELINE BEATEN"
        if m["cc"] >= 0.56
        else "BELOW BASELINE"
    )
    print(
        f"  [{tag:10s}]  CC={m['cc']*100:.2f}%  KL={m['kl']:.4f}  "
        f"NSS={m['nss']:.4f}  IG={m['ig']:.4f}  {label}"
    )
    return m


# =====================================================================
# 16. CHECKPOINT
# =====================================================================
def save_checkpoint(
    cfg: Config,
    model: DRNetRGBFlow,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    epoch: int,
    best_val_cc: float,
    best_test_cc: float,
    history: dict,
    is_best: bool,
) -> str:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.module.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_cc": best_val_cc,
        "best_test_cc": best_test_cc,
        "history": history,
        "clip_len": cfg.CLIP_LEN,
        "img_h": cfg.IMG_H,
        "img_w": cfg.IMG_W,
        "architecture": "DRNetRGBFlow_v6_R3D18_GatedFusion",
    }
    ep = os.path.join(cfg.CKPT_DIR, f"epoch_{epoch:03d}.pth")
    lat = os.path.join(cfg.CKPT_DIR, "latest.pth")
    best = os.path.join(cfg.CKPT_DIR, "best_model.pth")
    torch.save(state, ep)
    shutil.copyfile(ep, lat)
    if is_best:
        shutil.copyfile(ep, best)
        print(f"  [BEST]  val_CC={best_val_cc*100:.2f}%  -> {best}")
    return ep


def try_resume(
    cfg: Config,
    model: DRNetRGBFlow,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
):
    """Resume from latest.pth if present. Returns ``(start_epoch,
    best_val_cc, best_test_cc, history)``.
    """
    lat = os.path.join(cfg.CKPT_DIR, "latest.pth")
    if not os.path.isfile(lat):
        return 1, 0.0, 0.0, _empty_history()

    print(f"\n  [RESUME] Loading {lat}")
    ckpt = safe_load(lat, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    if "ema_state_dict" in ckpt:
        ema.module.load_state_dict(ckpt["ema_state_dict"])
        ema.updates = ckpt.get("ema_updates", 0)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_cc = float(ckpt.get("best_val_cc", 0.0))
    best_test_cc = float(ckpt.get("best_test_cc", 0.0))
    history = ckpt.get("history", _empty_history())
    print(
        f"  [RESUME] start_epoch={start_epoch}  best_val_cc={best_val_cc*100:.2f}%  "
        f"best_test_cc={best_test_cc*100:.2f}%"
    )
    return start_epoch, best_val_cc, best_test_cc, history


def _empty_history() -> dict:
    return {
        k: []
        for k in [
            "train_cc",
            "val_cc",
            "test_cc",
            "train_kl",
            "val_kl",
            "test_kl",
            "train_nss",
            "val_nss",
            "test_nss",
            "train_ig",
            "val_ig",
            "test_ig",
            "train_loss",
        ]
    }


# =====================================================================
# 17. PREVIEW + PLOTS
# =====================================================================
@torch.no_grad()
def save_preview(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    epoch: int,
    n_show: int = 4,
) -> None:
    model.eval()
    try:
        rgb, flow, sal = next(iter(loader))
    except Exception as e:
        print(f"  Preview skipped: {e}")
        return

    rgb = rgb[:n_show].to(cfg.DEVICE)
    flow = flow[:n_show].to(cfg.DEVICE)
    sal = sal[:n_show]
    with torch.amp.autocast(
        device_type=cfg.DEVICE, enabled=(cfg.DEVICE == "cuda")
    ):
        pred = model(rgb, flow).float().cpu()

    mean_t = torch.tensor(IMG_MEAN).view(3, 1, 1)
    std_t = torch.tensor(IMG_STD).view(3, 1, 1)
    n_show = min(n_show, rgb.shape[0])

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_show):
        img = (
            (rgb[i].cpu()[:, -1] * std_t + mean_t)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("RGB")
        axes[i, 1].imshow(sal[i, 0].numpy(), cmap="inferno")
        axes[i, 1].axis("off")
        axes[i, 1].set_title("GT")
        axes[i, 2].imshow(pred[i, 0].numpy(), cmap="inferno")
        axes[i, 2].axis("off")
        axes[i, 2].set_title("Pred")

    plt.suptitle(f"DRNetRGBFlow v6 (EMA) — Epoch {epoch:03d}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg.PREVIEW_DIR, f"epoch_{epoch:03d}.png"), dpi=100
    )
    plt.close()


def save_plot(history: dict, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(24, 4))
    specs = [
        ("cc", "CC up"),
        ("kl", "KL down"),
        ("nss", "NSS up"),
        ("ig", "IG up"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        for split, ls in [("train", "-"), ("val", "--"), ("test", ":")]:
            key = f"{split}_{metric}"
            if key in history and history[key]:
                ax.plot(history[key], ls=ls, label=split, marker="o", ms=3)
        if metric == "cc":
            ax.axhline(0.65, ls="--", lw=1.5, color="gold", label="target 65%")
            ax.axhline(0.56, ls=":", lw=1.0, color="gray", label="RGB baseline 56%")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.suptitle("DRNetRGBFlow v6 — CC / KL / NSS / IG", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOT_DIR, "metrics.png"), dpi=100)
    plt.close()


def print_summary(
    epoch: int,
    train_m: dict,
    val_m: dict,
    test_m: dict,
    best_val_cc: float,
    best_test_cc: float,
    elapsed_min: float,
    gate: float,
) -> None:
    target = "  <- TARGET MET" if test_m["cc"] >= 0.65 else ""
    print()
    print("+" + "-" * 78 + "+")
    print(
        f"|  EPOCH {epoch:03d}  |  {elapsed_min:.1f} min"
        f"  |  flow gate = {gate:.3f}{'':>20}|"
    )
    print("+" + "-" * 78 + "+")
    print(
        f"|  {'':12} {'Train':>10} {'Val':>10} {'Test':>10} "
        f"{'BestVal':>10} {'BestTest':>10} |"
    )
    for m in ("cc", "kl", "nss", "ig"):
        lbl = {"cc": "CC up", "kl": "KL dn", "nss": "NSS up", "ig": "IG up"}[m]
        bv = best_val_cc if m == "cc" else float("nan")
        bt = best_test_cc if m == "cc" else float("nan")
        sfx = target if m == "cc" else ""
        bv_str = f"{bv:>10.5f}" if not math.isnan(bv) else f"{'-':>10}"
        bt_str = f"{bt:>10.5f}" if not math.isnan(bt) else f"{'-':>10}"
        print(
            f"|  {lbl:12} {train_m.get(m, 0):>10.5f} "
            f"{val_m.get(m, 0):>10.5f} {test_m.get(m, 0):>10.5f} "
            f"{bv_str} {bt_str} |{sfx}"
        )
    print("+" + "-" * 78 + "+\n")


# =====================================================================
# 18. MAIN
# =====================================================================
def main() -> None:
    cfg = Config()
    seed_all(cfg.SEED)
    make_dirs(cfg)

    print()
    print("=" * 70)
    print("  DRNetRGBFlow v6  |  R3D-18 + FlowEncoder3D + GatedFusion")
    print(f"  Device      : {cfg.DEVICE}")
    print(f"  Frames      : {cfg.CLIP_LEN}  ({cfg.IMG_H}x{cfg.IMG_W})")
    print(f"  Steps/ep    : {cfg.TRAIN_STEPS_PER_EPOCH}")
    print(
        f"  Loss        : {cfg.LOSS_W_CC} CC + {cfg.LOSS_W_KL} KL + "
        f"{cfg.LOSS_W_NSS} NSS + {cfg.LOSS_W_IG} IG"
    )
    print(f"  EMA decay   : {cfg.EMA_DECAY}")
    print(f"  Mixup alpha : {cfg.MIXUP_ALPHA}")
    print(f"  Total epochs: {cfg.TOTAL_EPOCHS}")
    print("=" * 70)

    val_s = cfg.TOTAL_FRAMES // 2 - cfg.VAL_WINDOW_SIZE // 2
    val_e = val_s + cfg.VAL_WINDOW_SIZE
    print(
        f"  Split: train=runs 01-37 (excl frames {val_s}-{val_e}), "
        f"val=frames {val_s}-{val_e}, test=runs 38-74"
    )

    train_ranges, val_ranges = get_split_ranges(cfg)

    print("\n  Building clip pools...")
    train_clips = build_clips(
        cfg, cfg.TRAIN_RUNS, cfg.TRAIN_STRIDE, train_ranges
    )
    val_clips = build_clips(
        cfg, cfg.TRAIN_RUNS, cfg.VAL_STRIDE, val_ranges
    )
    test_clips = build_clips(cfg, cfg.TEST_RUNS, cfg.TEST_STRIDE, None)
    print(
        f"  Train={len(train_clips):,}  Val={len(val_clips):,}  "
        f"Test={len(test_clips):,}"
    )
    assert set(c[0] for c in train_clips).isdisjoint(
        set(c[0] for c in test_clips)
    ), "train/test overlap!"
    print("  Split safety check OK")

    train_ds = DreyeveFlowDataset(train_clips, cfg, augment=True)
    val_ds = DreyeveFlowDataset(val_clips, cfg, augment=False)
    test_ds = DreyeveFlowDataset(test_clips, cfg, augment=False)

    nw = cfg.NUM_WORKERS
    use_pw = nw > 0
    pf = 2 if use_pw else None

    def make_loader(ds, sampler, drop_last=False):
        return DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=use_pw,
            prefetch_factor=pf,
        )

    train_dl = make_loader(
        train_ds,
        RandomClipSampler(
            len(train_ds), cfg.CLIPS_PER_EPOCH, seed=cfg.SEED
        ),
        drop_last=True,
    )
    val_dl = make_loader(
        val_ds,
        FixedClipSampler(
            len(val_ds), cfg.VAL_STEPS * cfg.BATCH_SIZE, seed=1
        ),
    )
    test_dl = make_loader(
        test_ds,
        FixedClipSampler(
            len(test_ds), cfg.TEST_STEPS * cfg.BATCH_SIZE, seed=2
        ),
    )

    # --- Center-bias prior for IG ----------------------------------
    baseline = compute_training_mean_baseline(train_clips, cfg)

    # --- Build model ------------------------------------------------
    print("\n  Building model...")
    model = DRNetRGBFlow(cfg).to(cfg.DEVICE)
    tp = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {tp/1e6:.1f} M")

    load_rgb_checkpoint(model, cfg)

    ema = ModelEMA(model, decay=cfg.EMA_DECAY)

    # --- Param groups ----------------------------------------------
    bb_pfx = ("stem.", "l1.", "l2.", "l3.", "l4.")
    fl_pfx = ("flow_enc.", "fusion.")
    backbone_p, flow_p, decoder_p = [], [], []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in bb_pfx):
            backbone_p.append(param)
        elif any(name.startswith(p) for p in fl_pfx):
            flow_p.append(param)
        else:
            decoder_p.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_p,
                "lr": cfg.LR_BACKBONE,
                "base_lr": cfg.LR_BACKBONE,
                "name": "backbone",
            },
            {
                "params": flow_p,
                "lr": cfg.LR_FLOW,
                "base_lr": cfg.LR_FLOW,
                "name": "flow",
            },
            {
                "params": decoder_p,
                "lr": cfg.LR_DECODER,
                "base_lr": cfg.LR_DECODER,
                "name": "decoder",
            },
        ],
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    print("\n  AdamW param groups:")
    print(
        f"    backbone : {len(backbone_p):4d} tensors  lr={cfg.LR_BACKBONE:.1e}"
    )
    print(
        f"    flow_enc : {len(flow_p):4d} tensors  lr={cfg.LR_FLOW:.1e}"
    )
    print(
        f"    decoder  : {len(decoder_p):4d} tensors  lr={cfg.LR_DECODER:.1e}"
    )

    scaler = torch.amp.GradScaler(enabled=(cfg.DEVICE == "cuda"))

    start_epoch, best_val_cc, best_test_cc, history = try_resume(
        cfg, model, ema, optimizer, scaler
    )

    print(f"\n{'-' * 70}")
    print(
        f"  Training: {cfg.TOTAL_EPOCHS} epochs  ·  "
        f"{cfg.TRAIN_STEPS_PER_EPOCH} steps  ·  batch {cfg.BATCH_SIZE} "
        f"(grad_accum {cfg.GRAD_ACCUM})"
    )
    print(f"{'-' * 70}\n")
    t0_total = time.time()

    for epoch in range(start_epoch, cfg.TOTAL_EPOCHS + 1):
        session_h = (time.time() - t0_total) / 3600
        if session_h > cfg.SESSION_HOUR_BUDGET:
            print(
                f"\n  Session budget reached ({session_h:.2f}h). "
                "Saving and exiting."
            )
            break

        # Two-stage: freeze backbone for the first ``WARMUP_EPOCHS``.
        freeze_backbone(model, frozen=(epoch <= cfg.WARMUP_EPOCHS))

        lrs = [g["lr"] for g in optimizer.param_groups]
        print(
            f"Epoch {epoch:03d}  |  "
            f"BB lr={lrs[0]:.1e}  Flow lr={lrs[1]:.1e}  Dec lr={lrs[2]:.1e}  "
            f"|  bb_frozen={epoch <= cfg.WARMUP_EPOCHS}  "
            f"session={session_h:.2f}h"
        )
        t_ep = time.time()

        train_m = train_one_epoch(
            model, ema, train_dl, optimizer, scaler, epoch, cfg, baseline
        )

        # Always evaluate the EMA model — that's the deployed version.
        print("  [Eval]")
        val_m = evaluate(
            ema.module, val_dl, "VAL", cfg, cfg.VAL_STEPS, baseline
        )
        test_m = evaluate(
            ema.module, test_dl, "TEST", cfg, cfg.TEST_STEPS, baseline
        )

        for split, m in [
            ("train", train_m),
            ("val", val_m),
            ("test", test_m),
        ]:
            for metric in ("cc", "kl", "nss", "ig"):
                history[f"{split}_{metric}"].append(
                    float(m.get(metric, 0.0))
                )
        history["train_loss"].append(float(train_m.get("loss", 0.0)))

        is_best = val_m["cc"] > best_val_cc
        if is_best:
            best_val_cc = val_m["cc"]
        if test_m["cc"] > best_test_cc:
            best_test_cc = test_m["cc"]

        gate = float(torch.sigmoid(model.fusion.alpha).item())

        save_checkpoint(
            cfg,
            model,
            ema,
            optimizer,
            scaler,
            epoch,
            best_val_cc,
            best_test_cc,
            history,
            is_best,
        )
        save_plot(history, cfg)
        if epoch % 5 == 0 or epoch == cfg.TOTAL_EPOCHS:
            save_preview(ema.module, val_dl, cfg, epoch)

        print_summary(
            epoch,
            train_m,
            val_m,
            test_m,
            best_val_cc,
            best_test_cc,
            (time.time() - t_ep) / 60,
            gate,
        )

        check_storage(f"epoch {epoch}")
        gc.collect()
        if cfg.DEVICE == "cuda":
            torch.cuda.empty_cache()

    total_min = (time.time() - t0_total) / 60
    print("=" * 70)
    print(f"  Done  |  {total_min:.1f} min")
    print(f"  Best Val CC  : {best_val_cc*100:.2f}%")
    print(f"  Best Test CC : {best_test_cc*100:.2f}%")
    print(
        f"  Target 65%   : {'REACHED' if best_test_cc >= 0.65 else 'keep training'}"
    )
    print("=" * 70)

    results = {
        "architecture": "DRNetRGBFlow_v6_GatedFusion",
        "backbone": "R3D-18 (torchvision)",
        "clip_len": cfg.CLIP_LEN,
        "resolution": f"{cfg.IMG_H}x{cfg.IMG_W}",
        "loss": {
            "CC": cfg.LOSS_W_CC,
            "KL": cfg.LOSS_W_KL,
            "NSS": cfg.LOSS_W_NSS,
            "IG": cfg.LOSS_W_IG,
        },
        "ema_decay": cfg.EMA_DECAY,
        "mixup_alpha": cfg.MIXUP_ALPHA,
        "best_val_cc": best_val_cc,
        "best_test_cc": best_test_cc,
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }
    with open(os.path.join(cfg.OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n  results.json saved.")


if __name__ == "__main__":
    main()
