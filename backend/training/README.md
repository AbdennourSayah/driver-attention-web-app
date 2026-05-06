# Training scripts

This folder contains the training pipelines for the driver-attention
saliency model. The files here are **not used by the FastAPI inference
backend** — they exist so the Kaggle notebook code lives next to the
inference code and the architecture stays in sync.

| File | Purpose |
| ---- | ------- |
| `train_rgbflow_v6.py` | RGB + Optical-Flow training loop (target ≥ 65% TEST CC). Kaggle-ready. |

## Why v6 (and what was wrong with v5fix)

The previous `v5fix` notebook reached only **55% TEST CC**, *worse* than
the RGB-only baseline (~56%). The problems were:

1. **Random fusion destroyed the RGB initialisation.** v5fix concatenated
   512-D RGB features with 512-D *randomly-initialised* flow features and
   projected through a random `Conv2d(1024 → 512)`. At step 0 this
   demolishes the well-pretrained R3D-18 representation.
2. **KL/NSS NaN.** Metrics were computed on the autocast block's `pred`
   tensor (still fp16). `KL = q · log(q / p + EPS)` overflows in fp16
   when `p` is tiny → NaN that contaminated `np.mean`.
3. **Wrong train-CC printout.** `train_cc = 1 - train_loss`, but
   `loss = (1-CC) + 0.3·KL + 0.2·(1-NSS/6)` ≈ 1.0 even when CC ≈ 0.62 →
   the table showed `CC ≈ 0.10` while training was fine.
4. **Cosine warm-restart with `T0=5`** dropped backbone LR to 1.5e-5 by
   epoch 3 — too small to escape the bad local optimum.
5. **Generalisation gap.** Train+val drawn from runs 01–37, test from
   runs 38–74. v5fix had no EMA, no mixup, no center-bias prior — the
   model overfit val to 62.5% while test stuck at 55.1%.

## What v6 changes

| # | Change | Why it helps |
| --- | --- | --- |
| **A** | **Gated additive fusion** `out = rgb + sigmoid(α)·proj(flow)`, with `α` initialised to `−4.0` (so `sigmoid(α) ≈ 0.018`) and `proj` zero-initialised. | RGB checkpoint passes through *unchanged* on epoch 1. The network learns to *open* the gate when flow actually helps. The `gate=…` reading in the progress bar lets you watch this happen. |
| **B** | **Two-stage training**: epoch 1 freezes the R3D-18 backbone; from epoch 2 the backbone unfreezes at low LR. | Lets the freshly-initialised flow encoder + fusion + decoder warm up without back-propagating noise into the pretrained backbone. |
| **C** | **Numerically stable metrics**: `.float()` cast + KL formula `q · (log(q+ε) − log(p+ε))` (no division by `p`) + `np.nanmean` aggregation. | Eliminates the `KL=nan` and occasional `NSS=nan` reports. |
| **D** | **Real train-CC tracking** (running mean of per-batch CC) + a `gate=` indicator. | The per-epoch summary now prints the actual training-time CC and shows how much of the flow path is open. |
| **E** | **EMA model averaging** (decay 0.999, ramped over 2 000 updates). Evaluation is performed against the EMA copy. | Typically +1–2% CC on saliency tasks — for free at inference time. |
| **F** | **Information-Gain loss** vs. center-bias prior (mean training saliency map). Was in the RGB baseline, missing in v5fix. | Forces the model to learn the *deviation* from the trivial center prior, not just to match it. |
| **G** | **Mixup α=0.2** on `(rgb, flow, sal)` triplets. | Closes the train/test gap by smoothing the saliency target distribution. |
| **H** | **Robust flow normalisation**: `tanh(flow / 20)` instead of `clip(−50, 50) / 50`. | Most DR(eye)VE flow magnitudes are < 10 px; the previous normaliser placed nearly all signal in `[−0.2, 0.2]` and wasted dynamic range. |
| **I** | **Cosine schedule WITHOUT warm restart** + 0.5-epoch linear warmup. | No abrupt LR jumps that destabilise fine-tuning. |
| **J** | **Full resumability**: checkpoint includes model + EMA + optimizer + scaler + epoch + best metrics + history. The script auto-resumes from `checkpoints/latest.pth`. | Kaggle's 9h limit no longer wastes a session — you just restart and continue. |

## Realistic expectations

DR(eye)VE is a **hard** benchmark. Published numbers:

| Model | Test CC |
| --- | --- |
| RGB-only baseline (this repo) | ~56% |
| Multi-modal (RGB + flow + segmentation, in literature) | 60–66% |
| Best published (heavy backbone + extra losses) | 70%+ |

`v6` should **reliably** pass 60% within a few epochs and **target
65–68%** with 15+ epochs of training (≈ 4 Kaggle sessions). Reaching
70%+ is research-grade work that usually needs a heavier backbone (e.g.
R3D-50, Swin3D) or extra modalities (semantic segmentation). 80% is at
or beyond the state of the art on this benchmark.

## How to run on Kaggle

1. Open a new Kaggle notebook with the DR(eye)VE dataset and the
   `dreyeve-raft-npz-part01..15` flow datasets attached as inputs, plus
   your RGB checkpoint dataset.
2. Adjust the paths in `Config` to match your inputs:
   - `DATA_DIR` / `DATA_ALT_*` → `DREYEVE_DATA` folder
   - `RGB_CKPT` → your trained RGB-only `.pth`
   - `FLOW_OWNER`, `FLOW_PARTS` → the user/slug pattern of the flow datasets
3. Paste the contents of `train_rgbflow_v6.py` into a single cell and
   run, or run it as a module:
   ```python
   !python -m backend.training.train_rgbflow_v6
   ```
4. When the 9h session limit is approaching, the script saves
   `checkpoints/latest.pth`. Start a new session pointed at the same
   working directory and the script will resume automatically from the
   next epoch.
5. Watch the progress bar — you'll see `gate=0.018` at the start. As
   training progresses and flow becomes useful, that number climbs
   (typically into the `0.10–0.40` range). If it stays at the
   initialisation value, flow is not contributing and the model is
   acting as RGB-only.

## Local smoke test

The script can also be parsed and the model unit-instantiated locally
without any data:

```bash
cd backend
source .venv/bin/activate
python -c "
import torch
from backend.training.train_rgbflow_v6 import Config, DRNetRGBFlow
m = DRNetRGBFlow(Config())
rgb  = torch.zeros(1, 3, 16, 112, 192)
flow = torch.zeros(1, 2, 16, 112, 192)
print(m(rgb, flow).shape)  # -> torch.Size([1, 1, 112, 192])
"
```
