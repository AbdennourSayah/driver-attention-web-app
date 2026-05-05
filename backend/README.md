# dr(eye)ve backend (FastAPI + PyTorch)

This service wraps the **RGB-only** driver attention baseline
(`DRNetRGB`, R3D-18 backbone) trained in the project's Kaggle notebook.
It serves the Next.js frontend located at the repo root.

## Architecture

| Property         | Value                                |
| ---------------- | ------------------------------------ |
| Backbone         | `torchvision.models.video.r3d_18`    |
| Input shape      | `(B, 3, 16, 112, 192)` (B,C,T,H,W)   |
| Output shape     | `(B, 1, 112, 192)` sigmoid saliency  |
| Mean / Std       | `(0.43216, 0.394666, 0.37645)` / `(0.22803, 0.22145, 0.216989)` |

Single-frame uploads are tiled across the temporal dimension (T=16) so
the same checkpoint can serve both image and video requests.

## Layout

```
backend/
├── api.py              # FastAPI app
├── model.py            # DRNetRGB definition + checkpoint loader
├── requirements.txt
├── README.md
├── Dockerfile
└── weights/
    └── rgb.pth         # <- your trained checkpoint (NOT committed)
```

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place your trained weights at:

```
backend/weights/rgb.pth
```

Any of the following checkpoint formats are supported automatically:

- a raw `state_dict`,
- a dict with key `state_dict`, `model_state_dict`, `model_state`, `model`, or `net`,
- a `DataParallel` checkpoint with `module.` prefixes.

## Run

```bash
# from repo root
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Then point the frontend at it (in the repo root):

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Endpoints

| Method | Path                | Description                                                           |
| ------ | ------------------- | --------------------------------------------------------------------- |
| GET    | `/health`           | `{status, model_loaded, weights_path, device, error}`                |
| POST   | `/predict/image`    | multipart `image=<file>`, `model=rgb` -> `{original_image, saliency_map, overlay, ...}` |
| POST   | `/predict/video`    | multipart `video=<file>`, `model=rgb` -> `{output_video_url, ...}`   |
| POST   | `/chat`             | JSON `{message, history?}` -> `{response}` (canned answers)          |
| GET    | `/static/<file>`    | serves rendered overlay videos                                       |

`saliency_map` and `overlay` are returned as `data:image/png;base64,...`
so the frontend can render them directly without configuring a static
origin. Video output is served from `/static/saliency_<id>.mp4`.

## Environment variables

| Variable                       | Default                                          | Purpose                                  |
| ------------------------------ | ------------------------------------------------ | ---------------------------------------- |
| `DREYEVE_RGB_WEIGHTS`          | `backend/weights/rgb.pth`                        | Path to RGB checkpoint                   |
| `DREYEVE_WEIGHTS_DIR`          | `backend/weights`                                | Fallback weights folder                  |
| `DREYEVE_STATIC_DIR`           | `backend/static`                                 | Where rendered videos are written        |
| `DREYEVE_DEVICE`               | auto (`cuda` > `mps` > `cpu`)                    | Force a specific torch device            |
| `DREYEVE_ALLOWED_ORIGINS`      | `http://localhost:3000,http://127.0.0.1:3000`    | Comma-separated CORS allow-list          |
| `DREYEVE_PUBLIC_BASE`          | empty                                            | Prepended to video URLs (e.g. `https://api.example.com`) |
| `DREYEVE_MAX_IMAGE_BYTES`      | `20971520` (20 MB)                               | Image upload size cap                    |
| `DREYEVE_MAX_VIDEO_BYTES`      | `209715200` (200 MB)                             | Video upload size cap                    |
| `DREYEVE_MAX_VIDEO_FRAMES`     | `1500`                                           | Hard cap on processed frames per request |
| `DREYEVE_VIDEO_STRIDE`         | `1`                                              | Run inference every N frames             |
| `DREYEVE_VIDEO_OVERLAY_ALPHA`  | `0.5`                                            | Heatmap blend alpha                      |

## Smoke test (no GPU required)

```bash
python -c "import backend.model as m; net = m.DRNetRGB(); import torch; \
print(net(torch.zeros(1,3,16,112,192)).shape)"
# torch.Size([1, 1, 112, 192])
```

## Docker

```bash
cd backend
docker build -t dreyeve-api .
docker run --rm -p 8000:8000 -v $(pwd)/weights:/app/weights dreyeve-api
```
