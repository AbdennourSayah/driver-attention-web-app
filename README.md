# dr(eye)ve — driver attention web app

Web prototype for the master-thesis project _Analysis and Prediction of
Driver Attention in Real Driving Scenarios_. It is split into two
processes:

| Layer    | Path        | Stack                                      |
| -------- | ----------- | ------------------------------------------ |
| Frontend | `app/`      | Next.js 16, React 19, Tailwind, Radix UI   |
| Backend  | `backend/`  | FastAPI, PyTorch, OpenCV (RGB DRNet model) |

The backend exposes a FastAPI service that loads the trained RGB
baseline checkpoint (`rgb.pth`) and serves saliency-map predictions
for both images and videos. The frontend uploads to it from the
`/predict` page and renders side-by-side comparisons, an interactive
overlay viewer, and download buttons.

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p weights
cp /path/to/rgb.pth weights/rgb.pth
cd ..
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

See [`backend/README.md`](backend/README.md) for the full API reference,
checkpoint format requirements, environment variables, and the Docker
recipe.

### 2. Frontend

```bash
pnpm install
cp .env.local.example .env.local   # adjust NEXT_PUBLIC_API_URL if needed
pnpm dev
```

Open <http://localhost:3000> and head to **Prediction** to upload a
driving image or short video.

## Project structure

```
.
├── app/                   # Next.js App Router pages
│   ├── about/             # Architecture & methodology
│   ├── chat/              # Thesis assistant chat
│   ├── predict/           # Prediction interface
│   └── page.tsx           # Landing page
├── backend/
│   ├── api.py             # FastAPI service
│   ├── model.py           # DRNetRGB definition + checkpoint loader
│   ├── requirements.txt
│   ├── Dockerfile
│   └── weights/           # rgb.pth lives here (gitignored)
├── components/            # Reusable React components
├── lib/api.ts             # Typed API client
└── public/images/         # Static visual assets
```

## Model

`DRNetRGB` is a torchvision-based encoder-decoder:

- Encoder: `torchvision.models.video.r3d_18` (no pre-trained weights at
  module construction; we always load `rgb.pth`).
- Decoder: four `Conv3x3 → BN → ReLU → Dropout2d(p=0.3)` stages with
  bilinear upsampling and skip connections from temporally-pooled
  encoder features.
- Head: `Conv3x3 → ReLU → Conv1x1 → Sigmoid`.
- Input: `(B, 3, 16, 112, 192)` — 16-frame clip, 112×192, normalized
  with Kinetics stats `(0.43216, 0.39467, 0.37645)` /
  `(0.22803, 0.22145, 0.21699)`.
- Output: `(B, 1, 112, 192)` — sigmoid saliency.

The backend mirrors this architecture exactly and supports raw
`state_dict` checkpoints, dict checkpoints with `state_dict` /
`model_state_dict` / `model_state` / `model` / `net`, and
DataParallel-style `module.` prefixes.

## Scripts

| Command       | What it does                          |
| ------------- | -------------------------------------- |
| `pnpm dev`    | Run the Next.js dev server             |
| `pnpm build`  | Production build of the frontend      |
| `pnpm start`  | Serve the production build             |
| `pnpm lint`   | Run ESLint                             |

For the backend, see `backend/README.md`.
