"""FastAPI server for the dr(eye)ve driver-attention RGB baseline.

Endpoints
---------
GET  /health           Lightweight health/ready probe.
POST /predict/image    Upload an image -> JSON with base64 original /
                       saliency / overlay PNGs.
POST /predict/video    Upload a video -> JSON with a URL to a generated
                       overlay MP4 served from /static.
POST /chat             Minimal canned-response endpoint kept for the
                       existing frontend "Thesis Assistant" page.
GET  /static/<file>    Serves rendered video files.

The model implementation lives in :mod:`backend.model`. This file is
intentionally framework-light: load the weights once at startup, run
inference per request, and return data URLs so the Next.js frontend
can render results without configuring a separate static origin.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from backend.model import (
    RGB_MEAN,
    RGB_STD,
    DRNetRGB,
    ModelConfig,
    build_rgb_model,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("dreyeve.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path(__file__).resolve().parent

WEIGHTS_DIR = Path(os.getenv("DREYEVE_WEIGHTS_DIR", BACKEND_ROOT / "weights"))
RGB_WEIGHTS_PATH = Path(os.getenv("DREYEVE_RGB_WEIGHTS", WEIGHTS_DIR / "rgb.pth"))

STATIC_DIR = Path(os.getenv("DREYEVE_STATIC_DIR", BACKEND_ROOT / "static"))
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_BYTES = int(os.getenv("DREYEVE_MAX_IMAGE_BYTES", 20 * 1024 * 1024))   # 20 MB
MAX_VIDEO_BYTES = int(os.getenv("DREYEVE_MAX_VIDEO_BYTES", 200 * 1024 * 1024))  # 200 MB

VIDEO_STRIDE = int(os.getenv("DREYEVE_VIDEO_STRIDE", 1))
VIDEO_OVERLAY_ALPHA = float(os.getenv("DREYEVE_VIDEO_OVERLAY_ALPHA", 0.5))
MAX_VIDEO_FRAMES = int(os.getenv("DREYEVE_MAX_VIDEO_FRAMES", 1500))


# Allow the frontend dev server (and any deployment URL) to reach us.
# The wildcard fallback is only safe because every request is
# unauthenticated and we don't read cookies — adjust if you ever add auth.
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "DREYEVE_ALLOWED_ORIGINS",
        ",".join(DEFAULT_ALLOWED_ORIGINS),
    ).split(",")
    if origin.strip()
] or ["*"]


def _select_device() -> str:
    """Pick the best available device unless the user pinned one."""
    pinned = os.getenv("DREYEVE_DEVICE")
    if pinned:
        return pinned
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()
MODEL_CFG = ModelConfig()


# ---------------------------------------------------------------------------
# Global model state (populated in the lifespan handler)
# ---------------------------------------------------------------------------


class ModelState:
    model: Optional[DRNetRGB] = None
    info: Optional[dict] = None
    error: Optional[str] = None
    loaded_at: Optional[float] = None
    weights_path: Optional[Path] = None


STATE = ModelState()
INFER_LOCK = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RGB checkpoint once when the server starts."""
    weights = RGB_WEIGHTS_PATH if RGB_WEIGHTS_PATH.is_file() else None
    if weights is None:
        STATE.error = (
            f"RGB checkpoint not found at {RGB_WEIGHTS_PATH}. The server is "
            "running but /predict endpoints will return 503 until you copy "
            "rgb.pth into the weights directory."
        )
        logger.warning(STATE.error)
        # Still build an empty model so /health can report shape info.
        STATE.model = DRNetRGB(MODEL_CFG).to(DEVICE).eval()
    else:
        try:
            model, info = build_rgb_model(str(weights), device=DEVICE, cfg=MODEL_CFG)
            STATE.model = model
            STATE.info = info
            STATE.weights_path = weights
            STATE.loaded_at = time.time()
            logger.info(
                "Loaded RGB weights from %s (loaded=%s, missing=%s, skipped=%s)",
                weights,
                info.get("loaded") if info else None,
                len(info.get("missing", [])) if info else 0,
                len(info.get("skipped", [])) if info else 0,
            )
        except Exception as exc:  # pragma: no cover - depends on user weights
            STATE.error = f"Failed to load checkpoint {weights}: {exc}"
            STATE.model = DRNetRGB(MODEL_CFG).to(DEVICE).eval()
            logger.exception("Failed to load RGB checkpoint")

    yield

    STATE.model = None


app = FastAPI(
    title="dr(eye)ve API",
    version="0.2.0",
    description="RGB-only saliency prediction backend for the dr(eye)ve research prototype.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    weights_path: Optional[str] = None
    device: str
    error: Optional[str] = None
    input_shape: list[int] = Field(default_factory=lambda: [3, 16, 112, 192])
    output_shape: list[int] = Field(default_factory=lambda: [1, 112, 192])


class PredictionResponse(BaseModel):
    original_image: str
    saliency_map: str
    overlay: str
    inference_ms: float
    width: int
    height: int


class VideoPredictionResponse(BaseModel):
    output_video_url: str
    progress: float
    frames: int
    width: int
    height: int
    inference_ms: float


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Image / video helpers
# ---------------------------------------------------------------------------


_NORM_MEAN = np.array(RGB_MEAN, dtype=np.float32)
_NORM_STD = np.array(RGB_STD, dtype=np.float32)


def _normalize_frame(frame_rgb_uint8: np.ndarray) -> np.ndarray:
    """Apply training-time normalization to an HWC uint8 frame."""
    f = frame_rgb_uint8.astype(np.float32) / 255.0
    return (f - _NORM_MEAN) / _NORM_STD


def _decode_image_to_rgb(buffer: bytes) -> np.ndarray:
    """Decode arbitrary image bytes into a contiguous HWC RGB uint8 array."""
    try:
        img = Image.open(io.BytesIO(buffer)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    return np.asarray(img, dtype=np.uint8)


def _build_clip_from_image(rgb_uint8: np.ndarray) -> torch.Tensor:
    """Build a (1, 3, T, H, W) tensor by tiling a single image to T frames."""
    resized = cv2.resize(
        rgb_uint8,
        (MODEL_CFG.IMG_W, MODEL_CFG.IMG_H),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized = _normalize_frame(resized)                        # H, W, C
    frames = np.repeat(normalized[np.newaxis, ...], MODEL_CFG.CLIP_LEN, axis=0)
    # frames: T, H, W, C  ->  C, T, H, W
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous().float()
    return tensor.unsqueeze(0)  # add batch dim


def _build_clip_from_frames(rgb_frames: list[np.ndarray]) -> torch.Tensor:
    """Build a (1, 3, T, H, W) tensor from up to T already-resized RGB frames."""
    while len(rgb_frames) < MODEL_CFG.CLIP_LEN:
        rgb_frames.append(rgb_frames[-1])  # pad with last frame
    rgb_frames = rgb_frames[: MODEL_CFG.CLIP_LEN]
    normalized = np.stack([_normalize_frame(f) for f in rgb_frames], axis=0)
    tensor = torch.from_numpy(normalized).permute(3, 0, 1, 2).contiguous().float()
    return tensor.unsqueeze(0)


def _saliency_to_uint8(saliency: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    """Convert a (1,1,h,w) sigmoid map to an HxW uint8 saliency at target size."""
    sal = saliency.detach().squeeze().cpu().numpy().astype(np.float32)
    s_min = float(sal.min())
    s_max = float(sal.max())
    if s_max - s_min > 1e-6:
        sal = (sal - s_min) / (s_max - s_min)
    else:
        sal = np.zeros_like(sal)
    sal_resized = cv2.resize(sal, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
    return (np.clip(sal_resized, 0.0, 1.0) * 255.0).astype(np.uint8)


def _colorize_saliency(saliency_u8: np.ndarray) -> np.ndarray:
    """Apply the JET colormap and return an HxWx3 RGB uint8 image."""
    bgr = cv2.applyColorMap(saliency_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _make_overlay(rgb_image: np.ndarray, sal_color_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend the colorized saliency map over the original image."""
    if rgb_image.shape[:2] != sal_color_rgb.shape[:2]:
        sal_color_rgb = cv2.resize(
            sal_color_rgb,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    blended = cv2.addWeighted(rgb_image, 1.0 - alpha, sal_color_rgb, alpha, 0.0)
    return blended


def _encode_png_data_url(rgb_image: np.ndarray) -> str:
    """Encode an RGB uint8 image as a base64 ``data:image/png`` URL."""
    pil_image = Image.fromarray(rgb_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _ensure_model_ready() -> DRNetRGB:
    if STATE.model is None or STATE.weights_path is None:
        raise HTTPException(
            status_code=503,
            detail=STATE.error
            or "Model is not loaded. Place rgb.pth under backend/weights/.",
        )
    return STATE.model


@torch.inference_mode()
def _run_image_inference(rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a single forward pass and return (saliency_u8, overlay_rgb, ms)."""
    model = _ensure_model_ready()
    h, w = rgb_uint8.shape[:2]
    clip = _build_clip_from_image(rgb_uint8).to(DEVICE)

    started = time.perf_counter()
    saliency = model(clip)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    sal_u8 = _saliency_to_uint8(saliency, (h, w))
    sal_rgb = _colorize_saliency(sal_u8)
    overlay = _make_overlay(rgb_uint8, sal_rgb, alpha=0.5)
    return sal_u8, overlay, elapsed_ms


@torch.inference_mode()
def _run_clip_inference(clip: torch.Tensor) -> torch.Tensor:
    model = _ensure_model_ready()
    return model(clip.to(DEVICE))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if STATE.weights_path else "degraded",
        model_loaded=STATE.weights_path is not None,
        weights_path=str(STATE.weights_path) if STATE.weights_path else None,
        device=DEVICE,
        error=STATE.error,
    )


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(
    image: UploadFile = File(...),
    model: str = Form("rgb"),
) -> PredictionResponse:
    if model != "rgb":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available yet. Only 'rgb' is supported.",
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_IMAGE_BYTES} bytes.",
        )

    rgb = _decode_image_to_rgb(raw)

    async with INFER_LOCK:
        sal_u8, overlay_rgb, ms = await asyncio.get_event_loop().run_in_executor(
            None, _run_image_inference, rgb
        )

    sal_rgb = _colorize_saliency(sal_u8)
    return PredictionResponse(
        original_image=_encode_png_data_url(rgb),
        saliency_map=_encode_png_data_url(sal_rgb),
        overlay=_encode_png_data_url(overlay_rgb),
        inference_ms=ms,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
    )


@app.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video(
    video: UploadFile = File(...),
    model: str = Form("rgb"),
) -> VideoPredictionResponse:
    if model != "rgb":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available yet. Only 'rgb' is supported.",
        )

    raw = await video.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds {MAX_VIDEO_BYTES} bytes.",
        )

    upload_id = uuid.uuid4().hex
    upload_path = STATIC_DIR / f"input_{upload_id}.bin"
    upload_path.write_bytes(raw)

    cap = cv2.VideoCapture(str(upload_path))
    if not cap.isOpened():
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not decode video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if width == 0 or height == 0:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid video dimensions.")

    output_path = STATIC_DIR / f"saliency_{upload_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not open video writer.")

    started = time.perf_counter()

    async with INFER_LOCK:
        try:
            frames_written = await asyncio.get_event_loop().run_in_executor(
                None,
                _process_video_to_overlay,
                upload_path,
                writer,
                width,
                height,
            )
        finally:
            writer.release()
            upload_path.unlink(missing_ok=True)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Video rendering produced no output.")

    public_base = os.getenv("DREYEVE_PUBLIC_BASE", "").rstrip("/")
    relative_url = f"/static/{output_path.name}"
    output_url = f"{public_base}{relative_url}" if public_base else relative_url

    return VideoPredictionResponse(
        output_video_url=output_url,
        progress=100.0,
        frames=frames_written,
        width=width,
        height=height,
        inference_ms=elapsed_ms,
    )


def _process_video_to_overlay(
    upload_path: Path,
    writer: cv2.VideoWriter,
    width: int,
    height: int,
) -> int:
    """Sliding-window inference; returns the number of frames written."""
    cap = cv2.VideoCapture(str(upload_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not re-open video.")

    rolling_resized: list[np.ndarray] = []  # list of (H,W,3) RGB uint8 already resized
    rolling_originals: list[np.ndarray] = []  # raw RGB frames at original size

    frames_written = 0
    try:
        frames_read = 0
        while frames_read < MAX_VIDEO_FRAMES:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rolling_originals.append(frame_rgb)
            resized = cv2.resize(
                frame_rgb,
                (MODEL_CFG.IMG_W, MODEL_CFG.IMG_H),
                interpolation=cv2.INTER_LINEAR,
            )
            rolling_resized.append(resized)

            if len(rolling_resized) > MODEL_CFG.CLIP_LEN:
                rolling_resized.pop(0)
                rolling_originals.pop(0)

            frames_read += 1
            if frames_read % VIDEO_STRIDE != 0:
                continue

            clip_tensor = _build_clip_from_frames(list(rolling_resized))
            saliency = _run_clip_inference(clip_tensor)
            sal_u8 = _saliency_to_uint8(saliency, (height, width))
            sal_rgb = _colorize_saliency(sal_u8)
            overlay = _make_overlay(rolling_originals[-1], sal_rgb, alpha=VIDEO_OVERLAY_ALPHA)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            frames_written += 1
    finally:
        cap.release()

    return frames_written


# ---------------------------------------------------------------------------
# Chat — minimal canned responses (kept compatible with the existing
# frontend, which already falls back to mock answers on failure).
# ---------------------------------------------------------------------------


_CANNED = {
    "default": (
        "I'm the dr(eye)ve thesis assistant. Ask me about saliency maps, "
        "the DR(eye)VE dataset, the RGB baseline architecture, or the "
        "evaluation metrics (CC / KL / NSS / IG)."
    ),
    "saliency": (
        "Saliency maps represent where a driver is most likely to look in a "
        "given scene. They are 2D probability distributions normalized to "
        "[0, 1] (higher = more attention)."
    ),
    "rgb": (
        "The RGB baseline uses a torchvision r3d_18 backbone (R(2+1)D-style "
        "3D ResNet) followed by a 4-stage decoder with skip connections "
        "from temporally pooled encoder features. Input is a 16-frame clip "
        "at 112x192; output is a single-channel sigmoid saliency map."
    ),
    "metrics": (
        "We report CC (Pearson correlation), KL divergence, NSS (Normalized "
        "Scanpath Saliency) and IG (Information Gain over a center bias). "
        "Training optimizes a weighted sum: 1.0*(1-CC) + 0.3*KL + 0.2*(1-NSS/6)."
    ),
}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    msg = (request.message or "").lower()
    if "saliency" in msg:
        return ChatResponse(response=_CANNED["saliency"])
    if "rgb" in msg or "architecture" in msg or "model" in msg:
        return ChatResponse(response=_CANNED["rgb"])
    if any(k in msg for k in ("cc", "kl", "nss", "ig", "metric")):
        return ChatResponse(response=_CANNED["metrics"])
    return ChatResponse(response=_CANNED["default"])


@app.exception_handler(HTTPException)
async def _http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
