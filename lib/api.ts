// API client for the dr(eye)ve FastAPI backend.
//
// The Next.js frontend reads `NEXT_PUBLIC_API_URL` at build time. When
// it's not set we fall back to the local dev server. Every request has
// a sensible timeout and a normalized error so the UI can render a
// friendly message instead of a generic "Failed to fetch".

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const DEFAULT_TIMEOUT_MS = 60_000;
const VIDEO_TIMEOUT_MS = 10 * 60_000;

export class ApiError extends Error {
  status: number;
  detail?: string;

  constructor(message: string, status: number, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

// ---- Schemas -------------------------------------------------------------

export interface HealthResponse {
  status: "ok" | "degraded" | string;
  model_loaded: boolean;
  weights_path: string | null;
  weights_dir: string;
  available_weights: string[];
  device: string;
  error: string | null;
  input_shape: number[];
  output_shape: number[];
  loaded_tensors: number | null;
  skipped_tensors: number | null;
  missing_tensors: number | null;
}

export interface PredictionResponse {
  original_image: string;
  saliency_map: string;
  overlay: string;
  inference_ms: number;
  width: number;
  height: number;
}

export interface VideoPredictionResponse {
  output_video_url: string;
  progress: number;
  frames: number;
  width: number;
  height: number;
  inference_ms: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  response: string;
}

// ---- Internal helpers ----------------------------------------------------

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });
    if (!response.ok) {
      let detail: string | undefined;
      try {
        const body = await response.json();
        detail = body?.detail || body?.message;
      } catch {
        try {
          detail = await response.text();
        } catch {
          detail = undefined;
        }
      }
      throw new ApiError(
        detail || `Request to ${path} failed (${response.status})`,
        response.status,
        detail
      );
    }
    return (await response.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if ((err as Error)?.name === "AbortError") {
      throw new ApiError(
        `The request timed out after ${Math.round(timeoutMs / 1000)}s.`,
        408
      );
    }
    throw new ApiError(
      "Could not reach the prediction server. Is the FastAPI backend running?",
      0,
      (err as Error)?.message
    );
  } finally {
    clearTimeout(timer);
  }
}

function withApiBase(url: string): string {
  if (!url) return url;
  if (/^https?:\/\//i.test(url) || url.startsWith("data:")) return url;
  return `${API_BASE_URL}${url.startsWith("/") ? "" : "/"}${url}`;
}

// ---- Public API ----------------------------------------------------------

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { method: "GET" }, 8_000);
}

export async function predictImage(
  imageFile: File,
  model: string
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("model", model);

  return request<PredictionResponse>("/predict/image", {
    method: "POST",
    body: formData,
  });
}

export async function predictVideo(
  videoFile: File,
  model: string
): Promise<VideoPredictionResponse> {
  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("model", model);

  const response = await request<VideoPredictionResponse>(
    "/predict/video",
    { method: "POST", body: formData },
    VIDEO_TIMEOUT_MS
  );

  return {
    ...response,
    output_video_url: withApiBase(response.output_video_url),
  };
}

export async function sendChatMessage(
  message: string,
  history: ChatMessage[]
): Promise<ChatResponse> {
  return request<ChatResponse>(
    "/chat",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    },
    20_000
  );
}
