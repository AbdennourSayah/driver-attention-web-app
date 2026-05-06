// API configuration for FastAPI backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictionResponse {
  original_image: string;
  saliency_map: string;
  overlay: string;
}

export interface VideoPredictionResponse {
  output_video_url: string;
  progress: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  response: string;
}

// POST /predict/image - Predict saliency map from image
export async function predictImage(
  imageFile: File,
  model: string
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("model", model);

  const response = await fetch(`${API_BASE_URL}/predict/image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to predict saliency map");
  }

  return response.json();
}

// POST /predict/video - Predict saliency video
export async function predictVideo(
  videoFile: File,
  model: string,
  onProgress?: (progress: number) => void
): Promise<VideoPredictionResponse> {
  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("model", model);

  const response = await fetch(`${API_BASE_URL}/predict/video`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to predict video saliency");
  }

  return response.json();
}

// POST /chat - Chat with thesis assistant
export async function sendChatMessage(
  message: string,
  history: ChatMessage[]
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      history,
    }),
  });

  if (!response.ok) {
    throw new Error("Failed to get chat response");
  }

  return response.json();
}
