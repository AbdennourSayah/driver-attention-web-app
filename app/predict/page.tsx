"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Cpu,
  Image as ImageIcon,
  Loader2,
  Play,
  Sparkles,
  Timer,
  Video,
  Wand2,
  Eye,
} from "lucide-react";
import { toast } from "sonner";

import { BackendStatus } from "@/components/backend-status";
import { ImageUpload } from "@/components/image-upload";
import { ModelSelector, type ModelType } from "@/components/model-selector";
import { DownloadButton, OverlayViewer } from "@/components/overlay-viewer";
import { ResultCard } from "@/components/result-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { VideoUpload } from "@/components/video-upload";
import { ApiError, predictImage, predictVideo } from "@/lib/api";

type InputType = "image" | "video";

interface ImageResults {
  original: string | null;
  saliency: string | null;
  overlay: string | null;
  inferenceMs: number | null;
  width: number | null;
  height: number | null;
}

const EMPTY_IMAGE_RESULTS: ImageResults = {
  original: null,
  saliency: null,
  overlay: null,
  inferenceMs: null,
  width: null,
  height: null,
};

interface VideoResults {
  url: string | null;
  inferenceMs: number | null;
  frames: number | null;
}

const EMPTY_VIDEO_RESULTS: VideoResults = {
  url: null,
  inferenceMs: null,
  frames: null,
};

interface SampleImage {
  src: string;
  filename: string;
  label: string;
  mimeType: string;
}

const SAMPLE_IMAGES: SampleImage[] = [
  {
    src: "/images/saliency-example-1.png",
    filename: "sample-highway.png",
    label: "Highway",
    mimeType: "image/png",
  },
  {
    src: "/images/saliency-example-2.jpg",
    filename: "sample-urban.jpg",
    label: "Urban",
    mimeType: "image/jpeg",
  },
  {
    src: "/images/saliency-example-3.jpg",
    filename: "sample-crowded.jpg",
    label: "Crowded",
    mimeType: "image/jpeg",
  },
];

function formatMs(ms: number | null): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export default function PredictionPage() {
  const [inputType, setInputType] = useState<InputType>("image");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [model, setModel] = useState<ModelType>("rgb");
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  const [imageResults, setImageResults] = useState<ImageResults>(EMPTY_IMAGE_RESULTS);
  const [videoResults, setVideoResults] = useState<VideoResults>(EMPTY_VIDEO_RESULTS);

  const previewUrlRef = useRef<string | null>(null);

  const setObjectUrlPreview = useCallback((file: File) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    const url = URL.createObjectURL(file);
    previewUrlRef.current = url;
    setPreview(url);
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  const handleInputTypeChange = (value: string) => {
    setInputType(value as InputType);
    handleClear();
  };

  const handleImageSelect = useCallback(
    (file: File) => {
      setSelectedFile(file);
      setObjectUrlPreview(file);
      setImageResults(EMPTY_IMAGE_RESULTS);
      setError(null);
    },
    [setObjectUrlPreview]
  );

  const handleVideoSelect = useCallback(
    (file: File) => {
      setSelectedFile(file);
      setObjectUrlPreview(file);
      setVideoResults(EMPTY_VIDEO_RESULTS);
      setProgress(0);
      setError(null);
    },
    [setObjectUrlPreview]
  );

  const handleClear = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setSelectedFile(null);
    setPreview(null);
    setImageResults(EMPTY_IMAGE_RESULTS);
    setVideoResults(EMPTY_VIDEO_RESULTS);
    setProgress(0);
    setError(null);
  }, []);

  const handleSampleSelect = useCallback(
    async (sample: SampleImage) => {
      try {
        const res = await fetch(sample.src);
        if (!res.ok) throw new Error(`fetch ${sample.src} failed`);
        const blob = await res.blob();
        const file = new File([blob], sample.filename, { type: sample.mimeType });
        handleImageSelect(file);
        toast.success(`Loaded sample · ${sample.label}`);
      } catch (err) {
        toast.error(`Could not load sample: ${(err as Error).message}`);
      }
    },
    [handleImageSelect]
  );

  const formatErrorMessage = (err: unknown): string => {
    if (err instanceof ApiError) {
      if (err.status === 503) return err.message;
      if (err.status === 0)
        return "Could not reach the backend. Make sure FastAPI is running on the configured URL.";
      return err.message;
    }
    if (err instanceof Error) return err.message;
    return "Unexpected error.";
  };

  const handlePredictImage = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await predictImage(selectedFile, model);
      setImageResults({
        original: response.original_image,
        saliency: response.saliency_map,
        overlay: response.overlay,
        inferenceMs: response.inference_ms,
        width: response.width,
        height: response.height,
      });
      toast.success(`Saliency ready in ${formatMs(response.inference_ms)}`);
    } catch (err) {
      const msg = formatErrorMessage(err);
      setError(msg);
      toast.error(msg);
      setImageResults({
        ...EMPTY_IMAGE_RESULTS,
        original: preview,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredictVideo = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    setProgress(5);

    let intervalId: ReturnType<typeof setInterval> | null = null;
    try {
      intervalId = setInterval(() => {
        setProgress((prev) => (prev >= 92 ? prev : prev + 3));
      }, 750);
      const response = await predictVideo(selectedFile, model);
      setVideoResults({
        url: response.output_video_url,
        inferenceMs: response.inference_ms,
        frames: response.frames,
      });
      setProgress(100);
      toast.success(`Rendered ${response.frames} frames`);
    } catch (err) {
      const msg = formatErrorMessage(err);
      setError(msg);
      toast.error(msg);
    } finally {
      if (intervalId) clearInterval(intervalId);
      setIsLoading(false);
    }
  };

  const handlePredict = () => {
    if (inputType === "image") {
      handlePredictImage();
    } else {
      handlePredictVideo();
    }
  };

  const filenameStem = (selectedFile?.name || "saliency").replace(/\.[^.]+$/, "");

  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      {/* Header */}
      <div className="mb-8 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <Badge variant="outline" className="mb-3 gap-1.5 border-primary/30 bg-primary/5 text-primary">
            <Wand2 className="h-3 w-3" />
            Live inference
          </Badge>
          <h1 className="mb-2 text-3xl font-bold tracking-tight md:text-4xl">
            Saliency Prediction
          </h1>
          <p className="max-w-xl text-muted-foreground">
            Drop a driving image or short video clip and get the predicted
            attention heatmap rendered as a colored overlay.
          </p>
        </div>
        <BackendStatus
          onChange={(status) =>
            setBackendOk(status === "ok" ? true : status === "loading" ? null : false)
          }
        />
      </div>

      {/* Backend warning banner — shown only when degraded */}
      {backendOk === false && (
        <Card className="mb-6 border-amber-500/40 bg-amber-500/5">
          <CardContent className="flex flex-col items-start gap-3 p-4 sm:flex-row sm:items-center">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-amber-500/15 text-amber-500">
              <AlertCircle className="h-4 w-4" />
            </div>
            <div className="flex-1 text-sm">
              <p className="font-medium">Backend not ready</p>
              <p className="text-muted-foreground">
                The FastAPI server is unreachable or has no checkpoint loaded.
                Make sure{" "}
                <code className="rounded bg-background/60 px-1.5 py-0.5 font-mono text-xs">
                  uvicorn backend.api:app --port 8000
                </code>{" "}
                is running and that{" "}
                <code className="rounded bg-background/60 px-1.5 py-0.5 font-mono text-xs">
                  backend/weights/rgb.pth
                </code>{" "}
                exists.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Input type tabs */}
      <div className="mb-6">
        <Tabs value={inputType} onValueChange={handleInputTypeChange} className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="image" className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4" />
              Image
            </TabsTrigger>
            <TabsTrigger value="video" className="flex items-center gap-2">
              <Video className="h-4 w-4" />
              Video
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Sample image strip — only relevant for the image tab */}
      {inputType === "image" && (
        <div className="mb-8 rounded-xl border border-border/60 bg-card/60 p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Try a sample</p>
              <p className="text-xs text-muted-foreground">
                Click a thumbnail to load it as your input.
              </p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {SAMPLE_IMAGES.map((sample) => (
              <button
                key={sample.src}
                type="button"
                onClick={() => handleSampleSelect(sample)}
                disabled={isLoading}
                className="group relative aspect-video overflow-hidden rounded-lg border border-border bg-secondary/30 transition-all hover:-translate-y-0.5 hover:border-primary/50 hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={sample.src}
                  alt={sample.label}
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/0 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
                <div className="absolute inset-x-0 bottom-0 flex items-center justify-between px-2.5 py-1.5 text-[11px] font-medium text-white">
                  <span className="rounded-full bg-black/60 px-2 py-0.5 backdrop-blur">
                    {sample.label}
                  </span>
                  <span className="rounded-full bg-primary px-2 py-0.5 text-primary-foreground opacity-0 transition-opacity group-hover:opacity-100">
                    Use
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {inputType === "image" ? (
        <div className="grid gap-8 lg:grid-cols-3">
          {/* Upload & controls */}
          <div className="lg:col-span-1">
            <Card className="bg-card/60 backdrop-blur-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg">Input image</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-6">
                <ImageUpload
                  onImageSelect={handleImageSelect}
                  preview={preview}
                  onClear={handleClear}
                />

                <ModelSelector value={model} onChange={setModel} />

                <Button
                  onClick={handlePredict}
                  disabled={!selectedFile || isLoading || backendOk === false}
                  className="w-full"
                  size="lg"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Processing…
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Predict saliency
                    </>
                  )}
                </Button>

                {imageResults.inferenceMs !== null && (
                  <div className="grid grid-cols-2 gap-2 rounded-md border border-border bg-secondary/40 p-3 text-xs">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Timer className="h-3.5 w-3.5" /> Inference
                    </div>
                    <div className="text-right font-medium">
                      {formatMs(imageResults.inferenceMs)}
                    </div>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Cpu className="h-3.5 w-3.5" /> Output size
                    </div>
                    <div className="text-right font-medium">
                      {imageResults.width}×{imageResults.height}
                    </div>
                  </div>
                )}

                {error && (
                  <div className="flex items-start gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>{error}</span>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Results */}
          <div className="space-y-6 lg:col-span-2">
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              <ResultCard
                title="Original image"
                imageSrc={imageResults.original || preview}
                isLoading={isLoading && !imageResults.original}
                placeholder="Upload an image"
              />
              <ResultCard
                title="Predicted saliency"
                imageSrc={imageResults.saliency}
                isLoading={isLoading}
                placeholder="Awaiting prediction"
              />
              <ResultCard
                title="Overlay"
                imageSrc={imageResults.overlay}
                isLoading={isLoading}
                placeholder="Awaiting prediction"
              />
            </div>

            {imageResults.original && imageResults.saliency && (
              <Card className="bg-card/60 backdrop-blur-sm">
                <CardHeader>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Eye className="h-4 w-4 text-primary" />
                      Interactive overlay
                    </CardTitle>
                    <div className="flex flex-wrap gap-2">
                      <DownloadButton
                        href={imageResults.saliency}
                        filename={`${filenameStem}-saliency.png`}
                        label="Saliency"
                      />
                      <DownloadButton
                        href={imageResults.overlay}
                        filename={`${filenameStem}-overlay.png`}
                        label="Overlay"
                      />
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <OverlayViewer
                    baseImage={imageResults.original}
                    saliencyImage={imageResults.saliency}
                  />
                </CardContent>
              </Card>
            )}

            {!imageResults.original && (
              <Card className="bg-card/60 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-sm font-medium">How it works</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  <ol className="list-inside list-decimal space-y-2">
                    <li>Upload a driving image or pick a sample above.</li>
                    <li>
                      The image is resized to 192×112, normalized with the
                      training-time Kinetics statistics, and tiled across 16
                      frames so the same RGB R3D-18 baseline serves images and
                      videos.
                    </li>
                    <li>
                      The model produces a single-channel sigmoid saliency map
                      at 192×112 which is upsampled back to your input
                      resolution.
                    </li>
                    <li>
                      The map is colorized (JET) and blended with your image to
                      produce the overlay you can scrub with the slider.
                    </li>
                  </ol>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      ) : (
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Upload & controls */}
          <Card className="bg-card/60 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-lg">Input video</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-6">
              <VideoUpload
                onVideoSelect={handleVideoSelect}
                preview={preview}
                onClear={handleClear}
              />

              <ModelSelector value={model} onChange={setModel} />

              {isLoading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Processing video…</span>
                    <span className="font-medium">{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}

              <Button
                onClick={handlePredict}
                disabled={!selectedFile || isLoading || backendOk === false}
                className="w-full"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing video…
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Generate saliency video
                  </>
                )}
              </Button>

              {videoResults.inferenceMs !== null && (
                <div className="grid grid-cols-2 gap-2 rounded-md border border-border bg-secondary/40 p-3 text-xs">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Timer className="h-3.5 w-3.5" /> Render time
                  </div>
                  <div className="text-right font-medium">
                    {formatMs(videoResults.inferenceMs)}
                  </div>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Cpu className="h-3.5 w-3.5" /> Frames processed
                  </div>
                  <div className="text-right font-medium">
                    {videoResults.frames ?? "—"}
                  </div>
                </div>
              )}

              {error && (
                <div className="flex items-start gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Output */}
          <Card className="bg-card/60 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Predicted output</CardTitle>
                <DownloadButton
                  href={videoResults.url}
                  filename={`${filenameStem}-saliency.mp4`}
                  label="Download MP4"
                />
              </div>
            </CardHeader>
            <CardContent>
              {videoResults.url ? (
                // eslint-disable-next-line jsx-a11y/media-has-caption
                <video
                  src={videoResults.url}
                  controls
                  playsInline
                  className="w-full rounded-lg border border-border"
                />
              ) : (
                <div className="flex aspect-video w-full flex-col items-center justify-center rounded-lg bg-secondary/30">
                  <Play className="mb-2 h-12 w-12 text-muted-foreground/50" />
                  <p className="text-sm text-muted-foreground">
                    {isLoading
                      ? "Generating saliency video…"
                      : "Predicted video will appear here"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-card/60 backdrop-blur-sm lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-sm font-medium">
                Video processing details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 text-sm text-muted-foreground sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <h4 className="mb-1 font-medium text-foreground">
                    Temporal processing
                  </h4>
                  <p>
                    Each output frame uses the most recent 16 input frames as
                    temporal context, matching the training clip length.
                  </p>
                </div>
                <div>
                  <h4 className="mb-1 font-medium text-foreground">Frame rate</h4>
                  <p>
                    The output keeps the source frame rate so saliency timing
                    aligns with the original footage.
                  </p>
                </div>
                <div>
                  <h4 className="mb-1 font-medium text-foreground">Output format</h4>
                  <p>
                    Per-frame saliency maps are colorized with JET and blended
                    onto the original frames as MP4.
                  </p>
                </div>
                <div>
                  <h4 className="mb-1 font-medium text-foreground">Hardware</h4>
                  <p>
                    Runs on CPU or GPU. CUDA is detected automatically; expect
                    near real-time on a modern GPU and ~1–2 fps on CPU.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
