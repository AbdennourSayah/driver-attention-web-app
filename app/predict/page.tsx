"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ImageUpload } from "@/components/image-upload";
import { VideoUpload } from "@/components/video-upload";
import { ModelSelector, type ModelType } from "@/components/model-selector";
import { ResultCard } from "@/components/result-card";
import { predictImage, predictVideo } from "@/lib/api";
import { Loader2, Sparkles, Play, Image, Video } from "lucide-react";

type InputType = "image" | "video";

export default function PredictionPage() {
  const [inputType, setInputType] = useState<InputType>("image");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [model, setModel] = useState<ModelType>("rgb");
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Image results
  const [imageResults, setImageResults] = useState<{
    original: string | null;
    saliency: string | null;
    overlay: string | null;
  }>({
    original: null,
    saliency: null,
    overlay: null,
  });

  // Video results
  const [outputVideo, setOutputVideo] = useState<string | null>(null);

  const handleInputTypeChange = (value: string) => {
    setInputType(value as InputType);
    handleClear();
  };

  const handleImageSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setImageResults({ original: null, saliency: null, overlay: null });
    setError(null);
  }, []);

  const handleVideoSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setOutputVideo(null);
    setProgress(0);
    setError(null);
  }, []);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setPreview(null);
    setImageResults({ original: null, saliency: null, overlay: null });
    setOutputVideo(null);
    setProgress(0);
    setError(null);
  }, []);

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
      });
    } catch (err) {
      setError(
        "Failed to connect to the prediction server. Make sure the FastAPI backend is running."
      );
      setImageResults({
        original: preview,
        saliency: null,
        overlay: null,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredictVideo = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setProgress(0);

    try {
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 500);

      const response = await predictVideo(selectedFile, model, (prog) => {
        setProgress(prog);
      });

      clearInterval(progressInterval);
      setProgress(100);
      setOutputVideo(response.output_video_url);
    } catch (err) {
      setError(
        "Failed to connect to the prediction server. Make sure the FastAPI backend is running."
      );
    } finally {
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

  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold tracking-tight">
          Saliency Prediction
        </h1>
        <p className="text-muted-foreground">
          Upload a driving scene image or video to generate predicted saliency maps
          showing driver attention areas.
        </p>
      </div>

      {/* Input Type Selector */}
      <div className="mb-8">
        <Tabs value={inputType} onValueChange={handleInputTypeChange} className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="image" className="flex items-center gap-2">
              <Image className="h-4 w-4" />
              Image
            </TabsTrigger>
            <TabsTrigger value="video" className="flex items-center gap-2">
              <Video className="h-4 w-4" />
              Video
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {inputType === "image" ? (
        /* Image Prediction Layout */
        <div className="grid gap-8 lg:grid-cols-3">
          {/* Upload & Controls */}
          <div className="lg:col-span-1">
            <Card className="bg-card">
              <CardHeader>
                <CardTitle className="text-lg">Input Image</CardTitle>
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
                  disabled={!selectedFile || isLoading}
                  className="w-full"
                  size="lg"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Predict Saliency
                    </>
                  )}
                </Button>

                {error && (
                  <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                    {error}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Results */}
          <div className="lg:col-span-2">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <ResultCard
                title="Original Image"
                imageSrc={imageResults.original || preview}
                isLoading={isLoading}
                placeholder="Upload an image"
              />
              <ResultCard
                title="Predicted Saliency Map"
                imageSrc={imageResults.saliency}
                isLoading={isLoading}
                placeholder="Awaiting prediction"
              />
              <ResultCard
                title="Overlay Result"
                imageSrc={imageResults.overlay}
                isLoading={isLoading}
                placeholder="Awaiting prediction"
              />
            </div>

            {/* Info Section */}
            <Card className="mt-6 bg-card">
              <CardHeader>
                <CardTitle className="text-sm font-medium">
                  How it works
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                <ol className="list-inside list-decimal space-y-2">
                  <li>
                    Upload a driving scene image captured from a dashboard camera
                  </li>
                  <li>
                    Select the model architecture (currently RGB-only is available)
                  </li>
                  <li>
                    The model processes the image through a CNN encoder-decoder
                    network
                  </li>
                  <li>
                    A saliency map is generated showing predicted attention
                    hotspots
                  </li>
                  <li>
                    The overlay combines the original image with the attention
                    heatmap
                  </li>
                </ol>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : (
        /* Video Prediction Layout */
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Upload & Controls */}
          <Card className="bg-card">
            <CardHeader>
              <CardTitle className="text-lg">Input Video</CardTitle>
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
                    <span className="text-muted-foreground">
                      Processing video...
                    </span>
                    <span className="font-medium">{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}

              <Button
                onClick={handlePredict}
                disabled={!selectedFile || isLoading}
                className="w-full"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing Video...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Generate Saliency Video
                  </>
                )}
              </Button>

              {error && (
                <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Output */}
          <Card className="bg-card">
            <CardHeader>
              <CardTitle className="text-lg">Predicted Output</CardTitle>
            </CardHeader>
            <CardContent>
              {outputVideo ? (
                <video
                  src={outputVideo}
                  controls
                  className="w-full rounded-lg border border-border"
                />
              ) : (
                <div className="flex aspect-video w-full flex-col items-center justify-center rounded-lg bg-secondary/30">
                  <Play className="mb-2 h-12 w-12 text-muted-foreground/50" />
                  <p className="text-sm text-muted-foreground">
                    {isLoading
                      ? "Generating saliency video..."
                      : "Predicted video will appear here"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Info Section - Full Width */}
          <Card className="bg-card lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-sm font-medium">
                Video Processing Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 text-sm text-muted-foreground sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <h4 className="mb-1 font-medium text-foreground">
                    Temporal Processing
                  </h4>
                  <p>
                    The model processes 16-frame clips to capture temporal attention
                    dynamics.
                  </p>
                </div>
                <div>
                  <h4 className="mb-1 font-medium text-foreground">Frame Rate</h4>
                  <p>
                    Videos are processed at the native frame rate, preserving
                    temporal continuity.
                  </p>
                </div>
                <div>
                  <h4 className="mb-1 font-medium text-foreground">Output Format</h4>
                  <p>
                    Results include per-frame saliency maps overlaid on the original
                    footage.
                  </p>
                </div>
                <div>
                  <h4 className="mb-1 font-medium text-foreground">
                    Processing Time
                  </h4>
                  <p>
                    Expect approximately 2-5 seconds per frame depending on
                    resolution.
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
