"use client";

import { useCallback, useState } from "react";
import { Upload, X, Film } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface VideoUploadProps {
  onVideoSelect: (file: File) => void;
  preview: string | null;
  onClear: () => void;
}

export function VideoUpload({
  onVideoSelect,
  preview,
  onClear,
}: VideoUploadProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.type.startsWith("video/")) {
          onVideoSelect(file);
        }
      }
    },
    [onVideoSelect]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        onVideoSelect(files[0]);
      }
    },
    [onVideoSelect]
  );

  if (preview) {
    return (
      <div className="relative">
        <video
          src={preview}
          controls
          className="w-full rounded-lg border border-border"
          style={{ maxHeight: "300px" }}
        />
        <Button
          variant="destructive"
          size="icon"
          className="absolute right-2 top-2"
          onClick={onClear}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className={cn(
        "flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-secondary/30 p-8 transition-colors",
        isDragging && "border-primary bg-primary/5"
      )}
    >
      <input
        type="file"
        accept="video/*"
        onChange={handleFileSelect}
        className="hidden"
        id="video-upload"
      />
      <label
        htmlFor="video-upload"
        className="flex cursor-pointer flex-col items-center gap-4"
      >
        <div className="rounded-full bg-secondary p-4">
          {isDragging ? (
            <Film className="h-8 w-8 text-primary" />
          ) : (
            <Upload className="h-8 w-8 text-muted-foreground" />
          )}
        </div>
        <div className="text-center">
          <p className="font-medium">
            {isDragging ? "Drop video here" : "Drop a video or click to upload"}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            MP4, AVI, MOV up to 100MB
          </p>
        </div>
      </label>
    </div>
  );
}
