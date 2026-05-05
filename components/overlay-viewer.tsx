"use client";

import { useState } from "react";
import { Download, Eye, EyeOff } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";

interface OverlayViewerProps {
  baseImage: string | null;
  saliencyImage: string | null;
}

/** Adjustable saliency overlay viewer with download. */
export function OverlayViewer({ baseImage, saliencyImage }: OverlayViewerProps) {
  const [opacity, setOpacity] = useState(0.55);
  const [showOverlay, setShowOverlay] = useState(true);

  if (!baseImage) {
    return (
      <div className="flex aspect-video w-full items-center justify-center rounded-lg border border-dashed border-border bg-secondary/30 p-8 text-sm text-muted-foreground">
        Run a prediction to see the overlay here.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="relative overflow-hidden rounded-lg border border-border bg-secondary/30">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={baseImage} alt="Original" className="w-full" />
        {saliencyImage && showOverlay && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={saliencyImage}
            alt="Saliency overlay"
            className="absolute inset-0 h-full w-full mix-blend-screen"
            style={{ opacity }}
          />
        )}
      </div>

      {saliencyImage && (
        <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between gap-2">
            <Label className="text-xs uppercase tracking-wider text-muted-foreground">
              Overlay opacity · {Math.round(opacity * 100)}%
            </Label>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowOverlay((v) => !v)}
              className="h-7 px-2"
            >
              {showOverlay ? (
                <>
                  <EyeOff className="mr-1 h-3.5 w-3.5" /> Hide
                </>
              ) : (
                <>
                  <Eye className="mr-1 h-3.5 w-3.5" /> Show
                </>
              )}
            </Button>
          </div>
          <Slider
            value={[opacity]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={(values) => setOpacity(values[0] ?? 0.55)}
          />
        </div>
      )}
    </div>
  );
}

interface DownloadButtonProps {
  href: string | null;
  filename: string;
  label: string;
}

export function DownloadButton({ href, filename, label }: DownloadButtonProps) {
  if (!href) return null;
  return (
    <Button asChild variant="outline" size="sm">
      <a href={href} download={filename}>
        <Download className="mr-2 h-3.5 w-3.5" />
        {label}
      </a>
    </Button>
  );
}
