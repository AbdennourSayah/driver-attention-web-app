"use client";

import { useEffect, useState } from "react";
import {
  CheckCircle2,
  AlertCircle,
  Loader2,
  Cpu,
  Folder,
  RefreshCw,
} from "lucide-react";

import {
  API_BASE_URL,
  ApiError,
  getHealth,
  reloadBackend,
  type HealthResponse,
} from "@/lib/api";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type Status = "loading" | "ok" | "degraded" | "down";

const STATUS_LABEL: Record<Status, string> = {
  loading: "Checking…",
  ok: "Backend online",
  degraded: "Model not loaded",
  down: "Backend offline",
};

const STATUS_CLASSES: Record<Status, string> = {
  loading: "bg-muted text-muted-foreground border-border",
  ok: "bg-emerald-500/15 text-emerald-500 border-emerald-500/30",
  degraded: "bg-amber-500/15 text-amber-500 border-amber-500/30",
  down: "bg-destructive/15 text-destructive border-destructive/30",
};

export interface BackendStatusProps {
  className?: string;
  pollMs?: number;
  onChange?: (status: Status, health: HealthResponse | null) => void;
}

export function BackendStatus({
  className,
  pollMs = 30_000,
  onChange,
}: BackendStatusProps) {
  const [status, setStatus] = useState<Status>("loading");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const applyHealth = (result: HealthResponse) => {
    setHealth(result);
    const next: Status = result.model_loaded ? "ok" : "degraded";
    setStatus(next);
    onChange?.(next, result);
  };

  const probe = async () => {
    try {
      applyHealth(await getHealth());
    } catch {
      setStatus("down");
      setHealth(null);
      onChange?.("down", null);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (cancelled) return;
      await probe();
    };
    run();
    const id = setInterval(run, pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pollMs]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      // If the model isn't loaded yet, force the backend to re-scan the
      // weights directory and try to load — saves the user a uvicorn
      // restart when they drop in `rgb.pth` after launching the server.
      if (status !== "ok") {
        try {
          const reload = await reloadBackend();
          // /reload returns the same shape as /health (minus a few keys).
          // Synthesise a HealthResponse so the UI updates immediately.
          applyHealth({
            status: reload.reloaded ? "ok" : "degraded",
            model_loaded: reload.reloaded,
            weights_path: reload.weights_path,
            weights_dir: health?.weights_dir ?? "",
            available_weights: reload.available_weights,
            device: health?.device ?? "cpu",
            error: reload.error,
            input_shape: health?.input_shape ?? [3, 16, 112, 192],
            output_shape: health?.output_shape ?? [1, 112, 192],
            loaded_tensors: reload.loaded_tensors,
            skipped_tensors: reload.skipped_tensors,
            missing_tensors: reload.missing_tensors,
          });
          // Always re-probe to pick up any fields /reload doesn't return.
          await probe();
          return;
        } catch (err) {
          // Older backends (pre-/reload) — fall back to a plain health probe.
          if (!(err instanceof ApiError) || err.status !== 404) {
            // Surface non-404 errors via the regular probe path.
          }
        }
      }
      await probe();
    } finally {
      setRefreshing(false);
    }
  };

  const Icon =
    status === "ok"
      ? CheckCircle2
      : status === "loading"
        ? Loader2
        : AlertCircle;

  const apiHost = API_BASE_URL.replace(/^https?:\/\//, "");

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            STATUS_CLASSES[status],
            className
          )}
        >
          <Icon
            className={cn("h-3.5 w-3.5", status === "loading" && "animate-spin")}
          />
          <span>{STATUS_LABEL[status]}</span>
          {health?.device && status === "ok" && (
            <span className="hidden text-[11px] opacity-80 md:inline">
              · {health.device.toUpperCase()}
            </span>
          )}
        </button>
      </PopoverTrigger>

      <PopoverContent align="end" className="w-80 text-sm">
        <div className="space-y-3">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="font-semibold">{STATUS_LABEL[status]}</div>
              <div className="font-mono text-[11px] text-muted-foreground">
                {apiHost}
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={handleRefresh}
              disabled={refreshing}
              aria-label="Refresh status"
            >
              <RefreshCw
                className={cn("h-3.5 w-3.5", refreshing && "animate-spin")}
              />
            </Button>
          </div>

          {health?.device && (
            <div className="flex items-center gap-2 rounded-md border border-border bg-secondary/40 px-2 py-1.5 text-xs">
              <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="font-medium">{health.device.toUpperCase()}</span>
              <span className="text-muted-foreground">
                · input {health.input_shape.join("×")}
              </span>
            </div>
          )}

          {status === "ok" && health && (
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-2 text-xs">
              <div className="font-medium text-emerald-500">
                {health.weights_path?.split("/").pop()}
              </div>
              <div className="text-muted-foreground">
                {health.loaded_tensors ?? "?"} tensors loaded
                {(health.skipped_tensors ?? 0) > 0 && (
                  <> · {health.skipped_tensors} skipped</>
                )}
              </div>
            </div>
          )}

          {status === "degraded" && (
            <div className="space-y-2 rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-xs">
              <p className="text-amber-500">
                {health?.error ??
                  "The backend is running but no checkpoint was loaded."}
              </p>
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Folder className="h-3.5 w-3.5" />
                <code className="break-all font-mono">
                  {health?.weights_dir ?? "backend/weights/"}
                </code>
              </div>
              {health?.available_weights && health.available_weights.length > 0 ? (
                <div>
                  <div className="text-muted-foreground">
                    {health.error?.startsWith("Failed to load")
                      ? "Detected, but load failed:"
                      : "Detected in folder:"}
                  </div>
                  <ul className="mt-1 list-inside list-disc text-muted-foreground">
                    {health.available_weights.slice(0, 4).map((name) => (
                      <li key={name} className="font-mono">
                        {name}
                      </li>
                    ))}
                  </ul>
                  {!health.error?.startsWith("Failed to load") && (
                    <p className="mt-2 text-muted-foreground">
                      Click <span className="font-medium">Refresh</span> above
                      to load it now (no restart needed).
                    </p>
                  )}
                </div>
              ) : (
                <p className="text-muted-foreground">
                  Drop your trained{" "}
                  <code className="font-mono">rgb.pth</code> into that folder
                  and click <span className="font-medium">Refresh</span> above
                  (or restart{" "}
                  <code className="font-mono">uvicorn backend.api:app</code>).
                </p>
              )}
            </div>
          )}

          {status === "down" && (
            <div className="space-y-2 rounded-md border border-destructive/30 bg-destructive/5 p-2 text-xs">
              <p className="text-destructive">
                Cannot reach <code className="font-mono">{apiHost}</code>.
              </p>
              <p className="text-muted-foreground">
                Start the FastAPI server:
              </p>
              <pre className="overflow-x-auto rounded-sm bg-background/60 px-2 py-1 font-mono text-[11px]">
                uvicorn backend.api:app --reload --port 8000
              </pre>
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
