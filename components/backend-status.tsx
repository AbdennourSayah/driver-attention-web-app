"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, AlertCircle, Loader2 } from "lucide-react";

import { API_BASE_URL, getHealth, type HealthResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

type Status = "loading" | "ok" | "degraded" | "down";

const STATUS_LABEL: Record<Status, string> = {
  loading: "Checking…",
  ok: "Backend online",
  degraded: "Model not loaded",
  down: "Backend offline",
};

const STATUS_CLASSES: Record<Status, string> = {
  loading: "bg-muted text-muted-foreground",
  ok: "bg-emerald-500/15 text-emerald-500",
  degraded: "bg-amber-500/15 text-amber-500",
  down: "bg-destructive/15 text-destructive",
};

export interface BackendStatusProps {
  className?: string;
  pollMs?: number;
}

export function BackendStatus({ className, pollMs = 30_000 }: BackendStatusProps) {
  const [status, setStatus] = useState<Status>("loading");
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let cancelled = false;

    const probe = async () => {
      try {
        const result = await getHealth();
        if (cancelled) return;
        setHealth(result);
        setStatus(result.model_loaded ? "ok" : "degraded");
      } catch {
        if (cancelled) return;
        setStatus("down");
        setHealth(null);
      }
    };

    probe();
    const id = setInterval(probe, pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [pollMs]);

  const Icon =
    status === "ok"
      ? CheckCircle2
      : status === "loading"
        ? Loader2
        : AlertCircle;

  const detail =
    status === "ok"
      ? `${health?.device ?? "cpu"} · ${API_BASE_URL.replace(/^https?:\/\//, "")}`
      : status === "degraded"
        ? "Place rgb.pth in backend/weights/"
        : status === "down"
          ? `Cannot reach ${API_BASE_URL}`
          : "";

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium",
        STATUS_CLASSES[status],
        className
      )}
      title={detail}
    >
      <Icon
        className={cn("h-3.5 w-3.5", status === "loading" && "animate-spin")}
      />
      <span>{STATUS_LABEL[status]}</span>
      {detail && <span className="hidden text-[11px] opacity-80 md:inline">· {detail}</span>}
    </div>
  );
}
