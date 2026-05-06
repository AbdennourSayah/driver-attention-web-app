"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface ResultCardProps {
  title: string;
  imageSrc: string | null;
  isLoading?: boolean;
  placeholder?: string;
}

export function ResultCard({
  title,
  imageSrc,
  isLoading = false,
  placeholder = "Result will appear here",
}: ResultCardProps) {
  return (
    <Card className="bg-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="aspect-video w-full rounded-md" />
        ) : imageSrc ? (
          <img
            src={imageSrc}
            alt={title}
            className="aspect-video w-full rounded-md object-contain bg-secondary/30"
          />
        ) : (
          <div className="flex aspect-video w-full items-center justify-center rounded-md bg-secondary/30">
            <p className="text-sm text-muted-foreground">{placeholder}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
