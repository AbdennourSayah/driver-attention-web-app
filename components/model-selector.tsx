"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

export type ModelType = "rgb" | "rgb_flow" | "rgb_flow_seg";

interface ModelSelectorProps {
  value: ModelType;
  onChange: (value: ModelType) => void;
}

const models = [
  {
    value: "rgb" as const,
    label: "RGB Model",
    description: "Uses RGB frames only",
    available: true,
  },
  {
    value: "rgb_flow" as const,
    label: "RGB + Optical Flow",
    description: "RGB with motion cues",
    available: false,
  },
  {
    value: "rgb_flow_seg" as const,
    label: "RGB + Flow + Segmentation",
    description: "Full multi-modal input",
    available: false,
  },
];

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor="model-select">Model Architecture</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id="model-select" className="w-full">
          <SelectValue placeholder="Select a model" />
        </SelectTrigger>
        <SelectContent>
          {models.map((model) => (
            <SelectItem
              key={model.value}
              value={model.value}
              disabled={!model.available}
            >
              <div className="flex flex-col">
                <span>
                  {model.label}
                  {!model.available && (
                    <span className="ml-2 text-xs text-muted-foreground">
                      (Coming Soon)
                    </span>
                  )}
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {models.find((m) => m.value === value)?.description}
      </p>
    </div>
  );
}
