import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export interface ConfidenceMeterProps {
  value: number; // 0..1
  label?: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

function toneFor(v: number) {
  if (v >= 0.85) return "success" as const;
  if (v >= 0.6)  return "warning" as const;
  return "danger" as const;
}

function levelFor(v: number) {
  if (v >= 0.85) return "High";
  if (v >= 0.6) return "Medium";
  return "Low";
}

export function ConfidenceMeter({ value, label, size = "md", className }: ConfidenceMeterProps) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const tone = toneFor(value);
  const level = levelFor(value);

  return (
    <div className={cn("w-full", className)}>
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold text-[var(--color-text)]">
          {label ?? "Confidence"}
        </span>
        <span
          className={cn(
            "font-mono tabular-nums",
            tone === "success" && "text-[var(--color-success)]",
            tone === "warning" && "text-[var(--color-warning)]",
            tone === "danger" && "text-[var(--color-danger)]",
          )}
          aria-hidden
        >
          {level} · {pct}%
        </span>
      </div>
      <Progress
        value={pct}
        tone={tone}
        className={cn("mt-1.5", size === "lg" ? "h-3" : size === "sm" ? "h-1.5" : "h-2.5")}
        aria-label={`${label ?? "Confidence"}: ${level}, ${pct} percent`}
      />
    </div>
  );
}
