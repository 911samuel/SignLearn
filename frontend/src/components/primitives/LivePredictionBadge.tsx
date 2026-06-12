"use client";

import { cn } from "@/lib/utils";

export interface PredictionCandidate {
  label: string;
  confidence: number;
}

export interface LivePredictionBadgeProps {
  candidates: PredictionCandidate[];
  onSelect?: (label: string) => void;
  className?: string;
}

export function LivePredictionBadge({ candidates, onSelect, className }: LivePredictionBadgeProps) {
  if (candidates.length === 0) {
    return (
      <div
        aria-live="polite"
        className={cn("text-sm text-[var(--color-text-faint)]", className)}
      >
        Waiting for a sign…
      </div>
    );
  }

  const [top, ...rest] = candidates;

  return (
    <div className={cn("w-full", className)} aria-live="polite" aria-atomic="false">
      <p className="eyebrow">Detected</p>
      <p className="mt-1 heading-display text-[var(--color-text)] leading-none">
        {top.label}
      </p>
      <p className="mt-2 text-sm text-[var(--color-text-muted)]">
        {Math.round(top.confidence * 100)}% confidence
      </p>

      {rest.length > 0 && (
        <div className="mt-4">
          <p className="eyebrow mb-2">Other guesses</p>
          <div className="flex flex-wrap gap-2">
            {rest.slice(0, 4).map((c) => (
              <button
                key={c.label}
                type="button"
                onClick={() => onSelect?.(c.label)}
                className="inline-flex items-center gap-2 rounded-full border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-3 py-1.5 text-sm font-medium text-[var(--color-text)] transition hover:bg-[var(--color-surface-sunken)] focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]"
                aria-label={`Use ${c.label} instead — ${Math.round(c.confidence * 100)} percent confidence`}
              >
                <span>{c.label}</span>
                <span className="text-xs tabular-nums text-[var(--color-text-muted)]">
                  {Math.round(c.confidence * 100)}%
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
