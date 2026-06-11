"use client";

import { Hand, Mic } from "lucide-react";
import type { Caption } from "@/hooks/useRoom";
import { usePreferences } from "@/lib/preferences";
import { cn } from "@/lib/utils";

interface CaptionsPanelProps {
  captions: Caption[];
  filter?: "all" | "sign" | "speech";
  emptyHint?: string;
  className?: string;
}

export function CaptionsPanel({
  captions,
  filter = "all",
  emptyHint,
  className,
}: CaptionsPanelProps) {
  const { prefs } = usePreferences();
  const filtered = filter === "all" ? captions : captions.filter((c) => c.source === filter);
  const recent = filtered.slice(-6);

  const sizeClass =
    prefs.captionSize === "xlarge"
      ? "text-xl"
      : prefs.captionSize === "large"
        ? "text-lg"
        : "text-base";

  return (
    <div
      role="log"
      aria-live="polite"
      aria-atomic="false"
      aria-label="Live captions"
      className={cn(
        "min-h-[120px] rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3 flex flex-col gap-2",
        className,
      )}
    >
      {recent.length === 0 ? (
        <p className="text-sm text-[var(--color-text-faint)]">
          {emptyHint ?? "Captions will appear here as you sign or speak."}
        </p>
      ) : (
        recent.map((c) => {
          const isSign = c.source === "sign";
          const Icon = isSign ? Hand : Mic;
          return (
            <article
              key={c.id}
              className={cn(
                "flex items-start gap-2.5 rounded-[var(--radius-sm)] px-3 py-2 sl-fade-up",
                isSign
                  ? "bg-[var(--color-brand-subtle)]"
                  : "bg-[var(--color-surface-sunken)]",
              )}
            >
              <Icon
                className={cn(
                  "mt-1 size-4 shrink-0",
                  isSign ? "text-[var(--color-brand)]" : "text-[var(--color-text-muted)]",
                )}
                aria-hidden
              />
              <div className="flex-1">
                <p className="text-xs font-semibold text-[var(--color-text-muted)]">
                  <span className="sr-only">From </span>
                  {c.name ?? (isSign ? "Signer" : "Hearing partner")}
                </p>
                <p className={cn("font-medium text-[var(--color-text)] leading-snug", sizeClass)}>
                  {c.text}
                </p>
              </div>
              {c.confidence != null && (
                <span className="mt-1.5 shrink-0 font-mono text-xs text-[var(--color-text-muted)]">
                  {Math.round(c.confidence * 100)}%
                </span>
              )}
            </article>
          );
        })
      )}
    </div>
  );
}
