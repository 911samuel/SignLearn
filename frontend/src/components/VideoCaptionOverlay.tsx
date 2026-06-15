"use client";

import { Hand, Mic } from "lucide-react";
import type { Caption } from "@/hooks/useRoom";
import { usePreferences } from "@/lib/preferences";
import { cn } from "@/lib/utils";

interface VideoCaptionOverlayProps {
  captions: Caption[];
  filter: "sign" | "speech";
  emptyHint?: string;
  className?: string;
}

export function VideoCaptionOverlay({
  captions,
  filter,
  emptyHint,
  className,
}: VideoCaptionOverlayProps) {
  const { prefs } = usePreferences();
  const filtered = captions.filter((c) => c.source === filter);
  const latest = filtered.slice(-2);

  const sizeClass =
    prefs.captionSize === "xlarge"
      ? "text-xl"
      : prefs.captionSize === "large"
        ? "text-lg"
        : "text-base";

  const isSign = filter === "sign";
  const Icon = isSign ? Hand : Mic;

  return (
    <div
      role="log"
      aria-live="polite"
      aria-atomic="false"
      aria-label="Live captions"
      className={cn(
        "pointer-events-none absolute inset-x-0 bottom-0 flex flex-col gap-1 px-3 pb-3",
        "bg-gradient-to-t from-black/75 via-black/40 to-transparent pt-8",
        className,
      )}
    >
      {latest.length === 0 ? (
        emptyHint ? (
          <p className="text-xs font-medium text-white/70">{emptyHint}</p>
        ) : null
      ) : (
        latest.map((c, idx) => (
          <article
            key={c.id}
            className={cn(
              "flex items-start gap-2 sl-fade-up",
              idx < latest.length - 1 && "opacity-60",
            )}
          >
            <Icon className="mt-1 size-4 shrink-0 text-white/80" aria-hidden />
            <p
              className={cn(
                "flex-1 font-semibold leading-snug text-white drop-shadow-[0_1px_2px_rgba(0,0,0,0.6)]",
                sizeClass,
              )}
            >
              {c.text}
            </p>
            {c.confidence != null && (
              <span className="mt-1.5 shrink-0 font-mono text-xs text-white/70">
                {Math.round(c.confidence * 100)}%
              </span>
            )}
          </article>
        ))
      )}
    </div>
  );
}
