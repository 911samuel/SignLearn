"use client";

import * as React from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import { cn } from "@/lib/utils";

export const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> & { tone?: "brand" | "success" | "warning" | "danger" }
>(({ className, value, tone = "brand", ...props }, ref) => {
  const toneVar =
    tone === "success" ? "var(--color-success)"
    : tone === "warning" ? "var(--color-warning)"
    : tone === "danger" ? "var(--color-danger)"
    : "var(--color-brand)";
  return (
    <ProgressPrimitive.Root
      ref={ref}
      className={cn(
        "relative h-2.5 w-full overflow-hidden rounded-full bg-[var(--color-surface-sunken)] border border-[var(--color-border)]",
        className,
      )}
      {...props}
    >
      <ProgressPrimitive.Indicator
        className="h-full transition-transform duration-500 ease-out"
        style={{
          background: toneVar,
          transform: `translateX(-${100 - (value ?? 0)}%)`,
        }}
      />
    </ProgressPrimitive.Root>
  );
});
Progress.displayName = "Progress";
