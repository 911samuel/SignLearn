import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold leading-none",
  {
    variants: {
      tone: {
        neutral: "border-[var(--color-border)] bg-[var(--color-surface-sunken)] text-[var(--color-text)]",
        brand:   "border-transparent bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]",
        success: "border-transparent bg-[var(--color-success-subtle)] text-[var(--color-success)]",
        warning: "border-transparent bg-[var(--color-warning-subtle)] text-[var(--color-warning)]",
        danger:  "border-transparent bg-[var(--color-danger-subtle)] text-[var(--color-danger)]",
        info:    "border-transparent bg-[var(--color-info-subtle)] text-[var(--color-info)]",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
