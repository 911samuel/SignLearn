import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { AlertCircle, AlertTriangle, CheckCircle2, Info } from "lucide-react";
import { cn } from "@/lib/utils";

const alertVariants = cva(
  "relative grid grid-cols-[auto_1fr] gap-3 rounded-[var(--radius-md)] border p-4 text-sm",
  {
    variants: {
      tone: {
        info:    "border-[var(--color-info)]/30 bg-[var(--color-info-subtle)] text-[var(--color-info)]",
        success: "border-[var(--color-success)]/30 bg-[var(--color-success-subtle)] text-[var(--color-success)]",
        warning: "border-[var(--color-warning)]/30 bg-[var(--color-warning-subtle)] text-[var(--color-warning)]",
        danger:  "border-[var(--color-danger)]/30 bg-[var(--color-danger-subtle)] text-[var(--color-danger)]",
        neutral: "border-[var(--color-border)] bg-[var(--color-surface-sunken)] text-[var(--color-text)]",
      },
    },
    defaultVariants: { tone: "info" },
  },
);

const iconFor = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  danger: AlertCircle,
  neutral: Info,
} as const;

export interface AlertProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, "title">,
    VariantProps<typeof alertVariants> {
  title?: React.ReactNode;
}

export function Alert({ className, tone, title, children, ...props }: AlertProps) {
  const Icon = iconFor[(tone ?? "info") as keyof typeof iconFor];
  return (
    <div role="alert" className={cn(alertVariants({ tone }), className)} {...props}>
      <Icon className="size-5 shrink-0 mt-0.5" aria-hidden />
      <div className="text-[var(--color-text)]">
        {title && <p className="font-semibold mb-1">{title}</p>}
        {children && <div className="text-[var(--color-text-muted)]">{children}</div>}
      </div>
    </div>
  );
}
