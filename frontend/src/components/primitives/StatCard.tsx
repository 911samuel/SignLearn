import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface StatCardProps {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  icon?: React.ReactNode;
  trend?: { direction: "up" | "down" | "flat"; label: string };
  className?: string;
}

export function StatCard({ label, value, hint, icon, trend, className }: StatCardProps) {
  return (
    <Card className={cn("p-5", className)}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow">{label}</p>
          <p className="mt-2 heading-h2 text-[var(--color-text)]">{value}</p>
          {hint && <p className="mt-1 text-sm text-[var(--color-text-muted)]">{hint}</p>}
        </div>
        {icon && (
          <div
            className="flex h-10 w-10 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]"
            aria-hidden
          >
            {icon}
          </div>
        )}
      </div>
      {trend && (
        <p
          className={cn(
            "mt-3 inline-flex items-center gap-1 text-xs font-semibold",
            trend.direction === "up"
              ? "text-[var(--color-success)]"
              : trend.direction === "down"
                ? "text-[var(--color-danger)]"
                : "text-[var(--color-text-muted)]",
          )}
        >
          <span aria-hidden>{trend.direction === "up" ? "▲" : trend.direction === "down" ? "▼" : "•"}</span>
          {trend.label}
        </p>
      )}
    </Card>
  );
}
