import { CheckCircle2, Hand, Loader2, RadioTower, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";

export type Status = "idle" | "signing" | "processing" | "ok" | "disconnected";

const config: Record<Status, { label: string; cls: string; icon: React.ComponentType<{ className?: string }>; spin?: boolean }> = {
  idle:         { label: "Ready",                       cls: "bg-[var(--color-surface-sunken)] text-[var(--color-text-muted)] border-[var(--color-border)]", icon: Hand },
  signing:      { label: "Recognising sign",            cls: "bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)] border-transparent", icon: RadioTower },
  processing:   { label: "Processing…",                 cls: "bg-[var(--color-info-subtle)] text-[var(--color-info)] border-transparent", icon: Loader2, spin: true },
  ok:           { label: "Connected",                   cls: "bg-[var(--color-success-subtle)] text-[var(--color-success)] border-transparent", icon: CheckCircle2 },
  disconnected: { label: "Disconnected — reconnecting", cls: "bg-[var(--color-danger-subtle)] text-[var(--color-danger)] border-transparent", icon: WifiOff },
};

export function StatusPill({ status, className }: { status: Status; className?: string }) {
  const c = config[status];
  const Icon = c.icon;
  return (
    <span
      role="status"
      aria-live="polite"
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold",
        c.cls,
        className,
      )}
    >
      <Icon
        className={cn("size-3.5 shrink-0", c.spin && "sl-pulse-soft")}
        aria-hidden
      />
      {c.label}
    </span>
  );
}
