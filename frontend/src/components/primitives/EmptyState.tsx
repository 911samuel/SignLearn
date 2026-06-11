import { cn } from "@/lib/utils";

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center gap-4 rounded-[var(--radius-lg)] border border-dashed border-[var(--color-border-strong)] bg-[var(--color-surface-sunken)] px-6 py-12 text-center",
        className,
      )}
    >
      {icon && (
        <div
          className="flex h-12 w-12 items-center justify-center rounded-full bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]"
          aria-hidden
        >
          {icon}
        </div>
      )}
      <div className="max-w-md">
        <h3 className="heading-h3 text-[var(--color-text)]">{title}</h3>
        {description && <p className="mt-1.5 text-[var(--color-text-muted)]">{description}</p>}
      </div>
      {action}
    </div>
  );
}
