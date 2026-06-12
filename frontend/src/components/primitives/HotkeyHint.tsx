import { cn } from "@/lib/utils";

export function Kbd({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        "inline-flex h-6 min-w-6 items-center justify-center rounded-[var(--radius-xs)] border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-1.5 text-[0.72rem] font-semibold text-[var(--color-text)] shadow-[inset_0_-1px_0_var(--color-border)]",
        className,
      )}
    >
      {children}
    </kbd>
  );
}

export interface HotkeyHintProps {
  keys: string[];
  description: string;
  className?: string;
}

export function HotkeyHint({ keys, description, className }: HotkeyHintProps) {
  return (
    <p
      className={cn(
        "inline-flex items-center gap-2 text-xs text-[var(--color-text-muted)]",
        className,
      )}
    >
      <span aria-hidden className="inline-flex items-center gap-1">
        {keys.map((k, i) => (
          <span key={i} className="inline-flex items-center gap-1">
            <Kbd>{k}</Kbd>
            {i < keys.length - 1 && <span>+</span>}
          </span>
        ))}
      </span>
      <span className="sr-only">Keyboard shortcut: {keys.join(" plus ")} — </span>
      <span>{description}</span>
    </p>
  );
}
