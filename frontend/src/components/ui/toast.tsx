"use client";

import * as React from "react";
import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";
import { cn } from "@/lib/utils";

export type ToastTone = "success" | "info" | "warning" | "danger";
export interface Toast {
  id: string;
  title: string;
  description?: string;
  tone?: ToastTone;
  duration?: number;
}

interface ToastContextValue {
  toast: (t: Omit<Toast, "id">) => string;
  dismiss: (id: string) => void;
}

const ToastContext = React.createContext<ToastContextValue | null>(null);

export function useToast() {
  const ctx = React.useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within <Toaster>");
  return ctx;
}

const toneStyles: Record<ToastTone, { icon: typeof Info; cls: string }> = {
  success: { icon: CheckCircle2, cls: "border-[var(--color-success)]/40 bg-[var(--color-success-subtle)]" },
  info:    { icon: Info,         cls: "border-[var(--color-info)]/40 bg-[var(--color-info-subtle)]" },
  warning: { icon: AlertCircle,  cls: "border-[var(--color-warning)]/40 bg-[var(--color-warning-subtle)]" },
  danger:  { icon: AlertCircle,  cls: "border-[var(--color-danger)]/40 bg-[var(--color-danger-subtle)]" },
};

export function Toaster({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<Toast[]>([]);

  const dismiss = React.useCallback((id: string) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const toast = React.useCallback(
    ({ title, description, tone = "info", duration = 4500 }: Omit<Toast, "id">) => {
      const id = Math.random().toString(36).slice(2);
      setToasts((cur) => [...cur, { id, title, description, tone, duration }]);
      if (duration > 0) {
        window.setTimeout(() => dismiss(id), duration);
      }
      return id;
    },
    [dismiss],
  );

  return (
    <ToastContext.Provider value={{ toast, dismiss }}>
      {children}
      <ol
        role="region"
        aria-label="Notifications"
        className="pointer-events-none fixed bottom-4 right-4 z-[200] flex w-[min(380px,calc(100vw-2rem))] flex-col gap-2"
      >
        {toasts.map((t) => {
          const { icon: Icon, cls } = toneStyles[t.tone ?? "info"];
          return (
            <li
              key={t.id}
              role="status"
              className={cn(
                "pointer-events-auto flex items-start gap-3 rounded-[var(--radius-md)] border p-3 pr-2 shadow-[var(--shadow-overlay)] sl-fade-up",
                cls,
              )}
            >
              <Icon className="mt-0.5 size-5 shrink-0 text-[var(--color-text)]" aria-hidden />
              <div className="flex-1 text-sm">
                <p className="font-semibold text-[var(--color-text)]">{t.title}</p>
                {t.description && (
                  <p className="mt-0.5 text-[var(--color-text-muted)]">{t.description}</p>
                )}
              </div>
              <button
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss notification"
                className="ml-1 inline-flex h-7 w-7 items-center justify-center rounded-full text-[var(--color-text-muted)] hover:bg-black/5"
              >
                <X className="size-4" aria-hidden />
              </button>
            </li>
          );
        })}
      </ol>
    </ToastContext.Provider>
  );
}
