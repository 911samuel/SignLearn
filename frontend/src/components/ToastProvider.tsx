"use client";

import { useCallback, useRef, useState } from "react";
import { ToastContext, type ToastItem, type ToastKind } from "@/hooks/useToast";

let _id = 0;
const DURATION_MS = 4500;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const t = timers.current.get(id);
    if (t) { clearTimeout(t); timers.current.delete(id); }
  }, []);

  const push = useCallback((kind: ToastKind, message: string) => {
    const id = ++_id;
    setToasts((prev) => [...prev.slice(-4), { id, kind, message }]);
    timers.current.set(id, setTimeout(() => dismiss(id), DURATION_MS));
  }, [dismiss]);

  const api = {
    info: (m: string) => push("info", m),
    success: (m: string) => push("success", m),
    error: (m: string) => push("error", m),
    warn: (m: string) => push("warn", m),
  };

  const kindColor: Record<ToastKind, string> = {
    info: "var(--accent)",
    success: "var(--success)",
    error: "var(--danger)",
    warn: "var(--warn)",
  };

  const kindIcon: Record<ToastKind, string> = {
    info: "ℹ",
    success: "✓",
    error: "✕",
    warn: "⚠",
  };

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        role="region"
        aria-live="assertive"
        aria-atomic="false"
        aria-label="Notifications"
        style={styles.container}
      >
        {toasts.map((t) => (
          <div key={t.id} style={{ ...styles.toast, borderLeftColor: kindColor[t.kind] }}>
            <span style={{ ...styles.icon, color: kindColor[t.kind] }} aria-hidden>
              {kindIcon[t.kind]}
            </span>
            <span style={styles.msg}>{t.message}</span>
            <button
              type="button"
              onClick={() => dismiss(t.id)}
              style={styles.close}
              aria-label="Dismiss notification"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: "fixed",
    bottom: "1.25rem",
    left: "50%",
    transform: "translateX(-50%)",
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
    alignItems: "center",
    zIndex: 1000,
    pointerEvents: "none",
    width: "min(420px, 95vw)",
  },
  toast: {
    display: "flex",
    alignItems: "center",
    gap: "0.65rem",
    padding: "0.65rem 1rem",
    background: "var(--bg-card)",
    border: "1px solid var(--border)",
    borderLeft: "4px solid",
    borderRadius: "var(--radius)",
    boxShadow: "0 4px 16px rgba(0,0,0,0.45)",
    pointerEvents: "auto",
    width: "100%",
    animation: "sl-toast-in 200ms ease",
  },
  icon: { fontWeight: 700, fontSize: "1rem", flexShrink: 0 },
  msg: { flex: 1, fontSize: "0.9rem", color: "var(--text)", lineHeight: 1.4 },
  close: {
    background: "none",
    border: "none",
    cursor: "pointer",
    color: "var(--text-faint)",
    fontSize: "1.2rem",
    lineHeight: 1,
    padding: "0 0.2rem",
    flexShrink: 0,
  },
};
