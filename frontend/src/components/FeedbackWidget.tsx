"use client";

import { useRef, useState } from "react";
import { useToast } from "@/hooks/useToast";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5001";

interface FeedbackWidgetProps {
  roomId?: string;
}

export function FeedbackWidget({ roomId }: FeedbackWidgetProps) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [category, setCategory] = useState<"bug" | "praise" | "idea" | "accessibility">("bug");
  const [busy, setBusy] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const toast = useToast();

  function toggle() {
    setOpen((v) => {
      if (!v) setTimeout(() => textareaRef.current?.focus(), 60);
      return !v;
    });
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    try {
      await fetch(`${BACKEND_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, text: text.trim(), room_id: roomId ?? null }),
      });
      toast.success("Thanks! We read every message.");
      setText("");
      setOpen(false);
    } catch {
      toast.error("Couldn't send — please try again.");
    } finally {
      setBusy(false);
    }
  }

  const categories: { value: typeof category; label: string }[] = [
    { value: "bug", label: "🐛 Something broke" },
    { value: "praise", label: "✨ This helped me" },
    { value: "idea", label: "💡 Feature idea" },
    { value: "accessibility", label: "♿ Accessibility issue" },
  ];

  return (
    <>
      <button
        type="button"
        onClick={toggle}
        style={styles.fab}
        aria-label={open ? "Close feedback" : "Send feedback"}
        aria-expanded={open}
      >
        {open ? "✕" : "💬"}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Send feedback"
          style={styles.panel}
        >
          <div style={styles.panelHeader}>
            <strong style={styles.panelTitle}>Send feedback</strong>
            <p style={styles.panelSub}>We answer every message within 48&nbsp;hours.</p>
          </div>

          <form onSubmit={submit} style={styles.form}>
            <div style={styles.chips} role="group" aria-label="Feedback category">
              {categories.map((c) => (
                <button
                  key={c.value}
                  type="button"
                  className="sl-btn"
                  onClick={() => setCategory(c.value)}
                  style={{
                    ...styles.chip,
                    background: category === c.value ? "var(--primary)" : "var(--bg-input)",
                    color: category === c.value ? "#fff" : "var(--text-muted)",
                    borderColor: category === c.value ? "var(--primary)" : "var(--border)",
                  }}
                  aria-pressed={category === c.value}
                >
                  {c.label}
                </button>
              ))}
            </div>

            <label htmlFor="sl-feedback-text" style={styles.label}>
              What's on your mind?
            </label>
            <textarea
              id="sl-feedback-text"
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Tell us what happened, what you expected, or what would help…"
              rows={4}
              style={styles.textarea}
              required
            />

            <button
              type="submit"
              className="sl-btn-primary"
              disabled={busy || !text.trim()}
              style={styles.submit}
            >
              {busy ? "Sending…" : "Send feedback"}
            </button>
          </form>
        </div>
      )}
    </>
  );
}

const styles: Record<string, React.CSSProperties> = {
  fab: {
    position: "fixed",
    bottom: "1.25rem",
    right: "1.25rem",
    width: 52,
    height: 52,
    borderRadius: "50%",
    border: "none",
    background: "var(--bg-elevated)",
    boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
    cursor: "pointer",
    fontSize: "1.35rem",
    display: "grid",
    placeItems: "center",
    zIndex: 900,
    color: "var(--text)",
    borderColor: "var(--border)",
    outline: "1px solid var(--border)",
  },
  panel: {
    position: "fixed",
    bottom: "5rem",
    right: "1.25rem",
    width: "min(340px, calc(100vw - 2.5rem))",
    background: "var(--bg-card)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-lg)",
    boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    zIndex: 900,
    overflow: "hidden",
  },
  panelHeader: {
    padding: "1rem 1rem 0.5rem",
    borderBottom: "1px solid var(--border)",
  },
  panelTitle: { fontSize: "1rem" },
  panelSub: { margin: "0.25rem 0 0", fontSize: "0.8rem", color: "var(--text-muted)" },
  form: { padding: "0.85rem 1rem 1rem", display: "flex", flexDirection: "column", gap: "0.6rem" },
  chips: { display: "flex", flexWrap: "wrap", gap: "0.4rem" },
  chip: {
    padding: "0.3rem 0.6rem",
    borderRadius: 999,
    border: "1px solid",
    fontSize: "0.78rem",
    cursor: "pointer",
    fontFamily: "inherit",
    transition: "background 120ms, color 120ms",
  },
  label: { fontSize: "0.85rem", color: "var(--text-muted)" },
  textarea: {
    padding: "0.6rem 0.75rem",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    background: "var(--bg-input)",
    color: "var(--text)",
    fontSize: "0.9rem",
    fontFamily: "inherit",
    resize: "vertical",
    minHeight: 90,
  },
  submit: {
    padding: "0.7rem 1rem",
    borderRadius: "var(--radius)",
    border: "none",
    background: "var(--primary)",
    color: "#fff",
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "inherit",
    fontSize: "0.95rem",
  },
};
