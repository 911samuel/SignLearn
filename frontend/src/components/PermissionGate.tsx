"use client";

import { useState } from "react";

interface PermissionGateProps {
  /** What capability we'll ask for — drives copy. */
  kind: "camera" | "microphone";
  onAllow: () => void | Promise<void>;
  /** Optional escape hatch (e.g. "Try practice mode"). */
  secondaryAction?: { label: string; onClick: () => void };
}

/**
 * Shown BEFORE we call getUserMedia. The browser prompt is the most-rejected
 * dialog on the web; pre-explaining what we collect (landmarks, not video)
 * and what stays local converts the scariest moment into the most reassuring.
 */
export function PermissionGate({ kind, onAllow, secondaryAction }: PermissionGateProps) {
  const [requesting, setRequesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isCamera = kind === "camera";

  async function handleAllow() {
    setError(null);
    setRequesting(true);
    try {
      await onAllow();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Permission was denied or unavailable.");
    } finally {
      setRequesting(false);
    }
  }

  return (
    <div style={styles.shell}>
      <div style={styles.card}>
        <div style={styles.iconRow} aria-hidden>
          <span style={styles.icon}>{isCamera ? "📷" : "🎙️"}</span>
        </div>

        <h2 style={styles.title}>
          {isCamera ? "Turn on your camera to start signing" : "Turn on your microphone to start speaking"}
        </h2>

        <p style={styles.lead}>
          {isCamera
            ? "We use your camera locally to read your hand position. Your video never leaves this browser."
            : "We use your mic to transcribe what you say so the signer can read it. Audio is not stored."}
        </p>

        <ol style={styles.steps}>
          {isCamera ? (
            <>
              <li><strong>We turn on your camera</strong> — only in this tab.</li>
              <li><strong>We track 21 hand landmarks per hand</strong> (126 numbers per frame).</li>
              <li><strong>Your raw video stays on your device.</strong> Only the landmark numbers go to our server.</li>
            </>
          ) : (
            <>
              <li><strong>We turn on your microphone</strong> — only in this tab.</li>
              <li><strong>Your browser transcribes speech to text</strong> using the Web Speech API.</li>
              <li><strong>Only the text caption</strong> is sent to your partner. The audio is not stored.</li>
            </>
          )}
        </ol>

        <div style={styles.actions}>
          <button
            type="button"
            className="sl-btn-primary"
            onClick={handleAllow}
            disabled={requesting}
            style={styles.primary}
          >
            {requesting ? "Requesting…" : isCamera ? "Allow camera & start" : "Allow microphone & start"}
          </button>
          {secondaryAction && (
            <button
              type="button"
              className="sl-btn"
              onClick={secondaryAction.onClick}
              style={styles.secondary}
            >
              {secondaryAction.label}
            </button>
          )}
        </div>

        {error && (
          <p style={styles.error} role="alert">
            {error} You can re‑enable permission from your browser's address bar.
          </p>
        )}

        <p style={styles.fine}>
          By continuing you agree to our <a href="/privacy">privacy notes</a>. SignLearn is open source —
          you can verify every claim above in the code.
        </p>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  shell: { display: "grid", placeItems: "center", flex: 1, padding: "1.5rem" },
  card: {
    width: "min(520px, 100%)",
    display: "flex",
    flexDirection: "column",
    gap: "0.85rem",
    padding: "1.75rem",
    background: "var(--bg-card)",
    borderRadius: "var(--radius-lg)",
    boxShadow: "var(--shadow-card)",
  },
  iconRow: { display: "flex", justifyContent: "center" },
  icon: { fontSize: "2.5rem" },
  title: { margin: 0, fontSize: "1.35rem", textAlign: "center", lineHeight: 1.25 },
  lead: { margin: 0, color: "var(--text-muted)", textAlign: "center", lineHeight: 1.5 },
  steps: {
    margin: "0.5rem 0",
    paddingLeft: "1.25rem",
    lineHeight: 1.7,
    color: "var(--text)",
    fontSize: "0.95rem",
  },
  actions: { display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "0.25rem" },
  primary: {
    padding: "0.85rem 1rem",
    borderRadius: "var(--radius)",
    border: "none",
    background: "var(--primary)",
    color: "#fff",
    fontWeight: 600,
    fontSize: "1rem",
    cursor: "pointer",
  },
  secondary: {
    padding: "0.65rem 1rem",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    background: "transparent",
    color: "var(--text-muted)",
    cursor: "pointer",
    fontSize: "0.9rem",
  },
  error: { color: "var(--danger)", margin: 0, fontSize: "0.9rem", textAlign: "center" },
  fine: { margin: 0, fontSize: "0.78rem", color: "var(--text-faint)", textAlign: "center" },
};
