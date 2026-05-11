"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5001";

export function LandingCTA() {
  const router = useRouter();
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [showJoin, setShowJoin] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function createRoom() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/rooms`, { method: "POST" });
      if (!res.ok) throw new Error("Could not reach server");
      const { room_id } = await res.json();
      router.push(`/r/${room_id}/join`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create room");
    } finally {
      setBusy(false);
    }
  }

  function joinRoom(e: React.FormEvent) {
    e.preventDefault();
    const id = code.trim().toUpperCase();
    if (!id) return;
    router.push(`/r/${id}/join`);
  }

  return (
    <div style={styles.wrap}>
      <div style={styles.ctaRow}>
        <button
          type="button"
          onClick={createRoom}
          disabled={busy}
          style={styles.primary}
        >
          {busy ? "Creating room…" : "Start a conversation"}
        </button>
        <button
          type="button"
          onClick={() => setShowJoin((v) => !v)}
          style={styles.secondary}
          aria-expanded={showJoin}
        >
          Join with a code
        </button>
      </div>

      {showJoin && (
        <form onSubmit={joinRoom} style={styles.joinForm}>
          <label htmlFor="room-code" className="sr-only">
            Room code
          </label>
          <input
            id="room-code"
            value={code}
            onChange={(e) => setCode(e.target.value.toUpperCase())}
            placeholder="ABC123"
            maxLength={6}
            autoComplete="off"
            style={styles.input}
          />
          <button
            type="submit"
            disabled={code.trim().length === 0}
            style={styles.joinBtn}
          >
            Join →
          </button>
        </form>
      )}

      {error && (
        <p style={styles.error} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: { display: "flex", flexDirection: "column", gap: "0.75rem" },
  ctaRow: { display: "flex", flexWrap: "wrap", gap: "0.75rem" },
  primary: {
    padding: "0.95rem 1.5rem",
    borderRadius: "var(--radius)",
    border: "none",
    background: "var(--accent)",
    color: "#001016",
    fontWeight: 700,
    fontSize: "1.05rem",
    cursor: "pointer",
    minHeight: 56,
    fontFamily: "inherit",
  },
  secondary: {
    padding: "0.95rem 1.5rem",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    background: "transparent",
    color: "var(--text)",
    fontSize: "1rem",
    cursor: "pointer",
    minHeight: 56,
    fontFamily: "inherit",
  },
  joinForm: { display: "flex", gap: "0.5rem", maxWidth: 360, flexWrap: "wrap" },
  input: {
    flex: 1,
    minWidth: 0,
    padding: "0.7rem 0.9rem",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    background: "var(--bg-input)",
    color: "var(--text)",
    fontSize: "1.05rem",
    letterSpacing: "0.15em",
    textTransform: "uppercase",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
  },
  joinBtn: {
    padding: "0.7rem 1.1rem",
    borderRadius: "var(--radius)",
    border: "none",
    background: "var(--primary)",
    color: "#fff",
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "inherit",
  },
  error: { color: "var(--danger)", margin: 0, fontSize: "0.9rem" },
};
