"use client";

import { useEffect, useRef, useState } from "react";

interface ConfidenceMeterProps {
  /** 0..1 */
  value: number | null;
  label: string | null;
  ready: boolean;
  paused: boolean;
  /** Threshold above which we consider the prediction "committed". Default 0.7. */
  threshold?: number;
  /** Called when the signer confirms the correct word. */
  onCorrect?: (original: string, corrected: string) => void;
}

export function ConfidenceMeter({
  value,
  label,
  ready,
  paused,
  threshold = 0.7,
  onCorrect,
}: ConfidenceMeterProps) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  const pct = Math.round(v * 100);
  const committed = ready && v >= threshold && !!label;

  const [correcting, setCorrecting] = useState(false);
  const [correction, setCorrection] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Close the correction form whenever a new label arrives.
  const prevLabel = useRef(label);
  useEffect(() => {
    if (label !== prevLabel.current) {
      setCorrecting(false);
      setCorrection("");
      prevLabel.current = label;
    }
  }, [label]);

  useEffect(() => {
    if (correcting) inputRef.current?.focus();
  }, [correcting]);

  function submitCorrection(e: React.FormEvent) {
    e.preventDefault();
    const word = correction.trim();
    if (!word || !label) return;
    onCorrect?.(label, word);
    setCorrecting(false);
    setCorrection("");
  }

  const barColor = committed ? "var(--success)" : "var(--warn)";

  let statusText: string;
  if (paused) statusText = "Signing paused";
  else if (!ready) statusText = "Buffering frames…";
  else if (!label) statusText = "Hold steady — I'm listening";
  else if (committed) statusText = label;
  else statusText = `${label}? Hold steady…`;

  return (
    <div
      style={styles.wrap}
      role="status"
      aria-live="polite"
      aria-label={`Sign confidence ${pct} percent: ${statusText}`}
    >
      <div style={styles.topRow}>
        <span
          style={{
            ...styles.predLabel,
            color: committed ? "var(--accent)" : "var(--text-muted)",
            fontWeight: committed ? 700 : 500,
          }}
        >
          {statusText}
        </span>
        {ready && label && (
          <span style={styles.pct} aria-hidden>{pct}%</span>
        )}
      </div>

      <div style={styles.track} aria-hidden>
        <div style={{ ...styles.fill, width: `${pct}%`, background: barColor }} />
        <div style={{ ...styles.thresholdMark, left: `${threshold * 100}%` }} />
      </div>

      {committed && onCorrect && !correcting && (
        <button
          type="button"
          onClick={() => setCorrecting(true)}
          style={styles.wrongBtn}
          aria-label={`Not what you signed? Correct "${label}"`}
        >
          Not what I signed
        </button>
      )}

      {correcting && (
        <form onSubmit={submitCorrection} style={styles.correctionForm}>
          <label htmlFor="sl-correction" style={styles.correctionLabel}>
            What did you sign?
          </label>
          <div style={styles.correctionRow}>
            <input
              id="sl-correction"
              ref={inputRef}
              value={correction}
              onChange={(e) => setCorrection(e.target.value)}
              placeholder="Type the correct word…"
              style={styles.correctionInput}
              autoComplete="off"
            />
            <button type="submit" disabled={!correction.trim()} style={styles.correctionSubmit}>
              Send
            </button>
            <button
              type="button"
              onClick={() => { setCorrecting(false); setCorrection(""); }}
              style={styles.correctionCancel}
            >
              Cancel
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    display: "flex",
    flexDirection: "column",
    gap: "0.4rem",
    padding: "0.6rem 0.85rem",
    background: "var(--bg-card)",
    borderRadius: "var(--radius)",
    minHeight: 56,
  },
  topRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "0.75rem",
  },
  predLabel: {
    fontSize: "1.1rem",
    transition: "color 200ms ease",
    minWidth: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  pct: {
    fontSize: "0.8rem",
    color: "var(--text-faint)",
    fontVariantNumeric: "tabular-nums",
    flexShrink: 0,
  },
  track: {
    position: "relative",
    height: 6,
    background: "var(--bg-input)",
    borderRadius: 999,
    overflow: "hidden",
  },
  fill: {
    height: "100%",
    borderRadius: 999,
    transition: "width 180ms ease-out, background 200ms ease",
  },
  thresholdMark: {
    position: "absolute",
    top: -2,
    width: 2,
    height: 10,
    background: "var(--text-faint)",
    opacity: 0.35,
  },
  wrongBtn: {
    alignSelf: "flex-start",
    padding: "0.25rem 0.65rem",
    borderRadius: 999,
    border: "1px solid var(--border)",
    background: "transparent",
    color: "var(--text-muted)",
    fontSize: "0.78rem",
    cursor: "pointer",
    marginTop: "0.1rem",
  },
  correctionForm: {
    display: "flex",
    flexDirection: "column",
    gap: "0.35rem",
    marginTop: "0.1rem",
  },
  correctionLabel: {
    fontSize: "0.82rem",
    color: "var(--text-muted)",
  },
  correctionRow: {
    display: "flex",
    gap: "0.4rem",
    flexWrap: "wrap",
  },
  correctionInput: {
    flex: 1,
    minWidth: 120,
    padding: "0.4rem 0.65rem",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    background: "var(--bg-input)",
    color: "var(--text)",
    fontSize: "0.9rem",
    fontFamily: "inherit",
  },
  correctionSubmit: {
    padding: "0.4rem 0.85rem",
    borderRadius: "var(--radius)",
    border: "none",
    background: "var(--primary)",
    color: "#fff",
    cursor: "pointer",
    fontSize: "0.88rem",
    fontWeight: 600,
  },
  correctionCancel: {
    padding: "0.4rem 0.65rem",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    background: "transparent",
    color: "var(--text-muted)",
    cursor: "pointer",
    fontSize: "0.88rem",
  },
};
