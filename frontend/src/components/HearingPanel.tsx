import { useEffect, useRef } from "react";
import { useSpeechToText } from "../hooks/useSpeechToText";

export function HearingPanel() {
  const { transcript, listening, supported, start, stop, clear } =
    useSpeechToText();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to newest entry
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  return (
    <section style={styles.panel}>
      <h2 style={styles.heading}>Hearing User</h2>

      <div style={styles.log} role="log" aria-live="polite">
        {transcript.length === 0 ? (
          <p style={styles.empty}>
            {supported
              ? "Press the mic to start speaking…"
              : "Speech recognition not supported in this browser."}
          </p>
        ) : (
          transcript.map((entry) => (
            <div key={entry.id} style={styles.entry}>
              <span style={styles.entryText}>{entry.text}</span>
              <span style={styles.time}>
                {new Date(entry.ts).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })}
              </span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      <div style={styles.controls}>
        <button
          onClick={listening ? stop : start}
          disabled={!supported}
          style={{
            ...styles.micBtn,
            background: listening ? "#e53935" : "#1976d2",
          }}
          aria-label={listening ? "Stop recording" : "Start recording"}
        >
          {listening ? "⏹ Stop" : "🎤 Speak"}
        </button>

        <button
          onClick={clear}
          disabled={transcript.length === 0}
          style={styles.clearBtn}
        >
          Clear
        </button>
      </div>
    </section>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    padding: "1rem",
    gap: "0.75rem",
  },
  heading: { margin: 0, fontSize: "1.1rem", color: "#ccc" },
  log: {
    flex: 1,
    overflowY: "auto",
    background: "#0d0d1a",
    borderRadius: 8,
    padding: "0.75rem",
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
    minHeight: 200,
  },
  empty: { color: "#555", fontSize: "0.9rem", margin: 0 },
  entry: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "baseline",
    gap: "1rem",
    padding: "0.4rem 0.5rem",
    borderRadius: 6,
    background: "#1a1a2e",
  },
  entryText: { color: "#e0e0e0", fontSize: "1rem" },
  time: { color: "#555", fontSize: "0.75rem", flexShrink: 0 },
  controls: { display: "flex", gap: "0.75rem" },
  micBtn: {
    flex: 1,
    padding: "0.6rem 1rem",
    borderRadius: 8,
    border: "none",
    color: "#fff",
    fontSize: "1rem",
    cursor: "pointer",
    fontWeight: 600,
  },
  clearBtn: {
    padding: "0.6rem 1rem",
    borderRadius: 8,
    border: "none",
    background: "#333",
    color: "#aaa",
    cursor: "pointer",
  },
};
