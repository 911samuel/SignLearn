import { useSpeechToText } from "../hooks/useSpeechToText";

interface HearingPanelProps {
  onSpeech?: (text: string, ts: number) => void;
}

export function HearingPanel({ onSpeech }: HearingPanelProps) {
  const { listening, supported, start, stop } = useSpeechToText(onSpeech);

  return (
    <section style={styles.panel}>
      <h2 style={styles.heading}>Hearing User</h2>

      <p style={styles.hint}>
        {supported
          ? listening
            ? "Listening… speak now."
            : "Press the mic to start speaking."
          : "Speech recognition not supported in this browser."}
      </p>

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
    borderLeft: "1px solid #333",
  },
  heading: { margin: 0, fontSize: "1.1rem", color: "#ccc" },
  hint: { color: "#555", fontSize: "0.9rem", margin: 0 },
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
};
