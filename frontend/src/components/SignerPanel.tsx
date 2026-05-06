import { useEffect, useRef, useState } from "react";
import { useSignRecognition, type ConnectionStatus } from "../hooks/useSignRecognition";
import { LandmarkOverlay } from "./LandmarkOverlay";

const VIDEO_W = 640;
const VIDEO_H = 480;

export type CamStatus = "ok" | "lost" | "denied";

interface SignerPanelProps {
  onPrediction?: (label: string, confidence: number, ts: number) => void;
  onConnectionChange?: (status: ConnectionStatus) => void;
}

export function SignerPanel({ onPrediction, onConnectionChange }: SignerPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [camStatus, setCamStatus] = useState<CamStatus>("ok");
  const { prediction, connectionStatus, landmarkerResult, reset, paused, togglePaused, latencyMs } =
    useSignRecognition(videoRef);

  const onPredictionRef = useRef(onPrediction);
  useEffect(() => { onPredictionRef.current = onPrediction; }, [onPrediction]);

  const onConnectionChangeRef = useRef(onConnectionChange);
  useEffect(() => { onConnectionChangeRef.current = onConnectionChange; }, [onConnectionChange]);

  useEffect(() => {
    onConnectionChangeRef.current?.(connectionStatus);
  }, [connectionStatus]);

  // Fire callback when a new confident prediction arrives
  const lastLabelRef = useRef<string | null>(null);
  useEffect(() => {
    if (prediction.ready && prediction.label && prediction.label !== lastLabelRef.current) {
      lastLabelRef.current = prediction.label;
      onPredictionRef.current?.(prediction.label, prediction.confidence ?? 0, Date.now());
    }
    if (!prediction.ready || !prediction.label) {
      lastLabelRef.current = null;
    }
  }, [prediction]);

  // Start webcam and detect mid-stream loss
  useEffect(() => {
    let stream: MediaStream;

    navigator.mediaDevices
      .getUserMedia({ video: { width: VIDEO_W, height: VIDEO_H }, audio: false })
      .then((s) => {
        stream = s;
        setCamStatus("ok");
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.play();
        }
        // Detect camera revoked/disconnected mid-session
        s.getVideoTracks().forEach((track) => {
          track.onended = () => setCamStatus("lost");
        });
      })
      .catch(() => setCamStatus("denied"));

    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <section style={styles.panel}>
      <h2 style={styles.heading}>Signer</h2>

      {camStatus === "denied" && (
        <p style={styles.error}>Camera access denied — please allow camera permission and reload.</p>
      )}
      {camStatus === "lost" && (
        <p style={styles.error}>Camera disconnected — please reconnect your camera and reload.</p>
      )}
      {camStatus === "ok" && (
        <div style={styles.videoWrapper}>
          <video
            ref={videoRef}
            width={VIDEO_W}
            height={VIDEO_H}
            muted
            playsInline
            style={styles.video}
          />
          <LandmarkOverlay
            result={landmarkerResult}
            width={VIDEO_W}
            height={VIDEO_H}
          />
        </div>
      )}

      <div style={styles.predictionBox}>
        {paused ? (
          <span style={styles.waiting}>Signing paused</span>
        ) : prediction.ready && prediction.label ? (
          <>
            <span style={styles.label}>{prediction.label}</span>
            <span style={styles.conf}>
              {((prediction.confidence ?? 0) * 100).toFixed(0)}%
            </span>
          </>
        ) : (
          <span style={styles.waiting}>
            {prediction.ready ? "No confident prediction" : "Buffering…"}
          </span>
        )}
      </div>

      {import.meta.env.DEV && latencyMs !== null && (
        <div style={styles.latencyBadge}>
          ⏱ {latencyMs} ms {latencyMs > 2000 ? "⚠️" : ""}
        </div>
      )}

      <div style={styles.controls}>
        <button
          onClick={togglePaused}
          style={{ ...styles.btn, background: paused ? "#1976d2" : "#555" }}
          aria-label={paused ? "Start signing" : "Stop signing"}
        >
          {paused ? "▶ Start Signing" : "⏸ Stop Signing"}
        </button>
        <button onClick={reset} style={styles.btn}>
          Reset
        </button>
        <span style={{
          ...styles.dot,
          background: connectionStatus === "connected" ? "#4caf50"
            : connectionStatus === "reconnecting" ? "#ff9800"
            : "#f44336"
        }} />
        <span style={styles.connLabel}>
          {connectionStatus === "connected" ? "Connected"
            : connectionStatus === "reconnecting" ? "Reconnecting…"
            : "Disconnected"}
        </span>
      </div>
    </section>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "1rem",
    gap: "0.75rem",
    borderRight: "1px solid #333",
    overflowY: "auto",
  },
  heading: { margin: 0, fontSize: "1.1rem", color: "#ccc" },
  videoWrapper: { position: "relative", lineHeight: 0 },
  video: { display: "block", borderRadius: 8 },
  predictionBox: {
    minHeight: 56,
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    background: "#1a1a2e",
    borderRadius: 8,
    padding: "0.5rem 1.25rem",
  },
  label: { fontSize: "2rem", fontWeight: 700, color: "#00e5ff" },
  conf: { fontSize: "1rem", color: "#aaa" },
  waiting: { fontSize: "1rem", color: "#555" },
  controls: { display: "flex", alignItems: "center", gap: "0.75rem" },
  btn: {
    padding: "0.4rem 1rem",
    borderRadius: 6,
    border: "none",
    background: "#333",
    color: "#eee",
    cursor: "pointer",
  },
  dot: { width: 10, height: 10, borderRadius: "50%", display: "inline-block" },
  connLabel: { fontSize: "0.8rem", color: "#888" },
  error: { color: "#f44336" },
  latencyBadge: {
    fontSize: "0.75rem",
    color: "#888",
    background: "#111",
    borderRadius: 4,
    padding: "2px 8px",
  },
};
