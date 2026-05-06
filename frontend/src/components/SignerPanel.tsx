import { useEffect, useRef, useState } from "react";
import { useSignRecognition } from "../hooks/useSignRecognition";
import { LandmarkOverlay } from "./LandmarkOverlay";

const VIDEO_W = 640;
const VIDEO_H = 480;

export function SignerPanel() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [camError, setCamError] = useState<string | null>(null);
  const { prediction, connected, landmarkerResult, reset } =
    useSignRecognition(videoRef);

  // Start webcam
  useEffect(() => {
    let stream: MediaStream;

    navigator.mediaDevices
      .getUserMedia({ video: { width: VIDEO_W, height: VIDEO_H }, audio: false })
      .then((s) => {
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.play();
        }
      })
      .catch((err: Error) => setCamError(err.message));

    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <section style={styles.panel}>
      <h2 style={styles.heading}>Signer</h2>

      {camError ? (
        <p style={styles.error}>Camera error: {camError}</p>
      ) : (
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
        {prediction.ready && prediction.label ? (
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

      <div style={styles.controls}>
        <button onClick={reset} style={styles.btn}>
          Reset
        </button>
        <span style={{ ...styles.dot, background: connected ? "#4caf50" : "#f44336" }} />
        <span style={styles.connLabel}>{connected ? "Connected" : "Disconnected"}</span>
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
};
