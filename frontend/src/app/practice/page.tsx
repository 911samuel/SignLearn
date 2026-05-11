"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSignRecognition } from "@/hooks/useSignRecognition";
import { LandmarkOverlay } from "@/components/LandmarkOverlay";
import { ConfidenceMeter } from "@/components/ConfidenceMeter";
import { PermissionGate } from "@/components/PermissionGate";

const VIDEO_W = 640;
const VIDEO_H = 480;

type CamStatus = "pending" | "ok" | "denied" | "lost";

export default function PracticePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [camStatus, setCamStatus] = useState<CamStatus>("pending");
  const [history, setHistory] = useState<string[]>([]);

  const { prediction, landmarkerResult, reset, paused, togglePaused, latencyMs } =
    useSignRecognition(videoRef, null);

  const lastRef = useRef<string | null>(null);
  useEffect(() => {
    if (prediction.ready && prediction.label && prediction.label !== lastRef.current) {
      lastRef.current = prediction.label;
      setHistory((h) => [...h.slice(-49), prediction.label!]);
    }
    if (!prediction.ready || !prediction.label) lastRef.current = null;
  }, [prediction]);

  async function requestCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: VIDEO_W, height: VIDEO_H },
      audio: false,
    });
    setCamStatus("ok");
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }
    stream.getVideoTracks().forEach((t) => {
      t.onended = () => setCamStatus("lost");
    });
  }

  if (camStatus === "pending") {
    return (
      <div style={styles.shell}>
        <div style={styles.topbar}>
          <Link href="/" style={styles.back}>← Back</Link>
          <span style={styles.title}>Practice mode</span>
        </div>
        <PermissionGate
          kind="camera"
          onAllow={async () => {
            try {
              await requestCamera();
            } catch {
              setCamStatus("denied");
              throw new Error("Camera permission was denied.");
            }
          }}
        />
      </div>
    );
  }

  return (
    <div style={styles.shell}>
      <header style={styles.topbar}>
        <Link href="/" style={styles.back} aria-label="Back to home">← Back</Link>
        <span style={styles.title}>Practice mode</span>
        <span style={styles.badge}>No one else can see you</span>
      </header>

      <main style={styles.main}>
        <div style={styles.col}>
          <section style={styles.tile} aria-label="Your camera preview">
            {camStatus === "denied" && (
              <p style={styles.error} role="alert">
                Camera access denied. Allow permission from the address bar and reload.
              </p>
            )}
            {camStatus === "lost" && (
              <p style={styles.error} role="alert">Camera disconnected. Reload to reconnect.</p>
            )}
            {camStatus === "ok" && (
              <div style={styles.videoWrapper}>
                <video
                  ref={videoRef}
                  width={VIDEO_W}
                  height={VIDEO_H}
                  muted
                  playsInline
                  aria-label="Your camera preview with hand landmark overlay"
                  style={styles.video}
                />
                <LandmarkOverlay result={landmarkerResult} width={VIDEO_W} height={VIDEO_H} />
                {latencyMs !== null && (
                  <span style={styles.latencyPill} aria-hidden="true">⚡ {latencyMs} ms</span>
                )}
              </div>
            )}

            <ConfidenceMeter
              value={prediction.confidence ?? null}
              label={prediction.label ?? null}
              ready={prediction.ready}
              paused={paused}
            />

            <div style={styles.controls}>
              <button
                onClick={togglePaused}
                style={{ ...styles.btn, background: paused ? "var(--primary)" : "var(--bg-card)" }}
                aria-pressed={paused}
              >
                {paused ? "▶ Resume" : "⏸ Pause"}
              </button>
              <button onClick={reset} style={styles.btn}>✋ Reset window</button>
              <button onClick={() => setHistory([])} style={styles.btn}>Clear history</button>
            </div>
          </section>
        </div>

        <div style={styles.col}>
          <section style={styles.tile} aria-label="Practice history">
            <header style={styles.label}>What I signed</header>
            {history.length === 0 ? (
              <p style={styles.empty}>Signs you make will appear here. Try it!</p>
            ) : (
              <div style={styles.historyWrap} aria-live="polite">
                {history.map((word, i) => (
                  <span key={i} style={styles.chip}>{word}</span>
                ))}
              </div>
            )}

            <div style={styles.help}>
              <h2 style={styles.helpTitle}>Tips for better recognition</h2>
              <ul style={styles.helpList}>
                <li>Good lighting on your hands — no strong backlight.</li>
                <li>Keep hands fully in frame, roughly arm&apos;s-length away.</li>
                <li>Hold each sign for a beat — the model reads 30 frames at a time.</li>
                <li>Reset the window between distinct signs.</li>
              </ul>
              <p style={styles.readyLine}>
                Ready to have a real conversation?{" "}
                <Link href="/" style={styles.startLink}>Start a room →</Link>
              </p>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  shell: { minHeight: "100svh", display: "flex", flexDirection: "column" },
  topbar: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    padding: "0.7rem 1.25rem",
    borderBottom: "1px solid var(--border)",
    background: "var(--bg-elevated)",
  },
  back: { color: "var(--text-muted)", textDecoration: "none", fontSize: "0.9rem" },
  title: { fontWeight: 600, fontSize: "1rem" },
  badge: {
    marginLeft: "auto",
    padding: "0.25rem 0.65rem",
    borderRadius: 999,
    background: "var(--bg-card)",
    border: "1px solid var(--border)",
    color: "var(--text-muted)",
    fontSize: "0.78rem",
  },
  main: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
    gap: "1rem",
    padding: "1rem",
  },
  col: { display: "flex", flexDirection: "column" },
  tile: {
    display: "flex", flexDirection: "column", gap: "0.6rem",
    padding: "0.85rem", background: "var(--bg-elevated)",
    borderRadius: "var(--radius-lg)", flex: 1,
  },
  label: {
    fontSize: "0.82rem", color: "var(--text-muted)",
    textTransform: "uppercase", letterSpacing: "0.06em",
  },
  videoWrapper: { position: "relative", lineHeight: 0, width: "100%" },
  video: { display: "block", borderRadius: "var(--radius)", width: "100%", height: "auto", transform: "scaleX(-1)" },
  latencyPill: {
    position: "absolute", top: 8, right: 8,
    fontSize: "0.72rem", color: "var(--text)",
    background: "rgba(0,0,0,0.55)", backdropFilter: "blur(6px)",
    borderRadius: 999, padding: "3px 9px",
    fontVariantNumeric: "tabular-nums", pointerEvents: "none",
  },
  controls: { display: "flex", gap: "0.5rem", flexWrap: "wrap" },
  btn: {
    padding: "0.5rem 1rem", borderRadius: "var(--radius)", border: "1px solid var(--border)",
    background: "var(--bg-card)", color: "var(--text)", cursor: "pointer", fontSize: "0.88rem",
    minHeight: 40, fontFamily: "inherit",
  },
  error: { color: "var(--danger)", margin: 0 },
  empty: { color: "var(--text-faint)", fontSize: "0.9rem", margin: 0 },
  historyWrap: {
    display: "flex", flexWrap: "wrap", gap: "0.4rem",
    minHeight: 48, padding: "0.25rem 0",
  },
  chip: {
    padding: "0.3rem 0.65rem",
    borderRadius: 999,
    background: "var(--bg-card)",
    border: "1px solid var(--border)",
    fontSize: "0.9rem",
    color: "var(--accent)",
    fontWeight: 600,
  },
  help: {
    marginTop: "auto",
    paddingTop: "1rem",
    borderTop: "1px solid var(--border)",
  },
  helpTitle: { margin: "0 0 0.5rem", fontSize: "0.95rem" },
  helpList: { margin: 0, paddingLeft: "1.25rem", lineHeight: 1.7, color: "var(--text-muted)", fontSize: "0.9rem" },
  readyLine: { marginTop: "1rem", fontSize: "0.9rem", color: "var(--text-muted)" },
  startLink: { color: "var(--accent)", fontWeight: 600 },
};
