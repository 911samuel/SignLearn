"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";
import { useSignRecognition } from "@/hooks/useSignRecognition";
import { useWebRTC } from "@/hooks/useWebRTC";
import { LandmarkOverlay } from "./LandmarkOverlay";
import { RemoteVideo } from "./RemoteVideo";
import { CaptionsPanel } from "./CaptionsPanel";
import { ConfidenceMeter } from "./ConfidenceMeter";
import { PermissionGate } from "./PermissionGate";
import type { Caption } from "@/hooks/useRoom";

const VIDEO_W = 640;
const VIDEO_H = 480;

export type CamStatus = "ok" | "lost" | "denied" | "pending";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5001";

interface SignerViewProps {
  socket: Socket | null;
  captions: Caption[];
  peerPresent: boolean;
  roomId?: string;
  onPrediction?: (label: string, confidence: number, ts: number) => void;
}

export function SignerView({ socket, captions, peerPresent, roomId, onPrediction }: SignerViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [camStatus, setCamStatus] = useState<CamStatus>("pending");
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);

  const { prediction, landmarkerResult, reset, paused, togglePaused, latencyMs, landmarkerError } =
    useSignRecognition(videoRef, socket);

  // Signer is the WebRTC initiator (polite peer).
  const { remoteStream } = useWebRTC(socket, localStream, /*isInitiator*/ true, peerPresent);

  const onPredictionRef = useRef(onPrediction);
  useEffect(() => { onPredictionRef.current = onPrediction; }, [onPrediction]);

  const lastLabelRef = useRef<string | null>(null);
  useEffect(() => {
    if (prediction.ready && prediction.label && prediction.label !== lastLabelRef.current) {
      lastLabelRef.current = prediction.label;
      onPredictionRef.current?.(prediction.label, prediction.confidence ?? 0, Date.now());
      // Subtle haptic pulse on commit — respects device/browser support silently.
      try { navigator.vibrate?.(15); } catch {}
    }
    if (!prediction.ready || !prediction.label) {
      lastLabelRef.current = null;
    }
  }, [prediction]);

  // Tear-down on unmount.
  useEffect(() => {
    return () => {
      localStream?.getTracks().forEach((t) => t.stop());
    };
  }, [localStream]);

  const requestCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: VIDEO_W, height: VIDEO_H },
      audio: true,
    });
    stream.getVideoTracks().forEach((track) => {
      track.onended = () => setCamStatus("lost");
    });
    // Set stream first, then flip status. The useEffect below wires
    // srcObject after React mounts the <video> element.
    setLocalStream(stream);
    setCamStatus("ok");
  }, []);

  // Attach stream to the video element once it's in the DOM.
  // Must be an effect — the <video> only renders after setCamStatus("ok").
  useEffect(() => {
    if (camStatus !== "ok" || !localStream || !videoRef.current) return;
    videoRef.current.srcObject = localStream;
    videoRef.current.play().catch(() => {});
  }, [camStatus, localStream]);

  if (camStatus === "pending") {
    return (
      <PermissionGate
        kind="camera"
        onAllow={async () => {
          try {
            await requestCamera();
          } catch {
            setCamStatus("denied");
            throw new Error("Camera permission was denied. You can re‑enable it from your browser's address bar.");
          }
        }}
      />
    );
  }

  return (
    <div style={styles.grid}>
      <section style={styles.tile} aria-label="Your camera and current sign">
        <header style={styles.label}>You (Signer)</header>
        {camStatus === "denied" && (
          <p style={styles.error} role="alert">
            Camera access denied — allow camera permission from the address bar and reload.
          </p>
        )}
        {camStatus === "lost" && (
          <p style={styles.error} role="alert">
            Camera disconnected — reconnect and reload.
          </p>
        )}
        {landmarkerError && (
          <p style={styles.error} role="alert">
            MediaPipe failed to load: {landmarkerError}
          </p>
        )}
        {camStatus === "ok" && (
          <div style={styles.videoWrapper}>
            <video
              ref={videoRef}
              width={VIDEO_W}
              height={VIDEO_H}
              muted
              playsInline
              aria-label="Your live camera preview with hand landmark overlay"
              style={styles.video}
            />
            <LandmarkOverlay
              result={landmarkerResult}
              width={VIDEO_W}
              height={VIDEO_H}
            />
            {latencyMs !== null && (
              <span style={styles.latencyPill} aria-label={`Round-trip latency ${latencyMs} milliseconds`}>
                ⚡ {latencyMs} ms
              </span>
            )}
          </div>
        )}

        <ConfidenceMeter
          value={prediction.confidence ?? null}
          label={prediction.label ?? null}
          ready={prediction.ready}
          paused={paused}
          onCorrect={roomId ? (original, corrected) => {
            fetch(`${BACKEND_URL}/corrections`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                room_id: roomId,
                original_text: original,
                corrected_text: corrected,
                confidence: prediction.confidence ?? null,
              }),
            }).catch(() => {});
          } : undefined}
        />

        <div style={styles.controls}>
          <button
            className="sl-btn"
            onClick={togglePaused}
            style={{ ...styles.btn, background: paused ? "var(--primary)" : "var(--bg-card)" }}
            aria-pressed={paused}
          >
            {paused ? "▶ Resume" : "⏸ Pause"}
          </button>
          <button className="sl-btn" onClick={reset} style={styles.btn}>
            ✋ Reset sign window
          </button>
        </div>
      </section>

      <section style={styles.tile} aria-label="Hearing peer">
        <header style={styles.label}>
          {peerPresent ? "Hearing peer" : "Waiting for the hearing user to join…"}
        </header>
        <RemoteVideo stream={remoteStream} style={{ aspectRatio: "4 / 3" }} />
        <CaptionsPanel
          captions={captions}
          filter="speech"
          emptyHint="Speech captions from the hearing user will show here."
        />
      </section>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
    gap: "1rem",
    flex: 1,
  },
  tile: {
    display: "flex", flexDirection: "column", gap: "0.5rem",
    padding: "0.75rem", background: "var(--bg-elevated)", borderRadius: "var(--radius-lg)",
  },
  label: {
    fontSize: "0.85rem", color: "var(--text-muted)",
    textTransform: "uppercase", letterSpacing: "0.05em",
  },
  videoWrapper: { position: "relative", lineHeight: 0, width: "100%" },
  video: {
    display: "block", borderRadius: "var(--radius)",
    width: "100%", height: "auto", transform: "scaleX(-1)",
  },
  latencyPill: {
    position: "absolute", top: 8, right: 8,
    fontSize: "0.72rem",
    color: "var(--text)",
    background: "rgba(0, 0, 0, 0.55)",
    backdropFilter: "blur(6px)",
    borderRadius: 999,
    padding: "3px 9px",
    fontVariantNumeric: "tabular-nums",
    pointerEvents: "none",
  },
  controls: { display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" },
  btn: {
    padding: "0.5rem 1rem", borderRadius: "var(--radius)", border: "1px solid var(--border)",
    background: "var(--bg-card)", color: "var(--text)", cursor: "pointer",
    fontSize: "0.9rem", minHeight: 40,
  },
  error: { color: "var(--danger)" },
};
