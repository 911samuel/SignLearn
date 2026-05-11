"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";
import { useSpeechToText } from "@/hooks/useSpeechToText";
import { useWebRTC } from "@/hooks/useWebRTC";
import { RemoteVideo } from "./RemoteVideo";
import { CaptionsPanel } from "./CaptionsPanel";
import { PermissionGate } from "./PermissionGate";
import type { Caption } from "@/hooks/useRoom";

type MicStatus = "pending" | "ok" | "denied";

interface HearingViewProps {
  socket: Socket | null;
  captions: Caption[];
  peerPresent: boolean;
  onSpeech: (text: string, ts: number) => void;
}

export function HearingView({ socket, captions, peerPresent, onSpeech }: HearingViewProps) {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [micStatus, setMicStatus] = useState<MicStatus>("pending");
  const [micError, setMicError] = useState<string | null>(null);

  const { remoteStream } = useWebRTC(socket, localStream, /*isInitiator*/ false, peerPresent);
  const { listening, supported, start, stop } = useSpeechToText(onSpeech);

  // Push-to-talk: hold spacebar (when not focused in an input) to dictate.
  const heldRef = useRef(false);
  useEffect(() => {
    if (!supported || micStatus !== "ok") return;
    function isTextTarget(el: EventTarget | null) {
      if (!(el instanceof HTMLElement)) return false;
      return el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable;
    }
    function onDown(e: KeyboardEvent) {
      if (e.code !== "Space" || e.repeat || isTextTarget(e.target)) return;
      e.preventDefault();
      if (!heldRef.current) {
        heldRef.current = true;
        start();
      }
    }
    function onUp(e: KeyboardEvent) {
      if (e.code !== "Space" || isTextTarget(e.target)) return;
      if (heldRef.current) {
        heldRef.current = false;
        stop();
      }
    }
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, [supported, micStatus, start, stop]);

  useEffect(() => {
    return () => {
      localStream?.getTracks().forEach((t) => t.stop());
    };
  }, [localStream]);

  const requestMic = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    setLocalStream(stream);
    setMicStatus("ok");
  }, []);

  if (micStatus === "pending") {
    return (
      <PermissionGate
        kind="microphone"
        onAllow={async () => {
          try {
            await requestMic();
          } catch (err) {
            setMicStatus("denied");
            setMicError(err instanceof Error ? err.message : "Permission denied");
            throw err instanceof Error ? err : new Error("Permission denied");
          }
        }}
      />
    );
  }

  const pressStart = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    if (!listening) start();
  };
  const pressEnd = () => {
    if (listening) stop();
  };

  return (
    <div style={styles.grid}>
      <section style={styles.tile} aria-label="Signer's video and captions">
        <header style={styles.label}>
          {peerPresent ? "Signer" : "Waiting for the signer to join…"}
        </header>
        <RemoteVideo stream={remoteStream} style={{ aspectRatio: "4 / 3" }} />
        <CaptionsPanel
          captions={captions}
          filter="sign"
          emptyHint="Sign translations from the signer will show here."
        />
      </section>

      <section style={styles.tile} aria-label="Your microphone">
        <header style={styles.label}>You (Hearing)</header>
        <RemoteVideo stream={localStream} muted style={{ aspectRatio: "4 / 3" }} />
        {micError && <p style={styles.error} role="alert">Mic/camera unavailable: {micError}</p>}

        <p style={styles.hint}>
          {!supported
            ? "Speech recognition is not supported in this browser. Try Chrome or Edge."
            : listening
              ? "🎙 Listening — keep talking. The signer is reading your captions."
              : "Hold the button (or press & hold space) to talk. Release to send."}
        </p>

        <button
          type="button"
          disabled={!supported}
          onPointerDown={pressStart}
          onPointerUp={pressEnd}
          onPointerCancel={pressEnd}
          onPointerLeave={pressEnd}
          style={{
            ...styles.micBtn,
            background: listening ? "var(--danger)" : "var(--primary)",
            transform: listening ? "scale(1.02)" : "scale(1)",
          }}
          aria-pressed={listening}
          aria-label="Push to talk"
        >
          {listening ? "🎙 Listening…" : "🎙 Push to talk"}
        </button>
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
  hint: { color: "var(--text-muted)", fontSize: "0.9rem", margin: 0, lineHeight: 1.45 },
  micBtn: {
    padding: "1rem", borderRadius: "var(--radius)", border: "none",
    color: "#fff", fontSize: "1.1rem", cursor: "pointer", fontWeight: 700,
    minHeight: 64, transition: "transform 120ms ease, background 200ms ease",
    touchAction: "none", userSelect: "none",
  },
  error: { color: "var(--danger)", margin: 0, fontSize: "0.85rem" },
};
