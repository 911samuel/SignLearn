"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";
import { Pause, Play, RotateCcw, UserRound, Users, Zap } from "lucide-react";
import { useSignRecognition } from "@/hooks/useSignRecognition";
import { useWebRTC } from "@/hooks/useWebRTC";
import { LandmarkOverlay } from "./LandmarkOverlay";
import { RemoteVideo } from "./RemoteVideo";
import { VideoCaptionOverlay } from "./VideoCaptionOverlay";
import { PermissionGate } from "./PermissionGate";
import type { Caption } from "@/hooks/useRoom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { StatusPill, type Status } from "@/components/primitives/StatusPill";

const VIDEO_W = 640;
const VIDEO_H = 480;

export type CamStatus = "ok" | "lost" | "denied" | "pending";

interface SignerViewProps {
  socket: Socket | null;
  captions: Caption[];
  peerPresent: boolean;
  roomId?: string;
  onPrediction?: (label: string, confidence: number, ts: number) => void;
}

export function SignerView({ socket, captions, peerPresent, onPrediction }: SignerViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [camStatus, setCamStatus] = useState<CamStatus>("pending");
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);

  const {
    prediction,
    wordPrediction,
    captureStatus,
    captureProgress,
    landmarkerResult,
    reset,
    paused,
    togglePaused,
    latencyMs,
    landmarkerError,
  } = useSignRecognition(videoRef, socket);

  const { remoteStream } = useWebRTC(socket, localStream, /*isInitiator*/ true, peerPresent);

  const onPredictionRef = useRef(onPrediction);
  useEffect(() => {
    onPredictionRef.current = onPrediction;
  }, [onPrediction]);

  useEffect(() => {
    if (!wordPrediction || wordPrediction.error) return;
    const best = wordPrediction.top3?.[0];
    if (!best) return;
    onPredictionRef.current?.(best.label, best.confidence, Date.now());
    try {
      navigator.vibrate?.(15);
    } catch {}
  }, [wordPrediction]);

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
    setLocalStream(stream);
    setCamStatus("ok");
  }, []);

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
            throw new Error("Camera permission was denied. Re-enable it from your browser's address bar.");
          }
        }}
      />
    );
  }

  const statusForPill: Status =
    paused
      ? "idle"
      : captureStatus === "signing"
        ? "signing"
        : captureStatus === "processing"
          ? "processing"
          : "idle";

  const candidates = wordPrediction?.top3 ?? [];
  const topCandidate = candidates[0];

  return (
    <div className="flex flex-1 flex-col gap-3">
      {/* TILE GRID — symmetric, Meet-style */}
      <div className="grid flex-1 gap-3 lg:grid-cols-2">
        {/* OWN TILE */}
        <Card className="flex flex-col overflow-hidden">
          <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-2.5">
            <div className="inline-flex items-center gap-2">
              <UserRound className="size-4 text-[var(--color-text-muted)]" aria-hidden />
              <span className="eyebrow">You (Signer)</span>
            </div>
            <StatusPill status={statusForPill} />
          </div>

          <div className="relative flex-1 bg-[var(--color-surface-sunken)]">
            {camStatus === "denied" && (
              <Alert tone="danger" title="Camera access denied" className="m-4">
                Allow camera permission from the browser&apos;s address bar, then reload this page.
              </Alert>
            )}
            {camStatus === "lost" && (
              <Alert tone="warning" title="Camera disconnected" className="m-4">
                Reconnect your camera and reload to continue.
              </Alert>
            )}
            {landmarkerError && (
              <Alert tone="danger" title="MediaPipe failed to load" className="m-4">
                {landmarkerError}
              </Alert>
            )}

            {camStatus === "ok" && (
              <div className="relative h-full bg-black">
                <video
                  ref={videoRef}
                  width={VIDEO_W}
                  height={VIDEO_H}
                  muted
                  playsInline
                  aria-label="Your live camera preview with hand landmark overlay"
                  className="block h-full w-full -scale-x-100 object-cover"
                />
                <LandmarkOverlay
                  result={landmarkerResult}
                  width={VIDEO_W}
                  height={VIDEO_H}
                />
                {latencyMs !== null && (
                  <Badge
                    tone="neutral"
                    className="absolute right-3 top-3 border-none bg-black/55 text-white backdrop-blur"
                  >
                    <Zap className="size-3" aria-hidden />
                    <span className="font-mono tabular-nums">{latencyMs} ms</span>
                    <span className="sr-only">round-trip latency</span>
                  </Badge>
                )}
                {topCandidate && !wordPrediction?.error && (
                  <div className="absolute left-3 top-3 max-w-[calc(100%-1.5rem)]">
                    <div className="inline-flex items-center gap-2 rounded-[var(--radius-md)] bg-black/55 px-3 py-1.5 backdrop-blur">
                      <span className="text-xs uppercase tracking-wider text-white/70">Sign</span>
                      <span className="text-base font-bold text-white">{topCandidate.label}</span>
                    </div>
                  </div>
                )}
                {prediction?.ready && prediction.label && (
                  <div className="absolute left-3 top-14">
                    <div className="inline-flex items-center gap-2 rounded-[var(--radius-md)] bg-black/55 px-3 py-1.5 backdrop-blur">
                      <span className="text-xs uppercase tracking-wider text-white/70">Letter</span>
                      <span className="text-base font-bold text-white">{prediction.label}</span>
                    </div>
                  </div>
                )}
                {captureStatus === "signing" && !paused && (
                  <div className="absolute inset-x-3 bottom-3">
                    <Progress
                      value={Math.round(captureProgress * 100)}
                      tone="brand"
                      aria-label={`Capture progress ${Math.round(captureProgress * 100)} percent`}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>

        {/* PEER TILE */}
        <Card className="flex flex-col overflow-hidden">
          <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-2.5">
            <div className="inline-flex items-center gap-2">
              <Users className="size-4 text-[var(--color-text-muted)]" aria-hidden />
              <span className="eyebrow">Hearing partner</span>
            </div>
            {!peerPresent && (
              <Badge tone="warning" className="text-[0.65rem]">Waiting…</Badge>
            )}
          </div>
          <div className="relative flex-1 bg-black">
            <RemoteVideo
              stream={remoteStream}
              className="h-full w-full object-cover"
            />
            <VideoCaptionOverlay
              captions={captions}
              filter="speech"
              emptyHint={peerPresent ? "When your partner speaks, captions appear here." : "Waiting for partner…"}
            />
          </div>
        </Card>
      </div>

      {/* BOTTOM CONTROL BAR */}
      <Card className="px-4 py-3">
        {wordPrediction?.error ? (
          <Alert tone="danger" title="Couldn't read that sign">
            Try again — make sure your hand is fully in frame and hold the sign steadily.
          </Alert>
        ) : (
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant={paused ? "primary" : "secondary"}
              size="sm"
              onClick={togglePaused}
              aria-pressed={paused}
            >
              {paused ? <Play aria-hidden /> : <Pause aria-hidden />}
              {paused ? "Resume" : "Pause"}
            </Button>
            <Button variant="secondary" size="sm" onClick={reset}>
              <RotateCcw aria-hidden /> Reset
            </Button>
            <span className="ml-auto text-xs text-[var(--color-text-muted)]">
              {paused
                ? "Paused — press Resume when ready"
                : captureStatus === "signing"
                  ? "Hold the sign — we're reading it…"
                  : captureStatus === "processing"
                    ? "Recognising…"
                    : "Start signing whenever you're ready"}
            </span>
          </div>
        )}
      </Card>
    </div>
  );
}
