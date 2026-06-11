"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";
import { Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import { useSpeechToText } from "@/hooks/useSpeechToText";
import { useSpeakSignCaptions } from "@/hooks/useSpeakSignCaptions";
import { useWebRTC } from "@/hooks/useWebRTC";
import { RemoteVideo } from "./RemoteVideo";
import { CaptionsPanel } from "./CaptionsPanel";
import { PermissionGate } from "./PermissionGate";
import type { Caption } from "@/hooks/useRoom";
import { usePreferences } from "@/lib/preferences";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { HotkeyHint } from "@/components/primitives/HotkeyHint";
import { cn } from "@/lib/utils";

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
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const { prefs } = usePreferences();
  const isToggleMode = prefs.pushToTalkMode === "toggle";

  const { remoteStream } = useWebRTC(socket, localStream, /*isInitiator*/ false, peerPresent);
  const { listening, supported, start, stop } = useSpeechToText(onSpeech);

  useSpeakSignCaptions(captions, ttsEnabled);

  // Push-to-talk via spacebar — works for both hold and toggle modes.
  const heldRef = useRef(false);
  useEffect(() => {
    if (!supported || micStatus !== "ok") return;

    function isTextTarget(el: EventTarget | null) {
      if (!(el instanceof HTMLElement)) return false;
      return el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable;
    }

    function onDown(e: KeyboardEvent) {
      if (e.code !== "Space" || isTextTarget(e.target)) return;
      if (isToggleMode) {
        if (e.repeat) return;
        e.preventDefault();
        if (listening) stop();
        else start();
        return;
      }
      // Hold mode
      if (e.repeat) return;
      e.preventDefault();
      if (!heldRef.current) {
        heldRef.current = true;
        start();
      }
    }
    function onUp(e: KeyboardEvent) {
      if (isToggleMode) return;
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
  }, [supported, micStatus, start, stop, isToggleMode, listening]);

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
    if (isToggleMode) return;
    (e.target as Element).setPointerCapture(e.pointerId);
    if (!listening) start();
  };
  const pressEnd = (e: React.PointerEvent) => {
    if (isToggleMode) return;
    try {
      (e.target as Element).releasePointerCapture(e.pointerId);
    } catch {}
    if (listening) stop();
  };
  const toggleClick = () => {
    if (!isToggleMode) return;
    if (listening) stop();
    else start();
  };

  return (
    <div className="grid flex-1 gap-4 lg:grid-cols-[1.3fr_1fr]">
      {/* STAGE — peer video + captions */}
      <Card className="flex flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
          <div className="inline-flex items-center gap-2">
            <span className="eyebrow">Signer</span>
            {!peerPresent && (
              <Badge tone="warning" className="text-[0.65rem]">Waiting…</Badge>
            )}
          </div>
          <Button
            variant={ttsEnabled ? "primary" : "secondary"}
            size="sm"
            onClick={() => setTtsEnabled((v) => !v)}
            aria-pressed={ttsEnabled}
          >
            {ttsEnabled ? <Volume2 aria-hidden /> : <VolumeX aria-hidden />}
            {ttsEnabled ? "Reading aloud" : "Muted"}
          </Button>
        </div>
        <RemoteVideo
          stream={remoteStream}
          className="aspect-video w-full bg-[var(--color-surface-sunken)]"
        />
        <div className="border-t border-[var(--color-border)] p-3">
          <CaptionsPanel
            captions={captions}
            filter="sign"
            emptyHint="When the signer signs a word, it appears here as a caption."
          />
        </div>
      </Card>

      {/* RIGHT RAIL — your video + push-to-talk */}
      <div className="flex flex-col gap-4">
        <Card className="overflow-hidden">
          <div className="border-b border-[var(--color-border)] px-4 py-2 eyebrow">You</div>
          <RemoteVideo
            stream={localStream}
            muted
            className="aspect-video w-full bg-[var(--color-surface-sunken)]"
          />
        </Card>

        <Card className="p-5">
          {!supported ? (
            <Alert tone="warning" title="Speech recognition not available">
              This browser doesn&apos;t support the Web Speech API. Try Chrome or Edge.
            </Alert>
          ) : (
            <>
              <p className="eyebrow">Push to talk</p>
              <p className="mt-2 text-sm text-[var(--color-text-muted)] leading-relaxed">
                {isToggleMode
                  ? "Tap to start speaking, tap again to stop. Or press Space."
                  : "Hold the button (or press and hold Space) to talk. Release to send."}
              </p>

              <button
                type="button"
                disabled={!supported}
                onPointerDown={pressStart}
                onPointerUp={pressEnd}
                onPointerCancel={pressEnd}
                onPointerLeave={pressEnd}
                onClick={toggleClick}
                aria-pressed={listening}
                aria-label={listening ? "Listening — release to stop" : "Push to talk"}
                className={cn(
                  "mt-4 inline-flex h-20 w-full select-none items-center justify-center gap-3 rounded-[var(--radius-lg)] text-lg font-bold transition focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)] disabled:opacity-50",
                  listening
                    ? "bg-[var(--color-danger)] text-white shadow-[var(--shadow-card)] scale-[1.02]"
                    : "bg-[var(--color-brand)] text-[var(--color-brand-foreground)] hover:bg-[var(--color-brand-hover)]",
                )}
                style={{ touchAction: "none" }}
              >
                {listening ? (
                  <>
                    <Mic className="size-6 sl-pulse-soft" aria-hidden />
                    Listening…
                  </>
                ) : (
                  <>
                    <Mic className="size-6" aria-hidden />
                    {isToggleMode ? "Tap to talk" : "Push to talk"}
                  </>
                )}
              </button>

              <div className="mt-3 flex items-center justify-between gap-3">
                <HotkeyHint keys={["Space"]} description={isToggleMode ? "toggle listening" : "hold to talk"} />
                <span
                  className={cn(
                    "inline-flex items-center gap-1.5 text-xs font-semibold",
                    listening ? "text-[var(--color-danger)]" : "text-[var(--color-text-muted)]",
                  )}
                  aria-live="polite"
                >
                  {listening ? (
                    <>
                      <Mic className="size-3.5" aria-hidden /> On
                    </>
                  ) : (
                    <>
                      <MicOff className="size-3.5" aria-hidden /> Off
                    </>
                  )}
                </span>
              </div>

              {micError && (
                <Alert tone="danger" className="mt-3" title="Microphone unavailable">
                  {micError}
                </Alert>
              )}
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
