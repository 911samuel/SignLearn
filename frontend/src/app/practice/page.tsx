"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { CheckCircle2, Eye, EyeOff, Flame, RotateCcw, Shuffle, Target, XCircle } from "lucide-react";
import { useSignRecognition } from "@/hooks/useSignRecognition";
import { useRoom } from "@/hooks/useRoom";
import { BACKEND_URL } from "@/lib/api";

const DIGIT_NAMES = new Set([
  "zero", "one", "two", "three", "four",
  "five", "six", "seven", "eight", "nine",
]);
function isLetterOrDigit(t: string): boolean {
  return /^[a-z]$/.test(t) || DIGIT_NAMES.has(t);
}
import { LandmarkOverlay } from "@/components/LandmarkOverlay";
import { PermissionGate } from "@/components/PermissionGate";
import { PageShell } from "@/components/primitives/PageShell";
import { SectionHeader } from "@/components/primitives/SectionHeader";
import { ConfidenceMeter } from "@/components/primitives/ConfidenceMeter";
import { StatusPill, type Status } from "@/components/primitives/StatusPill";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ALL_LESSONS } from "@/data/curriculum";
import { recordAttempt, useProgress } from "@/lib/progress";
import { cn } from "@/lib/utils";
import { t } from "@/i18n";

const VIDEO_W = 640;
const VIDEO_H = 480;

type CamStatus = "pending" | "ok" | "denied" | "lost";

const TARGETS = Array.from(new Set(ALL_LESSONS.flatMap((l) => l.signs))).sort();

export default function PracticePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [camStatus, setCamStatus] = useState<CamStatus>("pending");
  const [target, setTarget] = useState<string>(TARGETS[0] ?? "hello");
  const [streak, setStreak] = useState(0);
  const [a11yMode, setA11yMode] = useState(false);
  const [outcome, setOutcome] = useState<"hit" | "miss" | null>(null);
  const progress = useProgress();

  // Allocate a one-off room so the backend treats us as a valid signer.
  // No peer ever joins; this is the minimum scaffolding to reuse the
  // existing recognition pipeline in a single-user practice page.
  const [roomId, setRoomId] = useState<string>("");
  useEffect(() => {
    let cancelled = false;
    fetch(`${BACKEND_URL}/rooms`, { method: "POST" })
      .then((r) => r.json())
      .then((d: { room_id?: string }) => {
        if (!cancelled && d.room_id) setRoomId(d.room_id);
      })
      .catch(() => { /* recognition just won't fire without a room */ });
    return () => { cancelled = true; };
  }, []);
  const { socket } = useRoom(roomId, "signer", "Learner");

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
  } = useSignRecognition(videoRef, socket);

  const targetIsLetterOrDigit = isLetterOrDigit(target);

  // Letter/digit target: score the latest letter-pipeline prediction.
  useEffect(() => {
    if (!targetIsLetterOrDigit) return;
    if (!prediction?.ready || !prediction.label) return;
    const hit = prediction.label === target;
    setOutcome(hit ? "hit" : "miss");
    setStreak((s) => (hit ? s + 1 : 0));
    recordAttempt(prediction.label, hit, prediction.confidence ?? 0);
    if (hit) {
      try { navigator.vibrate?.(25); } catch {}
    }
    const id = window.setTimeout(() => setOutcome(null), 1400);
    return () => window.clearTimeout(id);
  }, [prediction, target, targetIsLetterOrDigit]);

  // Word target: score the latest word-pipeline prediction.
  useEffect(() => {
    if (targetIsLetterOrDigit) return;
    if (!wordPrediction || wordPrediction.error) return;
    const best = wordPrediction.top3?.[0];
    if (!best) return;
    const hit = best.label === target;
    setOutcome(hit ? "hit" : "miss");
    setStreak((s) => (hit ? s + 1 : 0));
    recordAttempt(best.label, hit, best.confidence);
    if (hit) {
      try { navigator.vibrate?.(25); } catch {}
    }
    const id = window.setTimeout(() => setOutcome(null), 1400);
    return () => window.clearTimeout(id);
  }, [wordPrediction, target, targetIsLetterOrDigit]);

  const [localStream, setLocalStream] = useState<MediaStream | null>(null);

  async function requestCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: VIDEO_W, height: VIDEO_H },
      audio: false,
    });
    setLocalStream(stream);
    setCamStatus("ok");
    stream.getVideoTracks().forEach((tr) => {
      tr.onended = () => setCamStatus("lost");
    });
  }

  // Attach the stream once the <video> element is mounted (after
  // camStatus flips to "ok").  Assigning srcObject inside requestCamera
  // would no-op because videoRef.current is still null while the
  // "pending" branch is rendered.
  useEffect(() => {
    if (camStatus !== "ok" || !localStream || !videoRef.current) return;
    videoRef.current.srcObject = localStream;
    videoRef.current.play().catch(() => {});
  }, [camStatus, localStream]);

  useEffect(() => {
    return () => {
      localStream?.getTracks().forEach((t) => t.stop());
    };
  }, [localStream]);

  if (camStatus === "pending") {
    return (
      <PageShell>
        <div className="pt-10">
          <SectionHeader eyebrow="Practice" title={t("practice.title")} description={t("practice.subhead")} as="h1" />
        </div>
        <div className="mt-8">
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
      </PageShell>
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

  function pickRandom() {
    let next = target;
    while (next === target && TARGETS.length > 1) {
      next = TARGETS[Math.floor(Math.random() * TARGETS.length)];
    }
    setTarget(next);
    setOutcome(null);
  }

  return (
    <PageShell>
      <div className="pt-10">
        <SectionHeader
          eyebrow="Practice"
          title={t("practice.title")}
          description={t("practice.subhead")}
          as="h1"
        />
      </div>

      <div className="mt-8 grid gap-5 lg:grid-cols-[1.4fr_1fr] pb-12">
        {/* Camera stage */}
        <Card className="overflow-hidden">
          <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
            <Badge tone="brand">
              <Target className="size-3.5" aria-hidden />
              Target: <strong className="ml-1 font-semibold">{target}</strong>
            </Badge>
            <StatusPill status={statusForPill} />
          </div>

          <div className="relative bg-black">
            <video
              ref={videoRef}
              width={VIDEO_W}
              height={VIDEO_H}
              muted
              playsInline
              aria-label="Your camera preview with hand landmark overlay"
              className="block w-full -scale-x-100"
            />
            <LandmarkOverlay result={landmarkerResult} width={VIDEO_W} height={VIDEO_H} />

            {latencyMs !== null && (
              <Badge tone="neutral" className="absolute right-3 top-3 border-none bg-black/55 text-white backdrop-blur">
                <span className="font-mono">{latencyMs} ms</span>
              </Badge>
            )}

            {outcome && (
              <div
                role="status"
                aria-live="assertive"
                className={cn(
                  "absolute inset-0 grid place-items-center bg-black/40 text-white sl-pop-in pointer-events-none",
                )}
              >
                {outcome === "hit" ? (
                  <div className="flex flex-col items-center gap-3 text-center">
                    <CheckCircle2 className="size-20 text-[var(--color-success)]" aria-hidden />
                    <p className="heading-display">Nice!</p>
                    <p className="text-lg">You signed &ldquo;{target}&rdquo;</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3 text-center">
                    <XCircle className="size-20 text-[var(--color-warning)]" aria-hidden />
                    <p className="heading-display">Try again</p>
                    <p className="text-lg">We read another sign.</p>
                  </div>
                )}
              </div>
            )}

            {captureStatus === "signing" && !paused && (
              <div className="absolute inset-x-3 bottom-3 h-2.5 overflow-hidden rounded-full bg-white/25">
                <div
                  className="h-full bg-[var(--color-brand)]"
                  style={{ width: `${Math.round(captureProgress * 100)}%` }}
                />
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--color-border)] p-4">
            <div className="flex flex-wrap items-center gap-2">
              <Button variant={paused ? "primary" : "secondary"} onClick={togglePaused}>
                {paused ? "Resume" : "Pause"}
              </Button>
              <Button variant="secondary" onClick={reset}>
                <RotateCcw aria-hidden /> Reset
              </Button>
              <Button variant="secondary" onClick={pickRandom}>
                <Shuffle aria-hidden /> Random sign
              </Button>
            </div>
            <Button
              variant="ghost"
              onClick={() => setA11yMode((v) => !v)}
              aria-pressed={a11yMode}
            >
              {a11yMode ? <EyeOff aria-hidden /> : <Eye aria-hidden />}
              {a11yMode ? "Hide reference" : "Show reference"}
            </Button>
          </div>
        </Card>

        {/* Right rail — target picker, streak, prediction */}
        <div className="flex flex-col gap-4">
          <Card className="p-5">
            <p className="eyebrow">Pick a target sign</p>
            <div className="mt-3 flex flex-wrap gap-2 max-h-[16rem] overflow-y-auto pr-1" role="radiogroup" aria-label="Target sign">
              {TARGETS.map((s) => (
                <button
                  key={s}
                  type="button"
                  role="radio"
                  aria-checked={target === s}
                  onClick={() => {
                    setTarget(s);
                    setOutcome(null);
                  }}
                  className={cn(
                    "inline-flex h-9 items-center rounded-full border px-3 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]",
                    target === s
                      ? "border-transparent bg-[var(--color-brand)] text-[var(--color-brand-foreground)]"
                      : "border-[var(--color-border-strong)] bg-[var(--color-surface)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]",
                  )}
                >
                  {s}
                </button>
              ))}
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-4">
            <Card className="p-4">
              <p className="eyebrow">Current streak</p>
              <p className="mt-2 inline-flex items-center gap-2 heading-h2 text-[var(--color-text)]">
                <Flame className="size-6 text-[var(--color-warning)]" aria-hidden />
                {streak}
              </p>
            </Card>
            <Card className="p-4">
              <p className="eyebrow">All-time XP</p>
              <p className="mt-2 heading-h2 text-[var(--color-text)]">{progress.xp.toLocaleString()}</p>
            </Card>
          </div>

          <Card className="p-5">
            <p className="eyebrow">Latest reading</p>
            {(() => {
              const latestLabel = targetIsLetterOrDigit
                ? prediction?.label
                : wordPrediction?.top3?.[0]?.label;
              const latestConf = targetIsLetterOrDigit
                ? (prediction?.confidence ?? 0)
                : (wordPrediction?.top3?.[0]?.confidence ?? 0);
              return latestLabel ? (
                <>
                  <p className="mt-2 heading-h2 text-[var(--color-text)]">{latestLabel}</p>
                  <ConfidenceMeter value={latestConf} className="mt-3" />
                </>
              ) : (
                <p className="mt-2 text-[var(--color-text-faint)]">Sign something to see a reading.</p>
              );
            })()}
          </Card>

          {a11yMode && (
            <Card className="p-5">
              <p className="eyebrow">Reference</p>
              <p className="mt-3 text-[var(--color-text)]">
                Hold the sign for <strong>{target}</strong> in front of the camera. Make sure your
                hand is fully visible and lighting is even.
              </p>
              <p className="mt-2 text-sm text-[var(--color-text-muted)]">
                If recognition keeps missing, slow the motion down and check the landmark overlay is
                tracking all five fingers.
              </p>
            </Card>
          )}
        </div>
      </div>
    </PageShell>
  );
}
