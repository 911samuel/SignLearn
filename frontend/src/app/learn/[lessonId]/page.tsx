"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, ArrowRight, CheckCircle2, RotateCcw, Sparkles, Target, Trophy, XCircle } from "lucide-react";
import { useSignRecognition } from "@/hooks/useSignRecognition";
import { LandmarkOverlay } from "@/components/LandmarkOverlay";
import { PermissionGate } from "@/components/PermissionGate";
import { PageShell } from "@/components/primitives/PageShell";
import { ConfidenceMeter } from "@/components/primitives/ConfidenceMeter";
import { StatusPill, type Status } from "@/components/primitives/StatusPill";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert } from "@/components/ui/alert";
import { getLesson, getNextLesson, getUnitForLesson } from "@/data/curriculum";
import { completeLesson, recordAttempt } from "@/lib/progress";
import { cn } from "@/lib/utils";

const VIDEO_W = 640;
const VIDEO_H = 480;

type CamStatus = "pending" | "ok" | "denied";

export default function LessonPage() {
  const { lessonId } = useParams<{ lessonId: string }>();
  const router = useRouter();
  const lesson = getLesson(lessonId);
  const unit = getUnitForLesson(lessonId);
  const next = getNextLesson(lessonId);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [camStatus, setCamStatus] = useState<CamStatus>("pending");
  const [cardIdx, setCardIdx] = useState(0);
  const [attempts, setAttempts] = useState<Record<string, "hit" | "miss" | undefined>>({});
  const [done, setDone] = useState(false);

  const {
    wordPrediction,
    captureStatus,
    captureProgress,
    landmarkerResult,
    reset,
    paused,
  } = useSignRecognition(videoRef, null);

  const target = lesson?.signs[cardIdx];

  useEffect(() => {
    if (!target || !wordPrediction || wordPrediction.error) return;
    const best = wordPrediction.top3?.[0];
    if (!best) return;
    const hit = best.label === target;
    setAttempts((a) => ({ ...a, [target]: hit ? "hit" : "miss" }));
    recordAttempt(best.label, hit, best.confidence);
    if (hit) {
      try { navigator.vibrate?.(25); } catch {}
    }
  }, [wordPrediction, target]);

  const score = useMemo(() => {
    if (!lesson) return 0;
    const hits = lesson.signs.filter((s) => attempts[s] === "hit").length;
    return Math.round((hits / lesson.signs.length) * 100);
  }, [attempts, lesson]);

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
  }

  if (!lesson) {
    return (
      <PageShell>
        <div className="py-16">
          <Alert tone="warning" title="Lesson not found">
            We couldn&apos;t find that lesson.
            <div className="mt-3">
              <Button asChild size="sm" variant="secondary">
                <Link href="/learn"><ArrowLeft className="size-4" aria-hidden /> Back to lessons</Link>
              </Button>
            </div>
          </Alert>
        </div>
      </PageShell>
    );
  }

  if (camStatus === "pending") {
    return (
      <PageShell>
        <div className="pt-10">
          <LessonHeader unitTitle={unit?.title ?? ""} lessonTitle={lesson.title} />
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

  if (done) {
    return (
      <PageShell>
        <div className="grid place-items-center py-16">
          <Card className="w-full max-w-xl p-8 text-center">
            <Trophy className="mx-auto size-14 text-[var(--color-warning)]" aria-hidden />
            <h1 className="mt-4 heading-h1 text-[var(--color-text)]">Lesson complete</h1>
            <p className="mt-2 text-[var(--color-text-muted)]">
              You scored <strong>{score}%</strong> on <em>{lesson.title}</em>.
            </p>
            <div className="mt-6 flex flex-wrap justify-center gap-3">
              <Button asChild variant="secondary">
                <Link href="/learn"><ArrowLeft className="size-4" aria-hidden /> All lessons</Link>
              </Button>
              {next ? (
                <Button asChild>
                  <Link href={`/learn/${next.id}`}>
                    Continue to {next.title} <ArrowRight className="size-4" aria-hidden />
                  </Link>
                </Button>
              ) : (
                <Button asChild>
                  <Link href="/practice">
                    <Sparkles className="size-4" aria-hidden /> Keep practising
                  </Link>
                </Button>
              )}
            </div>
          </Card>
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

  const currentOutcome = target ? attempts[target] : undefined;
  const lastConfidence = wordPrediction?.top3?.[0]?.confidence ?? 0;

  function finishCard() {
    if (cardIdx + 1 < lesson!.signs.length) {
      setCardIdx((i) => i + 1);
      reset();
    } else {
      completeLesson(lesson!.id, score);
      setDone(true);
    }
  }

  return (
    <PageShell>
      <div className="pt-10 pb-12">
        <LessonHeader unitTitle={unit?.title ?? ""} lessonTitle={lesson.title} />

        <div className="mt-5 flex items-center gap-3">
          <Progress
            value={Math.round(((cardIdx + 1) / lesson.signs.length) * 100)}
            tone="brand"
            aria-label={`Lesson progress: card ${cardIdx + 1} of ${lesson.signs.length}`}
            className="h-2"
          />
          <span className="shrink-0 font-mono text-sm tabular-nums text-[var(--color-text-muted)]">
            {cardIdx + 1} / {lesson.signs.length}
          </span>
        </div>

        <div className="mt-8 grid gap-5 lg:grid-cols-[1.4fr_1fr]">
          <Card className="overflow-hidden">
            <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
              <Badge tone="brand">
                <Target className="size-3.5" aria-hidden />
                Sign &ldquo;<strong className="font-semibold">{target}</strong>&rdquo;
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

              {currentOutcome && (
                <div
                  role="status"
                  aria-live="assertive"
                  className="pointer-events-none absolute inset-0 grid place-items-center bg-black/40 text-white sl-pop-in"
                >
                  {currentOutcome === "hit" ? (
                    <div className="text-center">
                      <CheckCircle2 className="mx-auto size-16 text-[var(--color-success)]" aria-hidden />
                      <p className="mt-2 heading-display">Got it</p>
                    </div>
                  ) : (
                    <div className="text-center">
                      <XCircle className="mx-auto size-16 text-[var(--color-warning)]" aria-hidden />
                      <p className="mt-2 heading-display">Try once more</p>
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
              <Button variant="ghost" onClick={reset}>
                <RotateCcw aria-hidden /> Retry sign
              </Button>
              <div className="flex gap-2">
                {cardIdx > 0 && (
                  <Button variant="secondary" onClick={() => setCardIdx((i) => i - 1)}>
                    <ArrowLeft aria-hidden /> Previous
                  </Button>
                )}
                <Button onClick={finishCard}>
                  {cardIdx + 1 < lesson.signs.length ? "Next sign" : "Finish lesson"}
                  <ArrowRight className="size-4" aria-hidden />
                </Button>
              </div>
            </div>
          </Card>

          {/* Right rail */}
          <div className="flex flex-col gap-4">
            <Card className="p-5">
              <p className="eyebrow">Instructions</p>
              <p className="mt-3 text-lg text-[var(--color-text)]">
                Hold the sign for <strong>{target}</strong> steadily in front of the camera until
                the progress bar fills.
              </p>
              <p className="mt-2 text-sm text-[var(--color-text-muted)]">
                Make sure your hand is fully visible and lighting is even. The recogniser reads
                21 landmark points per hand.
              </p>
            </Card>

            <Card className="p-5">
              <p className="eyebrow">Latest reading</p>
              {wordPrediction?.top3?.[0] ? (
                <>
                  <p className="mt-2 heading-h2 text-[var(--color-text)]">
                    {wordPrediction.top3[0].label}
                  </p>
                  <ConfidenceMeter value={lastConfidence} className="mt-3" />
                </>
              ) : (
                <p className="mt-2 text-[var(--color-text-faint)]">Sign to see a reading.</p>
              )}
            </Card>

            <Card className="p-5">
              <p className="eyebrow">Lesson progress</p>
              <ul className="mt-3 flex flex-wrap gap-1.5">
                {lesson.signs.map((s, i) => (
                  <li
                    key={s}
                    className={cn(
                      "inline-flex h-8 min-w-8 items-center justify-center rounded-full px-2 text-xs font-mono font-semibold",
                      i === cardIdx
                        ? "bg-[var(--color-brand)] text-[var(--color-brand-foreground)]"
                        : attempts[s] === "hit"
                          ? "bg-[var(--color-success-subtle)] text-[var(--color-success)]"
                          : attempts[s] === "miss"
                            ? "bg-[var(--color-warning-subtle)] text-[var(--color-warning)]"
                            : "bg-[var(--color-surface-sunken)] text-[var(--color-text-muted)]",
                    )}
                    aria-label={`${s}: ${attempts[s] ?? "not attempted"}`}
                  >
                    {s}
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        </div>
      </div>
    </PageShell>
  );
}

function LessonHeader({ unitTitle, lessonTitle }: { unitTitle: string; lessonTitle: string }) {
  return (
    <div>
      <Link
        href="/learn"
        className="inline-flex items-center gap-1.5 text-sm font-semibold text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:no-underline"
      >
        <ArrowLeft className="size-4" aria-hidden /> All lessons
      </Link>
      <p className="mt-4 eyebrow">{unitTitle}</p>
      <h1 className="mt-1 heading-h1 text-[var(--color-text)]">{lessonTitle}</h1>
    </div>
  );
}
