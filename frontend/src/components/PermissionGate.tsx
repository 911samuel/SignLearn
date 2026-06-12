"use client";

import { useState } from "react";
import { Camera, CheckCircle2, Mic, ShieldCheck } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/alert";

interface PermissionGateProps {
  kind: "camera" | "microphone";
  onAllow: () => void | Promise<void>;
  secondaryAction?: { label: string; onClick: () => void };
}

export function PermissionGate({ kind, onAllow, secondaryAction }: PermissionGateProps) {
  const [requesting, setRequesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isCamera = kind === "camera";
  const Icon = isCamera ? Camera : Mic;

  async function handleAllow() {
    setError(null);
    setRequesting(true);
    try {
      await onAllow();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Permission was denied or unavailable.");
    } finally {
      setRequesting(false);
    }
  }

  const steps = isCamera
    ? [
        ["We turn on your camera", "only in this tab."],
        ["We track 21 hand landmarks per hand", "126 numbers per frame."],
        ["Your raw video stays on your device.", "Only landmark numbers go to our server."],
      ]
    : [
        ["We turn on your microphone", "only in this tab."],
        ["Your browser transcribes speech to text", "using the Web Speech API."],
        ["Only the text caption is sent to your partner.", "Audio is not stored."],
      ];

  return (
    <div className="grid place-items-center px-4 py-8">
      <Card className="w-full max-w-xl p-8">
        <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-[var(--radius-lg)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]">
          <Icon className="size-7" aria-hidden />
        </div>
        <h2 className="heading-h2 text-[var(--color-text)]">
          {isCamera
            ? "Turn on your camera to start signing"
            : "Turn on your microphone to start speaking"}
        </h2>
        <p className="mt-3 text-[var(--color-text-muted)] leading-relaxed">
          {isCamera
            ? "We use your camera locally to read your hand position. Your video never leaves this browser."
            : "We use your microphone to transcribe what you say so the signer can read it. Audio is never stored."}
        </p>

        <ul className="mt-6 space-y-3">
          {steps.map(([bold, rest]) => (
            <li key={bold} className="flex items-start gap-3">
              <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-[var(--color-success)]" aria-hidden />
              <p className="text-[var(--color-text)]">
                <strong>{bold}</strong>{" "}
                <span className="text-[var(--color-text-muted)]">— {rest}</span>
              </p>
            </li>
          ))}
        </ul>

        <div className="mt-7 flex flex-col gap-2 sm:flex-row">
          <Button size="lg" onClick={handleAllow} disabled={requesting} className="sm:flex-1">
            {requesting
              ? "Requesting…"
              : isCamera
                ? "Allow camera & start"
                : "Allow microphone & start"}
          </Button>
          {secondaryAction && (
            <Button variant="secondary" size="lg" onClick={secondaryAction.onClick}>
              {secondaryAction.label}
            </Button>
          )}
        </div>

        {error && (
          <Alert tone="danger" className="mt-4" title="Permission denied">
            {error} You can re-enable it from your browser&apos;s address bar.
          </Alert>
        )}

        <p className="mt-5 inline-flex items-center justify-center gap-1.5 text-center text-xs text-[var(--color-text-muted)] w-full">
          <ShieldCheck className="size-3.5" aria-hidden />
          SignLearn is open source — every claim above can be verified in the code.
        </p>
      </Card>
    </div>
  );
}
