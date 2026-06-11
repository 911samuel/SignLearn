"use client";

import { useEffect, useState } from "react";
import { Accessibility, Camera, KeyRound, Sparkles } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

const KEY = "signlearn.onboarded.v1";

const STEPS = [
  {
    icon: Sparkles,
    title: "Welcome to SignLearn",
    body: "A research project that recognises American Sign Language live in the browser — so Deaf and hearing people can have a real conversation without an interpreter in the room.",
  },
  {
    icon: Camera,
    title: "Your video stays on your device",
    body: "We use MediaPipe Hands in your browser to extract 21 landmark points per hand. Only those numbers are sent to our recogniser — never your video.",
  },
  {
    icon: Accessibility,
    title: "Make it work for you",
    body: "Open the accessibility menu (top right) to switch to high-contrast mode, enlarge text, reduce motion, or change how push-to-talk works.",
  },
  {
    icon: KeyRound,
    title: "Shortcuts that help",
    body: "Press Space to push-to-talk in the hearing view. Press Esc to close any dialog. Tab moves focus; the skip-nav link jumps to main content.",
  },
] as const;

export function OnboardingTour() {
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    try {
      if (!window.localStorage.getItem(KEY)) setOpen(true);
    } catch {}
  }, []);

  function finish() {
    try {
      window.localStorage.setItem(KEY, "1");
    } catch {}
    setOpen(false);
  }

  const current = STEPS[step];
  const Icon = current.icon;
  const isLast = step === STEPS.length - 1;

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? setOpen(true) : finish())}>
      <DialogContent>
        <DialogHeader>
          <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]">
            <Icon className="size-6" aria-hidden />
          </div>
          <DialogTitle>{current.title}</DialogTitle>
          <DialogDescription>{current.body}</DialogDescription>
        </DialogHeader>

        <div className="mt-5 flex items-center gap-1.5" aria-hidden>
          {STEPS.map((_, i) => (
            <span
              key={i}
              className={
                i === step
                  ? "h-1.5 w-6 rounded-full bg-[var(--color-brand)] transition-all"
                  : "h-1.5 w-1.5 rounded-full bg-[var(--color-border-strong)] transition-all"
              }
            />
          ))}
        </div>

        <DialogFooter>
          {step > 0 && (
            <Button variant="ghost" onClick={() => setStep(step - 1)}>Back</Button>
          )}
          <Button variant="ghost" onClick={finish}>Skip</Button>
          <Button onClick={() => (isLast ? finish() : setStep(step + 1))}>
            {isLast ? "Get started" : `Next (${step + 1}/${STEPS.length})`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
