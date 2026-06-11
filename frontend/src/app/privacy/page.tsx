import type { Metadata } from "next";
import Link from "next/link";
import { Camera, Database, Mic, ShieldCheck } from "lucide-react";
import { PageShell } from "@/components/primitives/PageShell";
import { SectionHeader } from "@/components/primitives/SectionHeader";
import { SiteFooter } from "@/components/primitives/Footer";
import { Card } from "@/components/ui/card";

export const metadata: Metadata = {
  title: "Privacy",
  description: "How SignLearn handles your camera, microphone, and data.",
};

const POINTS = [
  {
    icon: Camera,
    title: "Camera",
    body: "Your video is processed entirely in this browser using MediaPipe Hands. The raw video stream never leaves your device. We extract 21 landmark coordinates per hand (126 numbers per frame) and send only those to our recogniser.",
  },
  {
    icon: Mic,
    title: "Microphone",
    body: "Speech-to-text runs in your browser via the Web Speech API. Only the resulting text caption is sent to your conversation partner. Audio is never recorded or transmitted by SignLearn.",
  },
  {
    icon: Database,
    title: "Transcripts",
    body: "Captions sent between participants are stored on the SignLearn server for the duration of the room and a short audit window. You can export and delete transcripts from the room screen.",
  },
  {
    icon: ShieldCheck,
    title: "Your account",
    body: "There isn't one. Your preferences, lesson progress, and onboarding state live in your browser's localStorage. Visit /account to export or erase that data.",
  },
];

export default function PrivacyPage() {
  return (
    <PageShell>
      <div className="pt-10 pb-12">
        <SectionHeader
          eyebrow="Privacy"
          title="What stays on your device, and what doesn't."
          description="SignLearn is an open-source research project. Every claim below can be verified in the code."
          as="h1"
        />

        <div className="mt-10 grid gap-5 sm:grid-cols-2">
          {POINTS.map(({ icon: Icon, title, body }) => (
            <Card key={title} className="p-6">
              <div
                className="flex h-11 w-11 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]"
                aria-hidden
              >
                <Icon className="size-5" />
              </div>
              <h2 className="mt-4 heading-h3 text-[var(--color-text)]">{title}</h2>
              <p className="mt-2 text-sm text-[var(--color-text-muted)] leading-relaxed">{body}</p>
            </Card>
          ))}
        </div>

        <p className="mt-10 text-sm text-[var(--color-text-muted)]">
          Questions? Send them via{" "}
          <Link href="/research" className="font-semibold">
            Research & feedback
          </Link>
          .
        </p>
      </div>
      <SiteFooter />
    </PageShell>
  );
}
