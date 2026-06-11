import type { Metadata } from "next";
import Link from "next/link";
import {
  Accessibility,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Cpu,
  GraduationCap,
  HandHelping,
  Hand,
  Lock,
  MessageSquareText,
  Mic,
  Quote,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Users,
  Wifi,
} from "lucide-react";
import { PageShell } from "@/components/primitives/PageShell";
import { SectionHeader } from "@/components/primitives/SectionHeader";
import { SiteFooter } from "@/components/primitives/Footer";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { LandingCTA } from "./LandingCTA";

export const metadata: Metadata = {
  title: "SignLearn — Real-time ASL ↔ English in your browser",
  alternates: { canonical: "https://signlearn.app" },
};

const AUDIENCES = [
  {
    icon: HandHelping,
    title: "Deaf & hard-of-hearing",
    body: "Captions, transcripts, and TTS at the speed of conversation — designed with Deaf community contributors.",
  },
  {
    icon: GraduationCap,
    title: "Sign language learners",
    body: "Practice with live feedback, structured lessons, and a curriculum that matches how ASL is actually used.",
  },
  {
    icon: BookOpen,
    title: "Educators & interpreters",
    body: "A neutral, free tool for classroom demonstration, training, and assessment — no licensing, no lock-in.",
  },
  {
    icon: Users,
    title: "Researchers & NGOs",
    body: "Open data formats, exportable transcripts, on-device privacy. Deploy in studies, clinics, and programmes.",
  },
] as const;

const STEPS = [
  {
    icon: ScanLine,
    title: "Your browser sees your hands",
    body: "MediaPipe Hands extracts 21 landmark points per hand, 30 times a second — entirely in the browser. Your video never uploads.",
  },
  {
    icon: Cpu,
    title: "Our model reads the motion",
    body: "A TCN sequence model trained on 36 ASL classes recognises signs at 5.6 ms p95 latency — about 196 frames per second on a laptop CPU.",
  },
  {
    icon: MessageSquareText,
    title: "Everyone follows along",
    body: "Predicted signs become captions and speech in real time. Hearing partners speak back via push-to-talk. The whole conversation is logged for review.",
  },
] as const;

const TRUST = [
  { icon: ShieldCheck, label: "On-device hand detection" },
  { icon: Wifi,        label: "Works in any modern browser" },
  { icon: Accessibility, label: "WCAG 2.2 AA, keyboard-first" },
  { icon: Sparkles,    label: "Open source · MIT" },
] as const;

export default function LandingPage() {
  return (
    <PageShell noMaxWidth noPadding hideBottomNav>
      {/* HERO */}
      <section className="relative overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(1200px_600px_at_80%_-10%,color-mix(in_srgb,var(--color-brand)_22%,transparent),transparent_60%)]"
        />
        <div className="relative mx-auto grid max-w-7xl gap-12 px-[var(--spacing-page-x)] pb-16 pt-12 lg:grid-cols-[1.1fr_0.9fr] lg:gap-16 lg:px-[var(--spacing-page-x-lg)] lg:pb-24 lg:pt-20">
          <div>
            <Badge tone="brand" className="mb-6">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-[var(--color-brand)]" aria-hidden />
              Accessibility · Research · Open source
            </Badge>

            <h1 className="heading-display text-[var(--color-text)]">
              Sign and speak.
              <br />
              <span className="text-[var(--color-brand)]">In the same window.</span>
            </h1>

            <p className="mt-6 max-w-xl text-lg text-[var(--color-text-muted)] leading-relaxed">
              SignLearn is a research project that recognises American Sign Language live in the
              browser and turns it into captions and speech — so Deaf, hard-of-hearing, and
              hearing people can have a real conversation without an interpreter in the room.
            </p>

            <div className="mt-8">
              <LandingCTA />
            </div>

            <ul className="mt-10 flex flex-wrap gap-x-5 gap-y-3 text-sm text-[var(--color-text-muted)]" aria-label="Highlights">
              {TRUST.map(({ icon: Icon, label }) => (
                <li key={label} className="inline-flex items-center gap-2">
                  <Icon className="size-4 text-[var(--color-brand)]" aria-hidden />
                  {label}
                </li>
              ))}
            </ul>
          </div>

          <HeroVisual />
        </div>
      </section>

      {/* AUDIENCES */}
      <section className="border-y border-[var(--color-border)] bg-[var(--color-surface-sunken)] py-[var(--spacing-section)]">
        <div className="mx-auto max-w-7xl px-[var(--spacing-page-x)] lg:px-[var(--spacing-page-x-lg)]">
          <SectionHeader
            eyebrow="Who it's for"
            title="Built with the communities it serves"
            description="SignLearn is shaped by ongoing input from Deaf signers, learners, educators, and accessibility researchers."
          />
          <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {AUDIENCES.map(({ icon: Icon, title, body }) => (
              <Card key={title} className="p-6">
                <div className="flex h-11 w-11 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]" aria-hidden>
                  <Icon className="size-5" />
                </div>
                <h3 className="mt-4 heading-h3 text-[var(--color-text)]">{title}</h3>
                <p className="mt-2 text-sm text-[var(--color-text-muted)] leading-relaxed">
                  {body}
                </p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="py-[var(--spacing-section)]">
        <div className="mx-auto max-w-7xl px-[var(--spacing-page-x)] lg:px-[var(--spacing-page-x-lg)]">
          <SectionHeader eyebrow="How it works" title="Three steps, one tab" />
          <ol className="mt-12 grid gap-6 md:grid-cols-3">
            {STEPS.map((s, i) => (
              <li key={s.title} className="relative">
                <Card className="h-full p-6">
                  <p className="eyebrow">Step {i + 1}</p>
                  <div className="mt-3 flex h-12 w-12 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand)] text-[var(--color-brand-foreground)] shadow-[var(--shadow-sm)]" aria-hidden>
                    <s.icon className="size-6" />
                  </div>
                  <h3 className="mt-4 heading-h3 text-[var(--color-text)]">{s.title}</h3>
                  <p className="mt-2 text-[var(--color-text-muted)] leading-relaxed">{s.body}</p>
                </Card>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* WHY IT MATTERS — research framing */}
      <section className="border-y border-[var(--color-border)] bg-[var(--color-surface-sunken)] py-[var(--spacing-section)]">
        <div className="mx-auto grid max-w-7xl gap-12 px-[var(--spacing-page-x)] lg:grid-cols-[1fr_1fr] lg:px-[var(--spacing-page-x-lg)]">
          <div>
            <SectionHeader
              eyebrow="Why this matters"
              title="A free, immediate, privacy-respecting bridge"
              description="Roughly 70 million people worldwide use a sign language as a primary mode of communication. Professional interpreters are scarce, expensive, and rarely available on demand."
            />
            <p className="mt-6 text-[var(--color-text-muted)] leading-relaxed">
              SignLearn explores whether on-device machine learning can offer a free, immediate,
              privacy-respecting bridge — and whether learners can use the same engine to build fluency.
            </p>
          </div>

          <Card className="p-8">
            <p className="eyebrow">Research objectives</p>
            <h3 className="mt-2 heading-h2 text-[var(--color-text)]">What we're studying</h3>
            <ul className="mt-6 space-y-4">
              {[
                "Quantify recognition accuracy in real-world deployment, across signers and lighting.",
                "Surface failure modes through structured community feedback — including pairs of signs that are linguistically identical.",
                "Measure whether closed-loop practice with a recogniser improves learner outcomes.",
                "Publish open benchmarks and exportable transcripts to support replication.",
              ].map((line) => (
                <li key={line} className="flex items-start gap-3">
                  <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-[var(--color-success)]" aria-hidden />
                  <span className="text-[var(--color-text)]">{line}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </section>

      {/* DEPLOYMENT — quote + adopter strip */}
      <section className="py-[var(--spacing-section)]">
        <div className="mx-auto max-w-7xl px-[var(--spacing-page-x)] lg:px-[var(--spacing-page-x-lg)]">
          <Card className="overflow-hidden p-8 md:p-12">
            <Quote className="size-8 text-[var(--color-brand)]" aria-hidden />
            <p className="mt-4 heading-h2 text-[var(--color-text)] max-w-3xl">
              &ldquo;The tools that work for Deaf people are the tools the Deaf community helped
              build. SignLearn's open process and on-device defaults set the right baseline.&rdquo;
            </p>
            <p className="mt-4 text-sm text-[var(--color-text-muted)]">
              — Quotation illustrative. Real feedback from study participants will replace this when available.
            </p>

            <div className="mt-10 flex flex-wrap items-center gap-x-10 gap-y-6">
              <p className="eyebrow">Designed for deployment in</p>
              {["Universities", "NGOs", "Accessibility offices", "Deaf community orgs"].map((x) => (
                <span key={x} className="text-sm font-semibold text-[var(--color-text-muted)]">
                  {x}
                </span>
              ))}
            </div>
          </Card>
        </div>
      </section>

      {/* SECOND CTA */}
      <section className="pb-[var(--spacing-section)]">
        <div className="mx-auto max-w-3xl px-[var(--spacing-page-x)] text-center lg:px-[var(--spacing-page-x-lg)]">
          <h2 className="heading-h1 text-[var(--color-text)]">Try it now</h2>
          <p className="mt-4 text-lg text-[var(--color-text-muted)] leading-relaxed">
            Open a room, hand someone a code, and start signing. Or jump into a 5-minute lesson and
            see if the recogniser can spot your handshapes.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Button asChild size="lg">
              <Link href="/learn">
                <BookOpen aria-hidden />
                Try a lesson
                <ArrowRight className="ml-1 size-4" aria-hidden />
              </Link>
            </Button>
            <Button asChild size="lg" variant="secondary">
              <Link href="/practice">
                <Hand aria-hidden />
                Open practice
              </Link>
            </Button>
          </div>
        </div>
      </section>

      <SiteFooter />
    </PageShell>
  );
}

function HeroVisual() {
  return (
    <Card className="relative overflow-hidden border-[var(--color-border)] p-6" aria-hidden>
      <div className="grid gap-4 md:grid-cols-[1fr_1fr]">
        <div className="aspect-video rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-sunken)] p-4">
          <div className="flex items-center justify-between text-[10px] text-[var(--color-text-muted)]">
            <span className="inline-flex items-center gap-1.5 font-semibold">
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-success)] sl-pulse-soft" /> SIGNER · LIVE
            </span>
            <span className="font-mono">5.6 ms p95</span>
          </div>
          <div className="mt-6 grid place-items-center">
            <svg viewBox="0 0 120 120" className="size-28 text-[var(--color-brand)]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="32" cy="62" r="3" fill="currentColor" />
              <circle cx="46" cy="44" r="3" fill="currentColor" />
              <circle cx="58" cy="32" r="3" fill="currentColor" />
              <circle cx="72" cy="44" r="3" fill="currentColor" />
              <circle cx="86" cy="62" r="3" fill="currentColor" />
              <circle cx="58" cy="86" r="3" fill="currentColor" />
              <path d="M32 62 L58 32 L86 62 L58 86 Z" opacity="0.4" />
              <path d="M46 44 L58 32 L72 44" />
            </svg>
          </div>
        </div>
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-sunken)] p-4 text-[0.78rem]">
          <p className="eyebrow">Live captions</p>
          <ul className="mt-3 space-y-2.5">
            <li className="rounded-[var(--radius-sm)] bg-[var(--color-brand-subtle)] px-2.5 py-1.5">
              <span className="font-semibold text-[var(--color-brand-subtle-foreground)]">Hello</span>
              <span className="ml-2 text-[var(--color-text-muted)]">94%</span>
            </li>
            <li className="rounded-[var(--radius-sm)] bg-[var(--color-surface)] px-2.5 py-1.5 border border-[var(--color-border)]">
              <Mic className="inline size-3 text-[var(--color-text-muted)] mr-1.5" />
              <span className="text-[var(--color-text)]">Nice to meet you too.</span>
            </li>
            <li className="rounded-[var(--radius-sm)] bg-[var(--color-brand-subtle)] px-2.5 py-1.5">
              <span className="font-semibold text-[var(--color-brand-subtle-foreground)]">Thank you</span>
              <span className="ml-2 text-[var(--color-text-muted)]">91%</span>
            </li>
          </ul>
        </div>
      </div>
    </Card>
  );
}
