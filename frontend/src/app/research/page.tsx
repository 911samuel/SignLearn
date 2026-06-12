"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Accessibility, AlertTriangle, Bug, ClipboardList, Download, FlaskConical, Lightbulb, MessageSquareText, Send, ShieldCheck, Sparkles } from "lucide-react";
import { PageShell } from "@/components/primitives/PageShell";
import { SectionHeader } from "@/components/primitives/SectionHeader";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Input, Textarea } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/components/ui/toast";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useProgress } from "@/lib/progress";

export default function ResearchPage() {
  return (
    <PageShell>
      <Suspense fallback={null}>
        <ResearchInner />
      </Suspense>
    </PageShell>
  );
}

function ResearchInner() {
  const params = useSearchParams();
  const initialTab = (params?.get("tab") as "feedback" | "correction" | "study" | null) ?? "feedback";

  return (
    <>
      <div className="pt-10">
        <SectionHeader
          eyebrow="Research & feedback"
          title="Help us make this work better — for you and for everyone."
          description="Your input shapes the curriculum, the recogniser, and the accessibility of the whole platform. Every submission is logged and read."
          as="h1"
        />
      </div>

      <Tabs defaultValue={initialTab} className="mt-8 pb-16">
        <TabsList className="flex-wrap">
          <TabsTrigger value="feedback">
            <MessageSquareText className="size-4" aria-hidden />
            Send feedback
          </TabsTrigger>
          <TabsTrigger value="correction">
            <AlertTriangle className="size-4" aria-hidden />
            Report a wrong prediction
          </TabsTrigger>
          <TabsTrigger value="study">
            <FlaskConical className="size-4" aria-hidden />
            Participate in research
          </TabsTrigger>
        </TabsList>

        <TabsContent value="feedback">
          <FeedbackForm />
        </TabsContent>
        <TabsContent value="correction">
          <CorrectionForm />
        </TabsContent>
        <TabsContent value="study">
          <StudyForm />
        </TabsContent>
      </Tabs>
    </>
  );
}

const CATEGORIES = [
  { value: "bug", label: "Something broke", icon: Bug },
  { value: "praise", label: "This helped me", icon: Sparkles },
  { value: "idea", label: "Feature idea", icon: Lightbulb },
  { value: "accessibility", label: "Accessibility issue", icon: Accessibility },
] as const;

function FeedbackForm() {
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]["value"]>("bug");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const { toast } = useToast();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    try {
      await api.feedback({ category, message: text.trim() });
      toast({ tone: "success", title: "Thanks for the feedback", description: "We read every message." });
      setText("");
    } catch {
      toast({ tone: "danger", title: "Couldn't send", description: "Check your connection." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="p-6 md:p-8">
      <form onSubmit={submit} className="space-y-6">
        <fieldset>
          <legend className="text-sm font-semibold text-[var(--color-text)] mb-3">Category</legend>
          <div className="flex flex-wrap gap-2">
            {CATEGORIES.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                type="button"
                onClick={() => setCategory(value)}
                aria-pressed={category === value}
                className={cn(
                  "inline-flex h-10 items-center gap-2 rounded-full border px-3 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]",
                  category === value
                    ? "border-transparent bg-[var(--color-brand)] text-[var(--color-brand-foreground)]"
                    : "border-[var(--color-border-strong)] bg-[var(--color-surface)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]",
                )}
              >
                <Icon className="size-4" aria-hidden /> {label}
              </button>
            ))}
          </div>
        </fieldset>

        <div className="space-y-1.5">
          <Label htmlFor="feedback-text">What&apos;s on your mind?</Label>
          <Textarea
            id="feedback-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={7}
            placeholder="Tell us what happened, what you expected, or what would help…"
            required
          />
          <p className="text-xs text-[var(--color-text-muted)]">
            We never share your message publicly. If you include a transcript, please redact any
            sensitive details first.
          </p>
        </div>

        <div className="flex items-center justify-end gap-3">
          <Button type="submit" disabled={busy || !text.trim()} size="lg">
            <Send className="size-4" aria-hidden /> {busy ? "Sending…" : "Send feedback"}
          </Button>
        </div>
      </form>
    </Card>
  );
}

function CorrectionForm() {
  const progress = useProgress();
  const [predicted, setPredicted] = useState("");
  const [actual, setActual] = useState("");
  const [busy, setBusy] = useState(false);
  const { toast } = useToast();
  const recent = progress.attempts.slice(-10).reverse();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!predicted.trim() || !actual.trim()) return;
    setBusy(true);
    try {
      await api.correction({ predicted: predicted.trim(), actual: actual.trim() });
      toast({ tone: "success", title: "Correction logged", description: "Thanks — this helps the recogniser improve." });
      setPredicted("");
      setActual("");
    } catch {
      toast({ tone: "danger", title: "Couldn't send", description: "Try again in a moment." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
      <Card className="p-6 md:p-8">
        <form onSubmit={submit} className="space-y-5">
          <p className="text-sm text-[var(--color-text-muted)]">
            Saw a sign get recognised as the wrong word? Tell us what the recogniser said and what
            you actually signed. Reports go to our model improvement queue.
          </p>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="predicted">What the recogniser said</Label>
              <Input id="predicted" value={predicted} onChange={(e) => setPredicted(e.target.value)} placeholder="e.g. two" required />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="actual">What you actually signed</Label>
              <Input id="actual" value={actual} onChange={(e) => setActual(e.target.value)} placeholder="e.g. v" required />
            </div>
          </div>

          <Alert tone="info" title="Heads up about pairs">
            Some ASL handshapes are linguistically identical (<code>two</code>/<code>v</code>,{" "}
            <code>six</code>/<code>w</code>, <code>zero</code>/<code>o</code>). These reports help
            us decide when to treat them as semantically equivalent.
          </Alert>

          <div className="flex justify-end">
            <Button type="submit" disabled={busy || !predicted.trim() || !actual.trim()}>
              <Send className="size-4" aria-hidden /> Submit correction
            </Button>
          </div>
        </form>
      </Card>

      <Card className="p-6">
        <p className="eyebrow">Recent attempts on this device</p>
        {recent.length === 0 ? (
          <p className="mt-3 text-sm text-[var(--color-text-faint)]">
            Use /practice or a lesson first — then come here to flag any wrong predictions.
          </p>
        ) : (
          <ul className="mt-3 space-y-2">
            {recent.map((a, i) => (
              <li
                key={i}
                className="flex flex-wrap items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-sunken)] p-2.5"
              >
                <Badge tone={a.correct ? "success" : "warning"}>
                  {a.correct ? "Correct" : "Wrong"}
                </Badge>
                <code className="font-mono text-sm text-[var(--color-text)]">{a.sign}</code>
                <span className="ml-auto text-xs text-[var(--color-text-muted)]">
                  {Math.round(a.confidence * 100)}%
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setPredicted(a.sign);
                    setActual("");
                  }}
                >
                  Pre-fill
                </Button>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function StudyForm() {
  const [consent, setConsent] = useState(false);
  const [signerId, setSignerId] = useState("");
  const [proficiency, setProficiency] = useState<"beginner" | "intermediate" | "fluent" | "native">("beginner");
  const [submitted, setSubmitted] = useState(false);
  const { toast } = useToast();

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!consent) return;
    // Mock: research enrollment endpoint doesn't exist yet, persist locally.
    try {
      window.localStorage.setItem(
        "signlearn.research.v1",
        JSON.stringify({ signerId, proficiency, enrolledAt: Date.now() }),
      );
    } catch {}
    setSubmitted(true);
    toast({ tone: "success", title: "Enrolled in study", description: "Thank you. We'll be in touch about next steps." });
  }

  if (submitted) {
    return (
      <Card className="p-8 text-center">
        <Sparkles className="mx-auto size-12 text-[var(--color-brand)]" aria-hidden />
        <h3 className="mt-3 heading-h2 text-[var(--color-text)]">You&apos;re in. Thank you.</h3>
        <p className="mt-2 text-[var(--color-text-muted)]">
          Your consent and profile have been recorded on this device. The research team will be in
          touch with next steps when the current cohort opens.
        </p>
      </Card>
    );
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_1fr]">
      <Card className="p-6 md:p-8">
        <p className="eyebrow">About this study</p>
        <h3 className="mt-1 heading-h2 text-[var(--color-text)]">Recogniser accuracy in real deployment</h3>
        <p className="mt-3 text-[var(--color-text-muted)] leading-relaxed">
          We&apos;re measuring how well our recogniser performs across signers, lighting conditions, and
          camera angles outside the lab. Participants help us identify failure modes that are
          invisible from public benchmarks alone.
        </p>
        <ul className="mt-5 space-y-3 text-sm">
          {[
            "Anonymous signer ID — never linked to your name or email.",
            "Optional questionnaire after each session (≤ 2 minutes).",
            "All data stays on-device unless you explicitly export it.",
            "Withdraw any time from the Account page.",
          ].map((line) => (
            <li key={line} className="flex items-start gap-2.5">
              <ShieldCheck className="mt-0.5 size-4 shrink-0 text-[var(--color-success)]" aria-hidden />
              <span className="text-[var(--color-text)]">{line}</span>
            </li>
          ))}
        </ul>
        <Badge tone="info" className="mt-5">
          <ClipboardList className="size-3.5" aria-hidden />
          Preview — backend enrolment endpoint not yet wired
        </Badge>
      </Card>

      <Card className="p-6 md:p-8">
        <form onSubmit={submit} className="space-y-5">
          <div className="space-y-1.5">
            <Label htmlFor="signer-id">Anonymous signer ID</Label>
            <Input
              id="signer-id"
              value={signerId}
              onChange={(e) => setSignerId(e.target.value)}
              placeholder="Choose any string — e.g. owl42"
              maxLength={32}
              required
            />
            <p className="text-xs text-[var(--color-text-muted)]">
              Used to tie your sessions together without identifying you.
            </p>
          </div>

          <fieldset>
            <legend className="text-sm font-semibold text-[var(--color-text)] mb-2">
              ASL proficiency
            </legend>
            <div className="grid grid-cols-2 gap-2">
              {(["beginner", "intermediate", "fluent", "native"] as const).map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setProficiency(p)}
                  aria-pressed={proficiency === p}
                  className={cn(
                    "rounded-[var(--radius-md)] border px-3 py-2.5 text-left text-sm font-semibold capitalize transition focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]",
                    proficiency === p
                      ? "border-[var(--color-brand)] bg-[var(--color-brand-subtle)] text-[var(--color-text)]"
                      : "border-[var(--color-border-strong)] bg-[var(--color-surface)] text-[var(--color-text-muted)]",
                  )}
                >
                  {p}
                </button>
              ))}
            </div>
          </fieldset>

          <label className="flex items-start gap-3 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-sunken)] p-4">
            <Switch checked={consent} onCheckedChange={setConsent} aria-label="Informed consent" />
            <span className="text-sm text-[var(--color-text)]">
              I&apos;ve read the study description above and consent to participating under these terms.
              I can withdraw at any time.
            </span>
          </label>

          <Button type="submit" size="lg" disabled={!consent || !signerId.trim()} className="w-full">
            Enrol in study
          </Button>
        </form>
      </Card>
    </div>
  );
}
