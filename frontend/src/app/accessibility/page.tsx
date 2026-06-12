import type { Metadata } from "next";
import { Accessibility, CheckCircle2 } from "lucide-react";
import { PageShell } from "@/components/primitives/PageShell";
import { SectionHeader } from "@/components/primitives/SectionHeader";
import { SiteFooter } from "@/components/primitives/Footer";
import { Card } from "@/components/ui/card";

export const metadata: Metadata = {
  title: "Accessibility statement",
};

const STANDARDS = [
  "Conforms to WCAG 2.2 AA for colour contrast in light, dark, and high-contrast themes.",
  "Every interactive element is reachable and operable by keyboard alone.",
  "Live regions announce predictions, captions, and connection status to screen readers.",
  "Respects prefers-reduced-motion — all non-essential animation pauses automatically.",
  "Push-to-talk has both pointer and keyboard bindings, plus an alternate toggle mode for users who can't hold a key.",
  "Text size and high-contrast theme are persistent per device.",
];

export default function AccessibilityPage() {
  return (
    <PageShell>
      <div className="pt-10 pb-12">
        <SectionHeader
          eyebrow="Accessibility"
          title="Accessibility statement"
          description="SignLearn is designed accessibility-first. If you find a barrier, please flag it — every report is read."
          as="h1"
        />

        <Card className="mt-8 p-6 md:p-8">
          <div className="flex h-12 w-12 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]">
            <Accessibility className="size-6" aria-hidden />
          </div>
          <h2 className="mt-4 heading-h2 text-[var(--color-text)]">Our commitments</h2>
          <ul className="mt-5 space-y-3">
            {STANDARDS.map((line) => (
              <li key={line} className="flex items-start gap-3">
                <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-[var(--color-success)]" aria-hidden />
                <span className="text-[var(--color-text)]">{line}</span>
              </li>
            ))}
          </ul>

          <h2 className="mt-10 heading-h2 text-[var(--color-text)]">Found a problem?</h2>
          <p className="mt-2 text-[var(--color-text-muted)]">
            Use the &ldquo;Accessibility issue&rdquo; category on the Research & feedback page. We aim to
            acknowledge every accessibility report within 48 hours.
          </p>
        </Card>
      </div>
      <SiteFooter />
    </PageShell>
  );
}
