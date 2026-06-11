import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, BookOpen, CheckCircle2, Flame, Lock, Sparkles, Star } from "lucide-react";
import { PageShell } from "@/components/primitives/PageShell";
import { SectionHeader } from "@/components/primitives/SectionHeader";
import { SiteFooter } from "@/components/primitives/Footer";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CURRICULUM } from "@/data/curriculum";
import { LearnProgressClient } from "./LearnProgressClient";
import { t } from "@/i18n";

export const metadata: Metadata = {
  title: "Learning path",
  description:
    "Structured ASL lessons with live recognition feedback. Practise letters, numbers, and conversational vocabulary in your browser.",
};

export default function LearnPage() {
  return (
    <PageShell>
      <div className="pt-10 lg:pt-16">
        <SectionHeader
          eyebrow="Learning path"
          title={t("learn.title")}
          description={t("learn.subhead")}
          as="h1"
        />

        <LearnProgressClient />
      </div>

      <div className="mt-10 space-y-12 pb-16">
        {CURRICULUM.map((unit, unitIdx) => (
          <section key={unit.id} aria-labelledby={`unit-${unit.id}`}>
            <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="eyebrow">Unit {unitIdx + 1}</p>
                <h2 id={`unit-${unit.id}`} className="mt-1 heading-h2 text-[var(--color-text)]">
                  {unit.title}
                </h2>
                <p className="mt-1 max-w-2xl text-[var(--color-text-muted)]">
                  {unit.subtitle}
                </p>
              </div>
              <Badge tone="neutral" className="self-start">
                {unit.lessons.length} lessons
              </Badge>
            </header>

            <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {unit.lessons.map((lesson, idx) => (
                <li key={lesson.id} className="h-full">
                  <Card className="group flex h-full flex-col overflow-hidden transition hover:border-[var(--color-brand)]">
                    <Link
                      href={`/learn/${lesson.id}`}
                      className="flex h-full flex-col gap-3 p-5 text-[var(--color-text)] hover:no-underline focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]"
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className="inline-flex h-10 w-10 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)] font-bold"
                          aria-hidden
                        >
                          {idx + 1}
                        </span>
                        <div className="flex items-center gap-1.5" aria-label={`Difficulty ${lesson.difficulty} of 3`}>
                          {Array.from({ length: 3 }).map((_, i) => (
                            <Star
                              key={i}
                              className={
                                i < lesson.difficulty
                                  ? "size-3.5 fill-[var(--color-warning)] text-[var(--color-warning)]"
                                  : "size-3.5 text-[var(--color-border-strong)]"
                              }
                              aria-hidden
                            />
                          ))}
                        </div>
                      </div>
                      <h3 className="heading-h3 text-[var(--color-text)]">{lesson.title}</h3>
                      <p className="text-sm text-[var(--color-text-muted)] leading-relaxed">
                        {lesson.description}
                      </p>
                      <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-2">
                        {lesson.signs.slice(0, 6).map((s) => (
                          <code
                            key={s}
                            className="rounded-[var(--radius-xs)] bg-[var(--color-surface-sunken)] px-1.5 py-0.5 font-mono text-[0.72rem] text-[var(--color-text-muted)]"
                          >
                            {s}
                          </code>
                        ))}
                        {lesson.signs.length > 6 && (
                          <span className="text-[0.72rem] text-[var(--color-text-faint)]">
                            +{lesson.signs.length - 6}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center justify-between pt-3">
                        <span className="text-xs text-[var(--color-text-muted)]">
                          ~{lesson.durationMin} min
                        </span>
                        <span className="inline-flex items-center gap-1 text-sm font-semibold text-[var(--color-brand)] group-hover:underline">
                          Start lesson <ArrowRight className="size-4" aria-hidden />
                        </span>
                      </div>
                    </Link>
                  </Card>
                </li>
              ))}
            </ul>
          </section>
        ))}

        <Card className="p-8 md:p-10">
          <div className="flex flex-wrap items-center justify-between gap-6">
            <div className="max-w-xl">
              <Sparkles className="size-7 text-[var(--color-brand)]" aria-hidden />
              <h2 className="mt-3 heading-h2 text-[var(--color-text)]">Want to put it into practice?</h2>
              <p className="mt-2 text-[var(--color-text-muted)]">
                Open a room and share the link — your hearing partner joins in seconds. Or jump
                into solo practice to drill a specific sign.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Button asChild variant="secondary" size="lg">
                <Link href="/practice"><BookOpen aria-hidden /> Solo practice</Link>
              </Button>
              <Button asChild size="lg">
                <Link href="/">
                  Start a conversation <ArrowRight className="size-4" aria-hidden />
                </Link>
              </Button>
            </div>
          </div>
        </Card>
      </div>

      <SiteFooter />
    </PageShell>
  );
}
