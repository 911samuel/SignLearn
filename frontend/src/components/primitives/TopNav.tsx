"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { BookOpen, Dumbbell, FlaskConical, Hand, LineChart, MessageSquare, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { A11yPreferencesMenu } from "./A11yPreferencesMenu";
import { useToast } from "@/components/ui/toast";
import { BACKEND_URL } from "@/lib/api";
import { t } from "@/i18n";

const ITEMS = [
  { href: "/practice", label: "Practice", icon: Dumbbell },
  { href: "/learn",    label: "Learn",    icon: BookOpen },
  { href: "/analytics", label: "Analytics", icon: LineChart },
  { href: "/research", label: "Research", icon: FlaskConical },
] as const;

export function TopNav() {
  const pathname = usePathname();
  const router = useRouter();
  const { toast } = useToast();
  const [creating, setCreating] = useState(false);

  async function startConversation() {
    if (creating) return;
    setCreating(true);
    try {
      const res = await fetch(`${BACKEND_URL}/rooms`, { method: "POST" });
      if (!res.ok) throw new Error("Could not reach the SignLearn server.");
      const { room_id } = await res.json();
      router.push(`/r/${room_id}/join`);
    } catch (err) {
      toast({
        tone: "danger",
        title: "Couldn't start a conversation",
        description:
          err instanceof Error
            ? err.message
            : "Check your connection and try again.",
      });
      setCreating(false);
    }
  }

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--color-border)] bg-[color-mix(in_srgb,var(--color-bg)_92%,transparent)] backdrop-blur supports-[backdrop-filter]:bg-[color-mix(in_srgb,var(--color-bg)_75%,transparent)]">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-2 px-[var(--spacing-page-x)] lg:px-[var(--spacing-page-x-lg)]">
        <Link
          href="/"
          className="flex items-center gap-2 text-[var(--color-text)] hover:no-underline"
          aria-label="SignLearn — go to home"
        >
          <Logo />
          <span className="heading-h3 leading-none tracking-tight">SignLearn</span>
        </Link>

        <nav aria-label="Main navigation" className="ml-6 hidden items-center gap-1 md:flex">
          {ITEMS.map(({ href, label, icon: Icon }) => {
            const active = pathname === href || pathname.startsWith(href + "/");
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "inline-flex h-10 items-center gap-2 rounded-[var(--radius-md)] px-3 text-sm font-semibold transition hover:bg-[var(--color-surface-sunken)] hover:no-underline focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]",
                  active
                    ? "text-[var(--color-text)] bg-[var(--color-surface-sunken)]"
                    : "text-[var(--color-text-muted)]",
                )}
              >
                <Icon className="size-4" aria-hidden />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <A11yPreferencesMenu />
          <Link
            href="/account"
            aria-label="Account"
            className="inline-flex h-10 w-10 items-center justify-center rounded-full text-[var(--color-text-muted)] hover:bg-[var(--color-surface-sunken)] focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]"
          >
            <User className="size-5" aria-hidden />
          </Link>
          <Button
            variant="primary"
            size="md"
            className="hidden sm:inline-flex"
            onClick={startConversation}
            disabled={creating}
            aria-label={t("nav.startConversation")}
          >
            {creating ? (
              <>
                <Hand className="size-4 sl-pulse-soft" aria-hidden />
                <span>Starting…</span>
              </>
            ) : (
              <>
                <MessageSquare className="size-4" aria-hidden />
                <span>{t("nav.startConversation")}</span>
              </>
            )}
          </Button>
        </div>
      </div>
    </header>
  );
}

function Logo() {
  return (
    <span
      aria-hidden
      className="flex h-9 w-9 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand)] text-[var(--color-brand-foreground)] shadow-[var(--shadow-sm)]"
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M7 11V6a1.5 1.5 0 1 1 3 0v5" />
        <path d="M10 11V4.5a1.5 1.5 0 0 1 3 0V11" />
        <path d="M13 11V5.5a1.5 1.5 0 0 1 3 0V12" />
        <path d="M16 12V8a1.5 1.5 0 0 1 3 0v7a6 6 0 0 1-6 6h-1a6 6 0 0 1-6-6v-2l-2-3a1.5 1.5 0 0 1 2.5-1.7L7 11" />
      </svg>
    </span>
  );
}
