"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BookOpen, Dumbbell, FlaskConical, Home, LineChart } from "lucide-react";
import { cn } from "@/lib/utils";

const ITEMS = [
  { href: "/",          label: "Home",      icon: Home },
  { href: "/practice",  label: "Practice",  icon: Dumbbell },
  { href: "/learn",     label: "Learn",     icon: BookOpen },
  { href: "/analytics", label: "Analytics", icon: LineChart },
  { href: "/research",  label: "Research",  icon: FlaskConical },
] as const;

export function MobileBottomNav() {
  const pathname = usePathname();
  return (
    <nav
      aria-label="Bottom navigation"
      className="fixed inset-x-0 bottom-0 z-40 border-t border-[var(--color-border)] bg-[var(--color-surface-elevated)] pb-[env(safe-area-inset-bottom)] md:hidden"
    >
      <ul className="mx-auto grid max-w-md grid-cols-5">
        {ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || (href !== "/" && pathname.startsWith(href));
          return (
            <li key={href}>
              <Link
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "flex min-h-[56px] flex-col items-center justify-center gap-0.5 px-1 py-2 text-[0.7rem] font-semibold transition hover:no-underline",
                  active
                    ? "text-[var(--color-brand)]"
                    : "text-[var(--color-text-muted)]",
                )}
              >
                <Icon className="size-5" aria-hidden />
                <span>{label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
