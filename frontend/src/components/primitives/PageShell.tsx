import { TopNav } from "./TopNav";
import { MobileBottomNav } from "./MobileBottomNav";
import { cn } from "@/lib/utils";

export function PageShell({
  children,
  className,
  hideTopNav,
  hideBottomNav,
  noMaxWidth,
  noPadding,
}: {
  children: React.ReactNode;
  className?: string;
  hideTopNav?: boolean;
  hideBottomNav?: boolean;
  noMaxWidth?: boolean;
  noPadding?: boolean;
}) {
  return (
    <div className="flex min-h-screen flex-col bg-[var(--color-bg)] text-[var(--color-text)]">
      {!hideTopNav && <TopNav />}
      <main
        id="main-content"
        tabIndex={-1}
        className={cn(
          "flex-1 pb-20 md:pb-0 focus:outline-none",
          !noPadding && "px-[var(--spacing-page-x)] lg:px-[var(--spacing-page-x-lg)]",
          !noMaxWidth && "mx-auto w-full max-w-7xl",
          className,
        )}
      >
        {children}
      </main>
      {!hideBottomNav && <MobileBottomNav />}
    </div>
  );
}
