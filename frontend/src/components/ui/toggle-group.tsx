"use client";

import * as React from "react";
import * as ToggleGroupPrimitive from "@radix-ui/react-toggle-group";
import { cn } from "@/lib/utils";

export const ToggleGroup = React.forwardRef<
  React.ElementRef<typeof ToggleGroupPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Root>
>(({ className, ...props }, ref) => (
  <ToggleGroupPrimitive.Root
    ref={ref}
    className={cn(
      "inline-flex items-center gap-1 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-sunken)] p-1",
      className,
    )}
    {...props}
  />
));
ToggleGroup.displayName = "ToggleGroup";

export const ToggleGroupItem = React.forwardRef<
  React.ElementRef<typeof ToggleGroupPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <ToggleGroupPrimitive.Item
    ref={ref}
    className={cn(
      "inline-flex h-9 min-w-[44px] items-center justify-center gap-2 rounded-full px-3 text-sm font-semibold text-[var(--color-text-muted)] transition hover:text-[var(--color-text)] data-[state=on]:bg-[var(--color-surface)] data-[state=on]:text-[var(--color-text)] data-[state=on]:shadow-[var(--shadow-sm)] focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]",
      className,
    )}
    {...props}
  >
    {children}
  </ToggleGroupPrimitive.Item>
));
ToggleGroupItem.displayName = "ToggleGroupItem";
