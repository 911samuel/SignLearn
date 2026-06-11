"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius-md)] font-semibold transition disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-bg)] [&_svg]:size-[1.1em] [&_svg]:shrink-0 active:scale-[0.98]",
  {
    variants: {
      variant: {
        primary:
          "bg-[var(--color-brand)] text-[var(--color-brand-foreground)] shadow-[var(--shadow-sm)] hover:bg-[var(--color-brand-hover)]",
        secondary:
          "bg-[var(--color-surface)] text-[var(--color-text)] border border-[var(--color-border-strong)] hover:bg-[var(--color-surface-sunken)]",
        ghost:
          "bg-transparent text-[var(--color-text)] hover:bg-[var(--color-surface-sunken)]",
        outline:
          "bg-transparent text-[var(--color-text)] border border-[var(--color-border-strong)] hover:bg-[var(--color-surface-sunken)]",
        danger:
          "bg-[var(--color-danger)] text-white hover:opacity-90",
        success:
          "bg-[var(--color-success)] text-white hover:opacity-90",
        link:
          "bg-transparent text-[var(--color-brand)] underline-offset-4 hover:underline px-0 py-0 h-auto",
      },
      size: {
        sm: "h-9 px-3 text-sm",
        md: "h-11 px-5 text-[0.95rem]",
        lg: "h-12 px-6 text-base",
        xl: "h-14 px-7 text-lg",
        icon: "h-11 w-11 p-0",
        "icon-sm": "h-9 w-9 p-0",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { buttonVariants };
