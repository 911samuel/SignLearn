"use client";

import { useTheme } from "@/hooks/useTheme";

interface ThemeToggleProps {
  compact?: boolean;
}

export function ThemeToggle({ compact }: ThemeToggleProps) {
  const [theme, toggle] = useTheme();
  const label = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  const icon = theme === "dark" ? "☀️" : "🌙";

  return (
    <button
      type="button"
      onClick={toggle}
      title={label}
      aria-label={label}
      style={{
        padding: compact ? "0.3rem 0.5rem" : "0.4rem 0.65rem",
        borderRadius: "var(--radius)",
        border: "1px solid var(--border)",
        background: "transparent",
        cursor: "pointer",
        fontSize: compact ? "1rem" : "0.95rem",
        lineHeight: 1,
        color: "var(--text-muted)",
      }}
    >
      {icon}
    </button>
  );
}
