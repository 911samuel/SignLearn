"use client";

import type { Caption } from "@/hooks/useRoom";

interface CaptionsPanelProps {
  captions: Caption[];
  filter?: "all" | "sign" | "speech";
  emptyHint?: string;
}

export function CaptionsPanel({ captions, filter = "all", emptyHint }: CaptionsPanelProps) {
  const filtered = filter === "all" ? captions : captions.filter((c) => c.source === filter);
  const recent = filtered.slice(-5);

  return (
    <div style={styles.box} aria-live="polite">
      {recent.length === 0 ? (
        <span style={styles.empty}>{emptyHint ?? "Waiting for captions…"}</span>
      ) : (
        recent.map((c) => (
          <div key={c.id} style={styles.line}>
            <span style={styles.icon} aria-hidden>{c.source === "sign" ? "🤟" : "🗣"}</span>
            <span style={styles.name}>{c.name}:</span>
            <span style={styles.text}>{c.text}</span>
            {c.confidence != null && (
              <span style={styles.conf}>{Math.round(c.confidence * 100)}%</span>
            )}
          </div>
        ))
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  box: {
    minHeight: 96, padding: "0.5rem 0.75rem", background: "#0e0e1a",
    borderRadius: 8, display: "flex", flexDirection: "column", gap: "0.25rem",
  },
  empty: { color: "#555", fontSize: "0.85rem" },
  line: { display: "flex", alignItems: "baseline", gap: "0.4rem", fontSize: "0.95rem" },
  icon: { fontSize: "0.9rem" },
  name: { color: "#888", fontSize: "0.8rem" },
  text: { color: "#eee" },
  conf: { color: "#666", fontSize: "0.75rem", marginLeft: "auto" },
};
