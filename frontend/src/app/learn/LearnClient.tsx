"use client";

import { useState } from "react";
import Link from "next/link";

interface SignCard {
  word: string;
  category: "letter" | "number" | "word";
  description: string;
  tips?: string;
}

type Category = "all" | "letter" | "number" | "word";

const CATEGORY_LABELS: Record<Category, string> = {
  all: "All signs",
  letter: "Letters",
  number: "Numbers",
  word: "Common words",
};

interface Props {
  signs: SignCard[];
}

export function LearnClient({ signs }: Props) {
  const [category, setCategory] = useState<Category>("word");
  const [expanded, setExpanded] = useState<string | null>(null);

  const visible =
    category === "all" ? signs : signs.filter((s) => s.category === category);

  return (
    <>
      <div style={styles.tabs} role="tablist" aria-label="Sign categories">
        {(Object.keys(CATEGORY_LABELS) as Category[]).map((cat) => (
          <button
            key={cat}
            role="tab"
            aria-selected={category === cat}
            onClick={() => { setCategory(cat); setExpanded(null); }}
            style={{
              ...styles.tab,
              background: category === cat ? "var(--accent)" : "transparent",
              color: category === cat ? "#001016" : "var(--text-muted)",
              fontWeight: category === cat ? 700 : 400,
              borderColor: category === cat ? "var(--accent)" : "var(--border)",
            }}
          >
            {CATEGORY_LABELS[cat]}
          </button>
        ))}
      </div>

      <div style={styles.grid} role="tabpanel">
        {visible.map((sign) => {
          const isOpen = expanded === sign.word;
          return (
            <article
              key={sign.word}
              style={{
                ...styles.card,
                borderColor: isOpen ? "var(--accent)" : "var(--border)",
              }}
            >
              <button
                style={styles.cardBtn}
                onClick={() => setExpanded(isOpen ? null : sign.word)}
                aria-expanded={isOpen}
                aria-controls={`sign-${sign.word}`}
              >
                <span style={styles.cardWord}>
                  {sign.word.replace(/_/g, " ")}
                </span>
                <span style={styles.cardCaret} aria-hidden="true">
                  {isOpen ? "▲" : "▼"}
                </span>
              </button>

              <div id={`sign-${sign.word}`} hidden={!isOpen} style={styles.cardBody}>
                <p style={styles.cardDesc}>{sign.description}</p>
                {sign.tips && (
                  <p style={styles.cardTip}>
                    <strong>Tip:</strong> {sign.tips}
                  </p>
                )}
                <Link href="/practice" style={styles.tryBtn}>
                  Try it in practice mode →
                </Link>
              </div>
            </article>
          );
        })}
      </div>
    </>
  );
}

const styles: Record<string, React.CSSProperties> = {
  tabs: { display: "flex", flexWrap: "wrap", gap: "0.5rem" },
  tab: {
    padding: "0.45rem 1rem",
    borderRadius: 999,
    border: "1px solid",
    cursor: "pointer",
    fontSize: "0.9rem",
    transition: "background 150ms, color 150ms",
    fontFamily: "inherit",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
    gap: "0.75rem",
  },
  card: {
    background: "var(--bg-elevated)",
    border: "1px solid",
    borderRadius: "var(--radius-lg)",
    overflow: "hidden",
    transition: "border-color 150ms",
  },
  cardBtn: {
    width: "100%",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "0.85rem 1rem",
    background: "transparent",
    border: "none",
    cursor: "pointer",
    fontFamily: "inherit",
    textAlign: "left",
    gap: "0.5rem",
  },
  cardWord: {
    fontWeight: 600,
    fontSize: "1.05rem",
    color: "var(--text)",
    textTransform: "capitalize",
  },
  cardCaret: { fontSize: "0.65rem", color: "var(--text-faint)", flexShrink: 0 },
  cardBody: {
    padding: "0 1rem 1rem",
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  cardDesc: { margin: 0, fontSize: "0.9rem", lineHeight: 1.55, color: "var(--text)" },
  cardTip: {
    margin: 0,
    fontSize: "0.85rem",
    color: "var(--text-muted)",
    lineHeight: 1.5,
    padding: "0.45rem 0.65rem",
    borderLeft: "3px solid var(--accent)",
    background: "var(--bg-card)",
    borderRadius: "0 var(--radius) var(--radius) 0",
  },
  tryBtn: {
    alignSelf: "flex-start",
    marginTop: "0.25rem",
    color: "var(--accent)",
    fontSize: "0.85rem",
    fontWeight: 600,
    textDecoration: "none",
  },
};
