import type { Metadata } from "next";
import Link from "next/link";
import { LandingCTA } from "./LandingCTA";
import { ThemeToggle } from "@/components/ThemeToggle";

export const metadata: Metadata = {
  title: "SignLearn — Real-time ASL ↔ English in your browser",
  alternates: { canonical: "https://signlearn.app" },
};

export default function LandingPage() {
  return (
    <div style={styles.shell}>
      <header style={styles.topbar}>
        <span style={styles.brand} aria-label="SignLearn">
          <span style={styles.brandGlyph} aria-hidden="true">◐◑</span>{" "}
          SignLearn
        </span>
        <nav style={styles.nav} aria-label="Primary navigation">
          <Link href="/learn" style={styles.navLink}>Learn ASL</Link>
          <Link href="/practice" style={styles.navLink}>Practice</Link>
          <Link href="/privacy" style={styles.navLink}>Privacy</Link>
          <a
            href="https://github.com/"
            style={styles.navLink}
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub
          </a>
          <ThemeToggle compact />
        </nav>
      </header>

      <main id="main-content" style={styles.hero}>
        <h1 style={styles.h1}>
          Have a real conversation.
          <br />
          <span style={styles.h1Accent}>
            No interpreter. No app. Just a link.
          </span>
        </h1>
        <p style={styles.sub}>
          SignLearn translates American Sign Language and English in real time,
          right in your browser. Built with — and for — the Deaf and
          Hard‑of‑Hearing community.
        </p>

        <LandingCTA />

        <ul style={styles.trust} aria-label="Trust signals">
          <li style={styles.trustPill}>🔒 Video never leaves your device</li>
          <li style={styles.trustPill}>⚡ Real‑time, &lt;500&nbsp;ms latency</li>
          <li style={styles.trustPill}>♿ WCAG&nbsp;2.2&nbsp;AAA contrast</li>
          <li style={styles.trustPill}>📖 Open source</li>
        </ul>

        <section style={styles.how} aria-labelledby="how-title">
          <h2 id="how-title" style={styles.h2}>
            How it works
          </h2>
          <ol style={styles.howList}>
            <li>
              <strong>Open the link.</strong> Your camera runs in this tab —
              your video never uploads.
            </li>
            <li>
              <strong>Sign or speak.</strong> Hand landmarks become English;
              speech becomes captions.
            </li>
            <li>
              <strong>Share the room code.</strong> Anyone with a browser can
              join the other side.
            </li>
          </ol>
        </section>
      </main>

      <footer style={styles.footer}>
        <span>© SignLearn — built with the DHH community.</span>
        <span style={styles.sep}>·</span>
        <Link href="/privacy" style={styles.footerLink}>Privacy</Link>
        <span style={styles.sep}>·</span>
        <Link href="/learn" style={styles.footerLink}>Learn ASL</Link>
        <span style={styles.sep}>·</span>
        <a href="mailto:hello@signlearn.app" style={styles.footerLink}>
          Contact
        </a>
      </footer>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  shell: {
    minHeight: "100svh",
    display: "flex",
    flexDirection: "column",
    background:
      "radial-gradient(1200px 600px at 80% -10%, rgba(0,229,255,0.10), transparent 60%), var(--bg)",
  },
  topbar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "1rem 1.5rem",
    flexWrap: "wrap",
    gap: "0.5rem",
  },
  brand: { fontWeight: 700, fontSize: "1.1rem", letterSpacing: "0.01em" },
  brandGlyph: { color: "var(--accent)", marginRight: "0.35rem" },
  nav: { display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" },
  navLink: { color: "var(--text-muted)", textDecoration: "none", fontSize: "0.95rem" },

  hero: {
    flex: 1,
    width: "min(880px, 100%)",
    margin: "0 auto",
    padding: "3rem 1.5rem 2rem",
    display: "flex",
    flexDirection: "column",
    gap: "1.25rem",
  },
  h1: {
    margin: 0,
    fontSize: "clamp(2rem, 5vw, 3.25rem)",
    lineHeight: 1.1,
    fontWeight: 700,
    letterSpacing: "-0.015em",
  },
  h1Accent: { color: "var(--accent)" },
  sub: {
    margin: 0,
    fontSize: "1.1rem",
    color: "var(--text-muted)",
    maxWidth: 620,
    lineHeight: 1.5,
  },

  trust: {
    display: "flex",
    flexWrap: "wrap",
    gap: "0.6rem",
    padding: 0,
    margin: "0.5rem 0 0",
    listStyle: "none",
  },
  trustPill: {
    padding: "0.4rem 0.75rem",
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: 999,
    fontSize: "0.85rem",
    color: "var(--text-muted)",
  },

  how: {
    marginTop: "2rem",
    padding: "1.5rem",
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-lg)",
  },
  h2: { margin: "0 0 0.75rem", fontSize: "1.15rem" },
  howList: { margin: 0, paddingLeft: "1.25rem", lineHeight: 1.7, color: "var(--text)" },

  footer: {
    display: "flex",
    flexWrap: "wrap",
    gap: "0.5rem",
    padding: "1.5rem",
    borderTop: "1px solid var(--border)",
    color: "var(--text-faint)",
    fontSize: "0.85rem",
    justifyContent: "center",
  },
  sep: { opacity: 0.5 },
  footerLink: { color: "var(--text-muted)", textDecoration: "none" },
};
