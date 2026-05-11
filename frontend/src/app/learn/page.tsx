import type { Metadata } from "next";
import Link from "next/link";
import { LearnClient } from "./LearnClient";

export const metadata: Metadata = {
  title: "Learn ASL basics",
  description:
    "Interactive guide to 27 American Sign Language signs — letters, numbers, and common words. Practice each one with your camera.",
};

const SIGNS = [
  { word: "A", category: "letter" as const, description: "Closed fist, thumb resting on the side of the index finger.", tips: "Keep the thumb visible — don't tuck it under." },
  { word: "B", category: "letter" as const, description: "Four fingers straight up together, thumb folded across the palm.", tips: "Keep all four fingers parallel and together." },
  { word: "C", category: "letter" as const, description: "Curve all fingers and thumb into a C shape.", tips: "Both the fingers and thumb should curve — no rigid edges." },
  { word: "D", category: "letter" as const, description: "Index finger points up, other fingers and thumb form a circle.", tips: "The circle should be clear — touch thumb tip to middle fingertip." },
  { word: "E", category: "letter" as const, description: "All four fingers bent, touching the thumb.", tips: "Fingers hook down together; thumb presses up to meet them." },
  { word: "F", category: "letter" as const, description: "Index and thumb form a circle; other three fingers point up.", tips: "Three fingers stay extended and slightly separated." },
  { word: "I", category: "letter" as const, description: "Pinky finger extended up, other fingers closed in a fist.", tips: "Keep the pinky fully straight — it's easy to let it droop." },
  { word: "L", category: "letter" as const, description: "Index finger points up, thumb points out — classic L shape.", tips: "Hand is sideways so the L faces the viewer." },
  { word: "O", category: "letter" as const, description: "All fingers curve to meet the thumb in a round O shape.", tips: "Aim for symmetry — it should look like a circle from the front." },
  { word: "V", category: "letter" as const, description: "Index and middle fingers up in a V; other fingers closed.", tips: "Keep the two fingers fully extended and apart." },
  { word: "W", category: "letter" as const, description: "Index, middle, and ring fingers up and spread in a W.", tips: "Three fingers — not two. Ring finger is the common miss." },
  { word: "Y", category: "letter" as const, description: "Pinky and thumb extended; other three fingers closed.", tips: "Looks like a 'hang loose' gesture. Thumb must stay fully extended." },
  { word: "1", category: "number" as const, description: "Index finger points straight up; all other fingers closed.", tips: "Identical to pointing — keep the finger fully vertical." },
  { word: "2", category: "number" as const, description: "Index and middle fingers up in a V.", tips: "Palm usually faces the signer for numbers, faces out for letters." },
  { word: "5", category: "number" as const, description: "All five fingers spread wide open.", tips: "Spread them as wide as comfortable — a tight 5 looks like a B." },
  { word: "10", category: "number" as const, description: "Closed fist with thumb up, shaken slightly at the wrist.", tips: "The shake is small — two short twists, not a full wrist turn." },
  { word: "hello", category: "word" as const, description: "Open hand touches forehead (fingers together), then swings out and away — like a salute.", tips: "Start at the temple, not the top of the head. Keep fingers together." },
  { word: "thank_you", category: "word" as const, description: "Flat hand starts at chin, moves forward and slightly down toward the other person.", tips: "Think of 'blowing a kiss' from the chin — smooth and deliberate." },
  { word: "please", category: "word" as const, description: "Open hand, palm down, circles on the chest.", tips: "The circle is horizontal — palm stays flat against the chest." },
  { word: "yes", category: "word" as const, description: "Closed fist nods up and down at the wrist, like a nodding head.", tips: "The fist itself is the 'head' nodding — keep the motion small and clear." },
  { word: "no", category: "word" as const, description: "Index and middle fingers come together to tap the thumb, twice.", tips: "Fingers snap closed — not a sideways head shake." },
  { word: "help", category: "word" as const, description: "Thumb-up fist sits on the palm of the other hand; both hands rise together.", tips: "The flat hand lifts the fist — think of giving someone a boost." },
  { word: "sorry", category: "word" as const, description: "Closed fist circles on the chest over the heart.", tips: "Circle is small and sincere — over the heart, not the stomach." },
  { word: "water", category: "word" as const, description: "W-handshape taps the chin twice.", tips: "Form a clear W first, then two light taps. Don't rush the W." },
  { word: "more", category: "word" as const, description: "Both hands in O/pinch shapes tap fingertips together twice.", tips: "Both hands mirror each other. Two taps, not one." },
  { word: "stop", category: "word" as const, description: "One flat hand chops down onto the upturned palm of the other hand.", tips: "Sharp, decisive motion — it's a stop sign landing, not a tap." },
  { word: "i_love_you", category: "word" as const, description: "Pinky, index finger, and thumb extended; middle and ring folded — combined I-L-Y handshape.", tips: "One single handshape, not three separate letters. Hold it steady." },
];

export default function LearnPage() {
  return (
    <div style={styles.shell}>
      <header style={styles.topbar}>
        <Link href="/" style={styles.back}>← Back</Link>
        <h1 style={styles.title}>Learn ASL basics</h1>
        <Link href="/practice" style={styles.practiceLink}>
          Practice with your camera →
        </Link>
      </header>

      <main id="main-content" style={styles.main}>
        <p style={styles.intro}>
          SignLearn recognises <strong>93 signs</strong> — 26 letters, 10
          digits, and 57 words. Each card below describes the handshape so you
          can practise before opening the camera.
        </p>

        <LearnClient signs={SIGNS} />

        <section style={styles.cta} aria-label="Get started">
          <h2 style={styles.ctaTitle}>Ready for a real conversation?</h2>
          <p style={styles.ctaSub}>
            Start a room and share the link — your hearing partner joins in
            seconds, no install needed.
          </p>
          <div style={styles.ctaRow}>
            <Link href="/practice" style={styles.ctaBtnSecondary}>Practice solo first</Link>
            <Link href="/" style={styles.ctaBtnPrimary}>Start a conversation →</Link>
          </div>
        </section>
      </main>

      <footer style={styles.footer}>
        <Link href="/privacy" style={styles.footerLink}>Privacy</Link>
        <span style={styles.sep}>·</span>
        <Link href="/" style={styles.footerLink}>Home</Link>
      </footer>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  shell: { minHeight: "100svh", display: "flex", flexDirection: "column" },
  topbar: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    padding: "0.75rem 1.5rem",
    borderBottom: "1px solid var(--border)",
    background: "var(--bg-elevated)",
    flexWrap: "wrap",
  },
  back: { color: "var(--text-muted)", textDecoration: "none", fontSize: "0.9rem" },
  title: { margin: 0, fontSize: "1.15rem", fontWeight: 700, flex: 1 },
  practiceLink: {
    padding: "0.4rem 0.85rem",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    color: "var(--text-muted)",
    textDecoration: "none",
    fontSize: "0.88rem",
    whiteSpace: "nowrap",
  },
  main: {
    flex: 1,
    width: "min(960px, 100%)",
    margin: "0 auto",
    padding: "2rem 1.25rem",
    display: "flex",
    flexDirection: "column",
    gap: "1.5rem",
  },
  intro: { margin: 0, color: "var(--text-muted)", lineHeight: 1.6 },
  cta: {
    marginTop: "1rem",
    padding: "1.75rem",
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-lg)",
    display: "flex",
    flexDirection: "column",
    gap: "0.75rem",
  },
  ctaTitle: { margin: 0, fontSize: "1.25rem" },
  ctaSub: { margin: 0, color: "var(--text-muted)", lineHeight: 1.5 },
  ctaRow: { display: "flex", flexWrap: "wrap", gap: "0.75rem" },
  ctaBtnPrimary: {
    padding: "0.75rem 1.25rem",
    borderRadius: "var(--radius)",
    background: "var(--accent)",
    color: "#001016",
    fontWeight: 700,
    textDecoration: "none",
    fontSize: "0.95rem",
  },
  ctaBtnSecondary: {
    padding: "0.75rem 1.25rem",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    color: "var(--text-muted)",
    textDecoration: "none",
    fontSize: "0.95rem",
  },
  footer: {
    display: "flex",
    gap: "0.5rem",
    justifyContent: "center",
    padding: "1rem",
    borderTop: "1px solid var(--border)",
    fontSize: "0.85rem",
  },
  footerLink: { color: "var(--text-muted)", textDecoration: "none" },
  sep: { color: "var(--text-faint)" },
};
