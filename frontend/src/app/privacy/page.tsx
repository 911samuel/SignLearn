import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy",
  description:
    "How SignLearn handles your camera, microphone, and conversation data. Plain English, no legalese.",
};

export default function PrivacyPage() {
  return (
    <div style={styles.shell}>
      <main id="main-content" style={styles.main}>
        <Link href="/" style={styles.back}>← Back to SignLearn</Link>
        <h1 style={styles.h1}>Privacy &amp; how we handle your video</h1>
        <p style={styles.lead}>
          Plain English, no legalese. SignLearn is built for the Deaf and
          Hard‑of‑Hearing community first — privacy is the first feature, not
          the last.
        </p>

        <h2 style={styles.h2}>What stays on your device</h2>
        <ul style={styles.list}>
          <li>Your camera feed is rendered locally in your browser.</li>
          <li>
            MediaPipe runs in‑browser to extract hand landmarks — the raw video
            frames are never uploaded.
          </li>
          <li>
            Peer‑to‑peer video between you and your partner uses WebRTC; it is
            end‑to‑end between your two browsers.
          </li>
        </ul>

        <h2 style={styles.h2}>What is sent to our server</h2>
        <ul style={styles.list}>
          <li>
            <strong>126 floating‑point numbers per frame</strong> — the (x, y,
            z) positions of up to 21 landmarks per hand. This cannot be
            reconstructed into a video.
          </li>
          <li>
            The recognised word and a confidence score, so the other participant
            can see the caption.
          </li>
          <li>
            If you use speech‑to‑text, the transcribed text (not the audio) is
            sent so the signer can read it.
          </li>
        </ul>

        <h2 style={styles.h2}>What we store, and for how long</h2>
        <ul style={styles.list}>
          <li>
            The conversation transcript is stored for the lifetime of the room.
          </li>
          <li>You can export the transcript and clear the room at any time.</li>
          <li>
            We do not sell, share, or use your conversations for advertising.
            Ever.
          </li>
        </ul>

        <h2 style={styles.h2}>Open source</h2>
        <p style={styles.p}>
          The full SignLearn codebase is open source. If you want to verify any
          of these claims, you can read the code yourself — that is the
          strongest privacy guarantee we can offer.
        </p>

        <h2 style={styles.h2}>Questions or concerns</h2>
        <p style={styles.p}>
          Email{" "}
          <a href="mailto:privacy@signlearn.app">privacy@signlearn.app</a>. We
          answer every message within two business days.
        </p>
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  shell: {
    minHeight: "100svh",
    display: "flex",
    justifyContent: "center",
    padding: "2rem 1rem",
  },
  main: {
    width: "min(720px, 100%)",
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  back: { fontSize: "0.9rem", textDecoration: "none", marginBottom: "0.5rem" },
  h1: { fontSize: "2rem", margin: "0.5rem 0 0.25rem", lineHeight: 1.2 },
  lead: {
    fontSize: "1.05rem",
    color: "var(--text-muted)",
    margin: "0 0 1.5rem",
  },
  h2: { fontSize: "1.15rem", margin: "1.5rem 0 0.25rem" },
  p: { lineHeight: 1.6, margin: "0.25rem 0", color: "var(--text)" },
  list: { lineHeight: 1.7, color: "var(--text)", paddingLeft: "1.25rem" },
};
