"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import type { Role } from "@/hooks/useRoom";
import { usePreferences } from "@/hooks/usePreferences";
import { ThemeToggle } from "@/components/ThemeToggle";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5001";

interface RoomInfo {
  exists: boolean;
  members: { role: Role; name: string }[];
}

export default function JoinPage() {
  const params = useParams();
  const roomId = (params?.roomId as string) ?? "";
  const router = useRouter();
  const [prefs, updatePrefs] = usePreferences();
  const [info, setInfo] = useState<RoomInfo | null>(null);
  const [name, setName] = useState(prefs.name);
  const [role, setRole] = useState<Role | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${BACKEND_URL}/rooms/${roomId}`)
      .then(async (r) => {
        if (r.status === 404) return { exists: false, members: [] } as RoomInfo;
        return (await r.json()) as RoomInfo;
      })
      .then((data) => { if (!cancelled) setInfo(data); })
      .catch(() => { if (!cancelled) setError("Could not reach server"); });
    return () => { cancelled = true; };
  }, [roomId]);

  // Sync name if prefs load after initial render
  useEffect(() => {
    if (prefs.name && !name) setName(prefs.name);
  }, [prefs.name, name]);

  if (info && !info.exists) {
    return (
      <div style={styles.shell}>
        <div style={styles.card}>
          <h2 style={styles.title}>Room not found</h2>
          <p style={styles.muted}>No active room with code <code>{roomId}</code>.</p>
          <button style={styles.primary} onClick={() => router.push("/")}>Back to home</button>
        </div>
      </div>
    );
  }

  const takenRoles = new Set(info?.members.map((m) => m.role));
  const canSubmit = !!role && name.trim().length > 0;

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    updatePrefs({ name: name.trim() });
    // Pass role + name via URL search params — avoids navigation-state loss on refresh.
    router.push(`/r/${roomId}?role=${role}&name=${encodeURIComponent(name.trim())}`);
  }

  const roleHint = role === "signer"
    ? "Your camera will be used to recognise your signs. Your video stays on this device."
    : role === "hearing"
      ? "Your microphone will be used for speech-to-text captions. Audio is not recorded."
      : null;

  const joinUrl = typeof window !== "undefined"
    ? `${window.location.origin}/r/${roomId}/join`
    : `/r/${roomId}/join`;

  return (
    <div style={styles.shell}>
      <form onSubmit={submit} style={styles.card} aria-label="Join room form">
        <div style={styles.topRow}>
          <h2 style={styles.title}>
            Join room <code style={styles.code}>{roomId}</code>
          </h2>
          <div style={styles.topActions}>
            <ThemeToggle compact />
          </div>
        </div>

        <p style={styles.muted}>Share this link with the other participant:</p>
        <div style={styles.shareBox}>
          <code style={styles.url}>{joinUrl}</code>
          <button
            type="button"
            style={styles.copyBtn}
            onClick={() => navigator.clipboard?.writeText(joinUrl)}
            aria-label="Copy join link"
          >
            Copy
          </button>
        </div>

        <label style={styles.fieldLabel} htmlFor="join-name">Your name</label>
        <input
          id="join-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Alex"
          maxLength={40}
          style={styles.input}
          autoFocus={!prefs.name}
          autoComplete="nickname"
        />

        <fieldset style={styles.fieldset}>
          <legend style={styles.legend}>Your role in this conversation</legend>
          <RoleButton
            label="Signer"
            description="I will sign — uses my camera."
            selected={role === "signer"}
            disabled={takenRoles.has("signer")}
            onClick={() => setRole("signer")}
          />
          <RoleButton
            label="Hearing"
            description="I will speak — uses my microphone."
            selected={role === "hearing"}
            disabled={takenRoles.has("hearing")}
            onClick={() => setRole("hearing")}
          />
        </fieldset>

        {roleHint && <p style={styles.roleHint}>{roleHint}</p>}

        <div style={styles.textSizeRow}>
          <span style={styles.textSizeLabel}>Text size</span>
          <div style={styles.textSizeGroup} role="group" aria-label="Text size">
            {(["normal", "large"] as const).map((sz) => (
              <button
                key={sz}
                type="button"
                onClick={() => updatePrefs({ textSize: sz })}
                style={{
                  ...styles.szBtn,
                  background: prefs.textSize === sz ? "var(--primary)" : "transparent",
                  color: prefs.textSize === sz ? "#fff" : "var(--text-muted)",
                  borderColor: prefs.textSize === sz ? "var(--primary)" : "var(--border)",
                }}
                aria-pressed={prefs.textSize === sz}
              >
                {sz === "normal" ? "A" : "A+"}
              </button>
            ))}
          </div>
        </div>

        <button type="submit" disabled={!canSubmit} style={styles.primary}>
          Enter room
        </button>

        {error && <p style={styles.error} role="alert">{error}</p>}

        <p style={styles.fine}>
          <Link href="/privacy" style={styles.fineLink}>How we use your camera &amp; mic</Link>
        </p>
      </form>
    </div>
  );
}

function RoleButton({
  label, description, selected, disabled, onClick,
}: {
  label: string; description: string; selected: boolean; disabled?: boolean; onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      style={{
        ...styles.roleBtn,
        background: selected ? "var(--primary)" : disabled ? "var(--bg-input)" : "var(--bg-elevated)",
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
        borderColor: selected ? "var(--primary)" : "var(--border)",
      }}
    >
      <div style={{ fontWeight: 600, color: "var(--text)" }}>
        {label}{disabled ? " (taken)" : ""}
      </div>
      <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>{description}</div>
    </button>
  );
}

const styles: Record<string, React.CSSProperties> = {
  shell: { minHeight: "100svh", display: "grid", placeItems: "center", padding: "1rem" },
  card: {
    width: "min(460px, 100%)", display: "flex", flexDirection: "column", gap: "0.85rem",
    padding: "2rem", background: "var(--bg-card)", borderRadius: "var(--radius-lg)",
    color: "var(--text)", boxShadow: "var(--shadow-card)",
  },
  topRow: { display: "flex", justifyContent: "space-between", alignItems: "flex-start" },
  topActions: { display: "flex", gap: "0.5rem" },
  title: { margin: 0, fontSize: "1.25rem" },
  muted: { color: "var(--text-muted)", fontSize: "0.85rem", margin: 0 },
  shareBox: {
    display: "flex", alignItems: "center", gap: "0.5rem",
    background: "var(--bg-input)", borderRadius: "var(--radius)",
    padding: "0.4rem 0.6rem", overflow: "hidden",
  },
  url: {
    flex: 1, fontSize: "0.78rem", color: "var(--text-muted)",
    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
  },
  copyBtn: {
    padding: "0.25rem 0.6rem", borderRadius: 6, border: "none",
    background: "var(--bg-elevated)", color: "var(--text-muted)",
    fontSize: "0.8rem", cursor: "pointer", flexShrink: 0,
  },
  code: { background: "var(--bg-input)", padding: "0.05rem 0.4rem", borderRadius: 4 },
  fieldLabel: { fontSize: "0.9rem", color: "var(--text-muted)", marginBottom: "-0.4rem" },
  input: {
    padding: "0.65rem 0.85rem", borderRadius: "var(--radius)", border: "1px solid var(--border)",
    background: "var(--bg-input)", color: "var(--text)", fontSize: "1rem",
    fontFamily: "inherit",
  },
  fieldset: {
    display: "flex", flexDirection: "column", gap: "0.5rem",
    border: "none", padding: 0, margin: 0,
  },
  legend: { fontSize: "0.9rem", color: "var(--text-muted)", marginBottom: "0.35rem" },
  roleBtn: {
    textAlign: "left", padding: "0.8rem 1rem", borderRadius: "var(--radius)",
    border: "1px solid", color: "var(--text)", fontFamily: "inherit",
  },
  roleHint: {
    margin: 0, fontSize: "0.82rem", color: "var(--text-muted)",
    padding: "0.5rem 0.75rem", background: "var(--bg-input)",
    borderRadius: "var(--radius)", lineHeight: 1.5,
    borderLeft: "3px solid var(--accent)",
  },
  textSizeRow: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "0.5rem 0",
  },
  textSizeLabel: { fontSize: "0.9rem", color: "var(--text-muted)" },
  textSizeGroup: { display: "flex", gap: "0.35rem" },
  szBtn: {
    padding: "0.3rem 0.75rem", borderRadius: "var(--radius)",
    border: "1px solid", cursor: "pointer", fontFamily: "inherit",
    fontSize: "0.9rem", fontWeight: 600, transition: "background 120ms",
  },
  primary: {
    padding: "0.85rem 1rem", borderRadius: "var(--radius)", border: "none",
    background: "var(--primary)", color: "#fff", fontWeight: 600,
    cursor: "pointer", fontSize: "1rem", fontFamily: "inherit", minHeight: 52,
  },
  error: { color: "var(--danger)", margin: 0, fontSize: "0.85rem" },
  fine: { margin: 0, textAlign: "center", fontSize: "0.78rem" },
  fineLink: { color: "var(--text-faint)", textDecoration: "none" },
};
