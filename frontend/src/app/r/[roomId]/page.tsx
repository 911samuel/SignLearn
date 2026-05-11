"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import { useRoom, type Role } from "@/hooks/useRoom";
import { SignerView } from "@/components/SignerView";
import { HearingView } from "@/components/HearingView";
import { ConversationLog, type LogEntry } from "@/components/ConversationLog";
import { ThemeToggle } from "@/components/ThemeToggle";
import { RoomErrorBoundary } from "@/components/RoomErrorBoundary";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5001";

export default function RoomPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const roomId = (params?.roomId as string) ?? "";

  const role = searchParams?.get("role") as Role | null;
  const name = searchParams?.get("name") ?? "";

  // Redirect to join if params are missing or invalid.
  useEffect(() => {
    if (!role || !name || (role !== "signer" && role !== "hearing")) {
      router.replace(`/r/${roomId}/join`);
    }
  }, [role, name, roomId, router]);

  if (!role || !name || (role !== "signer" && role !== "hearing")) {
    return null;
  }

  return (
    <RoomInner
      roomId={roomId}
      role={role}
      name={name}
      onLeave={() => router.push("/")}
    />
  );
}

interface RoomInnerProps {
  roomId: string;
  role: Role;
  name: string;
  onLeave: () => void;
}

function RoomInner({ roomId, role, name, onLeave }: RoomInnerProps) {
  const { socket, status, joinError, members, captions, emitSpeech } = useRoom(roomId, role, name);
  const [log, setLog] = useState<LogEntry[]>([]);

  const peerPresent = members.some((m) => m.role !== role);

  useEffect(() => {
    fetch(`${BACKEND_URL}/transcript?room_id=${roomId}&limit=200`)
      .then((r) => r.ok ? r.json() : { messages: [] })
      .then((data) => {
        const hydrated: LogEntry[] = (
          data.messages as Array<{
            id: number; ts: string; source: "sign" | "speech";
            text: string; confidence: number | null;
          }>
        ).map((m) => ({
          id: m.id,
          source: m.source,
          text: m.text,
          confidence: m.confidence ?? undefined,
          ts: new Date(m.ts).getTime(),
        }));
        setLog(hydrated);
      })
      .catch((err) => console.warn("[RoomPage] transcript fetch failed:", err));
  }, [roomId]);

  useEffect(() => {
    if (captions.length === 0) return;
    const latest = captions[captions.length - 1];
    setLog((prev) => {
      if (prev.length > 0 && prev[prev.length - 1].id === latest.id) return prev;
      return [...prev, {
        id: latest.id,
        source: latest.source,
        text: latest.text,
        confidence: latest.confidence,
        ts: latest.ts,
      }];
    });
  }, [captions]);

  const handlePrediction = useCallback(() => {}, []);

  const handleSpeech = useCallback((text: string, ts: number) => {
    emitSpeech(text);
    setLog((prev) => [
      ...prev,
      { id: ts + Math.random(), source: "speech", text, ts },
    ]);
  }, [emitSpeech]);

  return (
    <div style={styles.shell}>
      <header style={styles.header}>
        <span style={styles.brand}>◐◑ SignLearn</span>
        <span style={styles.room}>Room <strong>{roomId}</strong></span>
        <span style={styles.you}>You: {name} ({role})</span>
        <span
          style={{ ...styles.dot, background: statusColor(status) }}
          aria-hidden="true"
        />
        <span style={styles.statusLabel} aria-live="polite">{statusLabel(status)}</span>
        <ThemeToggle compact />
        <button onClick={onLeave} style={styles.leave}>Leave</button>
      </header>

      {joinError && (
        <div style={styles.banner} role="alert">
          {joinError}
          <button onClick={onLeave} style={{ ...styles.leave, marginLeft: "0.75rem" }}>Back</button>
        </div>
      )}

      <main style={styles.main}>
        <RoomErrorBoundary onLeave={onLeave}>
          {role === "signer" ? (
            <SignerView
              socket={socket}
              captions={captions}
              peerPresent={peerPresent}
              roomId={roomId}
              onPrediction={handlePrediction}
            />
          ) : (
            <HearingView
              socket={socket}
              captions={captions}
              peerPresent={peerPresent}
              onSpeech={handleSpeech}
            />
          )}
        </RoomErrorBoundary>
      </main>

      <ConversationLog entries={log} roomId={roomId} />
    </div>
  );
}

function statusColor(s: string) {
  return s === "connected" ? "var(--success)" : s === "reconnecting" ? "#ff9800" : "var(--danger)";
}
function statusLabel(s: string) {
  return s === "connected" ? "Connected" : s === "reconnecting" ? "Reconnecting…" : "Disconnected";
}

const styles: Record<string, React.CSSProperties> = {
  shell: { minHeight: "100svh", display: "flex", flexDirection: "column" },
  header: {
    display: "flex", alignItems: "center", gap: "0.75rem",
    padding: "0.6rem 1rem", borderBottom: "1px solid var(--border)",
    background: "var(--bg-elevated)", color: "var(--text)", flexWrap: "wrap",
  },
  brand: { fontWeight: 700, color: "var(--accent)" },
  room: { color: "var(--text-muted)", fontSize: "0.9rem" },
  you: { color: "var(--text-faint)", fontSize: "0.85rem", marginRight: "auto" },
  dot: { width: 10, height: 10, borderRadius: "50%", display: "inline-block", flexShrink: 0 },
  statusLabel: { fontSize: "0.85rem", color: "var(--text-muted)" },
  leave: {
    padding: "0.4rem 0.85rem", borderRadius: 6, border: "1px solid var(--border)",
    background: "transparent", color: "var(--text-muted)", cursor: "pointer",
    fontFamily: "inherit", fontSize: "0.88rem",
  },
  banner: {
    padding: "0.6rem 1rem", background: "var(--danger)", color: "#fff",
    display: "flex", alignItems: "center", fontSize: "0.9rem",
  },
  main: { flex: 1, padding: "1rem", display: "flex", flexDirection: "column" },
};
