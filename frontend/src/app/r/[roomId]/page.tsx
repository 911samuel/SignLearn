"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Copy, DoorOpen, Hand, Mic } from "lucide-react";
import { useRoom, type Role } from "@/hooks/useRoom";
import { SignerView } from "@/components/SignerView";
import { HearingView } from "@/components/HearingView";
import { ConversationLog, type LogEntry } from "@/components/ConversationLog";
import { RoomErrorBoundary } from "@/components/RoomErrorBoundary";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert } from "@/components/ui/alert";
import { StatusPill, type Status } from "@/components/primitives/StatusPill";
import { A11yPreferencesMenu } from "@/components/primitives/A11yPreferencesMenu";
import { useToast } from "@/components/ui/toast";
import { BACKEND_URL } from "@/lib/api";

export default function RoomPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const roomId = (params?.roomId as string) ?? "";

  const role = searchParams?.get("role") as Role | null;
  const name = searchParams?.get("name") ?? "";

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

function RoomInner({
  roomId,
  role,
  name,
  onLeave,
}: {
  roomId: string;
  role: Role;
  name: string;
  onLeave: () => void;
}) {
  const { socket, status, joinError, members, captions, emitSpeech } = useRoom(roomId, role, name);
  const [log, setLog] = useState<LogEntry[]>([]);
  const { toast } = useToast();
  const peerPresent = members.some((m) => m.role !== role);

  useEffect(() => {
    fetch(`${BACKEND_URL}/transcript?room_id=${roomId}&limit=200`)
      .then((r) => (r.ok ? r.json() : { messages: [] }))
      .then((data) => {
        const hydrated: LogEntry[] = (
          data.messages as Array<{
            id: number;
            ts: string;
            source: "sign" | "speech";
            text: string;
            confidence: number | null;
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
      .catch(() => {});
  }, [roomId]);

  useEffect(() => {
    if (captions.length === 0) return;
    const latest = captions[captions.length - 1];
    setLog((prev) => {
      if (prev.length > 0 && prev[prev.length - 1].id === latest.id) return prev;
      return [
        ...prev,
        {
          id: latest.id,
          source: latest.source,
          text: latest.text,
          confidence: latest.confidence,
          ts: latest.ts,
        },
      ];
    });
  }, [captions]);

  const handlePrediction = useCallback(() => {}, []);

  const handleSpeech = useCallback(
    (text: string, ts: number) => {
      emitSpeech(text);
      setLog((prev) => [...prev, { id: ts + Math.random(), source: "speech", text, ts }]);
    },
    [emitSpeech],
  );

  // A WebSocket "connected" status without a successful room join is
  // misleading — the user can't actually do anything in the room.  Treat
  // join failure as disconnected so the header badge doesn't contradict the
  // red "Room does not exist" banner below.
  const connectionStatus: Status = joinError
    ? "disconnected"
    : status === "connected"
      ? "ok"
      : status === "reconnecting"
        ? "processing"
        : "disconnected";

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(roomId);
      toast({ tone: "success", title: "Room code copied", description: roomId });
    } catch {
      toast({ tone: "danger", title: "Couldn't copy", description: "Copy the code manually." });
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-[var(--color-bg)] text-[var(--color-text)]">
      {/* HEADER */}
      <header className="sticky top-0 z-30 border-b border-[var(--color-border)] bg-[color-mix(in_srgb,var(--color-bg)_92%,transparent)] backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl flex-wrap items-center gap-3 px-4 lg:px-6">
          <Link
            href="/"
            aria-label="SignLearn home"
            className="inline-flex items-center gap-2 text-[var(--color-text)] hover:no-underline"
          >
            <span
              aria-hidden
              className="flex h-8 w-8 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand)] text-[var(--color-brand-foreground)]"
            >
              SL
            </span>
          </Link>

          <button
            type="button"
            onClick={copyCode}
            className="inline-flex h-9 items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-sunken)] px-3 text-sm font-semibold text-[var(--color-text)] transition hover:bg-[var(--color-surface)] focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]"
            aria-label={`Copy room code ${roomId}`}
          >
            <span className="text-xs uppercase tracking-wider text-[var(--color-text-muted)]">Room</span>
            <code className="font-mono text-base text-[var(--color-text)]">{roomId}</code>
            <Copy className="size-3.5 text-[var(--color-text-muted)]" aria-hidden />
          </button>

          <Badge tone={role === "signer" ? "brand" : "info"} className="text-xs">
            {role === "signer" ? (
              <>
                <Hand className="size-3" aria-hidden /> Signer
              </>
            ) : (
              <>
                <Mic className="size-3" aria-hidden /> Hearing
              </>
            )}{" "}
            · {name}
          </Badge>

          <div className="ml-auto flex items-center gap-2">
            <StatusPill status={connectionStatus} />
            <A11yPreferencesMenu />
            <Button variant="outline" size="sm" onClick={onLeave}>
              <DoorOpen className="size-4" aria-hidden /> Leave
            </Button>
          </div>
        </div>
      </header>

      {joinError && (
        <Alert tone="danger" title="Couldn't join this room" className="mx-4 mt-4 lg:mx-6">
          {joinError}
          <div className="mt-3">
            <Button variant="secondary" size="sm" onClick={onLeave}>
              <DoorOpen className="size-4" aria-hidden /> Back to home
            </Button>
          </div>
        </Alert>
      )}

      <main
        id="main-content"
        tabIndex={-1}
        className="flex flex-1 flex-col gap-4 px-4 py-4 focus:outline-none lg:px-6"
      >
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
