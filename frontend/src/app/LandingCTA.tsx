"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, Hand, Lock, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert } from "@/components/ui/alert";
import { BACKEND_URL } from "@/lib/api";

export function LandingCTA() {
  const router = useRouter();
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [showJoin, setShowJoin] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function createRoom() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/rooms`, { method: "POST" });
      if (!res.ok) throw new Error("Could not reach the SignLearn server.");
      const { room_id } = await res.json();
      router.push(`/r/${room_id}/join`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to create room — check your connection and try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  function joinRoom(e: React.FormEvent) {
    e.preventDefault();
    const id = code.trim().toUpperCase();
    if (!id) return;
    router.push(`/r/${id}/join`);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button size="xl" onClick={createRoom} disabled={busy}>
          {busy ? (
            <>
              <Hand className="sl-pulse-soft" aria-hidden /> Creating room…
            </>
          ) : (
            <>
              <MessageSquare aria-hidden /> Start a conversation
              <ArrowRight className="ml-1 size-4" aria-hidden />
            </>
          )}
        </Button>
        <Button
          size="xl"
          variant="secondary"
          onClick={() => setShowJoin((v) => !v)}
          aria-expanded={showJoin}
          aria-controls="join-room-form"
        >
          Join with a code
        </Button>
      </div>

      {showJoin && (
        <form id="join-room-form" onSubmit={joinRoom} className="flex max-w-sm flex-wrap gap-2 sl-fade-up">
          <label htmlFor="room-code" className="sr-only">Room code</label>
          <Input
            id="room-code"
            value={code}
            onChange={(e) => setCode(e.target.value.toUpperCase())}
            placeholder="ABC123"
            maxLength={6}
            autoComplete="off"
            className="flex-1 min-w-0 font-mono tracking-[0.2em] uppercase text-lg"
            aria-describedby="room-code-help"
          />
          <Button type="submit" disabled={code.trim().length === 0}>
            Join
            <ArrowRight className="size-4" aria-hidden />
          </Button>
          <p id="room-code-help" className="basis-full text-xs text-[var(--color-text-muted)]">
            Got a 6-character code from someone? Enter it here to join their room.
          </p>
        </form>
      )}

      {error && (
        <Alert tone="danger" title="Couldn't start a room">
          {error}
        </Alert>
      )}

      <p className="inline-flex items-center gap-1.5 text-xs text-[var(--color-text-muted)]">
        <Lock className="size-3.5" aria-hidden />
        No sign-up required. Your video stays on your device.
      </p>
    </div>
  );
}
