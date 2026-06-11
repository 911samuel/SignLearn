"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { QRCodeSVG } from "qrcode.react";
import { ArrowRight, ArrowLeft, Camera, Check, Copy, Hand, Mic, ShieldCheck } from "lucide-react";
import type { Role } from "@/hooks/useRoom";
import { usePreferences } from "@/lib/preferences";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { BACKEND_URL } from "@/lib/api";
import { cn } from "@/lib/utils";

interface RoomInfo {
  exists: boolean;
  members: { role: Role; name: string }[];
}

export default function JoinPage() {
  const params = useParams();
  const roomId = (params?.roomId as string) ?? "";
  const router = useRouter();
  const { prefs, update, hydrated } = usePreferences();
  const [info, setInfo] = useState<RoomInfo | null>(null);
  const [name, setName] = useState("");
  const [role, setRole] = useState<Role | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [joinUrl, setJoinUrl] = useState(`/r/${roomId}/join`);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setJoinUrl(`${window.location.origin}/r/${roomId}/join`);
    }
  }, [roomId]);

  useEffect(() => {
    if (hydrated && prefs.name && !name) setName(prefs.name);
  }, [hydrated, prefs.name, name]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${BACKEND_URL}/rooms/${roomId}`)
      .then(async (r) => {
        if (r.status === 404) return { exists: false, members: [] } as RoomInfo;
        return (await r.json()) as RoomInfo;
      })
      .then((data) => { if (!cancelled) setInfo(data); })
      .catch(() => { if (!cancelled) setError("Couldn't reach the SignLearn server."); });
    return () => { cancelled = true; };
  }, [roomId]);

  if (info && !info.exists) {
    return (
      <main className="grid min-h-screen place-items-center bg-[var(--color-bg)] p-4">
        <Card className="w-full max-w-md p-8 text-center">
          <h1 className="heading-h2 text-[var(--color-text)]">Room not found</h1>
          <p className="mt-2 text-[var(--color-text-muted)]">
            No active room with code <code className="rounded bg-[var(--color-surface-sunken)] px-1.5 py-0.5 font-mono text-[var(--color-text)]">{roomId}</code>.
          </p>
          <Button asChild className="mt-6">
            <Link href="/"><ArrowLeft className="size-4" aria-hidden /> Back to home</Link>
          </Button>
        </Card>
      </main>
    );
  }

  const takenRoles = new Set(info?.members.map((m) => m.role));
  const canSubmit = !!role && name.trim().length > 0;

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    update({ name: name.trim() });
    router.push(`/r/${roomId}?role=${role}&name=${encodeURIComponent(name.trim())}`);
  }

  async function copyJoinUrl() {
    try {
      await navigator.clipboard.writeText(joinUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard API blocked */
    }
  }

  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="grid min-h-screen place-items-center bg-[var(--color-bg)] p-4 focus:outline-none"
    >
      <Card className="w-full max-w-2xl overflow-hidden">
        <div className="border-b border-[var(--color-border)] p-6 md:p-8">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="eyebrow">Room invitation</p>
              <h1 className="mt-1 heading-h2 text-[var(--color-text)]">
                Join room{" "}
                <code className="rounded-[var(--radius-sm)] bg-[var(--color-surface-sunken)] px-2 py-1 font-mono text-[0.85em] text-[var(--color-text)]">
                  {roomId}
                </code>
              </h1>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                Share the link or QR with the other participant.
              </p>
            </div>
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            >
              <ArrowLeft className="size-4" aria-hidden /> Exit
            </Link>
          </div>

          <div className="mt-5 grid gap-5 sm:grid-cols-[1fr_auto] sm:items-center">
            <div className="flex items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-sunken)] p-2 pr-1">
              <code className="flex-1 truncate px-2 font-mono text-xs text-[var(--color-text-muted)]">
                {joinUrl}
              </code>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={copyJoinUrl}
                aria-live="polite"
              >
                {copied ? (
                  <>
                    <Check className="size-4" aria-hidden /> Copied
                  </>
                ) : (
                  <>
                    <Copy className="size-4" aria-hidden /> Copy link
                  </>
                )}
              </Button>
            </div>
            <div
              className="hidden rounded-[var(--radius-md)] border border-[var(--color-border)] bg-white p-2 sm:block"
              aria-hidden
              title="QR code — scan to join on another device"
            >
              <QRCodeSVG value={joinUrl} size={84} bgColor="#ffffff" fgColor="#0f172a" includeMargin={false} />
            </div>
          </div>
        </div>

        <form onSubmit={submit} className="space-y-7 p-6 md:p-8">
          <div className="space-y-2">
            <Label htmlFor="join-name">Your display name</Label>
            <Input
              id="join-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Alex"
              maxLength={40}
              autoComplete="nickname"
              autoFocus={!prefs.name}
            />
          </div>

          <fieldset className="space-y-3">
            <legend className="text-sm font-semibold text-[var(--color-text)]">
              Your role in this conversation
            </legend>
            <div className="grid gap-3 sm:grid-cols-2">
              <RoleCard
                icon={Hand}
                title="Signer"
                description="I will sign — uses my camera."
                selected={role === "signer"}
                disabled={takenRoles.has("signer")}
                onClick={() => setRole("signer")}
              />
              <RoleCard
                icon={Mic}
                title="Hearing partner"
                description="I will speak — uses my microphone."
                selected={role === "hearing"}
                disabled={takenRoles.has("hearing")}
                onClick={() => setRole("hearing")}
              />
            </div>
            {role && (
              <Alert tone="info" className="mt-2">
                {role === "signer" ? (
                  <span className="inline-flex items-start gap-1.5">
                    <Camera className="mt-0.5 size-4 shrink-0" aria-hidden />
                    Your camera will be used to recognise your signs. Your video stays on this device — only landmark numbers are sent to the recogniser.
                  </span>
                ) : (
                  <span className="inline-flex items-start gap-1.5">
                    <Mic className="mt-0.5 size-4 shrink-0" aria-hidden />
                    Your microphone will be used for speech-to-text captions. Audio is not recorded.
                  </span>
                )}
              </Alert>
            )}
          </fieldset>

          <div className="flex flex-wrap items-center justify-between gap-3">
            <Label className="m-0">Text size</Label>
            <ToggleGroup
              type="single"
              value={prefs.textSize}
              onValueChange={(v) => v && update({ textSize: v as typeof prefs.textSize })}
              aria-label="Text size"
            >
              <ToggleGroupItem value="normal">A</ToggleGroupItem>
              <ToggleGroupItem value="large" className="text-base">A</ToggleGroupItem>
              <ToggleGroupItem value="xlarge" className="text-lg">A</ToggleGroupItem>
            </ToggleGroup>
          </div>

          {error && <Alert tone="danger" title="Network problem">{error}</Alert>}

          <Button type="submit" size="lg" disabled={!canSubmit} className="w-full">
            Enter room
            <ArrowRight className="ml-1 size-4" aria-hidden />
          </Button>

          <p className="inline-flex items-center justify-center gap-1.5 text-center text-xs text-[var(--color-text-muted)] w-full">
            <ShieldCheck className="size-3.5" aria-hidden />
            <Link href="/privacy" className="underline-offset-2 hover:underline">
              How we use your camera and microphone
            </Link>
          </p>
        </form>
      </Card>
    </main>
  );
}

function RoleCard({
  icon: Icon,
  title,
  description,
  selected,
  disabled,
  onClick,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  selected: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={cn(
        "relative flex h-full flex-col items-start gap-2 rounded-[var(--radius-lg)] border-2 p-4 text-left transition focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)] disabled:cursor-not-allowed disabled:opacity-50",
        selected
          ? "border-[var(--color-brand)] bg-[var(--color-brand-subtle)] text-[var(--color-text)]"
          : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-sunken)]",
      )}
    >
      <div
        className={cn(
          "flex h-10 w-10 items-center justify-center rounded-[var(--radius-md)]",
          selected
            ? "bg-[var(--color-brand)] text-[var(--color-brand-foreground)]"
            : "bg-[var(--color-surface-sunken)] text-[var(--color-text)]",
        )}
        aria-hidden
      >
        <Icon className="size-5" />
      </div>
      <div>
        <p className="font-semibold text-[var(--color-text)]">
          {title}
          {disabled && (
            <Badge tone="neutral" className="ml-2 text-[0.65rem]">
              Taken
            </Badge>
          )}
        </p>
        <p className="text-sm text-[var(--color-text-muted)]">{description}</p>
      </div>
      {selected && (
        <Check className="absolute right-3 top-3 size-5 text-[var(--color-brand)]" aria-hidden />
      )}
    </button>
  );
}
