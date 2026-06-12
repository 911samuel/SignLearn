"use client";

import { useEffect, useRef } from "react";
import { Download, Hand, History, Mic } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BACKEND_URL } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface LogEntry {
  id: number;
  source: "sign" | "speech";
  text: string;
  confidence?: number;
  ts: number;
}

type ExportFormat = "txt" | "md" | "csv";

interface ConversationLogProps {
  entries: LogEntry[];
  roomId: string;
}

function fmtTime(ts: number | string) {
  return new Date(ts).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function ConversationLog({ entries, roomId }: ConversationLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  async function fetchMessages() {
    try {
      const res = await fetch(`${BACKEND_URL}/transcript?room_id=${roomId}&limit=1000`);
      if (!res.ok) throw new Error("fetch failed");
      const data = await res.json();
      return data.messages as Array<{
        ts: string;
        source: string;
        text: string;
        confidence: number | null;
      }>;
    } catch {
      return entries.map((e) => ({
        ts: new Date(e.ts).toISOString(),
        source: e.source,
        text: e.text,
        confidence: e.confidence ?? null,
      }));
    }
  }

  async function doExport(format: ExportFormat) {
    const msgs = await fetchMessages();
    if (!msgs.length) return;

    let content: string;
    let mime: string;
    let ext: string;

    if (format === "md") {
      const header = `# SignLearn Transcript\nRoom: ${roomId}  \nExported: ${new Date().toLocaleString()}\n\n---\n\n`;
      content = header + msgs
        .map((m) => {
          const speaker = m.source === "sign" ? "Signer" : "Hearing";
          const conf = m.confidence != null ? ` *(${(m.confidence * 100).toFixed(0)}%)*` : "";
          return `**${speaker}** — ${fmtTime(m.ts)}${conf}\n> ${m.text}`;
        })
        .join("\n\n");
      mime = "text/markdown";
      ext = "md";
    } else if (format === "csv") {
      const header = "timestamp,source,text,confidence\n";
      content = header + msgs
        .map((m) => {
          const conf = m.confidence != null ? m.confidence.toFixed(4) : "";
          const text = `"${m.text.replace(/"/g, '""')}"`;
          return `${m.ts},${m.source},${text},${conf}`;
        })
        .join("\n");
      mime = "text/csv";
      ext = "csv";
    } else {
      content = msgs
        .map((m) => {
          const conf = m.confidence != null ? ` (${(m.confidence * 100).toFixed(0)}%)` : "";
          return `[${fmtTime(m.ts)}] [${m.source.toUpperCase()}] ${m.text}${conf}`;
        })
        .join("\n");
      mime = "text/plain";
      ext = "txt";
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `signlearn-${roomId}-${Date.now()}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (entries.length === 0) return null;

  return (
    <section
      aria-label="Conversation transcript"
      className="border-t border-[var(--color-border)] bg-[var(--color-surface)]"
    >
      <header className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-2">
        <div className="inline-flex items-center gap-2">
          <History className="size-4 text-[var(--color-text-muted)]" aria-hidden />
          <span className="eyebrow">Conversation</span>
          <Badge tone="neutral" className="text-[0.65rem]">
            {entries.length}
          </Badge>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm">
              <Download className="size-4" aria-hidden /> Export
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Export transcript</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => doExport("txt")}>Plain text (.txt)</DropdownMenuItem>
            <DropdownMenuItem onSelect={() => doExport("md")}>Markdown (.md)</DropdownMenuItem>
            <DropdownMenuItem onSelect={() => doExport("csv")}>Spreadsheet (.csv)</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </header>

      <div
        role="log"
        aria-live="polite"
        className="flex max-h-[200px] flex-col gap-1.5 overflow-y-auto px-4 py-3"
      >
        {entries.map((entry) => {
          const isSign = entry.source === "sign";
          const Icon = isSign ? Hand : Mic;
          return (
            <div
              key={entry.id}
              className={cn(
                "flex items-center gap-2.5 rounded-[var(--radius-sm)] px-2.5 py-1.5",
                isSign
                  ? "bg-[var(--color-brand-subtle)]"
                  : "bg-[var(--color-surface-sunken)]",
              )}
            >
              <Icon
                className={cn(
                  "size-3.5 shrink-0",
                  isSign ? "text-[var(--color-brand)]" : "text-[var(--color-text-muted)]",
                )}
                aria-hidden
              />
              <span className="flex-1 text-sm text-[var(--color-text)]">{entry.text}</span>
              {entry.confidence != null && (
                <span className="font-mono text-xs text-[var(--color-text-muted)]">
                  {(entry.confidence * 100).toFixed(0)}%
                </span>
              )}
              <time className="font-mono text-xs text-[var(--color-text-faint)]">
                {fmtTime(entry.ts)}
              </time>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </section>
  );
}
