"use client";

import { useEffect, useRef, useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5001";

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

export function ConversationLog({ entries, roomId }: ConversationLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  // Close dropdown when clicking outside.
  useEffect(() => {
    if (!menuOpen) return;
    const handler = () => setMenuOpen(false);
    window.addEventListener("click", handler, { capture: true, once: true });
    return () => window.removeEventListener("click", handler, { capture: true });
  }, [menuOpen]);

  async function fetchMessages() {
    try {
      const res = await fetch(`${BACKEND_URL}/transcript?room_id=${roomId}&limit=1000`);
      if (!res.ok) throw new Error("fetch failed");
      const data = await res.json();
      return data.messages as Array<{
        ts: string; source: string; text: string; confidence: number | null;
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

  function fmtTime(iso: string) {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function toTxt(msgs: Array<{ ts: string; source: string; text: string; confidence: number | null }>) {
    return msgs.map((m) => {
      const conf = m.confidence != null ? ` (${(m.confidence * 100).toFixed(0)}%)` : "";
      return `[${fmtTime(m.ts)}] [${m.source.toUpperCase()}] ${m.text}${conf}`;
    }).join("\n");
  }

  function toMd(msgs: Array<{ ts: string; source: string; text: string; confidence: number | null }>) {
    const header = `# SignLearn Transcript\nRoom: ${roomId}  \nExported: ${new Date().toLocaleString()}\n\n---\n\n`;
    const body = msgs
      .map((m) => {
        const speaker = m.source === "sign" ? "🤟 Signer" : "🗣 Hearing";
        const conf = m.confidence != null ? ` *(${(m.confidence * 100).toFixed(0)}%)*` : "";
        return `**${speaker}** — ${fmtTime(m.ts)}${conf}\n> ${m.text}`;
      }).join("\n\n");
    return header + body;
  }

  function toCsv(msgs: Array<{ ts: string; source: string; text: string; confidence: number | null }>) {
    const header = "timestamp,source,text,confidence\n";
    const rows = msgs.map((m) => {
      const conf = m.confidence != null ? m.confidence.toFixed(4) : "";
      const text = `"${m.text.replace(/"/g, '""')}"`;
      return `${m.ts},${m.source},${text},${conf}`;
    }).join("\n");
    return header + rows;
  }

  async function doExport(format: ExportFormat) {
    setMenuOpen(false);
    const msgs = await fetchMessages();
    if (!msgs.length) return;

    let content: string;
    let mime: string;
    let ext: string;

    if (format === "md") {
      content = toMd(msgs as never);
      mime = "text/markdown";
      ext = "md";
    } else if (format === "csv") {
      content = toCsv(msgs);
      mime = "text/csv";
      ext = "csv";
    } else {
      content = toTxt(msgs);
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
    <div className="conv-log">
      <div className="conv-log-header">
        <span className="conv-log-heading">Conversation</span>
        <div style={{ position: "relative" }}>
          <button
            className="conv-log-export"
            onClick={(e) => { e.stopPropagation(); setMenuOpen((v) => !v); }}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-label="Export transcript"
          >
            ⬇ Export ▾
          </button>
          {menuOpen && (
            <div
              role="menu"
              style={menuStyles.dropdown}
              onClick={(e) => e.stopPropagation()}
            >
              {(["txt", "md", "csv"] as ExportFormat[]).map((fmt) => (
                <button
                  key={fmt}
                  role="menuitem"
                  onClick={() => doExport(fmt)}
                  style={menuStyles.item}
                >
                  {fmt === "txt" && "Plain text (.txt)"}
                  {fmt === "md" && "Markdown (.md)"}
                  {fmt === "csv" && "Spreadsheet (.csv)"}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="conv-log-entries" role="log" aria-live="polite">
        {entries.map((entry) => (
          <div key={entry.id} className={`conv-entry conv-entry--${entry.source}`}>
            <span className="conv-entry-icon" aria-hidden="true">
              {entry.source === "sign" ? "🤟" : "🗣"}
            </span>
            <span className="conv-entry-text">{entry.text}</span>
            {entry.confidence != null && (
              <span className="conv-entry-conf">
                {(entry.confidence * 100).toFixed(0)}%
              </span>
            )}
            <span className="conv-entry-time">
              {new Date(entry.ts).toLocaleTimeString([], {
                hour: "2-digit", minute: "2-digit", second: "2-digit",
              })}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

const menuStyles: Record<string, React.CSSProperties> = {
  dropdown: {
    position: "absolute",
    right: 0,
    top: "calc(100% + 4px)",
    background: "var(--bg-card)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    boxShadow: "0 6px 20px rgba(0,0,0,0.4)",
    display: "flex",
    flexDirection: "column",
    minWidth: 180,
    zIndex: 50,
    overflow: "hidden",
  },
  item: {
    padding: "0.55rem 0.85rem",
    background: "transparent",
    border: "none",
    textAlign: "left",
    color: "var(--text)",
    fontSize: "0.85rem",
    cursor: "pointer",
    fontFamily: "inherit",
  },
};
