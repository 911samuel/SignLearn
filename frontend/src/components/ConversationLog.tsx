import { useEffect, useRef } from "react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:5001";

export interface LogEntry {
  id: number;
  source: "sign" | "speech";
  text: string;
  confidence?: number;
  ts: number;
}

interface ConversationLogProps {
  entries: LogEntry[];
}

export function ConversationLog({ entries }: ConversationLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  async function exportTranscript() {
    let lines: string;
    try {
      const res = await fetch(`${BACKEND_URL}/transcript?limit=1000`);
      if (!res.ok) throw new Error("fetch failed");
      const data = await res.json();
      lines = (data.messages as Array<{ ts: string; source: string; text: string; confidence: number | null }>)
        .map((m) => {
          const time = new Date(m.ts).toLocaleTimeString([], {
            hour: "2-digit", minute: "2-digit", second: "2-digit",
          });
          const conf = m.confidence != null ? ` (${(m.confidence * 100).toFixed(0)}%)` : "";
          return `[${time}] [${m.source.toUpperCase()}] ${m.text}${conf}`;
        })
        .join("\n");
    } catch {
      // Fallback to local entries when backend is unreachable
      lines = entries
        .map((e) => {
          const time = new Date(e.ts).toLocaleTimeString([], {
            hour: "2-digit", minute: "2-digit", second: "2-digit",
          });
          const conf = e.confidence != null ? ` (${(e.confidence * 100).toFixed(0)}%)` : "";
          return `[${time}] [${e.source.toUpperCase()}] ${e.text}${conf}`;
        })
        .join("\n");
    }

    if (!lines) return;
    const blob = new Blob([lines], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `signlearn-transcript-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (entries.length === 0) return null;

  return (
    <div className="conv-log">
      <div className="conv-log-header">
        <span className="conv-log-heading">Conversation</span>
        <button
          className="conv-log-export"
          onClick={exportTranscript}
          aria-label="Export transcript"
        >
          ⬇ Export
        </button>
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
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
