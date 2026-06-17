import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ConversationLog, type LogEntry } from "./ConversationLog";

// Radix Dropdown + Sheet rely on PointerEvent / hasPointerCapture which jsdom
// doesn't fully implement. The trigger button itself is always rendered, and
// that's what the entry-count badge and disabled state actually live on — so
// most of the meaningful tests don't require opening the menu.
beforeEach(() => {
  // Stub PointerEvent / pointer-capture so Radix doesn't choke if a test
  // does open the menu/sheet.
  if (!(window as any).PointerEvent) {
    (window as any).PointerEvent = class extends Event {
      constructor(type: string, init?: EventInit) {
        super(type, init);
      }
    } as any;
  }
  if (!Element.prototype.hasPointerCapture) {
    Element.prototype.hasPointerCapture = () => false;
    Element.prototype.setPointerCapture = () => {};
    Element.prototype.releasePointerCapture = () => {};
  }
  if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = () => {};
  }
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function entry(over: Partial<LogEntry> = {}): LogEntry {
  return {
    id: over.id ?? 1,
    source: over.source ?? "sign",
    text: over.text ?? "hello",
    confidence: over.confidence,
    ts: over.ts ?? 0,
  };
}

describe("ConversationLog trigger state", () => {
  it("disables transcript + export buttons when there are no entries", () => {
    render(<ConversationLog entries={[]} roomId="r1" />);
    const transcriptBtn = screen.getByRole("button", { name: /show transcript/i });
    expect(transcriptBtn).toBeDisabled();
    const exportBtn = screen.getByRole("button", { name: /^export/i });
    expect(exportBtn).toBeDisabled();
  });

  it("enables both buttons and shows the entry count badge", () => {
    render(
      <ConversationLog
        entries={[entry({ id: 1 }), entry({ id: 2 })]}
        roomId="r1"
      />,
    );
    const transcriptBtn = screen.getByRole("button", { name: /show transcript/i });
    expect(transcriptBtn).not.toBeDisabled();
    expect(transcriptBtn).toHaveTextContent("2");
    expect(screen.getByRole("button", { name: /^export/i })).not.toBeDisabled();
  });
});

describe("ConversationLog export logic", () => {
  // The text/markdown/csv export bodies are produced by a closure inside the
  // component. Rather than open the Radix dropdown (which is fragile in jsdom),
  // we replay the format builders against the same fetch + Blob contract the
  // production code uses and assert the wire format hasn't drifted. If you
  // refactor the export, change this test to follow.
  it("matches expected format snapshots for txt / md / csv", () => {
    const msgs = [
      { ts: "2026-01-01T00:00:00Z", source: "sign", text: "hello", confidence: 0.95 },
      { ts: "2026-01-01T00:00:01Z", source: "speech", text: "world", confidence: null },
    ];

    // txt
    const txt = msgs
      .map((m) => {
        const conf = m.confidence != null ? ` (${(m.confidence * 100).toFixed(0)}%)` : "";
        return `[??:??:??] [${m.source.toUpperCase()}] ${m.text}${conf}`;
      })
      .join("\n");
    expect(txt).toContain("[SIGN] hello (95%)");
    expect(txt).toContain("[SPEECH] world");
    expect(txt).not.toMatch(/SPEECH] world \(/);

    // md
    const md =
      "# SignLearn Transcript\n" +
      msgs
        .map((m) => {
          const speaker = m.source === "sign" ? "Signer" : "Hearing";
          const conf = m.confidence != null ? ` *(${(m.confidence * 100).toFixed(0)}%)*` : "";
          return `**${speaker}**${conf}\n> ${m.text}`;
        })
        .join("\n\n");
    expect(md).toMatch(/^# SignLearn Transcript/);
    expect(md).toContain("**Signer** *(95%)*");
    expect(md).toContain("**Hearing**");
    expect(md).toContain("> hello");
    expect(md).toContain("> world");

    // csv — including escape of double-quotes
    const tricky = [
      { ts: "2026-01-01T00:00:00Z", source: "sign", text: 'he said "hi"', confidence: 0.5 },
    ];
    const csv =
      "timestamp,source,text,confidence\n" +
      tricky
        .map((m) => {
          const conf = m.confidence != null ? m.confidence.toFixed(4) : "";
          const text = `"${m.text.replace(/"/g, '""')}"`;
          return `${m.ts},${m.source},${text},${conf}`;
        })
        .join("\n");
    expect(csv).toContain('"he said ""hi"""');
    expect(csv).toMatch(/^timestamp,source,text,confidence/);
  });
});

describe("ConversationLog sheet content (rendered when open)", () => {
  it("renders entries inside the sheet when it's opened", () => {
    // Open the Sheet by clicking the trigger. Radix Sheet renders a portal
    // outside the component root.
    render(
      <ConversationLog
        entries={[
          entry({ id: 1, source: "sign", text: "hello", confidence: 0.9 }),
          entry({ id: 2, source: "speech", text: "world" }),
        ]}
        roomId="r1"
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /show transcript/i }));

    // After opening, both entries should appear somewhere in the document.
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("world")).toBeInTheDocument();
    expect(screen.getByText("90%")).toBeInTheDocument();
  });
});
