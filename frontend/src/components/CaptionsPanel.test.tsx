import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { CaptionsPanel } from "./CaptionsPanel";
import type { Caption } from "@/hooks/useRoom";

function caption(over: Partial<Caption> = {}): Caption {
  return {
    id: over.id ?? 1,
    source: over.source ?? "sign",
    text: over.text ?? "hello",
    name: over.name ?? "Sam",
    confidence: over.confidence,
    ts: over.ts ?? 0,
  };
}

describe("CaptionsPanel", () => {
  it("shows the default hint when there are no captions", () => {
    render(<CaptionsPanel captions={[]} />);
    expect(screen.getByRole("log")).toHaveTextContent(
      /Captions will appear here as you sign or speak\./,
    );
  });

  it("shows a custom emptyHint when provided", () => {
    render(<CaptionsPanel captions={[]} emptyHint="Nothing yet" />);
    expect(screen.getByRole("log")).toHaveTextContent("Nothing yet");
  });

  it("renders captions newest-last and shows speaker + text", () => {
    render(
      <CaptionsPanel
        captions={[
          caption({ id: 1, source: "sign", text: "hello", name: "Sam" }),
          caption({ id: 2, source: "speech", text: "hi back", name: "Lee" }),
        ]}
      />,
    );
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("hi back")).toBeInTheDocument();
    expect(screen.getByText("Sam")).toBeInTheDocument();
    expect(screen.getByText("Lee")).toBeInTheDocument();
  });

  it("only shows the last 6 captions", () => {
    const captions: Caption[] = Array.from({ length: 8 }, (_, i) =>
      caption({ id: i + 1, text: `cap-${i + 1}` }),
    );
    render(<CaptionsPanel captions={captions} />);
    expect(screen.queryByText("cap-1")).toBeNull();
    expect(screen.queryByText("cap-2")).toBeNull();
    expect(screen.getByText("cap-3")).toBeInTheDocument();
    expect(screen.getByText("cap-8")).toBeInTheDocument();
  });

  it("filters by source when filter='sign'", () => {
    render(
      <CaptionsPanel
        filter="sign"
        captions={[
          caption({ id: 1, source: "sign", text: "signed" }),
          caption({ id: 2, source: "speech", text: "spoken" }),
        ]}
      />,
    );
    expect(screen.getByText("signed")).toBeInTheDocument();
    expect(screen.queryByText("spoken")).toBeNull();
  });

  it("renders rounded confidence percentage when present", () => {
    render(
      <CaptionsPanel
        captions={[caption({ id: 1, text: "ok", confidence: 0.876 })]}
      />,
    );
    expect(screen.getByText("88%")).toBeInTheDocument();
  });

  it("omits the confidence node when confidence is undefined", () => {
    render(<CaptionsPanel captions={[caption({ id: 1, text: "no conf" })]} />);
    expect(screen.queryByText(/%/)).toBeNull();
  });

  it("falls back to a role label when name is missing", () => {
    render(
      <CaptionsPanel
        captions={[
          { id: 1, source: "sign", text: "x", ts: 0, name: undefined as unknown as string },
          { id: 2, source: "speech", text: "y", ts: 0, name: undefined as unknown as string },
        ]}
      />,
    );
    expect(screen.getByText("Signer")).toBeInTheDocument();
    expect(screen.getByText("Hearing partner")).toBeInTheDocument();
  });
});
