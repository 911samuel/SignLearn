import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { VideoCaptionOverlay } from "./VideoCaptionOverlay";
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

describe("VideoCaptionOverlay", () => {
  it("renders nothing in the body when empty and no hint provided", () => {
    render(<VideoCaptionOverlay captions={[]} filter="sign" />);
    const log = screen.getByRole("log");
    expect(log.textContent?.trim()).toBe("");
  });

  it("renders emptyHint when present and no captions match", () => {
    render(
      <VideoCaptionOverlay captions={[]} filter="sign" emptyHint="Sign something" />,
    );
    expect(screen.getByText("Sign something")).toBeInTheDocument();
  });

  it("only renders captions of the given source", () => {
    render(
      <VideoCaptionOverlay
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

  it("shows only the last two matching captions", () => {
    const caps: Caption[] = [1, 2, 3, 4].map((n) =>
      caption({ id: n, text: `t-${n}` }),
    );
    render(<VideoCaptionOverlay captions={caps} filter="sign" />);
    expect(screen.queryByText("t-1")).toBeNull();
    expect(screen.queryByText("t-2")).toBeNull();
    expect(screen.getByText("t-3")).toBeInTheDocument();
    expect(screen.getByText("t-4")).toBeInTheDocument();
  });

  it("shows confidence percentage when present", () => {
    render(
      <VideoCaptionOverlay
        filter="sign"
        captions={[caption({ id: 1, text: "ok", confidence: 0.42 })]}
      />,
    );
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("omits the confidence node when confidence is undefined", () => {
    render(
      <VideoCaptionOverlay
        filter="sign"
        captions={[caption({ id: 1, text: "no conf" })]}
      />,
    );
    expect(screen.queryByText(/%/)).toBeNull();
  });
});
