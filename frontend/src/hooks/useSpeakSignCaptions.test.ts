import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useSpeakSignCaptions } from "./useSpeakSignCaptions";
import type { Caption } from "./useRoom";

const speak = vi.fn();

class FakeUtterance {
  rate = 1;
  pitch = 1;
  lang = "";
  text: string;
  constructor(text: string) {
    this.text = text;
  }
}

beforeEach(() => {
  speak.mockReset();
  (window as any).speechSynthesis = { speak };
  (globalThis as any).SpeechSynthesisUtterance = FakeUtterance;
});

afterEach(() => {
  delete (window as any).speechSynthesis;
  delete (globalThis as any).SpeechSynthesisUtterance;
});

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

describe("useSpeakSignCaptions", () => {
  it("does not speak captions present on first render", () => {
    renderHook(({ caps }) => useSpeakSignCaptions(caps), {
      initialProps: { caps: [caption({ id: 1, text: "old" })] },
    });
    expect(speak).not.toHaveBeenCalled();
  });

  it("speaks newly-added sign captions", () => {
    const { rerender } = renderHook(({ caps }) => useSpeakSignCaptions(caps), {
      initialProps: { caps: [] as Caption[] },
    });
    rerender({ caps: [caption({ id: 1, text: "hello" })] });
    expect(speak).toHaveBeenCalledOnce();
    const utter = speak.mock.calls[0][0] as FakeUtterance;
    expect(utter.text).toBe("hello");
    expect(utter.lang).toBe("en-US");
  });

  it("strips trailing ASL-LEX digit suffix before speaking", () => {
    const { rerender } = renderHook(({ caps }) => useSpeakSignCaptions(caps), {
      initialProps: { caps: [] as Caption[] },
    });
    rerender({ caps: [caption({ id: 1, text: "fine1" })] });
    const utter = speak.mock.calls[0][0] as FakeUtterance;
    expect(utter.text).toBe("fine");
  });

  it("ignores speech-source captions", () => {
    const { rerender } = renderHook(({ caps }) => useSpeakSignCaptions(caps), {
      initialProps: { caps: [] as Caption[] },
    });
    rerender({
      caps: [caption({ id: 1, source: "speech", text: "spoken aloud" })],
    });
    expect(speak).not.toHaveBeenCalled();
  });

  it("does not re-speak a caption seen before", () => {
    const { rerender } = renderHook(({ caps }) => useSpeakSignCaptions(caps), {
      initialProps: { caps: [] as Caption[] },
    });
    const c = caption({ id: 1, text: "hi" });
    rerender({ caps: [c] });
    rerender({ caps: [c] });
    rerender({ caps: [c, caption({ id: 2, text: "again" })] });
    expect(speak).toHaveBeenCalledTimes(2);
    expect((speak.mock.calls[0][0] as FakeUtterance).text).toBe("hi");
    expect((speak.mock.calls[1][0] as FakeUtterance).text).toBe("again");
  });

  it("is a no-op when enabled=false", () => {
    const { rerender } = renderHook(
      ({ caps, on }) => useSpeakSignCaptions(caps, on),
      { initialProps: { caps: [] as Caption[], on: false } },
    );
    rerender({ caps: [caption({ id: 1, text: "hello" })], on: false });
    expect(speak).not.toHaveBeenCalled();
  });

  it("skips captions whose text is only a digit (becomes empty after strip)", () => {
    const { rerender } = renderHook(({ caps }) => useSpeakSignCaptions(caps), {
      initialProps: { caps: [] as Caption[] },
    });
    rerender({ caps: [caption({ id: 1, text: "123" })] });
    expect(speak).not.toHaveBeenCalled();
  });

  it("swallows errors from speak() (e.g. autoplay block)", () => {
    speak.mockImplementation(() => {
      throw new Error("autoplay blocked");
    });
    const { rerender } = renderHook(({ caps }) => useSpeakSignCaptions(caps), {
      initialProps: { caps: [] as Caption[] },
    });
    expect(() =>
      rerender({ caps: [caption({ id: 1, text: "boom" })] }),
    ).not.toThrow();
  });
});
