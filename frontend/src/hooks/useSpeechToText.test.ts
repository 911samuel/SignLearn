import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useSpeechToText } from "./useSpeechToText";

class FakeRecognition {
  static instances: FakeRecognition[] = [];
  continuous = false;
  interimResults = false;
  lang = "";
  onresult: ((e: unknown) => void) | null = null;
  onend: (() => void) | null = null;
  start = vi.fn();
  stop = vi.fn(() => this.onend?.());
  constructor() {
    FakeRecognition.instances.push(this);
  }
  fireFinal(transcripts: string[]) {
    this.onresult?.({
      resultIndex: 0,
      results: transcripts.map((t) => {
        const arr: any = [{ transcript: t }];
        arr.isFinal = true;
        return arr;
      }),
    });
  }
}

beforeEach(() => {
  FakeRecognition.instances = [];
  (window as any).SpeechRecognition = FakeRecognition;
});

afterEach(() => {
  delete (window as any).SpeechRecognition;
  delete (window as any).webkitSpeechRecognition;
});

describe("useSpeechToText", () => {
  it("reports unsupported when the API is missing", () => {
    delete (window as any).SpeechRecognition;
    const { result } = renderHook(() => useSpeechToText());
    expect(result.current.supported).toBe(false);
  });

  it("reports supported and toggles listening through start/stop", () => {
    const { result } = renderHook(() => useSpeechToText());
    expect(result.current.supported).toBe(true);
    expect(result.current.listening).toBe(false);

    act(() => result.current.start());
    expect(result.current.listening).toBe(true);
    expect(FakeRecognition.instances).toHaveLength(1);
    expect(FakeRecognition.instances[0].start).toHaveBeenCalledOnce();
    expect(FakeRecognition.instances[0].continuous).toBe(true);

    act(() => result.current.stop());
    expect(result.current.listening).toBe(false);
  });

  it("does not start a second recognition while already listening", () => {
    const { result } = renderHook(() => useSpeechToText());
    act(() => result.current.start());
    act(() => result.current.start());
    expect(FakeRecognition.instances).toHaveLength(1);
  });

  it("appends finalized transcripts and invokes the onResult callback", () => {
    const onResult = vi.fn();
    const { result } = renderHook(() => useSpeechToText(onResult));
    act(() => result.current.start());
    act(() => FakeRecognition.instances[0].fireFinal(["hello world  "]));

    expect(result.current.transcript).toHaveLength(1);
    expect(result.current.transcript[0].text).toBe("hello world");
    expect(onResult).toHaveBeenCalledOnce();
    expect(onResult.mock.calls[0][0]).toBe("hello world");
  });

  it("ignores empty transcript strings", () => {
    const onResult = vi.fn();
    const { result } = renderHook(() => useSpeechToText(onResult));
    act(() => result.current.start());
    act(() => FakeRecognition.instances[0].fireFinal(["   "]));
    expect(result.current.transcript).toHaveLength(0);
    expect(onResult).not.toHaveBeenCalled();
  });

  it("clear() empties the transcript", () => {
    const { result } = renderHook(() => useSpeechToText());
    act(() => result.current.start());
    act(() => FakeRecognition.instances[0].fireFinal(["a", "b"]));
    expect(result.current.transcript).toHaveLength(2);
    act(() => result.current.clear());
    expect(result.current.transcript).toHaveLength(0);
  });

  it("falls back to webkitSpeechRecognition when SpeechRecognition is absent", () => {
    delete (window as any).SpeechRecognition;
    (window as any).webkitSpeechRecognition = FakeRecognition;
    const { result } = renderHook(() => useSpeechToText());
    expect(result.current.supported).toBe(true);
    act(() => result.current.start());
    expect(FakeRecognition.instances).toHaveLength(1);
  });
});
