import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { createRef, type RefObject } from "react";

// ---------- mediapipe mock ----------
vi.mock("@mediapipe/tasks-vision", () => ({
  FilesetResolver: { forVisionTasks: vi.fn(async () => ({})) },
  HandLandmarker: {
    createFromOptions: vi.fn(async () => ({
      detectForVideo: vi.fn(() => ({ landmarks: [] })),
      close: vi.fn(),
    })),
  },
}));

// ---------- fake socket ----------
type Handler = (...args: unknown[]) => void;

class FakeSocket {
  private handlers = new Map<string, Handler[]>();
  emit = vi.fn();
  on = vi.fn((event: string, fn: Handler) => {
    const list = this.handlers.get(event) ?? [];
    list.push(fn);
    this.handlers.set(event, list);
  });
  off = vi.fn((event: string, fn: Handler) => {
    const list = (this.handlers.get(event) ?? []).filter((h) => h !== fn);
    this.handlers.set(event, list);
  });
  fire(event: string, ...args: unknown[]) {
    for (const fn of this.handlers.get(event) ?? []) fn(...args);
  }
}

import { useSignRecognition } from "./useSignRecognition";

// Used everywhere the hook needs a video ref. Just needs to be a ref —
// the rAF loop bails when videoRef.current is null, so we don't actually
// touch a real <video>.
function nullRef(): RefObject<HTMLVideoElement | null> {
  return createRef<HTMLVideoElement | null>();
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useSignRecognition socket subscriptions", () => {
  it("subscribes to 'word_prediction' and 'prediction' when a socket is present", () => {
    const sock = new FakeSocket();
    renderHook(() => useSignRecognition(nullRef(), sock as any));
    const events = sock.on.mock.calls.map((c) => c[0]);
    expect(events).toContain("word_prediction");
    expect(events).toContain("prediction");
  });

  it("does not subscribe to socket events when socket is null", () => {
    const { result } = renderHook(() => useSignRecognition(nullRef(), null));
    expect(result.current.utterance).toBe("");
  });

  it("cleans up listeners on unmount", () => {
    const sock = new FakeSocket();
    const { unmount } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    unmount();
    const offEvents = sock.off.mock.calls.map((c) => c[0]);
    expect(offEvents).toContain("word_prediction");
    expect(offEvents).toContain("prediction");
  });
});

describe("word_prediction handling", () => {
  it("appends the top word to the utterance and clears capture state", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );

    act(() => {
      sock.fire("word_prediction", {
        top3: [
          { label: "hello", confidence: 0.9 },
          { label: "hi", confidence: 0.5 },
        ],
      });
    });

    expect(result.current.utterance).toBe("hello");
    expect(result.current.wordPrediction?.top3[0].label).toBe("hello");
    expect(result.current.captureStatus).toBe("idle");
    expect(result.current.captureProgress).toBe(0);
  });

  it("appends a second word with a space separator", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );

    act(() => sock.fire("word_prediction", { top3: [{ label: "hello", confidence: 0.9 }] }));
    act(() => sock.fire("word_prediction", { top3: [{ label: "world", confidence: 0.8 }] }));

    expect(result.current.utterance).toBe("hello world");
  });

  it("does not append when the prediction carries an error", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() =>
      sock.fire("word_prediction", { top3: [], error: "model unavailable" }),
    );
    expect(result.current.utterance).toBe("");
    expect(result.current.wordPrediction?.error).toBe("model unavailable");
  });

  it("safely handles an empty top3 array", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => sock.fire("word_prediction", { top3: [] }));
    expect(result.current.utterance).toBe("");
  });
});

describe("letter/digit (prediction) handling", () => {
  it("uppercases a-z letters and appends to an empty utterance", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() =>
      sock.fire("prediction", { label: "s", confidence: 0.99, ready: true }),
    );
    expect(result.current.utterance).toBe("S");
    expect(result.current.prediction.label).toBe("s");
  });

  it("concatenates a fingerspell run without spaces (S, A, M → SAM)", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => sock.fire("prediction", { label: "s", confidence: 1, ready: true }));
    act(() => sock.fire("prediction", { label: "a", confidence: 1, ready: true }));
    act(() => sock.fire("prediction", { label: "m", confidence: 1, ready: true }));
    expect(result.current.utterance).toBe("SAM");
  });

  it("inserts a space when a letter follows a word", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => sock.fire("word_prediction", { top3: [{ label: "hello", confidence: 0.9 }] }));
    act(() => sock.fire("prediction", { label: "s", confidence: 1, ready: true }));
    expect(result.current.utterance).toBe("hello S");
  });

  it("ignores repeats of the same letter", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => sock.fire("prediction", { label: "s", confidence: 1, ready: true }));
    act(() => sock.fire("prediction", { label: "s", confidence: 1, ready: true }));
    act(() => sock.fire("prediction", { label: "s", confidence: 1, ready: true }));
    expect(result.current.utterance).toBe("S");
  });

  it("ignores predictions where ready=false", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() =>
      sock.fire("prediction", { label: "s", confidence: 1, ready: false }),
    );
    expect(result.current.utterance).toBe("");
  });

  it("ignores predictions with a null label", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() =>
      sock.fire("prediction", { label: null, confidence: 0, ready: true }),
    );
    expect(result.current.utterance).toBe("");
  });

  it("keeps non-letter labels (e.g. digits) without uppercasing", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => sock.fire("prediction", { label: "5", confidence: 1, ready: true }));
    expect(result.current.utterance).toBe("5");
  });
});

describe("commitUtterance", () => {
  it("emits 'utterance_complete', flashes sentUtterance briefly, and clears the row", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );

    act(() => sock.fire("word_prediction", { top3: [{ label: "hello", confidence: 0.9 }] }));
    expect(result.current.utterance).toBe("hello");

    act(() => result.current.commitUtterance());
    expect(result.current.sentUtterance).toBe("hello");
    expect(result.current.utterance).toBe("");
    const emit = sock.emit.mock.calls.find((c) => c[0] === "utterance_complete");
    expect(emit).toBeTruthy();
    expect((emit![1] as { text: string }).text).toBe("hello");

    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(result.current.sentUtterance).toBeNull();
  });

  it("is a no-op when the utterance is already empty", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => result.current.commitUtterance());
    expect(sock.emit).not.toHaveBeenCalledWith(
      "utterance_complete",
      expect.anything(),
    );
    expect(result.current.sentUtterance).toBeNull();
  });
});

describe("idle clear timer", () => {
  it("clears the utterance after IDLE_CLEAR_MS of no events", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => sock.fire("prediction", { label: "a", confidence: 1, ready: true }));
    expect(result.current.utterance).toBe("A");

    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(result.current.utterance).toBe("");
  });

  it("each new event resets the idle timer", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => sock.fire("prediction", { label: "a", confidence: 1, ready: true }));
    act(() => {
      vi.advanceTimersByTime(8000);
    });
    act(() => sock.fire("prediction", { label: "b", confidence: 1, ready: true }));
    act(() => {
      vi.advanceTimersByTime(8000);
    });
    expect(result.current.utterance).toBe("AB");

    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(result.current.utterance).toBe("");
  });
});

describe("reset()", () => {
  it("clears every piece of state to its initial value", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => sock.fire("word_prediction", { top3: [{ label: "hello", confidence: 0.9 }] }));
    act(() => sock.fire("prediction", { label: "a", confidence: 1, ready: true }));
    expect(result.current.utterance).not.toBe("");

    act(() => result.current.reset());

    expect(result.current.utterance).toBe("");
    expect(result.current.sentUtterance).toBeNull();
    expect(result.current.wordPrediction).toBeNull();
    expect(result.current.prediction).toEqual({
      label: null,
      confidence: null,
      ready: false,
    });
    expect(result.current.captureStatus).toBe("idle");
    expect(result.current.captureProgress).toBe(0);
  });
});

describe("togglePaused", () => {
  it("flips the paused flag", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    expect(result.current.paused).toBe(false);
    act(() => result.current.togglePaused());
    expect(result.current.paused).toBe(true);
    act(() => result.current.togglePaused());
    expect(result.current.paused).toBe(false);
  });

  it("clears capture state when transitioning into paused", () => {
    const sock = new FakeSocket();
    const { result } = renderHook(() =>
      useSignRecognition(nullRef(), sock as any),
    );
    act(() => result.current.togglePaused());
    expect(result.current.captureStatus).toBe("idle");
    expect(result.current.captureProgress).toBe(0);
  });
});
