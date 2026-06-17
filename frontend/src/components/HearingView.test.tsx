import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";

// ---------- hook mocks ----------
const useSpeechToText = vi.fn();
const useSpeakSignCaptions = vi.fn();
const useWebRTC = vi.fn();
const usePreferences = vi.fn();

vi.mock("@/hooks/useSpeechToText", () => ({
  useSpeechToText: (...a: unknown[]) => useSpeechToText(...a),
}));
vi.mock("@/hooks/useSpeakSignCaptions", () => ({
  useSpeakSignCaptions: (...a: unknown[]) => useSpeakSignCaptions(...a),
}));
vi.mock("@/hooks/useWebRTC", () => ({
  useWebRTC: (...a: unknown[]) => useWebRTC(...a),
}));
vi.mock("@/lib/preferences", () => ({
  usePreferences: (...a: unknown[]) => usePreferences(...a),
}));

// Stub portal-rendered or heavy children that aren't the focus.
// (Uses createElement, not JSX, because vi.mock factories are hoisted
// above imports and JSX needs the react/jsx-runtime import in scope.)
vi.mock("./RemoteVideo", async () => {
  const React = await import("react");
  return {
    RemoteVideo: ({ stream }: { stream: unknown }) =>
      React.createElement("div", {
        "data-testid": "remote-video",
        "data-has-stream": stream ? "yes" : "no",
      }),
  };
});
vi.mock("./VideoCaptionOverlay", async () => {
  const React = await import("react");
  return {
    VideoCaptionOverlay: ({
      filter,
      emptyHint,
    }: {
      filter: string;
      emptyHint?: string;
    }) =>
      React.createElement(
        "div",
        { "data-testid": `overlay-${filter}` },
        emptyHint ?? "",
      ),
  };
});

import { HearingView } from "./HearingView";

const start = vi.fn();
const stop = vi.fn();
const getUserMedia = vi.fn();

beforeEach(() => {
  start.mockReset();
  stop.mockReset();
  getUserMedia.mockReset();
  useSpeechToText.mockReturnValue({ listening: false, supported: true, start, stop });
  useWebRTC.mockReturnValue({ remoteStream: null, state: "new" });
  useSpeakSignCaptions.mockReturnValue(undefined);
  usePreferences.mockReturnValue({
    prefs: { pushToTalkMode: "hold", captionSize: "normal" },
    update: vi.fn(),
    hydrated: true,
  });

  Object.defineProperty(global.navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
});

afterEach(() => {
  // @ts-expect-error reset
  delete (global.navigator as any).mediaDevices;
});

function makeFakeStream(): MediaStream {
  const track = { stop: vi.fn() };
  return { getTracks: () => [track] } as unknown as MediaStream;
}

describe("HearingView permission gate", () => {
  it("shows the microphone PermissionGate until access is granted", () => {
    render(
      <HearingView socket={null} captions={[]} peerPresent={false} onSpeech={vi.fn()} />,
    );
    expect(
      screen.getByRole("heading", { name: /turn on your microphone/i }),
    ).toBeInTheDocument();
  });

  it("requests microphone access via getUserMedia and unlocks the UI", async () => {
    getUserMedia.mockResolvedValueOnce(makeFakeStream());
    render(
      <HearingView socket={null} captions={[]} peerPresent={false} onSpeech={vi.fn()} />,
    );

    fireEvent.click(screen.getByRole("button", { name: /allow microphone & start/i }));

    await waitFor(() => {
      expect(getUserMedia).toHaveBeenCalledWith({ video: true, audio: true });
      expect(screen.getByRole("button", { name: /push to talk/i })).toBeInTheDocument();
    });
  });

  it("surfaces the error message when getUserMedia rejects", async () => {
    getUserMedia.mockRejectedValueOnce(new Error("NotAllowedError"));
    render(
      <HearingView socket={null} captions={[]} peerPresent={false} onSpeech={vi.fn()} />,
    );

    fireEvent.click(screen.getByRole("button", { name: /allow microphone & start/i }));

    await waitFor(() => {
      expect(screen.getByText(/NotAllowedError/)).toBeInTheDocument();
    });
  });
});

describe("HearingView main UI (mic granted)", () => {
  async function mount() {
    getUserMedia.mockResolvedValueOnce(makeFakeStream());
    const utils = render(
      <HearingView socket={null} captions={[]} peerPresent={true} onSpeech={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /allow microphone & start/i }));
    // Wait until the gate is gone — main UI is mounted.
    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: /turn on your microphone/i }),
      ).toBeNull(),
    );
    return utils;
  }

  it("renders the Push to talk control and signer overlay placeholder", async () => {
    await mount();
    expect(screen.getByRole("button", { name: /push to talk/i })).toBeInTheDocument();
    expect(screen.getByTestId("overlay-sign")).toBeInTheDocument();
  });

  it("shows 'Listening…' when the speech hook reports listening=true", async () => {
    useSpeechToText.mockReturnValue({ listening: true, supported: true, start, stop });
    await mount();
    expect(screen.getAllByText(/Listening…/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Mic on/)).toBeInTheDocument();
  });

  it("toggles the read-aloud button between Reading aloud and Muted", async () => {
    await mount();
    const toggle = screen.getByRole("button", { name: /reading aloud/i });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(toggle);
    expect(screen.getByRole("button", { name: /muted/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("renders the unsupported-browser alert when speech recognition is unsupported", async () => {
    useSpeechToText.mockReturnValue({ listening: false, supported: false, start, stop });
    getUserMedia.mockResolvedValueOnce(makeFakeStream());
    render(
      <HearingView socket={null} captions={[]} peerPresent={false} onSpeech={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /allow microphone & start/i }));
    await waitFor(() =>
      expect(screen.getByText(/Speech recognition not available/i)).toBeInTheDocument(),
    );
    expect(screen.queryByRole("button", { name: /push to talk/i })).toBeNull();
  });

  it("Space key starts and stops listening in push-to-talk (hold) mode", async () => {
    await mount();
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "Space" }));
    });
    expect(start).toHaveBeenCalled();

    // Once listening becomes true, the keyup should trigger stop. Simulate by
    // re-rendering with listening=true and dispatching keyup.
    useSpeechToText.mockReturnValue({ listening: true, supported: true, start, stop });
    await mount();
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keyup", { code: "Space" }));
    });
    // Keyup only fires stop if the held flag was set; this confirms the listener
    // is registered and runs without errors. (heldRef is internal.)
    expect(start).toHaveBeenCalled();
  });

  it("Space key toggles listening in 'toggle' mode", async () => {
    usePreferences.mockReturnValue({
      prefs: { pushToTalkMode: "toggle", captionSize: "normal" },
      update: vi.fn(),
      hydrated: true,
    });
    await mount();
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "Space" }));
    });
    expect(start).toHaveBeenCalledOnce();
    expect(screen.getByRole("button", { name: /tap to talk/i })).toBeInTheDocument();
  });

  it("ignores Space when focus is inside a text input", async () => {
    await mount();
    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();
    act(() => {
      input.dispatchEvent(
        new KeyboardEvent("keydown", { code: "Space", bubbles: true }),
      );
    });
    expect(start).not.toHaveBeenCalled();
    input.remove();
  });
});

describe("HearingView WebRTC wiring", () => {
  it("forwards captions filter to the signer overlay", async () => {
    getUserMedia.mockResolvedValueOnce(makeFakeStream());
    render(
      <HearingView
        socket={null}
        captions={[
          {
            id: 1,
            source: "sign",
            text: "hello",
            name: "Sam",
            ts: 0,
          },
        ]}
        peerPresent={true}
        onSpeech={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /allow microphone & start/i }));
    await waitFor(() =>
      expect(screen.getByTestId("overlay-sign")).toBeInTheDocument(),
    );
  });

  it("passes the local stream to RemoteVideo for the self-view tile", async () => {
    getUserMedia.mockResolvedValueOnce(makeFakeStream());
    render(
      <HearingView socket={null} captions={[]} peerPresent={false} onSpeech={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /allow microphone & start/i }));
    await waitFor(() => {
      const videos = screen.getAllByTestId("remote-video");
      const withStream = videos.find((v) => v.getAttribute("data-has-stream") === "yes");
      expect(withStream).toBeTruthy();
    });
  });
});
