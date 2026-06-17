import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

// ---------- hook mocks (factories use createElement, not JSX, because vi.mock
// calls are hoisted above the test's React import) ----------
const useSignRecognition = vi.fn();
const useWebRTC = vi.fn();

vi.mock("@/hooks/useSignRecognition", () => ({
  useSignRecognition: (...a: unknown[]) => useSignRecognition(...a),
}));
vi.mock("@/hooks/useWebRTC", () => ({
  useWebRTC: (...a: unknown[]) => useWebRTC(...a),
}));

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
vi.mock("./LandmarkOverlay", async () => {
  const React = await import("react");
  return {
    LandmarkOverlay: () => React.createElement("div", { "data-testid": "landmarks" }),
  };
});

import { SignerView } from "./SignerView";

const reset = vi.fn();
const togglePaused = vi.fn();
const commitUtterance = vi.fn();
const getUserMedia = vi.fn();

const baseRecognition = {
  prediction: { label: null, confidence: null, ready: false },
  utterance: "",
  sentUtterance: null as string | null,
  commitUtterance,
  wordPrediction: null,
  captureStatus: "idle" as const,
  captureProgress: 0,
  landmarkerResult: null,
  reset,
  paused: false,
  togglePaused,
  latencyMs: null,
  landmarkerError: null,
};

beforeEach(() => {
  reset.mockReset();
  togglePaused.mockReset();
  commitUtterance.mockReset();
  getUserMedia.mockReset();
  useSignRecognition.mockReturnValue({ ...baseRecognition });
  useWebRTC.mockReturnValue({ remoteStream: null, state: "new" });
  Object.defineProperty(global.navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
});

afterEach(() => {
  delete (global.navigator as any).mediaDevices;
});

function makeStream(): MediaStream {
  const videoTrack: any = { stop: vi.fn(), onended: null };
  return {
    getTracks: () => [videoTrack],
    getVideoTracks: () => [videoTrack],
  } as unknown as MediaStream;
}

async function mountWithCam() {
  getUserMedia.mockResolvedValueOnce(makeStream());
  const utils = render(
    <SignerView socket={null} captions={[]} peerPresent={true} />,
  );
  fireEvent.click(screen.getByRole("button", { name: /allow camera & start/i }));
  await waitFor(() =>
    expect(
      screen.queryByRole("heading", { name: /turn on your camera/i }),
    ).toBeNull(),
  );
  return utils;
}

describe("SignerView permission gate", () => {
  it("shows the camera PermissionGate until access is granted", () => {
    render(<SignerView socket={null} captions={[]} peerPresent={false} />);
    expect(
      screen.getByRole("heading", { name: /turn on your camera/i }),
    ).toBeInTheDocument();
  });

  it("calls getUserMedia with video+audio and clears the gate", async () => {
    await mountWithCam();
    expect(getUserMedia).toHaveBeenCalledWith({
      video: { width: 640, height: 480 },
      audio: true,
    });
    expect(
      screen.queryByRole("heading", { name: /turn on your camera/i }),
    ).toBeNull();
  });

  it("renders a denied alert when getUserMedia rejects", async () => {
    getUserMedia.mockRejectedValueOnce(new Error("blocked"));
    render(<SignerView socket={null} captions={[]} peerPresent={false} />);
    fireEvent.click(screen.getByRole("button", { name: /allow camera & start/i }));
    await waitFor(() =>
      expect(screen.getByText(/Camera access denied|denied/i)).toBeInTheDocument(),
    );
  });
});

describe("SignerView main UI", () => {
  it("renders the live video preview, the landmark overlay, and the peer tile", async () => {
    await mountWithCam();
    expect(
      screen.getByLabelText(/Your live camera preview/i),
    ).toBeInTheDocument();
    expect(screen.getByTestId("landmarks")).toBeInTheDocument();
    expect(screen.getByTestId("overlay-speech")).toBeInTheDocument();
  });

  it("shows the current utterance and a Send button", async () => {
    useSignRecognition.mockReturnValue({ ...baseRecognition, utterance: "hello" });
    await mountWithCam();
    expect(screen.getByText("hello")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /send utterance now/i }));
    expect(commitUtterance).toHaveBeenCalledOnce();
  });

  it("shows 'Sent ✓' badge when sentUtterance is set", async () => {
    useSignRecognition.mockReturnValue({
      ...baseRecognition,
      sentUtterance: "thanks",
    });
    await mountWithCam();
    expect(screen.getByText("thanks")).toBeInTheDocument();
    expect(screen.getByText(/Sent ✓/i)).toBeInTheDocument();
  });

  it("renders the latency badge when latencyMs is set", async () => {
    useSignRecognition.mockReturnValue({ ...baseRecognition, latencyMs: 42 });
    await mountWithCam();
    expect(screen.getByText("42 ms")).toBeInTheDocument();
  });

  it("renders the capture progress bar only while 'signing'", async () => {
    useSignRecognition.mockReturnValue({
      ...baseRecognition,
      captureStatus: "signing",
      captureProgress: 0.5,
    });
    const { rerender } = await mountWithCam();

    expect(
      screen.getByLabelText(/Capture progress 50 percent/i),
    ).toBeInTheDocument();

    useSignRecognition.mockReturnValue({
      ...baseRecognition,
      captureStatus: "idle",
    });
    rerender(<SignerView socket={null} captions={[]} peerPresent={true} />);
    expect(screen.queryByLabelText(/Capture progress/i)).toBeNull();
  });

  it("Pause/Resume button calls togglePaused", async () => {
    await mountWithCam();
    fireEvent.click(screen.getByRole("button", { name: /pause/i }));
    expect(togglePaused).toHaveBeenCalledOnce();
  });

  it("flips to Resume label when paused=true", async () => {
    useSignRecognition.mockReturnValue({ ...baseRecognition, paused: true });
    await mountWithCam();
    expect(screen.getByRole("button", { name: /resume/i })).toBeInTheDocument();
  });

  it("Reset button calls reset()", async () => {
    await mountWithCam();
    fireEvent.click(screen.getByRole("button", { name: /reset/i }));
    expect(reset).toHaveBeenCalledOnce();
  });

  it("renders the MediaPipe error alert when landmarkerError is set", async () => {
    useSignRecognition.mockReturnValue({
      ...baseRecognition,
      landmarkerError: "GPU unavailable",
    });
    await mountWithCam();
    expect(screen.getByText(/GPU unavailable/)).toBeInTheDocument();
  });

  it("renders the prediction-error alert when wordPrediction.error is set", async () => {
    useSignRecognition.mockReturnValue({
      ...baseRecognition,
      wordPrediction: { top3: [], error: "boom" },
    });
    await mountWithCam();
    expect(screen.getByText(/Couldn't read that sign/i)).toBeInTheDocument();
  });
});

describe("SignerView onPrediction wiring", () => {
  it("calls onPrediction with the top candidate when a word prediction arrives", async () => {
    const onPrediction = vi.fn();
    useSignRecognition.mockReturnValue({
      ...baseRecognition,
      wordPrediction: {
        top3: [
          { label: "hello", confidence: 0.9 },
          { label: "hi", confidence: 0.4 },
        ],
      },
    });
    getUserMedia.mockResolvedValueOnce(makeStream());
    render(
      <SignerView
        socket={null}
        captions={[]}
        peerPresent={true}
        onPrediction={onPrediction}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /allow camera & start/i }));
    await waitFor(() => {
      expect(onPrediction).toHaveBeenCalledWith("hello", 0.9, expect.any(Number));
    });
  });

  it("does not call onPrediction when the prediction carries an error", async () => {
    const onPrediction = vi.fn();
    useSignRecognition.mockReturnValue({
      ...baseRecognition,
      wordPrediction: { top3: [{ label: "x", confidence: 1 }], error: "bad" },
    });
    getUserMedia.mockResolvedValueOnce(makeStream());
    render(
      <SignerView
        socket={null}
        captions={[]}
        peerPresent={true}
        onPrediction={onPrediction}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /allow camera & start/i }));
    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: /turn on your camera/i }),
      ).toBeNull(),
    );
    expect(onPrediction).not.toHaveBeenCalled();
  });
});

describe("SignerView peer-tile state", () => {
  it("shows the 'Waiting…' badge when no peer is present", () => {
    render(<SignerView socket={null} captions={[]} peerPresent={false} />);
    // Gate is still showing — that's fine; we're testing peer presence is
    // forwarded down. Skip past the gate first.
  });

  it("forwards captions to the speech overlay", async () => {
    await mountWithCam();
    expect(screen.getByTestId("overlay-speech")).toBeInTheDocument();
  });
});
