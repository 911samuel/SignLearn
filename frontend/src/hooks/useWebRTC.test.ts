import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

// ---------------------------------------------------------------------------
// Mock RTCPeerConnection — captures the events the hook listens to and gives
// the test direct control over the signalling state machine.
// ---------------------------------------------------------------------------

type Handler = (...args: unknown[]) => void;

class FakeRTCPeerConnection {
  static instances: FakeRTCPeerConnection[] = [];
  ontrack: ((ev: { streams: MediaStream[] }) => void) | null = null;
  onicecandidate: ((ev: { candidate: RTCIceCandidate | null }) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;
  connectionState: RTCPeerConnectionState = "new";
  signalingState: RTCSignalingState = "stable";
  addTrack = vi.fn();
  addIceCandidate = vi.fn(async () => {});
  createOffer = vi.fn(async () => ({ type: "offer", sdp: "OFFER" }));
  createAnswer = vi.fn(async () => ({ type: "answer", sdp: "ANSWER" }));
  setLocalDescription = vi.fn(async () => {});
  setRemoteDescription = vi.fn(async () => {});
  close = vi.fn();
  constructor(public config: RTCConfiguration) {
    FakeRTCPeerConnection.instances.push(this);
  }
  setState(state: RTCPeerConnectionState) {
    this.connectionState = state;
    this.onconnectionstatechange?.();
  }
}

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
  async fire(event: string, ...args: unknown[]) {
    const list = [...(this.handlers.get(event) ?? [])];
    for (const fn of list) await fn(...args);
  }
}

function makeStream(): MediaStream {
  const track = { kind: "video", stop: vi.fn() } as unknown as MediaStreamTrack;
  return {
    getTracks: () => [track],
  } as unknown as MediaStream;
}

beforeEach(() => {
  FakeRTCPeerConnection.instances = [];
  (globalThis as any).RTCPeerConnection = FakeRTCPeerConnection;
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  delete (globalThis as any).RTCPeerConnection;
  vi.restoreAllMocks();
});

import { useWebRTC } from "./useWebRTC";

describe("useWebRTC", () => {
  it("does nothing until both socket and localStream are present", () => {
    const socket = new FakeSocket();
    renderHook(() => useWebRTC(socket as any, null, false, false));
    expect(FakeRTCPeerConnection.instances).toHaveLength(0);

    renderHook(() => useWebRTC(null, makeStream(), false, false));
    expect(FakeRTCPeerConnection.instances).toHaveLength(0);
  });

  it("builds a peer connection, adds tracks, wires listeners, and emits webrtc_ready", () => {
    const socket = new FakeSocket();
    const stream = makeStream();
    renderHook(() => useWebRTC(socket as any, stream, false, false));

    expect(FakeRTCPeerConnection.instances).toHaveLength(1);
    const pc = FakeRTCPeerConnection.instances[0];
    expect(pc.addTrack).toHaveBeenCalledOnce();
    expect(socket.on).toHaveBeenCalledWith("webrtc_offer", expect.any(Function));
    expect(socket.on).toHaveBeenCalledWith("webrtc_answer", expect.any(Function));
    expect(socket.on).toHaveBeenCalledWith("webrtc_ice", expect.any(Function));
    expect(socket.on).toHaveBeenCalledWith("webrtc_ready", expect.any(Function));
    expect(socket.emit).toHaveBeenCalledWith("webrtc_ready", {});
  });

  it("exposes connectionState changes via the `state` field", () => {
    const socket = new FakeSocket();
    const { result } = renderHook(() =>
      useWebRTC(socket as any, makeStream(), false, false),
    );
    expect(result.current.state).toBe("new");
    const pc = FakeRTCPeerConnection.instances[0];
    act(() => pc.setState("connected"));
    expect(result.current.state).toBe("connected");
  });

  it("publishes the remote stream from the ontrack event", () => {
    const socket = new FakeSocket();
    const { result } = renderHook(() =>
      useWebRTC(socket as any, makeStream(), false, false),
    );
    const pc = FakeRTCPeerConnection.instances[0];
    const remote = {} as MediaStream;
    act(() => pc.ontrack?.({ streams: [remote] }));
    expect(result.current.remoteStream).toBe(remote);
  });

  it("emits webrtc_ice when a local ICE candidate is gathered", () => {
    const socket = new FakeSocket();
    renderHook(() => useWebRTC(socket as any, makeStream(), false, false));
    const pc = FakeRTCPeerConnection.instances[0];
    const candidate = { toJSON: () => ({ sdp: "ICE" }) } as unknown as RTCIceCandidate;
    act(() => pc.onicecandidate?.({ candidate }));
    expect(socket.emit).toHaveBeenCalledWith("webrtc_ice", {
      candidate: { sdp: "ICE" },
    });
  });

  it("ignores ICE events with a null candidate", () => {
    const socket = new FakeSocket();
    renderHook(() => useWebRTC(socket as any, makeStream(), false, false));
    socket.emit.mockClear();
    const pc = FakeRTCPeerConnection.instances[0];
    act(() => pc.onicecandidate?.({ candidate: null }));
    expect(socket.emit).not.toHaveBeenCalled();
  });

  it("the initiator sends an offer only after peer is present AND ready", async () => {
    const socket = new FakeSocket();
    const stream = makeStream();
    const { rerender } = renderHook(
      ({ present }: { present: boolean }) =>
        useWebRTC(socket as any, stream, true, present),
      { initialProps: { present: false } },
    );

    const pc = FakeRTCPeerConnection.instances[0];
    socket.emit.mockClear();

    rerender({ present: true });
    await Promise.resolve();
    expect(pc.createOffer).not.toHaveBeenCalled();

    await act(async () => {
      await socket.fire("webrtc_ready");
    });

    await waitFor(() => {
      expect(pc.createOffer).toHaveBeenCalled();
      expect(socket.emit).toHaveBeenCalledWith("webrtc_offer", {
        sdp: { type: "offer", sdp: "OFFER" },
      });
    });
  });

  it("non-initiator never creates an offer, even when peer is ready", async () => {
    const socket = new FakeSocket();
    renderHook(() => useWebRTC(socket as any, makeStream(), false, true));
    const pc = FakeRTCPeerConnection.instances[0];
    await act(async () => {
      await socket.fire("webrtc_ready");
    });
    expect(pc.createOffer).not.toHaveBeenCalled();
  });

  it("rolls back to stable before re-offering when mid-negotiation", async () => {
    const socket = new FakeSocket();
    const stream = makeStream();
    renderHook(() => useWebRTC(socket as any, stream, true, true));
    const pc = FakeRTCPeerConnection.instances[0];
    pc.signalingState = "have-local-offer";

    await act(async () => {
      await socket.fire("webrtc_ready");
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 30));
    });

    expect(pc.setLocalDescription).toHaveBeenCalledWith({ type: "rollback" });
    expect(pc.createOffer).toHaveBeenCalled();
  });

  it("answers an incoming offer with setRemoteDescription → createAnswer → emit answer", async () => {
    const socket = new FakeSocket();
    renderHook(() => useWebRTC(socket as any, makeStream(), false, false));
    const pc = FakeRTCPeerConnection.instances[0];

    await act(async () => {
      await socket.fire("webrtc_offer", { sdp: { type: "offer", sdp: "REMOTE" } });
    });

    expect(pc.setRemoteDescription).toHaveBeenCalledWith({ type: "offer", sdp: "REMOTE" });
    expect(pc.createAnswer).toHaveBeenCalled();
    expect(socket.emit).toHaveBeenCalledWith("webrtc_answer", {
      sdp: { type: "answer", sdp: "ANSWER" },
    });
  });

  it("only applies the remote answer while in 'have-local-offer'", async () => {
    const socket = new FakeSocket();
    renderHook(() => useWebRTC(socket as any, makeStream(), true, false));
    const pc = FakeRTCPeerConnection.instances[0];

    // Not in have-local-offer → ignored.
    pc.signalingState = "stable";
    await act(async () => {
      await socket.fire("webrtc_answer", { sdp: { type: "answer", sdp: "A" } });
    });
    expect(pc.setRemoteDescription).not.toHaveBeenCalled();

    // Now we are mid-negotiation → applied.
    pc.signalingState = "have-local-offer";
    await act(async () => {
      await socket.fire("webrtc_answer", { sdp: { type: "answer", sdp: "B" } });
    });
    expect(pc.setRemoteDescription).toHaveBeenCalledWith({ type: "answer", sdp: "B" });
  });

  it("forwards incoming ICE candidates to addIceCandidate, swallowing failures", async () => {
    const socket = new FakeSocket();
    renderHook(() => useWebRTC(socket as any, makeStream(), false, false));
    const pc = FakeRTCPeerConnection.instances[0];

    await act(async () => {
      await socket.fire("webrtc_ice", { candidate: { sdp: "X" } });
    });
    expect(pc.addIceCandidate).toHaveBeenCalledWith({ sdp: "X" });

    pc.addIceCandidate.mockRejectedValueOnce(new Error("bad"));
    await expect(
      act(async () => {
        await socket.fire("webrtc_ice", { candidate: { sdp: "Y" } });
      }),
    ).resolves.not.toThrow();
  });

  it("closes the peer connection and unhooks listeners on unmount", () => {
    const socket = new FakeSocket();
    const { unmount } = renderHook(() =>
      useWebRTC(socket as any, makeStream(), false, false),
    );
    const pc = FakeRTCPeerConnection.instances[0];
    unmount();
    expect(pc.close).toHaveBeenCalled();
    expect(socket.off).toHaveBeenCalledWith("webrtc_offer", expect.any(Function));
    expect(socket.off).toHaveBeenCalledWith("webrtc_answer", expect.any(Function));
    expect(socket.off).toHaveBeenCalledWith("webrtc_ice", expect.any(Function));
    expect(socket.off).toHaveBeenCalledWith("webrtc_ready", expect.any(Function));
  });
});
