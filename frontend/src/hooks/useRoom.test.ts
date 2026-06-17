import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, renderHook } from "@testing-library/react";

type Handler = (...args: unknown[]) => void;

class FakeSocket {
  private handlers = new Map<string, Handler[]>();
  emit = vi.fn();
  disconnect = vi.fn();
  on(event: string, fn: Handler) {
    const list = this.handlers.get(event) ?? [];
    list.push(fn);
    this.handlers.set(event, list);
    return this;
  }
  fire(event: string, ...args: unknown[]) {
    for (const fn of this.handlers.get(event) ?? []) fn(...args);
  }
}

const lastSocket = { current: null as FakeSocket | null };
const ioMock = vi.fn(() => {
  const s = new FakeSocket();
  lastSocket.current = s;
  return s;
});

vi.mock("socket.io-client", () => ({ io: (...args: unknown[]) => ioMock(...args) }));

beforeEach(() => {
  ioMock.mockClear();
  lastSocket.current = null;
});

import { useRoom } from "./useRoom";

describe("useRoom", () => {
  it("does not connect until a roomId is provided", () => {
    renderHook(() => useRoom("", "signer", "Sam"));
    expect(ioMock).not.toHaveBeenCalled();
  });

  it("connects, joins the room, and tracks status", () => {
    const { result } = renderHook(() => useRoom("room-1", "signer", "Sam"));
    expect(ioMock).toHaveBeenCalledOnce();

    const sock = lastSocket.current!;
    expect(result.current.status).toBe("disconnected");

    act(() => sock.fire("connect"));
    expect(result.current.status).toBe("connected");
    expect(sock.emit).toHaveBeenCalledWith("join_room", {
      room_id: "room-1",
      role: "signer",
      name: "Sam",
    });

    act(() => sock.fire("disconnect"));
    expect(result.current.status).toBe("disconnected");

    act(() => sock.fire("reconnect_attempt"));
    expect(result.current.status).toBe("reconnecting");

    act(() => sock.fire("reconnect"));
    expect(result.current.status).toBe("connected");
  });

  it("stores join_ok payload and clears join errors", () => {
    const { result } = renderHook(() => useRoom("r", "hearing", "Lee"));
    const sock = lastSocket.current!;
    act(() => sock.fire("join_error", { message: "full" }));
    expect(result.current.joinError).toBe("full");

    act(() =>
      sock.fire("join_ok", { you: { role: "hearing", name: "Lee", sid: "abc" } }),
    );
    expect(result.current.joinError).toBeNull();
    expect(result.current.you).toEqual({ role: "hearing", name: "Lee", sid: "abc" });
  });

  it("appends captions with auto-incrementing ids and caps at 50", () => {
    const { result } = renderHook(() => useRoom("r", "signer", "Sam"));
    const sock = lastSocket.current!;

    act(() => {
      for (let i = 0; i < 55; i++) {
        sock.fire("caption", { source: "sign", text: `c${i}`, name: "Sam", ts: i });
      }
    });

    expect(result.current.captions).toHaveLength(50);
    expect(result.current.captions[0].text).toBe("c5");
    expect(result.current.captions.at(-1)!.text).toBe("c54");
    const ids = result.current.captions.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("emitSpeech proxies to the socket", () => {
    const { result } = renderHook(() => useRoom("r", "signer", "Sam"));
    const sock = lastSocket.current!;
    act(() => result.current.emitSpeech("hello"));
    expect(sock.emit).toHaveBeenCalledWith("speech", { text: "hello" });
  });

  it("emits leave_room and disconnects on unmount", () => {
    const { unmount } = renderHook(() => useRoom("r", "signer", "Sam"));
    const sock = lastSocket.current!;
    unmount();
    expect(sock.emit).toHaveBeenCalledWith("leave_room");
    expect(sock.disconnect).toHaveBeenCalled();
  });
});
