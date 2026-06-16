import { useEffect, useRef, useState, useCallback } from "react";
import { io, type Socket } from "socket.io-client";

export type Role = "signer" | "hearing";
export type ConnectionStatus = "connected" | "disconnected" | "reconnecting";

export interface Member {
  role: Role;
  name: string;
  sid?: string;
}

export interface Caption {
  id: number;
  source: "sign" | "speech";
  text: string;
  name: string;
  confidence?: number;
  ts: number;
}

export interface UseRoomResult {
  socket: Socket | null;
  status: ConnectionStatus;
  joinError: string | null;
  members: Member[];
  captions: Caption[];
  you: Member | null;
  emitSpeech: (text: string) => void;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5001";

export function useRoom(roomId: string, role: Role, name: string): UseRoomResult {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [joinError, setJoinError] = useState<string | null>(null);
  const [members, setMembers] = useState<Member[]>([]);
  const [captions, setCaptions] = useState<Caption[]>([]);
  const [you, setYou] = useState<Member | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const idRef = useRef(0);

  useEffect(() => {
    // Defer connecting until the caller has a real roomId.  Pages can
    // allocate a room asynchronously (e.g. POST /rooms) and pass the
    // result in once it arrives without the hook trying to join an
    // empty room.
    if (!roomId) return;

    const socket = io(BACKEND_URL, { transports: ["websocket"] });
    socketRef.current = socket;

    socket.on("connect", () => {
      setStatus("connected");
      socket.emit("join_room", { room_id: roomId, role, name });
    });
    socket.on("disconnect", () => setStatus("disconnected"));
    socket.on("reconnect_attempt", () => setStatus("reconnecting"));
    socket.on("reconnect", () => setStatus("connected"));

    socket.on("join_ok", (data: { you: Member }) => {
      setJoinError(null);
      setYou(data.you);
    });
    socket.on("join_error", (data: { message: string }) => setJoinError(data.message));

    socket.on("room_state", (data: { members: Member[] }) => setMembers(data.members));

    socket.on("caption", (data: Omit<Caption, "id">) => {
      setCaptions((prev) => {
        const next = [...prev, { ...data, id: ++idRef.current }];
        return next.slice(-50);
      });
    });

    return () => {
      socket.emit("leave_room");
      socket.disconnect();
      socketRef.current = null;
    };
  }, [roomId, role, name]);

  const emitSpeech = useCallback((text: string) => {
    socketRef.current?.emit("speech", { text });
  }, []);

  return { socket: socketRef.current, status, joinError, members, captions, you, emitSpeech };
}
