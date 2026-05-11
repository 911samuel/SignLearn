"""In-memory room registry for 2-party SignLearn calls.

A *room* pairs one Signer with one Hearing participant. Each browser tab is
a member identified by its Socket.IO session id (``sid``). The store is
process-local — fine for a single dev server, would need Redis to scale out.
"""

from __future__ import annotations

import secrets
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["signer", "hearing"]

_CODE_ALPHABET = string.ascii_uppercase + string.digits
_CODE_LEN = 6


@dataclass
class Member:
    sid: str
    role: Role
    name: str
    joined_at: float = field(default_factory=time.time)


@dataclass
class Room:
    id: str
    members: dict[str, Member] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def has_role(self, role: Role) -> bool:
        return any(m.role == role for m in self.members.values())

    def peer_sid(self, sid: str) -> str | None:
        for s in self.members:
            if s != sid:
                return s
        return None

    def roster(self) -> list[dict]:
        return [
            {"role": m.role, "name": m.name, "sid": m.sid}
            for m in self.members.values()
        ]


class RoomStore:
    """Thread-safe registry of active rooms."""

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}
        self._sid_to_room: dict[str, str] = {}
        self._lock = threading.RLock()

    def create(self) -> Room:
        with self._lock:
            for _ in range(20):
                code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))
                if code not in self._rooms:
                    room = Room(id=code)
                    self._rooms[code] = room
                    return room
            raise RuntimeError("Failed to allocate a unique room code")

    def get(self, room_id: str) -> Room | None:
        with self._lock:
            return self._rooms.get(room_id)

    def room_for_sid(self, sid: str) -> Room | None:
        with self._lock:
            rid = self._sid_to_room.get(sid)
            return self._rooms.get(rid) if rid else None

    def add_member(self, room_id: str, sid: str, role: Role, name: str) -> Member:
        with self._lock:
            room = self._rooms.get(room_id)
            if room is None:
                raise KeyError(f"Room {room_id!r} does not exist")
            if len(room.members) >= 2:
                raise ValueError("Room is full")
            if room.has_role(role):
                raise ValueError(f"Role {role!r} is already taken in this room")
            if role not in ("signer", "hearing"):
                raise ValueError(f"Unknown role {role!r}")
            member = Member(sid=sid, role=role, name=name)
            room.members[sid] = member
            self._sid_to_room[sid] = room_id
            return member

    def remove_member(self, sid: str) -> Room | None:
        with self._lock:
            rid = self._sid_to_room.pop(sid, None)
            if rid is None:
                return None
            room = self._rooms.get(rid)
            if room is None:
                return None
            room.members.pop(sid, None)
            if not room.members:
                self._rooms.pop(rid, None)
                return None
            return room


STORE = RoomStore()
