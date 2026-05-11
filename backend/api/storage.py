"""SQLite conversation log for the SignLearn backend.

Schema (single table)::

    messages(
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts        TEXT NOT NULL,          -- ISO-8601 UTC timestamp
        room_id   TEXT NOT NULL,          -- 6-char room code
        source    TEXT NOT NULL           -- 'sign' or 'speech'
                  CHECK(source IN ('sign', 'speech')),
        text      TEXT NOT NULL,
        confidence REAL                   -- NULL for speech entries
    )

Call :func:`set_db_path` before any other function when tests need an
isolated temporary database.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.api.config import CONFIG

_db_path: Path = CONFIG.db_path


def set_db_path(path: Path | str) -> None:
    """Override the database path (for testing)."""
    global _db_path
    _db_path = Path(path)


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_db_path))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create tables if they do not exist."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                room_id    TEXT    NOT NULL,
                source     TEXT    NOT NULL CHECK(source IN ('sign', 'speech')),
                text       TEXT    NOT NULL,
                confidence REAL
            )
        """)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_room_id ON messages(room_id)"
        )
        con.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ts             TEXT    NOT NULL,
                room_id        TEXT    NOT NULL,
                original_text  TEXT    NOT NULL,
                corrected_text TEXT    NOT NULL,
                confidence     REAL
            )
        """)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_corrections_room_id ON corrections(room_id)"
        )
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT    NOT NULL,
                category  TEXT    NOT NULL,
                text      TEXT    NOT NULL,
                room_id   TEXT
            )
        """)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append(room_id: str, source: str, text: str, confidence: float | None = None) -> int:
    """Insert a message row and return its id."""
    init_db()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO messages (ts, room_id, source, text, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now_iso(), room_id, source, text, confidence),
        )
        return cur.lastrowid  # type: ignore[return-value]


def fetch(room_id: str, limit: int = 100) -> list[dict]:
    """Return up to *limit* messages for *room_id*, oldest first."""
    init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT id, ts, room_id, source, text, confidence "
            "FROM messages WHERE room_id = ? ORDER BY id ASC LIMIT ?",
            (room_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def append_feedback(
    category: str,
    text: str,
    room_id: str | None = None,
) -> int:
    """Log a user's feedback message and return its id."""
    init_db()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO feedback (ts, category, text, room_id) VALUES (?, ?, ?, ?)",
            (_now_iso(), category, text, room_id),
        )
        return cur.lastrowid  # type: ignore[return-value]


def append_correction(
    room_id: str,
    original_text: str,
    corrected_text: str,
    confidence: float | None = None,
) -> int:
    """Log a signer's correction and return its id."""
    init_db()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO corrections (ts, room_id, original_text, corrected_text, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now_iso(), room_id, original_text, corrected_text, confidence),
        )
        return cur.lastrowid  # type: ignore[return-value]


def clear(room_id: str | None = None) -> int:
    """Delete messages (all, or just for *room_id*) and return the count removed."""
    init_db()
    with _conn() as con:
        if room_id is None:
            cur = con.execute("DELETE FROM messages")
        else:
            cur = con.execute("DELETE FROM messages WHERE room_id = ?", (room_id,))
        return cur.rowcount
