"""SQLite conversation log for the SignLearn backend.

Schema (single table)::

    messages(
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts        TEXT NOT NULL,          -- ISO-8601 UTC timestamp
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
    """Create the messages table if it does not exist."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                source     TEXT    NOT NULL CHECK(source IN ('sign', 'speech')),
                text       TEXT    NOT NULL,
                confidence REAL
            )
        """)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append(source: str, text: str, confidence: float | None = None) -> int:
    """Insert a message row and return its id."""
    init_db()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO messages (ts, source, text, confidence) VALUES (?, ?, ?, ?)",
            (_now_iso(), source, text, confidence),
        )
        return cur.lastrowid  # type: ignore[return-value]


def fetch(limit: int = 100) -> list[dict]:
    """Return up to *limit* messages, oldest first."""
    init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT id, ts, source, text, confidence FROM messages ORDER BY id ASC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def clear() -> int:
    """Delete all messages and return the count removed."""
    init_db()
    with _conn() as con:
        cur = con.execute("DELETE FROM messages")
        return cur.rowcount
