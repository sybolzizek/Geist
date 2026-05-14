"""Shared context store — SQLite-backed key-value channel for fractal communication.

The store lives in the adapter layer.  Fractals never read or write it
directly.  The adapter:

  1. reads ``context_updates`` from a fractal's ``own`` layer
  2. applies those updates to the store
  3. resolves ``refs`` from the store when building the next fractal's ``growth``

No fractal ever performs a "read" action.  Everything it receives
is already in its ``growth`` when the engine starts it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any


class SharedContext:
    """SQLite-backed key-value context store.

    ``db_path``
        Path to SQLite database file.  Use ``":memory:"`` for ephemeral
        (tests, quick demos).  Default: ``":memory:"``.

    Values are JSON-serialised before storage and deserialised on read.
    Keys must be strings.
    """

    _init_sql = """
        CREATE TABLE IF NOT EXISTS store (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(self._init_sql)
        self._conn.commit()
        self._lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def apply(self, updates: dict[str, Any]) -> None:
        """Merge *updates* into the store.  Existing keys are overwritten."""
        if not updates:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO store (key, value) VALUES (?, ?)",
                ((k, json.dumps(v, ensure_ascii=False)) for k, v in updates.items()),
            )

    def resolve(self, refs: list[str]) -> dict[str, Any]:
        """Resolve a list of reference keys against the store.

        Returns ``{key: value}`` for every key that exists.
        Missing keys are silently omitted.
        """
        if not refs:
            return {}
        placeholders = ",".join("?" * len(refs))
        rows = self._conn.execute(
            f"SELECT key, value FROM store WHERE key IN ({placeholders})",
            refs,
        ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def get(self, key: str, default: Any = None) -> Any:
        """Read a single key with an optional default."""
        row = self._conn.execute(
            "SELECT value FROM store WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the full store."""
        rows = self._conn.execute("SELECT key, value FROM store").fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
