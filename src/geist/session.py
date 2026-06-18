"""Session storage for Geist CLI runs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class SessionRef:
    workspace: Path
    session_id: str
    path: Path


class SessionStore:
    """JSONL-backed session store under `~/.geist/agent` by default."""

    def __init__(self, home: str | Path | None = None) -> None:
        self.home = Path(home or os.getenv("GEIST_HOME") or (Path.home() / ".geist")).resolve()
        self.root = self.home / "agent" / "sessions"

    def workspace_key(self, workspace: str | Path) -> str:
        resolved = Path(workspace).resolve()
        digest = hashlib.sha256(str(resolved).encode("utf-8", errors="replace")).hexdigest()[:16]
        return digest

    def open(
        self,
        workspace: str | Path,
        *,
        session_id: str | None = None,
        continue_latest: bool = False,
    ) -> SessionRef:
        root = self.root / self.workspace_key(workspace)
        root.mkdir(parents=True, exist_ok=True)
        if continue_latest and not session_id:
            session_id = self.latest_session_id(workspace)
        session_id = _safe_session_id(session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
        return SessionRef(Path(workspace).resolve(), session_id, root / f"{session_id}.jsonl")

    def latest_session_id(self, workspace: str | Path) -> str | None:
        root = self.root / self.workspace_key(workspace)
        if not root.exists():
            return None
        files = sorted(root.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
        return files[0].stem if files else None

    def append(self, ref: SessionRef, event: dict[str, Any]) -> None:
        ref.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"at": time.time(), **event}
        with ref.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def read(self, ref: SessionRef) -> list[dict[str, Any]]:
        if not ref.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in ref.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def recent_history(self, ref: SessionRef, *, limit: int = 8, max_chars: int = 12000) -> dict[str, Any]:
        rows = [
            item
            for item in self.read(ref)
            if item.get("event") in {"user", "assistant", "runtime_error"}
        ][-limit:]
        compact = [
            {
                "event": item.get("event"),
                "text": str(item.get("text") or item.get("response") or item.get("error") or "")[:2000],
            }
            for item in rows
        ]
        payload = {
            "type": "geist_recent_session_history",
            "schema": "geist.session.history.v1",
            "session_id": ref.session_id,
            "items": compact,
        }
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(text) <= max_chars:
            return payload
        payload["items"] = compact[-max(1, limit // 2):]
        payload["truncated"] = True
        return payload


def _safe_session_id(value: str) -> str:
    text = str(value or "").strip()
    safe = []
    for ch in text[:96]:
        safe.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(safe).strip("._-") or uuid4().hex[:12]
