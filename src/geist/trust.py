"""Project trust state for Geist."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class TrustStore:
    def __init__(self, home: str | Path | None = None) -> None:
        self.home = Path(home or os.getenv("GEIST_HOME") or (Path.home() / ".geist")).resolve()
        self.path = self.home / "agent" / "trusted_projects.json"

    def is_trusted(self, workspace: str | Path) -> bool:
        data = self._load()
        key = _workspace_key(workspace)
        return key in data.get("trusted", {})

    def trust(self, workspace: str | Path) -> dict[str, Any]:
        data = self._load()
        data.setdefault("trusted", {})[_workspace_key(workspace)] = {
            "path": str(Path(workspace).resolve()),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data["trusted"][_workspace_key(workspace)]

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"trusted": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"trusted": {}}
        if not isinstance(data, dict):
            return {"trusted": {}}
        if not isinstance(data.get("trusted"), dict):
            data["trusted"] = {}
        return data


def _workspace_key(workspace: str | Path) -> str:
    path = str(Path(workspace).resolve())
    return hashlib.sha256(path.encode("utf-8", errors="replace")).hexdigest()[:16]
