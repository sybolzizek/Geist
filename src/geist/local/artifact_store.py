"""Local managed artifact store for large runtime materials."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LocalArtifactStore:
    """File-backed material store for Geist runtime output.

    The store deliberately keeps no semantic ontology. It gives large text a
    stable handle, a readable path, and a bounded preview so later calls can
    choose how to use the material.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.jsonl"
        self._manifest_path.touch(exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def put_text(
        self,
        text: str,
        *,
        material_ref: str = "",
        kind: str = "text",
        source: dict[str, Any] | None = None,
        sha256: str = "",
    ) -> dict[str, Any]:
        value = str(text or "")
        digest = sha256 or hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        ref = _safe_ref(material_ref) or ("mat_" + digest[:16])
        path = self.root / f"{ref}.txt"
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            path.write_text(value, encoding="utf-8", errors="replace")
        entry = {
            "ref": ref,
            "kind": str(kind or "text"),
            "path": str(path),
            "relative_path": _relative_to_cwd(path),
            "chars": len(value),
            "lines": value.count("\n") + 1 if value else 0,
            "sha256": digest,
            "source": source if isinstance(source, dict) else {},
            "first_line": _first_nonempty_line(value),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._append_manifest(entry)
        return entry

    def read(
        self,
        ref: str,
        *,
        max_chars: int = 20000,
        offset: int = 0,
        line_start: int | None = None,
        line_count: int | None = None,
    ) -> dict[str, Any]:
        entry = self._find(ref)
        if entry is None:
            return {"ok": False, "error": f"artifact not found: {ref}"}
        path = Path(str(entry.get("path") or ""))
        if not path.exists() or not path.is_file():
            return {"ok": False, "error": f"artifact file missing: {path}", "artifact": entry}
        text = path.read_text(encoding="utf-8", errors="replace")
        content, range_info = _slice_text(
            text,
            max_chars=max_chars,
            offset=offset,
            line_start=line_start,
            line_count=line_count,
        )
        next_read = _next_read(ref=ref, range_info=range_info, max_chars=max_chars)
        return {
            "ok": True,
            "artifact": entry,
            "content": content,
            "truncated": bool(range_info.get("truncated")),
            "chars": len(text),
            "lines": text.count("\n") + 1 if text else 0,
            "range": range_info,
            "next_read": next_read,
        }

    def list(self, *, limit: int = 40) -> dict[str, Any]:
        entries = list(reversed(self._load_manifest()))
        return {
            "ok": True,
            "root": str(self.root),
            "manifest_path": str(self._manifest_path),
            "count": len(entries),
            "artifacts": [_brief(entry) for entry in entries[:limit]],
        }

    def search(self, query: str, *, limit: int = 20, max_chars: int = 20000) -> dict[str, Any]:
        needle = str(query or "").strip().lower()
        if not needle:
            return {"ok": False, "error": "query required"}
        matches: list[dict[str, Any]] = []
        excerpts: list[dict[str, Any]] = []
        for entry in reversed(self._load_manifest()):
            if len(matches) >= limit:
                break
            path = Path(str(entry.get("path") or ""))
            meta_text = json.dumps(entry, ensure_ascii=False, default=str)
            content = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            haystack = (meta_text + "\n" + content).lower()
            if needle not in haystack:
                continue
            brief = _brief(entry)
            matches.append(brief)
            excerpt, truncated_before, truncated_after = _excerpt(content, needle, max_chars=1200)
            excerpts.append({
                **brief,
                "excerpt": excerpt,
                "excerpt_chars": len(excerpt),
                "truncated_before": truncated_before,
                "truncated_after": truncated_after,
            })
        rendered, truncated = _render_search_text(query=query, items=excerpts, max_chars=max_chars)
        return {
            "ok": True,
            "query": query,
            "count": len(matches),
            "matches": matches,
            "excerpts": excerpts,
            "text": rendered,
            "truncated": truncated,
        }

    def manifest(self) -> dict[str, Any]:
        entries = self._load_manifest()
        return {
            "ok": True,
            "root": str(self.root),
            "manifest_path": str(self._manifest_path),
            "count": len(entries),
            "latest": _brief(entries[-1]) if entries else None,
        }

    def _append_manifest(self, entry: dict[str, Any]) -> None:
        with self._manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _find(self, ref: str) -> dict[str, Any] | None:
        wanted = _safe_ref(ref)
        if not wanted:
            return None
        for entry in reversed(self._load_manifest()):
            if str(entry.get("ref") or "") == wanted:
                return entry
        path = self.root / f"{wanted}.txt"
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            return {
                "ref": wanted,
                "kind": "text",
                "path": str(path),
                "relative_path": _relative_to_cwd(path),
                "chars": len(text),
                "sha256": digest,
                "source": {},
            }
        return None

    def _load_manifest(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self._manifest_path.exists():
            return rows
        seen: set[str] = set()
        for line in reversed(self._manifest_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            ref = str(entry.get("ref") or "")
            if not ref or ref in seen:
                continue
            seen.add(ref)
            rows.append(entry)
        return list(reversed(rows))


def _safe_ref(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    allowed = []
    for ch in text[:96]:
        allowed.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(allowed)


def _brief(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "ref": entry.get("ref"),
        "kind": entry.get("kind"),
        "relative_path": entry.get("relative_path"),
        "chars": entry.get("chars"),
        "lines": entry.get("lines"),
        "sha256": entry.get("sha256"),
        "first_line": entry.get("first_line"),
        "source": entry.get("source") if isinstance(entry.get("source"), dict) else {},
        "created_at": entry.get("created_at"),
    }


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _slice_text(
    text: str,
    *,
    max_chars: int,
    offset: int = 0,
    line_start: int | None = None,
    line_count: int | None = None,
) -> tuple[str, dict[str, Any]]:
    value = str(text or "")
    total_chars = len(value)
    total_lines = value.count("\n") + 1 if value else 0
    cap = max(int(max_chars or 0), 0)
    if line_start is not None:
        start_line = max(int(line_start), 1)
        count = max(int(line_count or 0), 0) or None
        lines = value.splitlines(keepends=True)
        start_index = min(start_line - 1, len(lines))
        end_index = len(lines) if count is None else min(start_index + count, len(lines))
        selected = "".join(lines[start_index:end_index])
        selected, char_truncated = _truncate(selected, cap)
        end_line = start_index + len(selected.splitlines()) if selected else start_index
        truncated = end_index < len(lines) or char_truncated
        return selected, {
            "mode": "lines",
            "line_start": start_line,
            "line_end": end_line,
            "requested_line_count": line_count,
            "total_lines": total_lines,
            "total_chars": total_chars,
            "max_chars": cap,
            "truncated": truncated,
        }

    start = min(max(int(offset or 0), 0), total_chars)
    end = total_chars if cap <= 0 else min(start + cap, total_chars)
    content = value[start:end]
    return content, {
        "mode": "chars",
        "offset": start,
        "end": end,
        "total_chars": total_chars,
        "total_lines": total_lines,
        "max_chars": cap,
        "truncated": end < total_chars,
    }


def _next_read(*, ref: str, range_info: dict[str, Any], max_chars: int) -> dict[str, Any] | None:
    if not range_info.get("truncated"):
        return None
    if range_info.get("mode") == "lines":
        next_line = int(range_info.get("line_end") or 0) + 1
        return {
            "tool": "artifact.read",
            "arguments": {"ref": ref, "line_start": next_line, "max_chars": max_chars},
        }
    return {
        "tool": "artifact.read",
        "arguments": {"ref": ref, "offset": range_info.get("end"), "max_chars": max_chars},
    }


def _excerpt(text: str, needle: str, *, max_chars: int) -> tuple[str, bool, bool]:
    if max_chars <= 0:
        return "", False, bool(text)
    lower = text.lower()
    index = lower.find(needle)
    if index < 0:
        excerpt, truncated = _truncate(text, max_chars)
        return excerpt, False, truncated
    start = max(0, index - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end], start > 0, end < len(text)


def _render_search_text(
    *,
    query: str,
    items: list[dict[str, Any]],
    max_chars: int,
) -> tuple[str, bool]:
    target = max(int(max_chars or 0), 0)
    text_items = [_search_text_item(item) for item in items]
    payload = {
        "type": "fractal_artifact_search",
        "schema": "geist.fractal.artifact_search.v1",
        "query": query,
        "count": len(items),
        "truncated": False,
        "items": text_items,
    }
    rendered = _json_compact(payload)
    if target <= 0 or len(rendered) <= target:
        return rendered, False

    per_item = max(64, target // max(len(items), 1) - 256)
    payload["truncated"] = True
    payload["items"] = [_shrink_search_item(item, max_excerpt_chars=per_item) for item in text_items]
    rendered = _json_compact(payload)
    if len(rendered) <= target:
        return rendered, True

    payload["items"] = [
        {
            "ref": item.get("ref"),
            "kind": item.get("kind"),
            "relative_path": item.get("relative_path"),
            "chars": item.get("chars"),
            "lines": item.get("lines"),
            "sha256": item.get("sha256"),
            "excerpt_chars": item.get("excerpt_chars"),
            "excerpt_omitted": True,
        }
        for item in text_items
    ]
    payload["omitted"] = {"excerpt": True}
    rendered = _json_compact(payload)
    if len(rendered) <= target:
        return rendered, True

    kept: list[dict[str, Any]] = []
    for item in reversed(payload["items"]):
        candidate = [item, *kept]
        payload["items"] = candidate
        payload["omitted_items"] = len(items) - len(candidate)
        candidate_text = _json_compact(payload)
        if len(candidate_text) <= target:
            kept = candidate
            rendered = candidate_text
    if kept:
        return rendered, True
    return _json_compact({
        "type": "fractal_artifact_search",
        "schema": "geist.fractal.artifact_search.v1",
        "query": query,
        "count": len(text_items),
        "truncated": True,
        "omitted_items": len(text_items),
        "items": [],
    }), True


def _search_text_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "ref": item.get("ref"),
        "kind": item.get("kind"),
        "relative_path": item.get("relative_path"),
        "excerpt": item.get("excerpt"),
        "excerpt_chars": item.get("excerpt_chars"),
        "truncated_before": item.get("truncated_before"),
        "truncated_after": item.get("truncated_after"),
    }


def _shrink_search_item(item: dict[str, Any], *, max_excerpt_chars: int) -> dict[str, Any]:
    shrunk = dict(item)
    excerpt = str(shrunk.get("excerpt") or "")
    if len(excerpt) > max_excerpt_chars:
        shrunk["excerpt"] = excerpt[:max_excerpt_chars]
        shrunk["excerpt_truncated"] = True
    return shrunk


def _json_compact(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return ""


def _relative_to_cwd(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)
