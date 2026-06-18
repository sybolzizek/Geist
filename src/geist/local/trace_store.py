"""Local trace object store for fractal runtime motion history."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LocalTraceStore:
    """Append-only readable trace layer for Geist.

    The store intentionally does not model a workflow. It records immutable
    objects and lets a later runtime decide which slice of history to read,
    summarize, pass onward, or build on.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root
        self._memory: list[dict[str, Any]] = []
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
            self._objects_path.parent.mkdir(parents=True, exist_ok=True)
            self._objects_path.touch(exist_ok=True)

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def objects_path(self) -> Path | None:
        return self._objects_path if self._root is not None else None

    @property
    def _objects_path(self) -> Path:
        if self._root is None:
            raise ValueError("trace store is in-memory")
        return self._root / "objects.jsonl"

    def append(
        self,
        *,
        title: str,
        text: str = "",
        data: dict[str, Any] | None = None,
        source: str = "runtime",
    ) -> dict[str, Any]:
        data = data if isinstance(data, dict) else {}
        now = datetime.now(timezone.utc).isoformat()
        canonical = json.dumps(
            {"at": now, "source": source, "title": title, "text": text, "data": data},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        object_id = "tr_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        row = {
            "id": object_id,
            "at": now,
            "source": str(source or "runtime"),
            "title": str(title or "trace object"),
            "text": str(text or ""),
            "data": data,
        }
        if self._root is None:
            self._memory.append(row)
        else:
            with self._objects_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def record_runtime_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event_name = str(event.get("event") or "runtime")
        title = f"runtime:{event_name}"
        text = _event_text(event)
        return self.append(title=title, text=text, data=dict(event), source="runtime")

    def read(
        self,
        *,
        query: str = "",
        object_id: str = "",
        event: str = "",
        run_id: str = "",
        call_id: str = "",
        parent_call_id: str = "",
        branch_path: str = "",
        tool: str = "",
        path: str = "",
        change_id: str = "",
        source: str = "",
        limit: int = 20,
        offset: int = 0,
        sequence_min: int | None = None,
        sequence_max: int | None = None,
        max_chars: int = 20000,
        max_text_chars: int | None = None,
        include_data: bool = False,
        order: str = "latest",
    ) -> dict[str, Any]:
        order_value = str(order or "latest").strip().lower()
        chronological = order_value in {"chronological", "oldest", "forward"}
        rows = self._load_all()
        source_rows = rows if chronological else list(reversed(rows))
        matched: list[dict[str, Any]] = []
        skipped = 0
        for row in source_rows:
            if object_id and str(row.get("id") or "") != object_id:
                continue
            if source and str(row.get("source") or "") != source:
                continue
            data = row.get("data") if isinstance(row.get("data"), dict) else {}
            if event and str(data.get("event") or "") != event:
                continue
            if run_id and str(data.get("run_id") or "") != run_id:
                continue
            if call_id and str(data.get("call_id") or "") != call_id:
                continue
            if parent_call_id and str(data.get("parent_call_id") or "") != parent_call_id:
                continue
            if branch_path and str(data.get("branch_path") or "") != branch_path:
                continue
            sequence = _coerce_optional_int(data.get("sequence"))
            if sequence_min is not None and (sequence is None or sequence < sequence_min):
                continue
            if sequence_max is not None and (sequence is None or sequence > sequence_max):
                continue
            haystack = _search_text(row)
            if query and query.lower() not in haystack.lower():
                continue
            if tool and tool.lower() not in haystack.lower():
                continue
            if path and path.replace("\\", "/").lower() not in haystack.replace("\\", "/").lower():
                continue
            if change_id and change_id.lower() not in haystack.lower():
                continue
            if skipped < max(offset, 0):
                skipped += 1
                continue
            matched.append(row)
            if len(matched) >= limit:
                break
        rendered_rows = matched if chronological else list(reversed(matched))
        rendered, truncated = _render_rows(
            rendered_rows,
            max_chars=max_chars,
            include_data=include_data,
            max_text_chars=max_text_chars,
        )
        return {
            "ok": True,
            "count": len(matched),
            "query": {
                "query": query,
                "id": object_id,
                "event": event,
                "run_id": run_id,
                "call_id": call_id,
                "parent_call_id": parent_call_id,
                "branch_path": branch_path,
                "tool": tool,
                "path": path,
                "change_id": change_id,
                "source": source,
                "offset": max(offset, 0),
                "sequence_min": sequence_min,
                "sequence_max": sequence_max,
                "order": "chronological" if chronological else "latest",
            },
            "objects": [_brief_row(row) for row in rendered_rows],
            "text": rendered,
            "truncated": truncated,
            "store": str(self._objects_path) if self._root is not None else "memory",
        }

    def stats(self) -> dict[str, Any]:
        rows = self._load_all()
        return {
            "ok": True,
            "count": len(rows),
            "store": str(self._objects_path) if self._root is not None else "memory",
            "latest": _brief_row(rows[-1]) if rows else None,
        }

    def _load_all(self) -> list[dict[str, Any]]:
        if self._root is None:
            return list(self._memory)
        rows: list[dict[str, Any]] = []
        if not self._objects_path.exists():
            return rows
        for line in self._objects_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows


def _event_text(event: dict[str, Any]) -> str:
    event_name = str(event.get("event") or "runtime")
    lines = [f"event: {event_name}"]
    for key in (
        "instruction",
        "response",
        "continuation_context",
        "observation",
        "error",
        "reason",
    ):
        value = event.get(key)
        if value:
            lines.append(f"{key}: {value}")
    tools = event.get("tools")
    if tools:
        lines.append("tools: " + ", ".join(str(item) for item in tools))
    instructions = event.get("instructions")
    if instructions:
        lines.append("instructions: " + " | ".join(str(item) for item in instructions))
    return "\n".join(lines)


def _search_text(row: dict[str, Any]) -> str:
    return "\n".join([
        str(row.get("id") or ""),
        str(row.get("source") or ""),
        str(row.get("title") or ""),
        str(row.get("text") or ""),
        json.dumps(row.get("data") or {}, ensure_ascii=False, default=str),
    ])


def _brief_row(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    return {
        "id": row.get("id"),
        "at": row.get("at"),
        "source": row.get("source"),
        "title": row.get("title"),
        "event": data.get("event"),
        "run_id": data.get("run_id"),
        "sequence": data.get("sequence"),
        "call_id": data.get("call_id"),
        "parent_call_id": data.get("parent_call_id"),
        "branch_path": data.get("branch_path"),
        "tools": data.get("tools") if isinstance(data.get("tools"), list) else None,
    }


def _render_rows(
    rows: list[dict[str, Any]],
    *,
    max_chars: int,
    include_data: bool,
    max_text_chars: int | None,
) -> tuple[str, bool]:
    target = max(int(max_chars or 0), 0)
    payload = _trace_rows_payload(rows, include_data=include_data, max_text_chars=max_text_chars)
    rendered = _json_compact(payload)
    if target <= 0 or len(rendered) <= target:
        return rendered, False

    payload["truncated"] = True
    payload["rows"] = [
        _trace_row_payload(
            row,
            include_data=include_data,
            max_text_chars=min(
                max_text_chars,
                max(64, target // max(len(rows), 1) - 256),
            ) if max_text_chars is not None else max(64, target // max(len(rows), 1) - 256),
        )
        for row in rows
    ]
    rendered = _json_compact(payload)
    if len(rendered) <= target:
        return rendered, True

    payload["rows"] = [
        {
            **_brief_row(row),
            "text_chars": len(str(row.get("text") or "")),
            "text_omitted": True,
            "data_omitted": bool(include_data and isinstance(row.get("data"), dict) and row.get("data")),
        }
        for row in rows
    ]
    payload["omitted"] = {"text": True, "data": bool(include_data)}
    rendered = _json_compact(payload)
    if len(rendered) <= target:
        return rendered, True

    kept: list[dict[str, Any]] = []
    omitted_count = 0
    for row in reversed(rows):
        candidate = [
            {
                **_brief_row(row),
                "text_chars": len(str(row.get("text") or "")),
                "text_omitted": True,
                "data_omitted": bool(include_data and isinstance(row.get("data"), dict) and row.get("data")),
            },
            *kept,
        ]
        payload["rows"] = candidate
        payload["omitted_rows"] = len(rows) - len(candidate)
        candidate_text = _json_compact(payload)
        if len(candidate_text) > target:
            omitted_count += 1
            continue
        kept = candidate
        rendered = candidate_text
    if kept:
        payload["rows"] = kept
        payload["omitted_rows"] = len(rows) - len(kept)
        return _json_compact(payload), True

    return _json_compact({
        "type": "fractal_trace_rows",
        "schema": "geist.fractal.trace_rows.v1",
        "count": len(rows),
        "truncated": True,
        "omitted_rows": len(rows) or omitted_count,
        "rows": [],
    }), True


def _trace_rows_payload(
    rows: list[dict[str, Any]],
    *,
    include_data: bool,
    max_text_chars: int | None,
) -> dict[str, Any]:
    return {
        "type": "fractal_trace_rows",
        "schema": "geist.fractal.trace_rows.v1",
        "count": len(rows),
        "include_data": include_data,
        "max_text_chars": max_text_chars,
        "truncated": False,
        "rows": [
            _trace_row_payload(row, include_data=include_data, max_text_chars=max_text_chars)
            for row in rows
        ],
    }


def _trace_row_payload(
    row: dict[str, Any],
    *,
    include_data: bool,
    max_text_chars: int | None = None,
) -> dict[str, Any]:
    text = str(row.get("text") or "").strip()
    text_truncated = False
    if max_text_chars is not None and len(text) > max_text_chars:
        text = text[:max_text_chars]
        text_truncated = True
    payload: dict[str, Any] = {
        **_brief_row(row),
        "text": text,
        "text_chars": len(str(row.get("text") or "")),
        "text_truncated": text_truncated,
    }
    if include_data:
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        payload["data"] = data
    return payload


def _json_compact(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except Exception:
        return None
