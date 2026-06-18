"""Project context loading for Geist agents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geist.core.agent import ToolSpec
from geist.core.fractal import FractalCall, NATIVE_FRACTAL_PROTOCOL


GEIST_AGENT_SYSTEM = """
You are Geist, a local coding agent running from the current workspace.
Use real local tools when the task requires inspecting, changing, or verifying
the workspace. Keep movement explicit: tool results are observations, large
materials should become artifact refs, and later fractal calls should receive
concrete paths, refs, trace ids, process ids, or URLs rather than vague roles.
""".strip()


@dataclass(frozen=True)
class ContextDocument:
    path: str
    kind: str
    content: str
    trusted: bool


@dataclass(frozen=True)
class ContextBundle:
    workspace: Path
    trusted: bool
    documents: tuple[ContextDocument, ...]
    blocked: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "type": "geist_project_context",
            "schema": "geist.context.project.v1",
            "workspace": str(self.workspace),
            "trusted": self.trusted,
            "documents": [
                {
                    "path": item.path,
                    "kind": item.kind,
                    "chars": len(item.content),
                    "content": item.content,
                    "trusted": item.trusted,
                }
                for item in self.documents
            ],
            "blocked": list(self.blocked),
        }


def load_context_bundle(
    workspace: str | Path,
    *,
    home: str | Path | None = None,
    trusted: bool = False,
    max_chars_per_doc: int = 24000,
) -> ContextBundle:
    root = Path(workspace).resolve()
    documents: list[ContextDocument] = []
    blocked: list[str] = []
    agent_home = Path(home or os.getenv("GEIST_HOME") or (Path.home() / ".geist")).resolve() / "agent"
    global_agents = agent_home / "AGENTS.md"
    if global_agents.exists():
        documents.append(_read_doc(global_agents, "global_agents", trusted=True, max_chars=max_chars_per_doc))

    for path in _agent_files_for_workspace(root):
        documents.append(_read_doc(path, "project_agents", trusted=True, max_chars=max_chars_per_doc))

    for name, kind in (("SYSTEM.md", "project_system"), ("APPEND_SYSTEM.md", "project_append_system")):
        path = root / ".geist" / name
        if path.exists():
            if trusted:
                documents.append(_read_doc(path, kind, trusted=True, max_chars=max_chars_per_doc))
            else:
                blocked.append(str(path))

    return ContextBundle(workspace=root, trusted=trusted, documents=tuple(documents), blocked=tuple(blocked))


def build_runtime_context(
    call: FractalCall,
    *,
    bundle: ContextBundle,
    tools: dict[str, ToolSpec],
    recent_history: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": NATIVE_FRACTAL_PROTOCOL},
        {"role": "system", "content": GEIST_AGENT_SYSTEM},
        {"role": "system", "content": json.dumps(_tool_surface_payload(tools), ensure_ascii=False, separators=(",", ":"), default=str)},
    ]
    context_payload = bundle.as_payload()
    if context_payload.get("documents") or context_payload.get("blocked"):
        messages.append({"role": "system", "content": json.dumps(context_payload, ensure_ascii=False, separators=(",", ":"), default=str)})
    if recent_history and recent_history.get("items"):
        messages.append({"role": "user", "content": json.dumps(recent_history, ensure_ascii=False, separators=(",", ":"), default=str)})
    messages.append({"role": "user", "content": json.dumps(_call_packet(call), ensure_ascii=False, separators=(",", ":"), default=str)})
    return messages


def _tool_surface_payload(tools: dict[str, ToolSpec]) -> dict[str, Any]:
    rows = []
    for name, spec in sorted(tools.items()):
        if not spec.executable:
            continue
        rows.append({
            "name": name,
            "description": spec.description,
            "arguments": spec.arguments,
            "side_effect": spec.side_effect,
            "kind": spec.kind,
            "reads": sorted(spec.reads),
            "writes": sorted(spec.writes),
            "serial": spec.serial,
        })
    return {
        "type": "geist_tool_surface",
        "schema": "geist.tool_surface.v1",
        "tools": rows,
    }


def _call_packet(call: FractalCall) -> dict[str, Any]:
    return {
        "type": "fractal_api_call",
        "schema": "geist.fractal.api_call.v1",
        "root_task": call.root_task,
        "call": {
            "call_id": call.call_id,
            "parent_call_id": call.parent_call_id,
            "spawn_kind": call.spawn_kind,
            "branch_path": call.branch_path,
            "sibling_index": call.sibling_index,
            "sibling_count": call.sibling_count,
            "expansion_round": call.expansion_round,
            "tool_round": call.tool_round,
        },
        "instruction": call.instruction or call.root_task,
        "continuation_context": call.continuation_context,
        "observations": list(call.observations or []),
    }


def _agent_files_for_workspace(workspace: Path) -> list[Path]:
    parts = [workspace, *workspace.parents]
    ordered = list(reversed(parts))
    return [path / "AGENTS.md" for path in ordered if (path / "AGENTS.md").exists()]


def _read_doc(path: Path, kind: str, *, trusted: bool, max_chars: int) -> ContextDocument:
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[truncated]"
    return ContextDocument(path=str(path), kind=kind, content=content, trusted=trusted)
