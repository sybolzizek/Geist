"""Persistent generated tool registry for Geist.

The registry is intentionally separate from production domain tools. It lets a
Geist run turn a small Python handler into a later `local.*` tool surface
without editing the main plugin module.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geist.core.agent.tool_spec import ToolSpec


LOCAL_TOOL_NAME_RE = re.compile(r"^local\.[a-z0-9][a-z0-9_.-]{0,96}$")


class LocalToolApi:
    """Async bridge that lets generated tools call the lab tool surface."""

    def __init__(
        self,
        dispatcher: Any,
        *,
        state: Any,
        project_name: str,
        workspace: str | Path,
        stack: tuple[str, ...],
        max_depth: int = 8,
    ) -> None:
        self._dispatcher = dispatcher
        self._state = state
        self._project_name = project_name
        self._workspace = Path(workspace)
        self._stack = tuple(stack)
        self._max_depth = max_depth

    @property
    def workspace(self) -> Path:
        return self._workspace

    def available_tools(self) -> list[str]:
        getter = getattr(self._dispatcher, "get_tools", None)
        if not callable(getter):
            return []
        tools = getter()
        if not isinstance(tools, dict):
            return []
        return sorted(str(name) for name in tools.keys())

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        tool_name = str(tool or "").strip()
        if not tool_name:
            return {"ok": False, "error": "tool name is required"}
        if tool_name in self._stack:
            return {
                "ok": False,
                "error": f"local tool recursion blocked: {' -> '.join([*self._stack, tool_name])}",
            }
        if len(self._stack) >= self._max_depth:
            return {"ok": False, "error": "local tool call depth exceeded"}
        available = self.available_tools()
        if available and tool_name not in available:
            return {"ok": False, "error": f"tool is not available: {tool_name}"}
        return await self._dispatcher.execute(
            {
                "tool": tool_name,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "_local_tool_api_stack": list(self._stack),
            },
            self._state,
            self._project_name,
        )


class LocalToolRegistry:
    """JSON-manifest backed registry for generated local Python tools."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.handlers_dir = self.root / "handlers"
        self.manifest_path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.handlers_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self._load_manifest()

    def specs(self) -> dict[str, ToolSpec]:
        specs: dict[str, ToolSpec] = {}
        for name, entry in self._manifest.get("tools", {}).items():
            if not isinstance(entry, dict):
                continue
            try:
                specs[name] = ToolSpec(
                    name=name,
                    description=str(entry.get("description") or "Geist-local generated tool."),
                    arguments=entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {},
                    authority=str(entry.get("authority") or "geist_local"),
                    side_effect=str(entry.get("side_effect") or "write"),
                    kind=str(entry.get("kind") or "local_generated_tool"),
                    reads=set(entry.get("reads") or []),
                    writes=set(entry.get("writes") or []),
                    serial=bool(entry.get("serial", True)),
                )
            except Exception:
                continue
        return specs

    def list_tools(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for name, entry in sorted(self._manifest.get("tools", {}).items()):
            if not isinstance(entry, dict):
                continue
            items.append({
                "name": name,
                "description": str(entry.get("description") or ""),
                "arguments": entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {},
                "handler_path": str(self._handler_path(entry)),
                "relative_handler_path": self._relative_handler_path(entry),
                "created_at": entry.get("created_at"),
                "updated_at": entry.get("updated_at"),
            })
        return items

    def register(
        self,
        *,
        name: str,
        description: str,
        arguments: dict[str, Any],
        code: str,
        overwrite: bool,
        reserved_names: set[str],
    ) -> dict[str, Any]:
        normalized = self._normalize_name(name)
        if normalized in reserved_names:
            return {"ok": False, "error": f"tool name collides with an existing tool: {normalized}"}
        if normalized in self._manifest.get("tools", {}) and not overwrite:
            return {"ok": False, "error": f"local tool already exists: {normalized}"}

        source = _render_handler_source(code)
        try:
            compile(source, f"<{normalized}>", "exec")
        except SyntaxError as exc:
            return {
                "ok": False,
                "error": "local tool code has syntax error",
                "message": exc.msg,
                "lineno": exc.lineno,
                "offset": exc.offset,
            }

        handler_path = self._handler_file_for(normalized)
        handler_path.parent.mkdir(parents=True, exist_ok=True)
        handler_path.write_text(source, encoding="utf-8")
        probe = self._load_handler(normalized, handler_path)
        if probe.get("ok") is not True:
            return probe

        now = datetime.now(timezone.utc).isoformat()
        existing = self._manifest.setdefault("tools", {}).get(normalized, {})
        created_at = existing.get("created_at") if isinstance(existing, dict) else None
        entry = {
            "name": normalized,
            "description": description.strip() or "Geist-local generated tool.",
            "arguments": arguments,
            "handler_path": handler_path.relative_to(self.root).as_posix(),
            "authority": "geist_local",
            "side_effect": "write",
            "kind": "local_generated_tool",
            "reads": ["workspace_files", "lab_state"],
            "writes": ["workspace_files", "lab_state"],
            "serial": True,
            "created_at": created_at or now,
            "updated_at": now,
        }
        self._manifest["tools"][normalized] = entry
        self._save_manifest()
        return {
            "ok": True,
            "tool": normalized,
            "description": entry["description"],
            "arguments": arguments,
            "handler_path": str(handler_path),
            "relative_handler_path": handler_path.relative_to(Path.cwd().resolve()).as_posix()
            if _is_relative_to(handler_path, Path.cwd().resolve()) else str(handler_path),
        }

    async def execute(
        self,
        name: str,
        arguments: Any,
        state: Any,
        workspace: str | Path,
        *,
        tool_api: LocalToolApi | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_name(name)
        entry = self._manifest.get("tools", {}).get(normalized)
        if not isinstance(entry, dict):
            return {"ok": False, "error": f"unknown local tool: {normalized}"}
        handler_path = self._handler_path(entry)
        loaded = self._load_handler(normalized, handler_path)
        if loaded.get("ok") is not True:
            return loaded
        func = loaded["execute"]
        safe_args = arguments if isinstance(arguments, dict) else {}
        try:
            result = _call_handler(func, safe_args, state, Path(workspace), tool_api)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, dict):
                return {"ok": True, **result} if "ok" not in result else result
            return {"ok": True, "data": result}
        except Exception as exc:
            return {"ok": False, "error": f"local tool execution failed: {exc}", "tool": normalized}

    def has_tool(self, name: str) -> bool:
        try:
            normalized = self._normalize_name(name)
        except ValueError:
            return False
        return normalized in self._manifest.get("tools", {})

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"tools": {}}
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"tools": {}}
        if not isinstance(raw, dict):
            return {"tools": {}}
        tools = raw.get("tools")
        if not isinstance(tools, dict):
            raw["tools"] = {}
        return raw

    def _save_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _normalize_name(self, name: str) -> str:
        normalized = str(name or "").strip().lower()
        if not LOCAL_TOOL_NAME_RE.match(normalized):
            raise ValueError("local tool name must match local.<lowercase-name>")
        return normalized

    def _handler_file_for(self, name: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "_", name.removeprefix("local.")).strip("_")
        return (self.handlers_dir / f"{slug}.py").resolve()

    def _handler_path(self, entry: dict[str, Any]) -> Path:
        raw = str(entry.get("handler_path") or "")
        path = (self.root / raw).resolve()
        if not _is_relative_to(path, self.root.resolve()):
            raise ValueError("local tool handler path escaped registry root")
        return path

    def _relative_handler_path(self, entry: dict[str, Any]) -> str:
        try:
            path = self._handler_path(entry)
            return path.relative_to(Path.cwd().resolve()).as_posix()
        except Exception:
            return str(entry.get("handler_path") or "")

    def _load_handler(self, name: str, handler_path: Path) -> dict[str, Any]:
        try:
            path = handler_path.resolve()
            if not _is_relative_to(path, self.root.resolve()):
                return {"ok": False, "error": "local tool handler path escaped registry root"}
            if not path.exists():
                return {"ok": False, "error": f"local tool handler not found: {path}"}
            module_name = "geist_local_" + re.sub(r"[^a-zA-Z0-9_]", "_", name)
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return {"ok": False, "error": f"could not import local tool handler: {path}"}
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            execute = getattr(module, "execute", None)
            if not callable(execute):
                return {"ok": False, "error": "local tool handler must define execute(arguments, state, workspace)"}
            return {"ok": True, "execute": execute}
        except Exception as exc:
            return {"ok": False, "error": f"local tool handler import failed: {exc}"}


def _render_handler_source(code: str) -> str:
    stripped = str(code or "").strip()
    if not stripped:
        stripped = (
            "async def execute(arguments, state, workspace, tool_api):\n"
            "    return {\"ok\": True, \"message\": \"local tool template\", \"arguments\": arguments}\n"
        )
    header = "# Generated by Geist tool.scaffold. Edit through local tools.\n"
    return header + stripped.rstrip() + "\n"


def _call_handler(
    func: Any,
    arguments: dict[str, Any],
    state: Any,
    workspace: Path,
    tool_api: LocalToolApi | None,
) -> Any:
    try:
        signature = inspect.signature(func)
    except Exception:
        return func(arguments, state, workspace)
    parameters = list(signature.parameters.values())
    if "tool_api" in signature.parameters:
        return func(arguments, state, workspace, tool_api=tool_api)
    accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if accepts_varargs or len(positional) >= 4:
        return func(arguments, state, workspace, tool_api)
    return func(arguments, state, workspace)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False
