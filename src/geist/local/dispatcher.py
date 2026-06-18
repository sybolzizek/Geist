"""Default local tool dispatcher for Geist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geist.core.agent import ToolSpec
from geist.local.artifact_store import LocalArtifactStore
from geist.local.generated_tools import LocalToolApi, LocalToolRegistry
from geist.local.tool_specs import DEFAULT_LOCAL_TOOL_SPECS
from geist.local.trace_store import LocalTraceStore
from geist.local.workspace import LocalWorkspace


class LocalToolDispatcher:
    """Dispatch the default local Geist tool surface."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        state_root: str | Path | None = None,
        artifacts: LocalArtifactStore | None = None,
        trace: LocalTraceStore | None = None,
        local_tools: LocalToolRegistry | None = None,
    ) -> None:
        self.workspace = LocalWorkspace(workspace)
        root = Path(state_root).resolve() if state_root is not None else Path(workspace).resolve() / ".geist_state"
        self.artifacts = artifacts or LocalArtifactStore(root / "artifacts")
        self.trace = trace or LocalTraceStore(root / "trace")
        self.local_tools = local_tools or LocalToolRegistry(root / "local_tools")

    def get_tools(self) -> dict[str, ToolSpec]:
        tools = dict(DEFAULT_LOCAL_TOOL_SPECS)
        tools.update(self.local_tools.specs())
        return tools

    async def execute(self, tool_call: dict[str, Any], state: Any = None, project_name: str = "geist") -> dict[str, Any]:
        tool = str(tool_call.get("tool") or "").strip()
        args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        if tool.startswith("local."):
            api = LocalToolApi(
                self,
                state=state,
                project_name=project_name,
                workspace=self.workspace.root,
                stack=tuple(tool_call.get("_local_tool_api_stack") or [tool]),
            )
            return await self.local_tools.execute(tool, args, state, self.workspace.root, tool_api=api)
        if tool == "read":
            return self.workspace.read(
                args.get("path") or "",
                max_chars=_int(args.get("max_chars"), 20000),
                offset=_int(args.get("offset"), 0),
                line_start=_optional_int(args.get("line_start")),
                line_count=_optional_int(args.get("line_count")),
                encoding=str(args.get("encoding") or "utf-8"),
            )
        if tool == "write":
            return self.workspace.write(
                args.get("path") or "",
                str(args.get("content") or ""),
                append=bool(args.get("append")),
                overwrite=_bool(args.get("overwrite"), True),
                expected_sha256=str(args.get("expected_sha256") or ""),
                encoding=str(args.get("encoding") or "utf-8"),
            )
        if tool == "edit":
            return self.workspace.edit(
                args.get("path") or "",
                old_text=str(args.get("old_text") or ""),
                new_text=str(args.get("new_text") or ""),
                replace_all=bool(args.get("replace_all")),
                expected_sha256=str(args.get("expected_sha256") or ""),
                dry_run=bool(args.get("dry_run")),
                encoding=str(args.get("encoding") or "utf-8"),
            )
        if tool == "ls":
            return self.workspace.list(
                args.get("path") or ".",
                pattern=str(args.get("pattern") or "*"),
                recursive=bool(args.get("recursive")),
                max_entries=_int(args.get("max_entries"), 200),
            )
        if tool in {"bash", "shell"}:
            return self.workspace.run(
                args.get("command") or "",
                cwd=args.get("cwd") or ".",
                timeout_ms=_int(args.get("timeout_ms"), 30000),
                max_chars=_int(args.get("max_chars") or args.get("max_output_chars"), 20000),
            )
        if tool == "git.status":
            return self.workspace.git_status(
                path=args.get("path") or ".",
                max_files=_int(args.get("max_files"), 80),
                include_untracked=_bool(args.get("include_untracked"), True),
            )
        if tool == "git.diff_summary":
            return self.workspace.git_diff_summary(
                path=args.get("path") or ".",
                paths=args.get("paths"),
                staged=bool(args.get("staged")),
                max_files=_int(args.get("max_files"), 80),
            )
        if tool == "git.diff_read":
            return self.workspace.git_diff_read(
                path=args.get("path") or ".",
                paths=args.get("paths"),
                staged=bool(args.get("staged")),
                max_chars=_int(args.get("max_chars"), 20000),
            )
        if tool == "git.snapshot":
            return self.workspace.git_snapshot(path=args.get("path") or ".", max_files=_int(args.get("max_files"), 200))
        if tool == "git.delta":
            return self.workspace.git_delta(args.get("snapshot"), path=args.get("path") or ".", max_files=_int(args.get("max_files"), 200))
        if tool == "artifact.read":
            return self.artifacts.read(
                str(args.get("ref") or ""),
                max_chars=_int(args.get("max_chars"), 20000),
                offset=_int(args.get("offset"), 0),
                line_start=_optional_int(args.get("line_start")),
                line_count=_optional_int(args.get("line_count")),
            )
        if tool == "artifact.list":
            return self.artifacts.list(limit=_int(args.get("limit"), 40))
        if tool == "artifact.search":
            return self.artifacts.search(
                str(args.get("query") or ""),
                limit=_int(args.get("limit"), 20),
                max_chars=_int(args.get("max_chars"), 20000),
            )
        if tool in {"trace.read", "trace.search", "trace.tail"}:
            return self.trace.read(
                query=str(args.get("query") or ""),
                object_id=str(args.get("id") or ""),
                event=str(args.get("event") or ""),
                run_id=str(args.get("run_id") or ""),
                call_id=str(args.get("call_id") or ""),
                parent_call_id=str(args.get("parent_call_id") or ""),
                branch_path=str(args.get("branch_path") or ""),
                tool=str(args.get("tool") or ""),
                path=str(args.get("path") or ""),
                change_id=str(args.get("change_id") or ""),
                source=str(args.get("source") or ""),
                limit=_int(args.get("limit"), 20),
                offset=_int(args.get("offset"), 0),
                max_chars=_int(args.get("max_chars"), 20000),
                max_text_chars=_optional_int(args.get("max_text_chars")),
                include_data=bool(args.get("include_data")),
                order=str(args.get("order") or "latest"),
            )
        if tool == "trace.write":
            return {
                "ok": True,
                "object": self.trace.append(
                    title=str(args.get("title") or "trace object"),
                    text=str(args.get("text") or ""),
                    data=args.get("data") if isinstance(args.get("data"), dict) else {},
                    source=str(args.get("source") or "agent"),
                ),
            }
        if tool == "tool.list_local":
            return {"ok": True, "tools": self.local_tools.list_tools()}
        if tool == "tool.scaffold":
            return self.local_tools.register(
                name=str(args.get("name") or ""),
                description=str(args.get("description") or ""),
                arguments=args.get("arguments") if isinstance(args.get("arguments"), dict) else {},
                code=str(args.get("code") or ""),
                overwrite=bool(args.get("overwrite")),
                reserved_names=set(DEFAULT_LOCAL_TOOL_SPECS),
            )
        return {"ok": False, "error": f"unknown local tool: {tool}"}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default
