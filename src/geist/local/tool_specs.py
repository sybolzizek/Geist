"""Tool specs for the default local Geist substrate."""

from __future__ import annotations

from geist.core.agent import ToolSpec


READ_TOOL = ToolSpec(
    name="read",
    description="Read one workspace-local text file.",
    arguments={
        "path": "Workspace-local file path.",
        "max_chars": "Optional max characters to return. Default 20000.",
        "offset": "Optional character offset.",
        "line_start": "Optional 1-based line number.",
        "line_count": "Optional line count.",
        "encoding": "Optional text encoding. Default utf-8.",
    },
    authority="local_workspace",
    side_effect="none",
    kind="local_file_read",
    reads={"workspace_files"},
)

WRITE_TOOL = ToolSpec(
    name="write",
    description="Write or append one workspace-local text file.",
    arguments={
        "path": "Workspace-local file path.",
        "content": "Text content.",
        "append": "Optional boolean.",
        "overwrite": "Optional boolean. Default true.",
        "expected_sha256": "Optional current file sha256.",
        "encoding": "Optional text encoding. Default utf-8.",
    },
    authority="local_workspace",
    side_effect="write",
    kind="local_file_write",
    reads={"workspace_files"},
    writes={"workspace_files"},
    serial=True,
)

EDIT_TOOL = ToolSpec(
    name="edit",
    description="Apply one exact text replacement to a workspace-local file.",
    arguments={
        "path": "Workspace-local file path.",
        "old_text": "Exact text to replace.",
        "new_text": "Replacement text.",
        "replace_all": "Optional boolean. Default false.",
        "expected_sha256": "Optional current file sha256.",
        "dry_run": "Optional boolean.",
        "encoding": "Optional text encoding. Default utf-8.",
    },
    authority="local_workspace",
    side_effect="write",
    kind="local_file_edit",
    reads={"workspace_files"},
    writes={"workspace_files"},
    serial=True,
)

LS_TOOL = ToolSpec(
    name="ls",
    description="List workspace-local files or directories with metadata.",
    arguments={
        "path": "Workspace-local file or directory. Default .",
        "pattern": "Optional glob pattern. Default *.",
        "recursive": "Optional boolean.",
        "max_entries": "Optional max entries. Default 200.",
    },
    authority="local_workspace",
    side_effect="none",
    kind="local_file_list",
    reads={"workspace_files"},
)

BASH_TOOL = ToolSpec(
    name="bash",
    description="Run a local executable directly in the workspace without shell chaining.",
    arguments={
        "command": "Command as a string or argv array. Prefer argv arrays.",
        "cwd": "Optional workspace-local working directory.",
        "timeout_ms": "Optional timeout in milliseconds.",
        "max_chars": "Optional stdout/stderr cap.",
    },
    authority="local_workspace",
    side_effect="write",
    kind="local_command",
    reads={"workspace_files"},
    writes={"workspace_files"},
    serial=True,
)

GIT_STATUS_TOOL = ToolSpec(
    name="git.status",
    description="Read workspace git status as structured data.",
    arguments={
        "path": "Workspace-local anchor path. Default .",
        "max_files": "Optional max changed files.",
        "include_untracked": "Optional boolean. Default true.",
    },
    authority="local_workspace",
    side_effect="none",
    kind="local_git_status",
    reads={"workspace_files", "git_index"},
)

GIT_DIFF_SUMMARY_TOOL = ToolSpec(
    name="git.diff_summary",
    description="Return compact git diff statistics without full patch bodies.",
    arguments={
        "path": "Workspace-local anchor path. Default .",
        "paths": "Optional file path or list of paths.",
        "staged": "Optional boolean.",
        "max_files": "Optional max files.",
    },
    authority="local_workspace",
    side_effect="none",
    kind="local_git_diff_summary",
    reads={"workspace_files", "git_index"},
)

GIT_DIFF_READ_TOOL = ToolSpec(
    name="git.diff_read",
    description="Read bounded git patch text for selected files.",
    arguments={
        "path": "Workspace-local anchor path. Default .",
        "paths": "Optional file path or list of paths.",
        "staged": "Optional boolean.",
        "max_chars": "Optional patch cap.",
    },
    authority="local_workspace",
    side_effect="none",
    kind="local_git_diff_read",
    reads={"workspace_files", "git_index"},
)

GIT_SNAPSHOT_TOOL = ToolSpec(
    name="git.snapshot",
    description="Return a compact immutable snapshot object for current git status.",
    arguments={"path": "Workspace-local anchor path. Default .", "max_files": "Optional max files."},
    authority="local_workspace",
    side_effect="none",
    kind="local_git_snapshot",
    reads={"workspace_files", "git_index"},
)

GIT_DELTA_TOOL = ToolSpec(
    name="git.delta",
    description="Compare a prior git.snapshot object with current git status.",
    arguments={
        "snapshot": "Snapshot object returned by git.snapshot.",
        "path": "Workspace-local anchor path. Default .",
        "max_files": "Optional max files.",
    },
    authority="local_workspace",
    side_effect="none",
    kind="local_git_delta",
    reads={"workspace_files", "git_index"},
)

ARTIFACT_READ_TOOL = ToolSpec(
    name="artifact.read",
    description="Read one managed Geist artifact by ref.",
    arguments={"ref": "Artifact ref.", "max_chars": "Optional cap.", "offset": "Optional offset."},
    authority="geist_local",
    side_effect="none",
    kind="local_artifact_read",
    reads={"local_artifacts"},
)

ARTIFACT_LIST_TOOL = ToolSpec(
    name="artifact.list",
    description="List recent managed Geist artifacts.",
    arguments={"limit": "Optional max artifacts."},
    authority="geist_local",
    side_effect="none",
    kind="local_artifact_list",
    reads={"local_artifacts"},
)

ARTIFACT_SEARCH_TOOL = ToolSpec(
    name="artifact.search",
    description="Search managed Geist artifact metadata and stored text.",
    arguments={"query": "Text to search for.", "limit": "Optional max matches.", "max_chars": "Optional cap."},
    authority="geist_local",
    side_effect="none",
    kind="local_artifact_search",
    reads={"local_artifacts"},
)

TRACE_READ_TOOL = ToolSpec(
    name="trace.read",
    description="Read selected Geist trace objects.",
    arguments={"query": "Optional text query.", "event": "Optional event filter.", "limit": "Optional max rows."},
    authority="geist_local",
    side_effect="none",
    kind="local_trace_read",
    reads={"local_trace"},
)

TRACE_WRITE_TOOL = ToolSpec(
    name="trace.write",
    description="Append a readable object to the Geist trace layer.",
    arguments={"title": "Trace title.", "text": "Trace text.", "data": "Optional JSON object."},
    authority="geist_local",
    side_effect="write",
    kind="local_trace_write",
    writes={"local_trace"},
    serial=True,
)

LOCAL_TOOL_LIST_TOOL = ToolSpec(
    name="tool.list_local",
    description="List generated local tools.",
    arguments={},
    authority="geist_local",
    side_effect="none",
    kind="local_tool_catalog",
    reads={"local_tools"},
)

LOCAL_TOOL_SCAFFOLD_TOOL = ToolSpec(
    name="tool.scaffold",
    description="Register or update one generated local Python tool.",
    arguments={
        "name": "Tool name matching local.<name>.",
        "description": "Tool description.",
        "arguments": "Argument schema object.",
        "code": "Python source defining execute(arguments, state, workspace, tool_api=None).",
        "overwrite": "Optional boolean.",
    },
    authority="geist_local",
    side_effect="write",
    kind="local_tool_scaffold",
    writes={"local_tools"},
    serial=True,
)


DEFAULT_LOCAL_TOOL_SPECS = {
    item.name: item
    for item in (
        READ_TOOL,
        WRITE_TOOL,
        EDIT_TOOL,
        LS_TOOL,
        BASH_TOOL,
        GIT_STATUS_TOOL,
        GIT_DIFF_SUMMARY_TOOL,
        GIT_DIFF_READ_TOOL,
        GIT_SNAPSHOT_TOOL,
        GIT_DELTA_TOOL,
        ARTIFACT_READ_TOOL,
        ARTIFACT_LIST_TOOL,
        ARTIFACT_SEARCH_TOOL,
        TRACE_READ_TOOL,
        TRACE_WRITE_TOOL,
        LOCAL_TOOL_LIST_TOOL,
        LOCAL_TOOL_SCAFFOLD_TOOL,
    )
}
