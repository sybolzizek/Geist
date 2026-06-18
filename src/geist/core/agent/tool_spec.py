"""Generic agent tool specification and profile.

Domain-agnostic.  No manga, VinEnd, or ShortTermMemory coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Generic tool definition.

    Every domain plugin registers its tools via this spec.  The agent runtime
    uses reads/writes/serial to batch independent calls in parallel.
    """

    name: str
    description: str
    arguments: dict[str, Any] = field(default_factory=dict)
    authority: str = "geist"
    side_effect: str = "none"
    kind: str = "tool"
    doc: str = ""
    executable: bool = True
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    serial: bool = False

    def prompt_line(self) -> str:
        args = self.arguments if self.arguments else {}
        return (
            f"- `{self.name}` | {self.description} | "
            f"args={args} | side_effect={self.side_effect} | doc={self.doc or '-'}"
        )


def render_tool_manifest(tool_registry: dict[str, ToolSpec]) -> str:
    """Render a concise tool manifest for LLM context."""
    lines = ["## 真实工具清单", ""]
    for name, spec in sorted(tool_registry.items()):
        if not spec.executable:
            continue
        lines.append(spec.prompt_line())
    lines.append("")
    lines.append(
        "工具调用规则：\n"
        "- tool 必须是上面清单里的精确名称。\n"
        "- arguments 必须按对应工具 schema 填写。\n"
        "- 工具调用放在 ```runtime JSON 代码块里的 tool_calls 字段。\n"
        "- 不要发明工具；不确定时说明缺少什么。"
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class ToolProfile:
    """Resolved profile for a single tool call in a batch."""

    index: int
    call: dict[str, Any]
    spec: ToolSpec | None
    reads: set[str]
    writes: set[str]
    serial: bool = False
