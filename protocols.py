"""Protocols and dataclasses for the selfcall kernel.

The kernel models recursive runtime re-entry, not parent/child hierarchy.
Every selfcall invocation is a bounded, isolated run of the same core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

StateT = TypeVar("StateT")


class StateSnapshotter(Protocol[StateT]):
    """Protocol for cloning and inspecting runtime state.

    The kernel does not know the concrete state type.  Adapters
    implement this protocol to provide clone / dump / delta operations.
    """

    def clone(self, state: StateT | None) -> StateT:
        """Return an isolated copy suitable for a selfcall invocation."""
        ...

    def dump(self, state: StateT) -> dict[str, Any]:
        """Serialize state to a plain dict for diff/persist."""
        ...


class RuntimeCore(Protocol):
    """Protocol for the core loop of a selfcall-capable runtime.

    The kernel calls ``run()`` to recursively enter the same core with
    a different context.  There is no parent/child authority implied
    by this protocol — only a call-stack relationship.
    """

    async def run(
        self,
        user_input: str,
        session: Any,
        /,
        *,
        state: Any | None = None,
        max_tool_rounds: int | None = None,
        selfcall_depth: int | None = None,
        selfcall_registry: Any | None = None,
    ) -> "FractalResult":
        ...


class ToolGate(Protocol):
    """Protocol for building a selfcall-appropriate tool registry.

    Adaptors decide which tools are safe for recursive invocations
    and return a registry that the kernel passes to the runtime core.
    """

    def build_selfcall_registry(
        self,
        requested_tools: Any,
        current_depth: int,
        max_depth: int,
    ) -> Any:
        """Return a filtered tool registry for a selfcall invocation."""
        ...


@dataclass
class FractalResult:
    """Result returned by RuntimeCore.run after a selfcall invocation."""

    ok: bool
    response: str = ""
    error: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tool_rounds: int = 0
    continuation_context: str = ""
