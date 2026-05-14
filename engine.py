"""FractalEngine — bounded fractal self-invocation.

This is the kernel's central piece.  It accepts a pre-built
FractalContext (constructed by the adapter), guards it against
depth/time limits, isolates state, enters the runtime core,
and captures the result into the context's ``own`` layer.

The engine does NOT construct, inherit, or transform the context.
Flow is managed entirely by the adapter.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from geist.time import reserve_slot
from geist.config import FractalConfig
from geist.context import FractalContext
from geist.protocols import RuntimeCore, StateSnapshotter, ToolGate
from geist.depth import current_depth, would_exceed


class FractalEngine:
    """Bounded fractal self-invocation.

    Usage sketch::

        engine = FractalEngine(config, snapshotter, tool_gate, invariant={...})

        ctx = FractalContext(invariant=engine.invariant, growth=adapter_built_growth)
        result = await engine.execute(ctx, state=..., runtime_core=self, ...)
        # result.own now contains this fractal's generation
    """

    def __init__(
        self,
        config: FractalConfig,
        snapshotter: StateSnapshotter[Any],
        tool_gate: ToolGate,
        invariant: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._snapshotter = snapshotter
        self._tool_gate = tool_gate
        self._invariant = invariant if invariant is not None else {}

    @property
    def invariant(self) -> dict[str, Any]:
        """The immutable self-anchor shared by all fractal instances."""
        return self._invariant

    async def execute(
        self,
        context: FractalContext,
        *,
        state: Any,
        project_name: str,
        runtime_core: RuntimeCore,
        session_factory: Any,  # callable(**kwargs) returning a session object
        next_depth: int | None = None,
    ) -> FractalContext:
        """Run *context* as a bounded fractal invocation.

        Guards (depth, budget) are checked first.
        State is cloned for isolation.
        Tools are filtered through the ToolGate.

        The passed *context* is mutated: ``own`` and ``ok``/``error``
        are set after the runtime core completes.  The same object
        is returned for convenience.
        """
        depth = next_depth if next_depth is not None else current_depth()

        # -- guard: depth ---------------------------------------------------
        if would_exceed(self._config.max_depth, depth):
            return FractalContext(
                invariant=context.invariant,
                growth={},
                ok=False,
                error=f"max depth={self._config.max_depth}",
            )

        # -- guard: time ----------------------------------------------------
        if not await reserve_slot():
            return FractalContext(
                invariant=context.invariant,
                growth={},
                ok=False,
                error="budget exhausted",
            )

        # -- state isolation ------------------------------------------------
        state_snapshot = self._snapshotter.clone(state)
        before = self._snapshotter.dump(state_snapshot)

        # -- tool policy ----------------------------------------------------
        max_rounds = self._config.max_tool_rounds
        selfcall_registry = self._tool_gate.build_selfcall_registry(
            None, current_depth=depth, max_depth=self._config.max_depth
        )

        # -- recursive entry ------------------------------------------------
        session_id = (
            f"{getattr(state_snapshot, 'session_id', 'session')}"
            f":fractal:{uuid4().hex[:8]}"
        )
        session = session_factory(
            session_id=session_id,
            workspace_label=getattr(state_snapshot, "workspace_label", project_name),
        )

        try:
            result = await runtime_core.run(
                json.dumps(context.resolve(), ensure_ascii=False, default=str),
                session,
                state=state_snapshot,
                max_tool_rounds=max_rounds,
                selfcall_depth=depth + 1,
                selfcall_registry=selfcall_registry,
            )
        except Exception as exc:
            context.ok = False
            context.error = str(exc)
            return context

        # -- capture own layer ----------------------------------------------
        after = self._snapshotter.dump(state_snapshot)
        context.ok = result.ok
        context.own = {
            "response": result.response,
            "tool_calls": result.tool_calls,
            "tool_results": result.tool_results,
            "state_delta_preview": self._delta(before, after),
            "tool_rounds": result.tool_rounds,
            "continuation_context": result.continuation_context,
        }
        if not result.ok:
            context.error = result.error

        return context

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        changed = [k for k, v in after.items() if before.get(k) != v]
        preview = {k: str(after.get(k))[:500] for k in changed[:8]}
        return {"changed_keys": changed, "preview": preview}
