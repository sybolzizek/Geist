"""Fractal context — a pure container for fractal self-invocation.

The kernel provides the container and the execution guard.
The adapter builds the context and reads the results.
Flow is external: adapter reads own, constructs next context,
passes it to the runner.  No inheritance, no unfold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FractalContext:
    """The context of a single fractal invocation.

    ``invariant``
        The immutable self-anchor, shared by ALL fractals.
        "You can change everything except this sentence."
        Never modified after root construction.

    ``growth``
        The context of this fractal invocation.  Fully constructed
        by the adapter before being passed to the runner.
        No automatic inheritance from any previous fractal.

    ``own``
        This fractal's own generation, set by the runner *after*
        the runtime core completes.  The adapter reads ``own``
        to extract blueprints and context updates for subsequent
        fractal invocations.

    ``ok`` / ``error``
        Execution status set by the runner.
    """

    invariant: dict[str, Any]
    growth: dict[str, Any]
    own: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""

    def resolve(self) -> dict[str, Any]:
        """Collapse invariant + growth + own into a single flat view.

        The runtime core consumes this view.  own overrides growth
        overrides invariant on key conflict.
        """
        return {**self.invariant, **self.growth, **self.own}
