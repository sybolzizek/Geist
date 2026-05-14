"""geist — fractal self-invocation kernel.

A portable, protocol-driven mechanism for an agent runtime to
recursively unfold itself in a bounded, layered context space.

It does NOT introduce subagents, parent/child hierarchies, role-play,
or expert-group semantics.  Every fractal invocation is the same self
running with a different context slice.
"""

from geist.config import FractalConfig
from geist.context import FractalContext
from geist.depth import current_depth, enter_invocation, leave_invocation, would_exceed
from geist.engine import FractalEngine
from geist.loop import FractalLoop
from geist.protocols import FractalResult, RuntimeCore, StateSnapshotter, ToolGate
from geist.store import SharedContext
from geist.time import FractalTime, ensure_time, reserve_slot, reset_time

__all__ = [
    # Configuration
    "FractalConfig",
    # Fractal context
    "FractalContext",
    # Depth
    "current_depth",
    "enter_invocation",
    "leave_invocation",
    "would_exceed",
    # Time
    "FractalTime",
    "ensure_time",
    "reset_time",
    "reserve_slot",
    # Protocols
    "FractalResult",
    "RuntimeCore",
    "StateSnapshotter",
    "ToolGate",
    # Engine
    "FractalEngine",
    # Loop
    "FractalLoop",
    "SharedContext",
]
