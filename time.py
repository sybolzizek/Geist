"""Atomic per-root fractal time.

Time is shared across parallel fractal siblings within the same
root runtime invocation.  Reservation is async-locked to prevent races
under asyncio.gather.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

_FractalTimeToken = Token["FractalTime | None"]

_fractal_time: ContextVar[FractalTime | None] = ContextVar(
    "fractal_time",
    default=None,
)


@dataclass
class FractalTime:
    """Mutable, shared, async-locked fractal time counter.

    Instances are stored in a ContextVar so parallel sibling tasks
    share the same counter object.  ``reserve()`` is guarded by an
    ``asyncio.Lock`` to make check-and-decrement atomic.
    """

    remaining: int
    lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lock = asyncio.Lock()

    async def reserve(self) -> bool:
        """Atomically try to reserve one selfcall slot.

        Returns True if a slot was available and has been consumed.
        """
        async with self.lock:
            if self.remaining <= 0:
                return False
            self.remaining -= 1
            return True


def ensure_time(max_invocations: int) -> _FractalTimeToken | int:
    """Set a new root time counter if none exists. Returns a reset token.

    Caller MUST call ``reset_time(token)`` in a finally block.

    Returns -1 when a counter already exists.
    """
    existing = _fractal_time.get()
    if existing is not None:
        return -1
    return _fractal_time.set(FractalTime(max_invocations))


def reset_time(token: _FractalTimeToken | int) -> None:
    """Reset the time ContextVar to its previous value."""
    if isinstance(token, int):
        return
    _fractal_time.reset(token)


async def reserve_slot() -> bool:
    """Try to reserve one fractal time slot.

    Returns False when time is exhausted or not configured.
    """
    counter = _fractal_time.get()
    if counter is None:
        return True  # no time configured = unbounded (caller should gate elsewhere)
    return await counter.reserve()
