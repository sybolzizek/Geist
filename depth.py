"""Depth guard for recursive selfcall invocations.

Uses a ContextVar so recursive sibling invocations (via asyncio.gather)
each inherit the current depth correctly.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_fractal_depth: ContextVar[int] = ContextVar("fractal_depth", default=0)


def current_depth() -> int:
    """Return the current selfcall depth (root = 0)."""
    return _fractal_depth.get()


def enter_invocation(depth: int | None = None) -> Token[int]:
    """Set the ContextVar and return a reset token.

    The caller MUST call ``leave_invocation(token)`` in a finally block.
    """
    value = depth if depth is not None else _fractal_depth.get()
    return _fractal_depth.set(value)


def leave_invocation(token: Token[int]) -> None:
    """Reset the depth ContextVar."""
    _fractal_depth.reset(token)


def would_exceed(configured_max: int, depth: int | None = None) -> bool:
    """Return True if entering one more level would exceed the configured max."""
    current = depth if depth is not None else _fractal_depth.get()
    return current >= configured_max
