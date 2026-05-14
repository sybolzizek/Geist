"""Configuration for a selfcall-capable runtime.

Every bound is explicit and finite.  The kernel refuses to run
without a concrete configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FractalConfig:
    """Immutable configuration for bounded recursive self-invocation.

    All limits are non-negative integers.  Setting a limit to 0
    disables the corresponding capability entirely.
    """

    max_depth: int = 2
    """Maximum recursive depth.  Root invocation = 0."""

    max_invocations_per_root: int = 4
    """Maximum total selfcall invocations across all depths in a root run."""

    max_tool_rounds: int = 2
    """Default tool-call rounds for a single selfcall invocation."""
