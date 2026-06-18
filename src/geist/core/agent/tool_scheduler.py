"""Dependency-aware scheduler for agent tool calls.

Domain-agnostic.  No manga, VinEnd, or ShortTermMemory coupling.
"""

from __future__ import annotations

import asyncio
from typing import Any

from geist.core.agent.tool_spec import ToolProfile, ToolSpec


class ToolScheduler:
    """Run independent tool calls in parallel, serial when writes conflict.

    State-key aware: tools that write disjoint keys and do not read each
    other's writes can share a batch.  Unknown tools (spec=None) and tools
    marked ``serial=True`` always run alone.
    """

    def __init__(self, dispatcher: Any, tool_registry: dict[str, ToolSpec]) -> None:
        # dispatcher: callable with signature execute(tool_call, state, project_name) -> dict
        self._dispatcher = dispatcher
        self._tool_registry = tool_registry

    async def execute_round(
        self,
        tool_calls: list[dict[str, Any]],
        state: Any,
        project_name: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        batches = self.plan_batches(tool_calls)
        executed_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for batch_index, batch in enumerate(batches, start=1):
            if len(batch) == 1:
                rows = [await self._execute_one(batch[0], state, project_name, batch_index, parallel=False)]
            else:
                rows = await asyncio.gather(
                    *[
                        self._execute_one(profile, state, project_name, batch_index, parallel=True)
                        for profile in batch
                    ]
                )

            for call_with_result, result_row in rows:
                executed_calls.append(call_with_result)
                tool_results.append(result_row)

        return executed_calls, tool_results

    def plan_batches(self, tool_calls: list[dict[str, Any]]) -> list[list[ToolProfile]]:
        batches: list[list[ToolProfile]] = []
        for index, call in enumerate(tool_calls):
            profile = self._profile(index, call)
            if profile.serial:
                batches.append([profile])
                continue

            if batches and self._can_join_batch(profile, batches[-1]):
                batches[-1].append(profile)
            else:
                batches.append([profile])
        return batches

    async def _execute_one(
        self,
        profile: ToolProfile,
        state: Any,
        project_name: str,
        batch_index: int,
        *,
        parallel: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        tool_name = str(profile.call.get("tool") or "")
        try:
            result = await self._dispatcher.execute(profile.call, state, project_name)
        except Exception as exc:
            result = {
                "ok": False,
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "tool": tool_name,
            }
        call_with_result = dict(profile.call)
        call_with_result["batch"] = batch_index
        call_with_result["parallel"] = parallel
        call_with_result["result"] = result
        result_row = {
            "batch": batch_index,
            "parallel": parallel,
            "call_id": str(profile.call.get("call_id") or profile.call.get("id") or f"call_{batch_index}_{profile.index + 1}"),
            "tool": tool_name,
            "data": result,
        }
        return call_with_result, result_row

    def _profile(self, index: int, call: dict[str, Any]) -> ToolProfile:
        tool_name = str(call.get("tool") or "").strip()
        spec = self._tool_registry.get(tool_name)
        if spec is None:
            return ToolProfile(index=index, call=call, spec=None, reads=set(), writes={"*"}, serial=True)

        reads = set(spec.reads)
        writes = set(spec.writes)
        if tool_name == "state.read":
            raw_args = call.get("arguments")
            args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}
            target_parts = str(args.get("target") or "").split(None, 1)
            target = target_parts[0] if target_parts else ""
            reads = {target} if target else {"*"}

        return ToolProfile(
            index=index,
            call=call,
            spec=spec,
            reads=reads,
            writes=writes,
            serial=spec.serial,
        )

    def _can_join_batch(self, profile: ToolProfile, batch: list[ToolProfile]) -> bool:
        if any(item.serial for item in batch):
            return False
        return all(self._can_run_together(profile, item) for item in batch)

    @staticmethod
    def _can_run_together(a: ToolProfile, b: ToolProfile) -> bool:
        if "*" in a.writes or "*" in b.writes:
            return False
        if "*" in a.reads and b.writes:
            return False
        if "*" in b.reads and a.writes:
            return False
        if a.writes & b.writes:
            return False
        if a.writes & b.reads:
            return False
        if b.writes & a.reads:
            return False
        return True
