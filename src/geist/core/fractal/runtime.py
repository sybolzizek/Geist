"""Runtime-native fractal API-call scheduler.

The unit here is one LLM API call. A call receives an explicit input packet,
may request local tools, may create follow-up calls through ``fractals``, or
may complete with natural-language output.

The scheduler does not clone conversation history, create child agents, or
give the model hidden ancestry. Tool results and handoff text move forward only
as explicit observations or continuation text.

Continuation text is a replaceable projection of what the next call should
carry. Observations are current events, not a hidden transcript; the runtime
keeps loop fingerprints internally instead of forcing old observations back
through model context.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from geist.core.agent.decision_parser import DecisionParser
from geist.core.agent.tool_scheduler import ToolScheduler
from geist.core.agent.tool_spec import ToolSpec
from geist.core.fractal.protocol import NATIVE_FRACTAL_PROTOCOL


RUNTIME_PROTOCOL_RETRY_PROMPT = """
Your previous response did not follow the native fractal runtime protocol.
The entire next message must be exactly one fenced `runtime` JSON object.
Start with ```runtime and end with ```.
Do not include explanations, analysis, markdown prose, or a ```json block.
Do not use provider-native tool syntax such as XML invoke tags.
Required JSON fields: final_response, continuation_context,
clear_continuation_context, tool_calls, fractals.
""".strip()


DEFAULT_FINAL_RESPONSE_INSTRUCTION = """
Return the final user-facing response for the original task from the terminal
runtime results and trace material in observations. This is a new API call
after the runtime frontier has ended. Do not call tools. Do not emit fractals.
""".strip()


OBSERVATION_TEXT_KEYS = {
    "body",
    "content",
    "html",
    "output",
    "raw",
    "response",
    "stderr",
    "stdout",
    "text",
}
OBSERVATION_INLINE_TEXT_CHARS = 800
OBSERVATION_INLINE_OUTPUT_CHARS = 1200
OBSERVATION_FIRST_LINE_CHARS = 180
OBSERVATION_CONTEXT_INLINE_TOTAL_CHARS = 16000
OBSERVATION_CONTEXT_INLINE_ITEM_CHARS = 8000
OBSERVATION_CONTEXT_MAX_ITEMS = 32
OBSERVATION_CONTEXT_VALUE_TEXT_CHARS = 280
OBSERVATION_CONTEXT_TOOL_ITEMS = 8
TRACE_SYNTHESIS_HEAD_EVENTS = 80
TRACE_SYNTHESIS_TAIL_EVENTS = 160
COMPLETION_GUARD_TYPE = "fractal_completion_guard"
VERIFICATION_EVIDENCE_TOOLS = {
    "bash",
    "git.diff_summary",
    "git.status",
    "process",
    "serve",
}


@dataclass(frozen=True)
class FractalLimits:
    """Optional bounds for one native fractal run; ``None`` means unbounded."""

    max_expansion_rounds: int | None = None
    max_expanded_calls: int | None = None
    max_fractals_per_call: int | None = None
    max_tool_rounds_per_call: int | None = None
    max_llm_calls: int | None = None
    max_continuation_chars: int | None = None
    max_observations: int | None = None
    max_observation_chars: int | None = None
    max_total_observation_chars: int | None = None
    max_repeated_tool_observations_per_call: int = 3
    max_repeated_structural_motions_per_run: int = 6


@dataclass
class FractalCall:
    """Input packet for one LLM API call."""

    root_task: str
    instruction: str = ""
    continuation_context: str = ""
    observations: list[str] = field(default_factory=list)
    tool_state: Any = None
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    workspace_label: str = ""
    expansion_round: int = 0
    tool_round: int = 0
    call_id: str = ""
    parent_call_id: str = ""
    spawn_kind: str = "root"
    spawn_sequence: int | None = None
    sibling_index: int = 1
    sibling_count: int = 1
    branch_path: str = "1"
    observation_fingerprints: tuple[str, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class FractalCompleted:
    """Natural output produced by one completed call."""

    response: str
    continuation_context: str = ""
    observations: tuple[str, ...] = ()
    tool_state: Any = None
    ok: bool = True
    error: str = ""
    pending_verification: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FractalRun:
    """Observable result of one peer-fractal execution."""

    results: tuple[FractalCompleted, ...]
    expansion_count: int
    trace: tuple[dict[str, Any], ...] = ()
    trace_synthesis: FractalCompleted | None = None
    terminal_results: tuple[FractalCompleted, ...] = ()


@dataclass(frozen=True)
class _Decision:
    response: str
    tool_calls: tuple[dict[str, Any], ...]
    fractals: tuple["_FractalInstruction", ...]
    continuation_context: str | None = None
    clear_continuation_context: bool = False


@dataclass(frozen=True)
class _FractalInstruction:
    instruction: str
    continuation_context: str = ""


@dataclass(frozen=True)
class _CallStep:
    next_calls: tuple[FractalCall, ...] = ()
    terminal_results: tuple[FractalCompleted, ...] = ()


class _RuntimeLimitReached(RuntimeError):
    """Internal signal that a runtime-wide bound was hit during async work."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FractalRuntime:
    """Schedule API calls until every path completes or hits a bound."""

    def __init__(
        self,
        llm: Any,
        dispatcher: Any,
        *,
        context_builder: Callable[[FractalCall], list[dict[str, Any]]] | None = None,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
        material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        wave_token: str | None = None,
        limits: FractalLimits | None = None,
        stream_llm: bool = False,
        live_llm_trace_chars: int = 300,
    ) -> None:
        self._llm = llm
        self._dispatcher = dispatcher
        self._context_builder = context_builder or self._default_context
        self._trace_sink = trace_sink
        self._material_store = material_store
        self._wave_token = wave_token
        self._limits = limits or FractalLimits()
        self._stream_llm = stream_llm
        self._live_llm_trace_chars = live_llm_trace_chars
        self._decision_parser = DecisionParser()
        self._trace: list[dict[str, Any]] = []
        self._expanded_call_count = 0
        self._call_id_counter = 0
        self._llm_calls = 0
        self._started_at = 0.0
        self._run_id = ""
        self._structural_motion_counts: dict[str, int] = {}

    async def run(
        self,
        call: FractalCall,
        *,
        trace_synthesis_instruction: str = "",
        final_response_instruction: str = DEFAULT_FINAL_RESPONSE_INSTRUCTION,
    ) -> FractalRun:
        """Run one call packet; expansion creates more call packets."""
        self._trace = []
        self._expanded_call_count = 0
        self._call_id_counter = 0
        self._llm_calls = 0
        self._started_at = time.monotonic()
        self._run_id = f"fr_{int(time.time() * 1000)}"
        self._structural_motion_counts = {}
        call = self._prepare_initial_call(call)
        terminal_run = await self._run_frontier(call)
        results = terminal_run.results
        terminal_results: tuple[FractalCompleted, ...] = ()
        if final_response_instruction.strip() and self._expanded_call_count > 0:
            terminal_results = terminal_run.results
            final_result = await self._finalize_from_terminal_results(
                call,
                terminal_run,
                instruction=final_response_instruction.strip(),
            )
            results = (final_result,)
        trace_synthesis = None
        if trace_synthesis_instruction.strip():
            trace_synthesis = await self._synthesize_from_trace(
                call,
                terminal_run,
                instruction=trace_synthesis_instruction.strip(),
            )
        return FractalRun(
            results=results,
            expansion_count=self._expanded_call_count,
            trace=tuple(self._trace),
            trace_synthesis=trace_synthesis,
            terminal_results=terminal_results,
        )

    def trace_snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return the trace collected so far without mutating the runtime."""
        return tuple(dict(item) for item in self._trace)

    async def _run_frontier(self, initial_call: FractalCall) -> FractalRun:
        terminal_results: list[FractalCompleted] = []
        active: dict[asyncio.Task[_CallStep], FractalCall] = {
            asyncio.create_task(self._run_call_once(initial_call)): initial_call,
        }

        while active:
            done, _pending = await asyncio.wait(
                active.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                call = active.pop(task)
                try:
                    step = task.result()
                except Exception as exc:
                    terminal_results.extend(self._error_result(call, exc).results)
                    continue
                terminal_results.extend(step.terminal_results)
                for next_call in step.next_calls:
                    active[asyncio.create_task(self._run_call_once(next_call))] = next_call

        if self._expanded_call_count:
            self._record(
                "terminal_pool_ready",
                result_count=len(terminal_results),
            )
        return FractalRun(
            results=tuple(terminal_results),
            expansion_count=self._expanded_call_count,
        )

    def _prepare_initial_call(self, call: FractalCall) -> FractalCall:
        observation_fingerprints = call.observation_fingerprints or tuple(
            item
            for item in (_observation_fingerprint(text) for text in call.observations)
            if item
        )
        if call.call_id:
            if observation_fingerprints == call.observation_fingerprints:
                return call
            return replace(call, observation_fingerprints=observation_fingerprints)
        return replace(
            call,
            call_id=self._new_call_id(),
            parent_call_id="",
            spawn_kind="root",
            spawn_sequence=None,
            sibling_index=1,
            sibling_count=1,
            branch_path="1",
            observation_fingerprints=observation_fingerprints,
        )

    async def _run_call_once(self, call: FractalCall) -> _CallStep:
        self._record_call(
            call,
            "start",
            instruction=_truncate_for_trace(call.instruction or call.root_task),
            observation_count=len(call.observations),
        )
        if _limit_reached(self._llm_calls, self._limits.max_llm_calls):
            return _CallStep(terminal_results=self._forced_result(call, reason="max_llm_calls").results)

        try:
            decision = await self._decide(call)
        except _RuntimeLimitReached as exc:
            return _CallStep(terminal_results=self._forced_result(call, reason=exc.reason).results)
        except Exception as exc:
            return _CallStep(terminal_results=self._error_result(call, exc).results)

        continuation = self._resolve_continuation(call, decision)

        pre_expansion_observations: list[str] = list(call.observations)
        pre_expansion_fingerprints: tuple[str, ...] = tuple(call.observation_fingerprints)
        if decision.tool_calls and decision.fractals:
            if _limit_reached(call.tool_round, self._limits.max_tool_rounds_per_call):
                return _CallStep(terminal_results=self._forced_result(
                    call,
                    reason="max_tool_rounds_per_call",
                    response=decision.response,
                    continuation_context=continuation,
                ).results)
            self._record_call(
                call,
                "tools",
                tools=[str(item.get("tool") or "") for item in decision.tool_calls],
                tool_calls=_summarize_tool_calls_for_trace(decision.tool_calls),
                before_expansion=True,
            )
            observation, _observation_sequence = await self._execute_tools(call, list(decision.tool_calls))
            pre_expansion_observations = self._append_observation(call.observations, observation, call=call)
            pre_expansion_fingerprints = self._append_observation_fingerprints(call, observation)
            stopped = self._structural_loop_result(
                call,
                decision=decision,
                tool_calls=list(decision.tool_calls),
                observation=observation,
                continuation=continuation,
                next_observations=pre_expansion_observations,
            )
            if stopped is not None:
                return _CallStep(terminal_results=stopped.results)

        if decision.fractals:
            if _limit_reached(call.expansion_round, self._limits.max_expansion_rounds):
                return _CallStep(terminal_results=self._forced_result(
                    call,
                    reason="max_expansion_rounds",
                    response=decision.response,
                    continuation_context=continuation,
                ).results)
            if _limit_reached(self._expanded_call_count, self._limits.max_expanded_calls):
                return _CallStep(terminal_results=self._forced_result(
                    call,
                    reason="max_expanded_calls",
                    response=decision.response,
                    continuation_context=continuation,
                ).results)
            fractal_limit = self._limits.max_fractals_per_call
            allowed = (
                decision.fractals[:fractal_limit]
                if _limit_enabled(fractal_limit)
                else decision.fractals
            )
            if len(allowed) < len(decision.fractals):
                self._record_call(
                    call,
                    "limit",
                    reason="max_fractals_per_call",
                    requested=len(decision.fractals),
                    allowed=len(allowed),
                )
            if not allowed:
                return _CallStep(terminal_results=self._forced_result(
                    call,
                    reason="no_fractals_allowed",
                    response=decision.response,
                    continuation_context=continuation,
                ).results)
            self._expanded_call_count += 1
            expand_event = self._record_call(
                call,
                "expand",
                count=len(allowed),
                continuation_context=continuation,
                instructions=[item.instruction for item in allowed],
            )
            calls = [
                self._next_call(
                    call,
                    instruction=item.instruction,
                    continuation_context=_child_continuation(continuation, item.continuation_context),
                    observations=pre_expansion_observations,
                    observation_fingerprints=pre_expansion_fingerprints,
                    expansion_round=call.expansion_round + 1,
                    tool_round=0,
                    spawn_kind="fractal",
                    spawn_sequence=int(expand_event["sequence"]),
                    sibling_index=index + 1,
                    sibling_count=len(allowed),
                    branch_path=f"{call.branch_path}.{index + 1}",
                )
                for index, item in enumerate(allowed)
            ]
            return _CallStep(next_calls=tuple(calls))

        if decision.tool_calls:
            if _limit_reached(call.tool_round, self._limits.max_tool_rounds_per_call):
                return _CallStep(terminal_results=self._forced_result(
                    call,
                    reason="max_tool_rounds_per_call",
                    response=decision.response,
                    continuation_context=continuation,
                ).results)
            self._record_call(
                call,
                "tools",
                tools=[str(item.get("tool") or "") for item in decision.tool_calls],
                tool_calls=_summarize_tool_calls_for_trace(decision.tool_calls),
            )
            observation, observation_sequence = await self._execute_tools(call, list(decision.tool_calls))
            next_observations = self._append_observation(call.observations, observation, call=call)
            stopped = self._structural_loop_result(
                call,
                decision=decision,
                tool_calls=list(decision.tool_calls),
                observation=observation,
                continuation=continuation,
                next_observations=next_observations,
            )
            if stopped is not None:
                return _CallStep(terminal_results=stopped.results)
            repeat_count = max(
                self._repeated_observation_count(call.observations, observation),
                self._repeated_observation_fingerprint_count(call, observation),
            )
            repeat_limit = self._limits.max_repeated_tool_observations_per_call
            if repeat_limit > 0 and repeat_count >= repeat_limit:
                next_fingerprints = self._append_observation_fingerprints(call, observation)
                stopped_call = self._next_call(
                    call,
                    instruction=call.instruction,
                    continuation_context=continuation,
                    observations=next_observations,
                    observation_fingerprints=next_fingerprints,
                    expansion_round=call.expansion_round,
                    tool_round=call.tool_round + 1,
                    spawn_kind="tool_observation",
                    spawn_sequence=observation_sequence,
                    sibling_index=1,
                    sibling_count=1,
                    branch_path=call.branch_path,
                )
                self._record_call(
                    stopped_call,
                    "tool_loop_stopped",
                    reason="repeated_tool_observation",
                    repeat_count=repeat_count,
                    repeat_limit=repeat_limit,
                    observation=_truncate_for_trace(observation, max_chars=self._limits.max_observation_chars),
                )
                return _CallStep(terminal_results=self._forced_result(
                    stopped_call,
                    reason="repeated_tool_observation",
                    response=(
                        "fractal runtime stopped a repeated tool loop: the same "
                        f"tool observation appeared {repeat_count} times in this API call."
                    ),
                    continuation_context=continuation,
                ).results)
            next_fingerprints = self._append_observation_fingerprints(call, observation)
            return _CallStep(next_calls=(self._next_call(
                call,
                instruction=call.instruction,
                continuation_context=continuation,
                observations=next_observations,
                observation_fingerprints=next_fingerprints,
                expansion_round=call.expansion_round,
                tool_round=call.tool_round + 1,
                spawn_kind="tool_observation",
                spawn_sequence=observation_sequence,
                sibling_index=1,
                sibling_count=1,
                branch_path=call.branch_path,
            ),))

        pending_verification = _pending_verification_from_observations(call.observations)
        auto_verification_calls = _auto_verification_tool_calls(call, pending_verification)
        if auto_verification_calls:
            self._record_call(
                call,
                "auto_verification",
                reason="pending_verification",
                pending_verification=pending_verification,
                tools=[str(item.get("tool") or "") for item in auto_verification_calls],
                tool_calls=_summarize_tool_calls_for_trace(auto_verification_calls),
            )
            observation, observation_sequence = await self._execute_tools(call, auto_verification_calls)
            next_observations = self._append_observation(call.observations, observation, call=call)
            next_fingerprints = self._append_observation_fingerprints(call, observation)
            return _CallStep(next_calls=(self._next_call(
                call,
                instruction=call.instruction,
                continuation_context=continuation,
                observations=next_observations,
                observation_fingerprints=next_fingerprints,
                expansion_round=call.expansion_round,
                tool_round=call.tool_round + 1,
                spawn_kind="auto_verification",
                spawn_sequence=observation_sequence,
                sibling_index=1,
                sibling_count=1,
                branch_path=call.branch_path,
            ),))
        if _should_guard_completion(call, pending_verification):
            guard_observation = _build_completion_guard_observation(
                pending_verification,
                attempted_response=decision.response,
            )
            event = self._record_call(
                call,
                "completion_guard",
                reason="pending_verification",
                pending_verification=pending_verification,
                attempted_response=_truncate_for_trace(decision.response, max_chars=1200),
            )
            next_observations = self._append_observation(call.observations, guard_observation, call=call)
            next_fingerprints = self._append_observation_fingerprints(call, guard_observation)
            return _CallStep(next_calls=(self._next_call(
                call,
                instruction=call.instruction,
                continuation_context=continuation,
                observations=next_observations,
                observation_fingerprints=next_fingerprints,
                expansion_round=call.expansion_round,
                tool_round=call.tool_round,
                spawn_kind="completion_guard",
                spawn_sequence=int(event["sequence"]),
                sibling_index=1,
                sibling_count=1,
                branch_path=call.branch_path,
            ),))

        if _should_guard_empty_completion(call, decision.response):
            guard_observation = _build_empty_completion_guard_observation()
            event = self._record_call(
                call,
                "completion_guard",
                reason="empty_final_response",
            )
            next_observations = self._append_observation(call.observations, guard_observation, call=call)
            next_fingerprints = self._append_observation_fingerprints(call, guard_observation)
            return _CallStep(next_calls=(self._next_call(
                call,
                instruction=call.instruction,
                continuation_context=continuation,
                observations=next_observations,
                observation_fingerprints=next_fingerprints,
                expansion_round=call.expansion_round,
                tool_round=call.tool_round,
                spawn_kind="completion_guard",
                spawn_sequence=int(event["sequence"]),
                sibling_index=1,
                sibling_count=1,
                branch_path=call.branch_path,
            ),))

        self._record_call(
            call,
            "complete",
            response=decision.response,
            continuation_context=continuation,
            pending_verification=pending_verification or None,
        )
        pending_error = "verification_pending" if pending_verification else ""
        empty_error = "empty_final_response" if _empty_completion_error(call, decision.response) else ""
        error = pending_error or empty_error
        return _CallStep(
            terminal_results=(FractalCompleted(
                response=decision.response,
                continuation_context=continuation,
                observations=tuple(call.observations),
                tool_state=call.tool_state,
                ok=not bool(error),
                error=error,
                pending_verification=pending_verification,
            ),),
        )

    async def _decide(self, call: FractalCall) -> _Decision:
        messages = self._context_builder(call)
        self._record_call(
            call,
            "context_budget",
            **_inspect_context_budget(messages, call=call, limits=self._limits),
        )
        raw = await self._generate(messages, call=call)
        text = str(raw or "")
        payload = self._extract_payload(text)
        provider_tool_calls = _extract_provider_tool_calls(text, call.tools)
        if not payload and provider_tool_calls:
            self._record_call(
                call,
                "provider_tool_syntax_translated",
                response=_truncate_for_trace(_text_before_provider_tool_calls(text), max_chars=2000),
                tools=[str(item.get("tool") or "") for item in provider_tool_calls],
            )
            return _Decision(
                response=_text_before_provider_tool_calls(text),
                tool_calls=tuple(provider_tool_calls),
                fractals=(),
                continuation_context=call.continuation_context,
            )
        looks_like_runtime_attempt = _looks_like_runtime_json_attempt(text)
        partial_tool_calls = _extract_partial_runtime_tool_calls(text, call.tools) if looks_like_runtime_attempt else []
        if not payload and partial_tool_calls:
            self._record_call(
                call,
                "protocol_partial_tool_recovery",
                response=_truncate_for_trace(text, max_chars=2000),
                tools=[str(item.get("tool") or "") for item in partial_tool_calls],
            )
            return _Decision(
                response="",
                tool_calls=tuple(partial_tool_calls),
                fractals=(),
                continuation_context=call.continuation_context,
            )
        if (
            not payload
            and "```runtime" not in text.lower()
            and not _contains_provider_tool_syntax(text)
            and not looks_like_runtime_attempt
        ):
            self._record_call(
                call,
                "protocol_plain_text_fallback",
                response=_truncate_for_trace(text, max_chars=2000),
            )
            return _Decision(
                response=_plain_text_fallback(text),
                tool_calls=(),
                fractals=(),
                continuation_context=call.continuation_context,
            )
        if not payload and not _limit_reached(self._llm_calls, self._limits.max_llm_calls):
            self._record_call(
                call,
                "protocol_retry",
                reason=(
                    "invalid_runtime_json"
                    if "```runtime" in text.lower() or looks_like_runtime_attempt else "missing_runtime_json"
                ),
                response=_truncate_for_trace(text, max_chars=2000),
            )
            raw = await self._generate([
                *messages,
                {"role": "assistant", "content": _truncate_for_trace(text, max_chars=2000)},
                {"role": "user", "content": RUNTIME_PROTOCOL_RETRY_PROMPT},
            ], call=call)
            text = str(raw or "")
            payload = self._extract_payload(text)
            provider_tool_calls = _extract_provider_tool_calls(text, call.tools)
            looks_like_runtime_attempt = _looks_like_runtime_json_attempt(text)
            partial_tool_calls = _extract_partial_runtime_tool_calls(text, call.tools) if looks_like_runtime_attempt else []
            if not payload and provider_tool_calls:
                self._record_call(
                    call,
                    "provider_tool_syntax_translated",
                    response=_truncate_for_trace(_text_before_provider_tool_calls(text), max_chars=2000),
                    tools=[str(item.get("tool") or "") for item in provider_tool_calls],
                )
                return _Decision(
                    response=_text_before_provider_tool_calls(text),
                    tool_calls=tuple(provider_tool_calls),
                    fractals=(),
                    continuation_context=call.continuation_context,
                )
            if not payload and partial_tool_calls:
                self._record_call(
                    call,
                    "protocol_partial_tool_recovery",
                    response=_truncate_for_trace(text, max_chars=2000),
                    tools=[str(item.get("tool") or "") for item in partial_tool_calls],
                )
                return _Decision(
                    response="",
                    tool_calls=tuple(partial_tool_calls),
                    fractals=(),
                    continuation_context=call.continuation_context,
                )
        if not payload and not _contains_provider_tool_syntax(text) and not _looks_like_runtime_json_attempt(text):
            self._record_call(
                call,
                "protocol_plain_text_fallback",
                response=_truncate_for_trace(text, max_chars=2000),
            )
            return _Decision(
                response=_plain_text_fallback(text),
                tool_calls=(),
                fractals=(),
                continuation_context=call.continuation_context,
            )
        if not payload:
            self._record_call(
                call,
                "protocol_error",
                response=_truncate_for_trace(text, max_chars=2000),
            )
            return _Decision(
                response=(
                    "geist protocol error: model did not return a valid "
                    "runtime JSON block after retry. No tools were executed."
                ),
                tool_calls=(),
                fractals=(),
                continuation_context=call.continuation_context,
            )

        response = str(payload.get("final_response") or "").strip()
        if not response:
            response = DecisionParser._text_before_runtime_payload(text)
        tool_calls = _normalize_runtime_tool_calls(payload.get("tool_calls"))
        continuation_context = payload.get("continuation_context")
        continuation_context = (
            str(continuation_context).strip()
            if continuation_context is not None
            else None
        )
        if (
            payload
            and "final_response" not in payload
            and not tool_calls
            and not payload.get("fractals")
            and payload.get("status")
        ):
            response = str(payload.get("status") or "").strip()
        if continuation_context is None and payload.get("continuation_content") is not None:
            continuation_context = str(payload.get("continuation_content") or "").strip()
        return _Decision(
            response=response,
            tool_calls=tuple(tool_calls),
            fractals=tuple(self._normalize_fractals(payload.get("fractals"))),
            continuation_context=continuation_context,
            clear_continuation_context=bool(payload.get("clear_continuation_context")),
        )

    async def _generate(self, messages: list[dict[str, Any]], *, call: FractalCall) -> Any:
        if _limit_reached(self._llm_calls, self._limits.max_llm_calls):
            raise _RuntimeLimitReached("max_llm_calls")
        self._llm_calls += 1
        llm_call = self._llm_calls
        if self._stream_llm and hasattr(self._llm, "stream_generate"):
            self._record_call(
                call,
                "llm_call_start",
                llm_call=llm_call,
                message_count=len(messages),
            )
            parts: list[str] = []
            delta_buffer: list[str] = []
            delta_buffer_chars = 0
            done_text = ""
            meta: dict[str, Any] = {}

            def flush_delta_buffer(*, force: bool = False) -> None:
                nonlocal delta_buffer_chars
                if not delta_buffer:
                    return
                if not force and delta_buffer_chars < max(1, self._live_llm_trace_chars):
                    return
                text = "".join(delta_buffer)
                self._record_call(
                    call,
                    "llm_delta",
                    llm_call=llm_call,
                    chars=len(text),
                    text=_truncate_for_trace(
                        text,
                        max_chars=self._live_llm_trace_chars,
                    ),
                )
                delta_buffer.clear()
                delta_buffer_chars = 0

            async for chunk in self._llm.stream_generate(
                messages,
                temperature=0.5,
                api_key=self._wave_token,
            ):
                if not isinstance(chunk, dict):
                    continue
                kind = str(chunk.get("type") or "")
                if kind == "delta":
                    text = str(chunk.get("text") or "")
                    if not text:
                        continue
                    parts.append(text)
                    delta_buffer.append(text)
                    delta_buffer_chars += len(text)
                    flush_delta_buffer()
                elif kind == "done":
                    done_text = str(chunk.get("text") or "")
                    if isinstance(chunk.get("meta"), dict):
                        meta = dict(chunk["meta"])
            flush_delta_buffer(force=True)
            full_text = "".join(parts) or done_text
            if not meta:
                meta = dict(getattr(self._llm, "last_response_meta", {}) or {})
            self._record_call(
                call,
                "llm_call_complete",
                llm_call=llm_call,
                chars=len(full_text),
                usage=meta.get("usage") if isinstance(meta, dict) else None,
                cost=meta.get("cost") if isinstance(meta, dict) else None,
                modelId=meta.get("modelId") if isinstance(meta, dict) else None,
            )
            return full_text
        return await self._llm.generate(
            messages,
            temperature=0.5,
            api_key=self._wave_token,
        )

    async def _execute_tools(self, call: FractalCall, tool_calls: list[dict[str, Any]]) -> tuple[str, int]:
        scheduler = ToolScheduler(self._dispatcher, call.tools)
        results = await self._execute_tool_round(scheduler, tool_calls, call)
        observation = self._summarize_results(results)
        event = self._record_call(
            call,
            "tool_observation",
            tools=[str(item.get("tool") or "") for item in tool_calls],
            observation=_truncate_for_trace(observation, max_chars=self._limits.max_observation_chars),
        )
        return observation, int(event["sequence"])

    def _structural_loop_result(
        self,
        call: FractalCall,
        *,
        decision: _Decision,
        tool_calls: list[dict[str, Any]],
        observation: str,
        continuation: str,
        next_observations: list[str],
    ) -> FractalRun | None:
        signature = _structural_motion_signature(tool_calls, tools=call.tools)
        if not signature:
            return None
        repeat_limit = self._limits.max_repeated_structural_motions_per_run
        if repeat_limit <= 0:
            return None
        repeat_count = self._structural_motion_counts.get(signature, 0) + 1
        self._structural_motion_counts[signature] = repeat_count
        if repeat_count < repeat_limit:
            return None
        structural_motion = _decode_structural_motion_signature(signature)
        self._record_call(
            call,
            "structural_loop_stopped",
            reason="repeated_structural_motion",
            repeat_count=repeat_count,
            repeat_limit=repeat_limit,
            signature=signature,
            structural_motion=structural_motion,
            repeated_target=_structural_motion_label(structural_motion),
            tools=[str(item.get("tool") or "") for item in tool_calls],
            response=_truncate_for_trace(decision.response, max_chars=1200),
            observation=_truncate_for_trace(observation, max_chars=self._limits.max_observation_chars),
        )
        return FractalRun(
            results=(FractalCompleted(
                response=(
                    decision.response.strip()
                    or "fractal runtime stopped a repeated read-only motion before further expansion."
                ),
                continuation_context=continuation,
                observations=tuple(next_observations),
                tool_state=call.tool_state,
                ok=False,
                error="repeated_structural_motion",
            ),),
            expansion_count=0,
        )

    async def _execute_tool_round(
        self,
        scheduler: ToolScheduler,
        tool_calls: list[dict[str, Any]],
        call: FractalCall,
    ) -> list[dict[str, Any]]:
        batches = scheduler.plan_batches(tool_calls)
        results: list[dict[str, Any]] = []
        prior_failures: list[dict[str, Any]] = []

        for batch_index, batch in enumerate(batches, start=1):
            pending: list[tuple[int, Any]] = []
            batch_rows: list[dict[str, Any] | None] = [None] * len(batch)
            for index, profile in enumerate(batch):
                blocked = _blocked_trace_write_result(profile.call, prior_failures)
                if blocked is not None:
                    batch_rows[index] = {
                        "batch": batch_index,
                        "parallel": False,
                        "call_id": _tool_call_id(profile.call, index=profile.index),
                        "tool": str(profile.call.get("tool") or ""),
                        "data": blocked,
                    }
                else:
                    pending.append((index, profile))

            if pending:
                if len(pending) == 1:
                    index, profile = pending[0]
                    batch_rows[index] = await self._execute_tool_profile(
                        profile,
                        call,
                        batch_index=batch_index,
                        parallel=False,
                    )
                else:
                    executed = await asyncio.gather(
                        *[
                            self._execute_tool_profile(
                                profile,
                                call,
                                batch_index=batch_index,
                                parallel=True,
                            )
                            for _, profile in pending
                        ]
                    )
                    for (index, _profile), row in zip(pending, executed):
                        batch_rows[index] = row

            for row in batch_rows:
                if row is None:
                    continue
                results.append(row)
                failure = _tool_failure_for_trace_guard(row)
                if failure:
                    prior_failures.append(failure)
        return results

    async def _execute_tool_profile(
        self,
        profile: Any,
        call: FractalCall,
        *,
        batch_index: int,
        parallel: bool,
    ) -> dict[str, Any]:
        tool_name = str(profile.call.get("tool") or "")
        try:
            result = await self._dispatcher.execute(profile.call, call.tool_state, call.workspace_label)
        except Exception as exc:
            result = {
                "ok": False,
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "tool": tool_name,
            }
        return {
            "batch": batch_index,
            "parallel": parallel,
            "call_id": _tool_call_id(profile.call, index=getattr(profile, "index", 0)),
            "tool": tool_name,
            "data": result,
        }

    async def _finalize_from_terminal_results(
        self,
        call: FractalCall,
        run: FractalRun,
        *,
        instruction: str,
    ) -> FractalCompleted:
        observation = _render_trace_synthesis_observation(
            trace=tuple(self._trace),
            results=run.results,
            expansion_count=self._expanded_call_count,
            max_chars=self._limits.max_observation_chars,
            material_store=self._material_store,
        )
        final_call = FractalCall(
            root_task=call.root_task,
            instruction=instruction,
            continuation_context=call.continuation_context,
            observations=[observation],
            tool_state=call.tool_state,
            tools={},
            workspace_label=call.workspace_label,
            expansion_round=call.expansion_round,
            tool_round=0,
            call_id=self._new_call_id(),
            parent_call_id=call.call_id,
            spawn_kind="final",
            spawn_sequence=None,
            sibling_index=1,
            sibling_count=1,
            branch_path=f"{call.branch_path}.final",
        )
        self._record_call(
            final_call,
            "final_call_start",
            result_count=len(run.results),
            observation_chars=len(observation),
        )
        if _limit_reached(self._llm_calls, self._limits.max_llm_calls):
            forced = self._forced_result(final_call, reason="max_llm_calls")
            return forced.results[0]

        try:
            decision = await self._decide(final_call)
        except _RuntimeLimitReached as exc:
            forced = self._forced_result(final_call, reason=exc.reason)
            return forced.results[0]
        except Exception as exc:
            failed = self._error_result(final_call, exc)
            return failed.results[0]

        continuation = self._resolve_continuation(final_call, decision)
        action_count = len(decision.tool_calls) + len(decision.fractals)
        if action_count:
            self._record_call(
                final_call,
                "final_call_ignored_actions",
                tool_count=len(decision.tool_calls),
                expansion_count=len(decision.fractals),
            )
        response = decision.response.strip()
        error = "final_call_requested_actions_without_response" if not response and action_count else ""
        self._record_call(
            final_call,
            "final_call_complete",
            response=response,
            continuation_context=continuation,
        )
        return FractalCompleted(
            response=response or "fractal runtime final call did not produce final_response",
            continuation_context=continuation,
            observations=tuple(final_call.observations),
            tool_state=final_call.tool_state,
            ok=not error,
            error=error,
        )

    async def _synthesize_from_trace(
        self,
        call: FractalCall,
        run: FractalRun,
        *,
        instruction: str,
    ) -> FractalCompleted:
        observation = _render_trace_synthesis_observation(
            trace=tuple(self._trace),
            results=run.results,
            expansion_count=self._expanded_call_count,
            max_chars=self._limits.max_observation_chars,
            material_store=self._material_store,
        )
        synthesis_call = FractalCall(
            root_task=call.root_task,
            instruction=instruction,
            continuation_context=call.continuation_context,
            observations=[observation],
            tool_state=call.tool_state,
            tools={},
            workspace_label=call.workspace_label,
            expansion_round=call.expansion_round,
            tool_round=0,
            call_id=self._new_call_id(),
            parent_call_id=call.call_id,
            spawn_kind="trace_synthesis",
            spawn_sequence=None,
            sibling_index=1,
            sibling_count=1,
            branch_path=f"{call.branch_path}.trace",
        )
        self._record_call(
            synthesis_call,
            "trace_synthesis_start",
            result_count=len(run.results),
            observation_chars=len(observation),
        )
        if _limit_reached(self._llm_calls, self._limits.max_llm_calls):
            forced = self._forced_result(synthesis_call, reason="max_llm_calls")
            return forced.results[0]

        try:
            decision = await self._decide(synthesis_call)
        except _RuntimeLimitReached as exc:
            forced = self._forced_result(synthesis_call, reason=exc.reason)
            return forced.results[0]
        except Exception as exc:
            failed = self._error_result(synthesis_call, exc)
            return failed.results[0]

        continuation = self._resolve_continuation(synthesis_call, decision)
        action_count = len(decision.tool_calls) + len(decision.fractals)
        if action_count:
            self._record_call(
                synthesis_call,
                "trace_synthesis_ignored_actions",
                tool_count=len(decision.tool_calls),
                expansion_count=len(decision.fractals),
            )
        response = decision.response.strip()
        error = "trace_synthesis_requested_actions_without_response" if not response and action_count else ""
        self._record_call(
            synthesis_call,
            "trace_synthesis_complete",
            response=response,
            continuation_context=continuation,
        )
        return FractalCompleted(
            response=response or "trace synthesis did not produce final_response",
            continuation_context=continuation,
            observations=tuple(synthesis_call.observations),
            tool_state=synthesis_call.tool_state,
            ok=not error,
            error=error,
        )

    def _resolve_continuation(self, call: FractalCall, decision: _Decision) -> str:
        if decision.clear_continuation_context:
            return ""
        if decision.continuation_context is None or not decision.continuation_context.strip():
            return call.continuation_context
        return _truncate_for_trace(
            decision.continuation_context,
            max_chars=self._limits.max_continuation_chars,
        )

    def _next_call(
        self,
        call: FractalCall,
        *,
        instruction: str,
        continuation_context: str,
        observations: list[str],
        expansion_round: int,
        tool_round: int,
        spawn_kind: str,
        spawn_sequence: int | None,
        sibling_index: int,
        sibling_count: int,
        branch_path: str,
        observation_fingerprints: tuple[str, ...] | None = None,
    ) -> FractalCall:
        return FractalCall(
            root_task=call.root_task,
            instruction=instruction,
            continuation_context=_truncate_for_trace(
                continuation_context,
                max_chars=self._limits.max_continuation_chars,
            ),
            observations=self._trim_observations(observations),
            observation_fingerprints=(
                tuple(observation_fingerprints)
                if observation_fingerprints is not None
                else tuple(call.observation_fingerprints)
            ),
            tool_state=call.tool_state,
            tools=self._refresh_tools(call),
            workspace_label=call.workspace_label,
            expansion_round=expansion_round,
            tool_round=tool_round,
            call_id=self._new_call_id(),
            parent_call_id=call.call_id,
            spawn_kind=spawn_kind,
            spawn_sequence=spawn_sequence,
            sibling_index=sibling_index,
            sibling_count=sibling_count,
            branch_path=branch_path,
        )

    def _refresh_tools(self, call: FractalCall) -> dict[str, ToolSpec]:
        state_getter = getattr(self._dispatcher, "tools_for_state", None)
        if callable(state_getter):
            try:
                tools = state_getter(call.tool_state)
            except Exception as exc:
                self._record_call(call, "tool_surface_refresh_failed", error=str(exc))
                return call.tools
            if isinstance(tools, dict):
                return self._record_tool_surface_refresh(call, call.tools, tools)
        getter = getattr(self._dispatcher, "get_tools", None)
        if not callable(getter):
            return call.tools
        try:
            tools = getter()
        except Exception as exc:
            self._record_call(call, "tool_surface_refresh_failed", error=str(exc))
            return call.tools
        if not isinstance(tools, dict):
            return call.tools
        return self._record_tool_surface_refresh(call, call.tools, tools)

    def _record_tool_surface_refresh(
        self,
        call: FractalCall,
        old_tools: dict[str, ToolSpec],
        new_tools: dict[str, ToolSpec],
    ) -> dict[str, ToolSpec]:
        old_names = set(old_tools.keys())
        new_names = set(str(name) for name in new_tools.keys())
        if old_names != new_names:
            self._record_call(
                call,
                "tool_surface_refreshed",
                added=sorted(new_names - old_names),
                removed=sorted(old_names - new_names),
            )
        return new_tools

    def _append_observation(
        self,
        observations: list[str],
        observation: str,
        *,
        call: FractalCall | None = None,
    ) -> list[str]:
        del observations
        return self._trim_observations([observation], call=call)

    def _append_observation_fingerprints(
        self,
        call: FractalCall,
        observation: str,
    ) -> tuple[str, ...]:
        current = _observation_fingerprint(observation)
        existing = tuple(call.observation_fingerprints)
        if not existing and call.observations:
            existing = tuple(
                item
                for item in (_observation_fingerprint(text) for text in call.observations)
                if item
            )
        if not current:
            return existing[-256:]
        return (*existing, current)[-256:]

    def _repeated_observation_count(self, observations: list[str], observation: str) -> int:
        current = _observation_fingerprint(observation)
        if not current:
            return 0
        count = 1
        for item in reversed(observations):
            if _observation_fingerprint(item) != current:
                break
            count += 1
        return count

    def _repeated_observation_fingerprint_count(self, call: FractalCall, observation: str) -> int:
        current = _observation_fingerprint(observation)
        if not current:
            return 0
        fingerprints = call.observation_fingerprints
        if not fingerprints and call.observations:
            fingerprints = tuple(
                item
                for item in (_observation_fingerprint(text) for text in call.observations)
                if item
            )
        count = 1
        for item in reversed(fingerprints):
            if item != current:
                break
            count += 1
        return count

    def _trim_observations(
        self,
        observations: list[str],
        *,
        call: FractalCall | None = None,
    ) -> list[str]:
        max_chars = self._limits.max_observation_chars
        trimmed: list[str] = []
        for index, item in enumerate(observations):
            text = str(item or "").strip()
            if not text:
                continue
            trimmed.append(self._project_observation_if_needed(
                text,
                reason="max_observation_chars",
                max_chars=max_chars,
                call=call,
                source={"observation_index": index},
            ))
        max_observations = self._limits.max_observations
        if _limit_enabled(max_observations) and len(trimmed) > int(max_observations or 0):
            keep_count = max(int(max_observations or 0), 0)
            if keep_count <= 0:
                trimmed = []
            elif keep_count == 1:
                trimmed = [trimmed[-1]]
            else:
                dropped = trimmed[: len(trimmed) - keep_count + 1]
                trimmed = [
                    self._project_observation_group(
                        dropped,
                        reason="max_observations",
                        call=call,
                    ),
                    *trimmed[-(keep_count - 1):],
                ]

        max_total = self._limits.max_total_observation_chars
        if not _limit_enabled(max_total):
            return trimmed

        remaining = int(max_total or 0)
        final_trimmed: list[str] = []
        overflow: list[str] = []
        reversed_trimmed = list(reversed(trimmed))
        for index, text in enumerate(reversed_trimmed):
            if len(text) <= remaining:
                final_trimmed.append(text)
                remaining -= len(text)
                continue
            overflow.extend(reversed_trimmed[index:])
            break
        final_trimmed = list(reversed(final_trimmed))
        if not overflow:
            return final_trimmed

        projected_overflow = self._project_observation_group(
            list(reversed(overflow)),
            reason="max_total_observation_chars",
            call=call,
            target_chars=remaining,
        )
        if remaining <= 0:
            if final_trimmed:
                return final_trimmed
            return [projected_overflow]
        if len(projected_overflow) <= remaining:
            return [projected_overflow, *final_trimmed]
        if final_trimmed:
            return final_trimmed
        return [projected_overflow]

    def _project_observation_if_needed(
        self,
        text: str,
        *,
        reason: str,
        max_chars: int | None,
        call: FractalCall | None = None,
        source: dict[str, Any] | None = None,
    ) -> str:
        if not _limit_enabled(max_chars) or len(text) <= int(max_chars or 0):
            return text
        if _is_context_projection(text):
            return _shrink_projection_text(text, target_chars=max_chars)
        projected = self._project_observation_text(
            text,
            reason=reason,
            projection_kind="observation",
            call=call,
            source=source,
            target_chars=max_chars,
        )
        return projected

    def _project_observation_group(
        self,
        observations: list[str],
        *,
        reason: str,
        call: FractalCall | None = None,
        target_chars: int | None = None,
    ) -> str:
        rows = [str(item or "").strip() for item in observations if str(item or "").strip()]
        if not rows:
            return ""
        group_text = json.dumps(
            {
                "type": "fractal_observation_group",
                "schema": "geist.fractal.observation_group.v1",
                "reason": reason,
                "count": len(rows),
                "items": rows,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        source: dict[str, Any] = {
            "observation_count": len(rows),
            "total_chars": sum(len(item) for item in rows),
        }
        projected = self._project_observation_text(
            group_text,
            reason=reason,
            projection_kind="observation_group",
            call=call,
            source=source,
            target_chars=target_chars,
            item_projections=[_observation_projection_brief(item) for item in rows[:24]],
            omitted_items=max(len(rows) - 24, 0),
        )
        return projected

    def _project_observation_text(
        self,
        text: str,
        *,
        reason: str,
        projection_kind: str,
        call: FractalCall | None,
        source: dict[str, Any] | None,
        target_chars: int | None = None,
        item_projections: list[dict[str, Any]] | None = None,
        omitted_items: int = 0,
    ) -> str:
        value = str(text or "")
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        payload: dict[str, Any] = {
            "type": "fractal_context_projection",
            "schema": "geist.fractal.context_projection.v1",
            "projection_kind": projection_kind,
            "reason": reason,
            "original_chars": len(value),
            "sha256": digest,
        }
        if source:
            payload["source"] = source
        if call is not None:
            payload["call"] = {
                "run_id": self._run_id,
                "call_id": call.call_id,
                "parent_call_id": call.parent_call_id,
                "branch_path": call.branch_path,
                "spawn_kind": call.spawn_kind,
                "expansion_round": call.expansion_round,
                "tool_round": call.tool_round,
            }
        if item_projections:
            payload["items"] = item_projections
        if omitted_items:
            payload["omitted_items"] = omitted_items
        first_line = _first_nonempty_line(value)
        if first_line:
            payload["first_line"] = _truncate_for_trace(first_line, max_chars=OBSERVATION_FIRST_LINE_CHARS)
        payload["preview"] = _projection_preview(value, target_chars=target_chars)
        stored = self._store_projection_material(
            value,
            material_ref=("ctx_" if projection_kind == "observation" else "ctxg_") + digest[:16],
            kind=f"fractal_context_{projection_kind}",
            source={
                "reason": reason,
                "projection_kind": projection_kind,
                **({"call_id": call.call_id, "branch_path": call.branch_path} if call is not None else {}),
            },
            sha256=digest,
        )
        artifact_ref = ""
        if isinstance(stored, dict):
            artifact_ref = str(stored.get("ref") or "")
            payload["artifact"] = {
                key: stored.get(key)
                for key in ("ref", "path", "relative_path")
                if stored.get(key)
            }
            payload["retrieve"] = {
                "tool": "artifact.read",
                "arguments": {"ref": artifact_ref},
            }
        rendered = _render_context_projection(payload, target_chars=target_chars)
        if call is not None:
            self._record_call(
                call,
                "context_projection",
                projection_kind=projection_kind,
                reason=reason,
                original_chars=len(value),
                projected_chars=len(rendered),
                sha256=digest,
                artifact_ref=artifact_ref,
            )
        return rendered

    def _store_projection_material(
        self,
        text: str,
        *,
        material_ref: str,
        kind: str,
        source: dict[str, Any],
        sha256: str,
    ) -> dict[str, Any] | None:
        if self._material_store is None:
            return None
        try:
            stored = self._material_store({
                "material_ref": material_ref,
                "kind": kind,
                "text": text,
                "sha256": sha256,
                "source": source,
            })
        except Exception:
            return None
        return stored if isinstance(stored, dict) else None

    def _new_call_id(self) -> str:
        self._call_id_counter += 1
        return f"call_{self._call_id_counter:06d}"

    def _record_call(self, call: FractalCall, event: str, **payload: Any) -> dict[str, Any]:
        return self._record(
            event,
            **self._call_trace_metadata(call),
            **payload,
        )

    @staticmethod
    def _call_trace_metadata(call: FractalCall) -> dict[str, Any]:
        return {
            "call_id": call.call_id,
            "parent_call_id": call.parent_call_id,
            "spawn_kind": call.spawn_kind,
            "spawn_sequence": call.spawn_sequence,
            "branch_path": call.branch_path,
            "sibling_index": call.sibling_index,
            "sibling_count": call.sibling_count,
            "expansion_round": call.expansion_round,
            "tool_round": call.tool_round,
        }

    def _record(self, event: str, **payload: Any) -> dict[str, Any]:
        entry = {
            "event": event,
            "run_id": self._run_id,
            "sequence": len(self._trace),
            "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
            **payload,
        }
        self._trace.append(entry)
        if self._trace_sink is not None:
            try:
                self._trace_sink(dict(entry))
            except Exception:
                pass
        return entry

    def _default_context(self, call: FractalCall) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": NATIVE_FRACTAL_PROTOCOL},
            {"role": "user", "content": _render_call_packet(call, material_store=self._material_store)},
        ]

    @staticmethod
    def _extract_payload(text: str) -> dict[str, Any]:
        raw = text.strip()
        for match in re.finditer(r"```(?:runtime|json)[^\S\r\n]*\r?\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            candidate = match.group(1).strip()
            payload = _load_json_object(candidate)
            if payload is not None:
                return payload
        payload = _load_json_object(raw)
        if payload is not None:
            return payload
        first = raw.find("{")
        last = raw.rfind("}")
        if first >= 0 and last > first:
            payload = _load_json_object(raw[first:last + 1])
            if payload is not None:
                return payload
        return {}

    @staticmethod
    def _normalize_fractals(raw: Any) -> list[_FractalInstruction]:
        if not isinstance(raw, list):
            return []
        instructions: list[_FractalInstruction] = []
        for item in raw:
            if isinstance(item, str):
                instruction = item.strip()
                continuation_context = ""
            elif isinstance(item, dict):
                instruction = str(item.get("instruction") or item.get("task") or "").strip()
                continuation_context = str(
                    item.get("continuation_context")
                    or item.get("continuation_content")
                    or ""
                ).strip()
            else:
                instruction = ""
                continuation_context = ""
            if instruction:
                instructions.append(_FractalInstruction(
                    instruction=instruction,
                    continuation_context=continuation_context,
                ))
        return instructions

    def _summarize_results(self, results: list[dict[str, Any]]) -> str:
        return json.dumps(
            _build_tool_observation_payload(results, material_store=self._material_store),
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

    def _forced_result(
        self,
        call: FractalCall,
        *,
        reason: str,
        response: str = "",
        continuation_context: str | None = None,
    ) -> FractalRun:
        continuation = call.continuation_context if continuation_context is None else continuation_context
        self._record_call(
            call,
            "limit",
            reason=reason,
            continuation_context=continuation,
            llm_calls=self._llm_calls,
            max_llm_calls=self._limits.max_llm_calls,
            expanded_call_count=self._expanded_call_count,
            max_expanded_calls=self._limits.max_expanded_calls,
            max_expansion_rounds=self._limits.max_expansion_rounds,
            max_tool_rounds_per_call=self._limits.max_tool_rounds_per_call,
        )
        return FractalRun(
            results=(FractalCompleted(
                response=response or f"fractal runtime stopped by limit: {reason}",
                continuation_context=continuation,
                observations=tuple(call.observations),
                tool_state=call.tool_state,
                ok=False,
                error=reason,
            ),),
            expansion_count=0,
        )

    def _error_result(self, call: FractalCall, exc: Exception) -> FractalRun:
        error = f"{type(exc).__name__}: {exc}"
        self._record_call(call, "error", error=error, continuation_context=call.continuation_context)
        return FractalRun(
            results=(FractalCompleted(
                response=f"fractal runtime call failed: {error}",
                continuation_context=call.continuation_context,
                observations=tuple(call.observations),
                tool_state=call.tool_state,
                ok=False,
                error=error,
            ),),
            expansion_count=0,
        )


def _load_json_object(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _inspect_context_budget(
    messages: list[dict[str, Any]],
    *,
    call: FractalCall,
    limits: FractalLimits,
) -> dict[str, Any]:
    message_rows: list[dict[str, Any]] = []
    role_chars: dict[str, int] = {}
    total_chars = 0
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        chars = len(content)
        total_chars += chars
        role_chars[role] = role_chars.get(role, 0) + chars
        row = _context_message_budget_row(index=index, role=role, content=content)
        message_rows.append(row)
    layer_rows = _context_layer_budget_rows(message_rows)
    observation_chars = [len(str(item or "")) for item in call.observations]
    payload: dict[str, Any] = {
        "type": "fractal_context_budget",
        "schema": "geist.fractal.context_budget.v1",
        "message_count": len(messages),
        "total_chars": total_chars,
        "role_chars": role_chars,
        "layers": layer_rows,
        "largest_layer": layer_rows[0] if layer_rows else None,
        "messages": message_rows[:32],
        "omitted_messages": max(len(message_rows) - 32, 0),
        "call_input": {
            "root_task_chars": len(call.root_task or ""),
            "instruction_chars": len(call.instruction or ""),
            "continuation_chars": len(call.continuation_context or ""),
            "observation_count": len(call.observations),
            "observation_chars": observation_chars[:40],
            "omitted_observation_char_entries": max(len(observation_chars) - 40, 0),
            "observation_total_chars": sum(observation_chars),
            "tool_count": len(call.tools or {}),
            "visible_tool_count": len(call.tools or {}),
        },
        "limits": {
            "max_continuation_chars": limits.max_continuation_chars,
            "max_observations": limits.max_observations,
            "max_observation_chars": limits.max_observation_chars,
            "max_total_observation_chars": limits.max_total_observation_chars,
        },
    }
    return payload


def _context_message_budget_row(*, index: int, role: str, content: str) -> dict[str, Any]:
    payload = _load_json_object(content)
    row: dict[str, Any] = {
        "index": index,
        "role": role,
        "chars": len(content),
        "layer": _context_message_layer(index=index, role=role, content=content, payload=payload),
    }
    if isinstance(payload, dict):
        row["type"] = payload.get("type")
        row["schema"] = payload.get("schema")
        if isinstance(payload.get("tools"), list):
            row["tool_count"] = len(payload["tools"])
        if isinstance(payload.get("observations"), list):
            row["observation_count"] = len(payload["observations"])
            row["context_observation_chars"] = sum(len(str(item or "")) for item in payload["observations"])
        observation_policy = payload.get("observation_policy")
        if isinstance(observation_policy, dict):
            row["observation_policy"] = {
                key: observation_policy.get(key)
                for key in ("embedded", "reason", "original_count", "original_chars", "projected_count")
                if observation_policy.get(key) not in (None, "", [], {})
            }
        data = payload.get("data")
        if isinstance(data, dict):
            row["data_keys"] = sorted(str(key) for key in data.keys())[:32]
    else:
        row["type"] = "text"
    return {key: value for key, value in row.items() if value not in (None, "", [], {})}


def _context_message_layer(
    *,
    index: int,
    role: str,
    content: str,
    payload: dict[str, Any] | None,
) -> str:
    if isinstance(payload, dict):
        kind = str(payload.get("type") or "").strip()
        if kind == "fractal_tool_surface":
            return "tool_surface"
        if kind == "fractal_context_capsule":
            return "context_capsule"
        if kind == "fractal_api_call":
            return "turn_packet"
        if kind == "fractal_cli_recent_history":
            return "recent_history"
        if kind == "fractal_context_projection":
            return "context_projection"
        if kind:
            return kind
        return "json_context"

    text = str(content or "").strip()
    if text == NATIVE_FRACTAL_PROTOCOL:
        return "native_fractal_protocol"
    if text.startswith("For native fractal API calls, return exactly one fenced `runtime` JSON object"):
        return "runtime_tool_protocol"
    if text == RUNTIME_PROTOCOL_RETRY_PROMPT or text.startswith("Your previous response did not follow"):
        return "protocol_retry"
    if text == DEFAULT_FINAL_RESPONSE_INSTRUCTION or text.startswith("Return the final user-facing response"):
        return "final_response_instruction"
    if role == "system" and index == 0:
        return "system_prompt"
    return f"{role or 'message'}_text"


def _context_layer_budget_rows(message_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[str, dict[str, Any]] = {}
    for row in message_rows:
        layer = str(row.get("layer") or "unknown")
        bucket = by_layer.setdefault(
            layer,
            {
                "layer": layer,
                "chars": 0,
                "message_count": 0,
                "message_indexes": [],
                "roles": {},
                "types": [],
                "schemas": [],
            },
        )
        bucket["chars"] += int(row.get("chars") or 0)
        bucket["message_count"] += 1
        bucket["message_indexes"].append(row.get("index"))
        role = str(row.get("role") or "")
        if role:
            bucket["roles"][role] = bucket["roles"].get(role, 0) + 1
        row_type = str(row.get("type") or "")
        if row_type and row_type not in bucket["types"]:
            bucket["types"].append(row_type)
        schema = str(row.get("schema") or "")
        if schema and schema not in bucket["schemas"]:
            bucket["schemas"].append(schema)
        if "tool_count" in row:
            bucket["tool_count"] = int(row.get("tool_count") or 0)
        if "observation_count" in row:
            bucket["observation_count"] = int(row.get("observation_count") or 0)
        if "data_keys" in row:
            bucket["data_keys"] = row.get("data_keys")

    rows = list(by_layer.values())
    rows.sort(key=lambda item: (-int(item.get("chars") or 0), str(item.get("layer") or "")))
    return [{key: value for key, value in row.items() if value not in (None, "", [], {})} for row in rows]


def _is_context_projection(text: str) -> bool:
    payload = _load_json_object(str(text or "").strip())
    return isinstance(payload, dict) and payload.get("type") == "fractal_context_projection"


def _shrink_projection_text(text: str, *, target_chars: int | None) -> str:
    payload = _load_json_object(str(text or "").strip())
    if not isinstance(payload, dict):
        return _truncate_for_trace(text, max_chars=target_chars)
    return _render_context_projection(payload, target_chars=target_chars)


def _render_context_projection(payload: dict[str, Any], *, target_chars: int | None) -> str:
    rendered = _json_compact(payload)
    if not _limit_enabled(target_chars) or len(rendered) <= int(target_chars or 0):
        return rendered

    compact = dict(payload)
    preview = compact.get("preview")
    if isinstance(preview, dict):
        compact["preview"] = {
            key: _truncate_for_trace(value, max_chars=120)
            for key, value in preview.items()
            if isinstance(value, str)
        }
    rendered = _json_compact(compact)
    if len(rendered) <= int(target_chars or 0):
        return rendered

    compact.pop("preview", None)
    compact.pop("first_line", None)
    compact.pop("items", None)
    compact.pop("source", None)
    compact.pop("call", None)
    rendered = _json_compact(compact)
    if len(rendered) <= int(target_chars or 0):
        return rendered

    minimal = {
        "type": "fractal_context_projection",
        "schema": "geist.fractal.context_projection.v1",
        "projection_kind": payload.get("projection_kind"),
        "reason": payload.get("reason"),
        "original_chars": payload.get("original_chars"),
        "sha256": payload.get("sha256"),
        "retrieve": payload.get("retrieve"),
    }
    return _json_compact({key: value for key, value in minimal.items() if value not in (None, "", [], {})})


def _projection_preview(text: str, *, target_chars: int | None) -> dict[str, Any]:
    value = str(text or "")
    if not value:
        return {}
    if not _limit_enabled(target_chars):
        size = 600
    else:
        size = max(80, min(600, int(target_chars or 0) // 4))
    if len(value) <= size * 2:
        return {"text": value}
    return {
        "head": value[:size],
        "tail": value[-size:],
        "omitted_middle_chars": max(len(value) - size * 2, 0),
    }


def _observation_projection_brief(text: str) -> dict[str, Any]:
    value = str(text or "")
    payload = _load_json_object(value)
    if isinstance(payload, dict) and payload.get("type") == "fractal_context_projection":
        return {
            "type": "fractal_context_projection",
            "projection_kind": payload.get("projection_kind"),
            "reason": payload.get("reason"),
            "original_chars": payload.get("original_chars"),
            "sha256": payload.get("sha256"),
            "retrieve": payload.get("retrieve"),
        }
    return {
        "type": "observation",
        "chars": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest(),
    }


def _json_compact(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _summarize_tool_calls_for_trace(tool_calls: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for item in list(tool_calls)[:16]:
        if not isinstance(item, dict):
            continue
        summarized.append({
            "tool": str(item.get("tool") or ""),
            "arguments": _summarize_tool_arguments_for_trace(item.get("arguments")),
        })
    if len(tool_calls) > 16:
        summarized.append({"tool": "...", "arguments": {"remaining": len(tool_calls) - 16}})
    return summarized


def _structural_motion_signature(
    tool_calls: list[dict[str, Any]],
    *,
    tools: dict[str, ToolSpec],
) -> str:
    if not tool_calls:
        return ""
    rows: list[dict[str, Any]] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            return ""
        name = str(item.get("tool") or "")
        spec = tools.get(name)
        if spec is None or str(spec.side_effect or "none") != "none":
            return ""
        rows.append({
            "tool": name,
            "target": _structural_motion_target(item.get("arguments")),
        })
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _decode_structural_motion_signature(signature: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(signature)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _structural_motion_label(rows: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    for row in rows:
        tool = str(row.get("tool") or "").strip() or "tool"
        target = row.get("target")
        if isinstance(target, dict):
            target_bits = [
                f"{key}={_truncate_for_trace(value, max_chars=80)}"
                for key, value in sorted(target.items())
                if value not in (None, "", [], {})
            ]
            labels.append(f"{tool}({', '.join(target_bits)})" if target_bits else tool)
        else:
            labels.append(tool)
    return " + ".join(labels)


def _structural_motion_target(arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return {}
    target: dict[str, Any] = {}
    for key in (
        "path",
        "paths",
        "ref",
        "id",
        "query",
        "tool",
        "tools",
        "groups",
        "action",
        "line_start",
        "line_count",
        "offset",
        "max_chars",
        "max_text_chars",
    ):
        if key in arguments:
            target[key] = arguments.get(key)
    if not target:
        for key, value in list(arguments.items())[:8]:
            if isinstance(value, (str, int, float, bool)) or value is None:
                target[str(key)] = value
            elif isinstance(value, list):
                target[str(key)] = value[:8]
    return target


def _summarize_tool_arguments_for_trace(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return _truncate_for_trace(value, max_chars=240)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:24]:
            key_text = str(key)
            if key_text in {"content", "text", "body"} and isinstance(item, str):
                result[key_text] = f"<{len(item)} chars>"
            else:
                result[key_text] = _summarize_tool_arguments_for_trace(item, depth=depth + 1)
        if len(value) > 24:
            result["..."] = f"+{len(value) - 24} keys"
        return result
    if isinstance(value, list):
        items = [_summarize_tool_arguments_for_trace(item, depth=depth + 1) for item in value[:24]]
        if len(value) > 24:
            items.append(f"... +{len(value) - 24}")
        return items
    if isinstance(value, str):
        return _truncate_for_trace(value, max_chars=500)
    return value


def _build_tool_observation_payload(
    results: list[dict[str, Any]],
    *,
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "fractal_tool_observation",
        "schema": "geist.fractal.tool_observation.v1",
        "items": [],
    }
    anchors = _collect_reality_anchors(results)
    if anchors:
        payload["anchors"] = anchors

    items: list[dict[str, Any]] = []
    for fallback_index, row in enumerate(results):
        data = row.get("data")
        summarized_data = _summarize_observation_value(data, material_store=material_store)
        item: dict[str, Any] = {
            "type": "tool_result",
            "call_id": str(row.get("call_id") or _tool_call_id(row, index=fallback_index)),
            "tool": str(row.get("tool") or ""),
            "batch": row.get("batch"),
            "parallel": bool(row.get("parallel")),
            "status": _tool_result_status(data),
            "data": summarized_data,
            "body_policy": _tool_result_body_policy(summarized_data),
        }
        if isinstance(data, dict) and "ok" in data:
            item["ok"] = data.get("ok") is True
        items.append(item)
    payload["items"] = items
    health = _tool_observation_health_from_items(items)
    if health:
        payload["health"] = health
    return payload


def _tool_result_body_policy(data: Any) -> dict[str, Any]:
    has_material = _contains_material_ref(data)
    if has_material:
        return {
            "inline_enough": False,
            "retrieve_if_needed": True,
        }
    return {
        "inline_enough": True,
        "retrieve_if_needed": False,
    }


def _contains_material_ref(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("material_ref") or value.get("retrieve") or value.get("artifact"):
            return True
        return any(_contains_material_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_material_ref(item) for item in value)
    return False


def _tool_call_id(call: dict[str, Any], *, index: int) -> str:
    for key in ("call_id", "id"):
        value = call.get(key)
        if value not in (None, ""):
            return str(value)
    digest = hashlib.sha256(
        json.dumps(
            {
                "index": index,
                "tool": call.get("tool"),
                "arguments": call.get("arguments"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8", errors="replace")
    ).hexdigest()
    return "call_" + digest[:16]


def _tool_result_status(data: Any) -> str:
    if not isinstance(data, dict):
        return "unknown"
    if _result_has_conflict(data):
        return "conflicted"
    if data.get("ok") is True:
        return "completed"
    if data.get("ok") is False:
        return "failed"
    returncode = data.get("returncode")
    if returncode is not None:
        try:
            return "completed" if int(returncode) == 0 else "failed"
        except Exception:
            return "unknown"
    if data.get("error") or data.get("exception_type"):
        return "failed"
    return "unknown"


def _tool_observation_health_from_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    status_counts: dict[str, int] = {}
    failed_items: list[dict[str, Any]] = []
    write_items: list[dict[str, Any]] = []
    changed_paths: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        tool = str(item.get("tool") or "")
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        if _tool_item_is_failure(item):
            failed_items.append(_tool_failure_brief(item))
        if _tool_item_is_workspace_write(item):
            write_items.append(item)
            for path in _paths_from_tool_item(item):
                if path not in changed_paths:
                    changed_paths.append(path)

    health: dict[str, Any] = {
        "status_counts": status_counts,
    }
    if failed_items:
        health["has_failed_tools"] = True
        health["failed_count"] = len(failed_items)
        health["failed_items"] = failed_items[:OBSERVATION_CONTEXT_TOOL_ITEMS]
        if len(failed_items) > OBSERVATION_CONTEXT_TOOL_ITEMS:
            health["omitted_failed_items"] = len(failed_items) - OBSERVATION_CONTEXT_TOOL_ITEMS
    if write_items:
        health["workspace_write_count"] = len(write_items)
        if changed_paths:
            health["changed_paths"] = changed_paths[:24]
            if len(changed_paths) > 24:
                health["omitted_changed_paths"] = len(changed_paths) - 24
    verification = _verification_cue_from_tool_items(
        items,
        failed_count=len(failed_items),
        write_count=len(write_items),
        changed_paths=changed_paths,
    )
    if verification:
        health["verification"] = verification
    return {key: value for key, value in health.items() if value not in (None, "", [], {})}


def _tool_item_is_failure(item: dict[str, Any]) -> bool:
    status = str(item.get("status") or "").lower()
    if status in {"failed", "conflicted"}:
        return True
    if item.get("ok") is False:
        return True
    data = item.get("data")
    if isinstance(data, dict):
        if data.get("ok") is False:
            return True
        if data.get("error") or data.get("exception_type"):
            return True
        returncode = data.get("returncode")
        if returncode is not None:
            try:
                return int(returncode) != 0
            except Exception:
                return False
    return False


def _tool_item_is_workspace_write(item: dict[str, Any]) -> bool:
    tool = str(item.get("tool") or "")
    if tool in {"write", "edit", "file", "sandbox"}:
        status = str(item.get("status") or "").lower()
        return status in {"completed", "unknown"} and not _tool_item_is_failure(item)
    data = item.get("data")
    if isinstance(data, dict):
        action = str(data.get("action") or "").lower()
        if action in {"write", "edit", "apply", "patch", "package"}:
            return not _tool_item_is_failure(item)
    return False


def _paths_from_tool_item(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    data = item.get("data")
    if isinstance(data, dict):
        _collect_paths_from_value(data, paths=paths)
    return paths


def _collect_paths_from_value(value: Any, *, paths: list[str], key: str = "", depth: int = 0) -> None:
    if len(paths) >= 80 or depth > 5:
        return
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key_text = str(raw_key)
            if key_text.lower() in REALITY_ANCHOR_PATH_KEYS and isinstance(item, (str, int, float)):
                text = str(item).strip()
                if text and text not in paths:
                    paths.append(text)
            _collect_paths_from_value(item, paths=paths, key=key_text, depth=depth + 1)
            if len(paths) >= 80:
                return
    elif isinstance(value, list):
        for item in value[:80]:
            _collect_paths_from_value(item, paths=paths, key=key, depth=depth + 1)
            if len(paths) >= 80:
                return


def _tool_failure_brief(item: dict[str, Any]) -> dict[str, Any]:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    brief: dict[str, Any] = {
        "call_id": item.get("call_id"),
        "tool": item.get("tool"),
        "status": item.get("status"),
    }
    for key in (
        "error",
        "exception_type",
        "message",
        "returncode",
        "command",
        "missing_executable",
        "path",
        "relative_path",
        "hint",
    ):
        if key in data:
            brief[key] = _context_value_brief(data.get(key))
    return {key: value for key, value in brief.items() if value not in (None, "", [], {})}


def _verification_cue_from_tool_items(
    items: list[dict[str, Any]],
    *,
    failed_count: int,
    write_count: int,
    changed_paths: list[str],
) -> dict[str, Any]:
    if items:
        return _pending_verification_from_tool_items(items)
    if write_count <= 0:
        return {}
    reasons: list[str] = []
    if failed_count:
        reasons.append("tool_failure_after_workspace_changes")
    if write_count >= 2:
        reasons.append("multiple_workspace_writes")
    if not reasons:
        return {}
    return {
        "needed": True,
        "reasons": reasons,
        "changed_paths": changed_paths[:16],
        "suggested_tools": ["bash", "git.status"],
    }


def _pending_verification_from_tool_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    pending_writes: list[dict[str, Any]] = []
    pending_failed: list[dict[str, Any]] = []
    pending_paths: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if pending_writes and _tool_item_is_successful_verification_evidence(item):
            pending_writes = []
            pending_failed = []
            pending_paths = []
            continue
        if _tool_item_is_workspace_write(item):
            pending_writes.append(item)
            for path in _paths_from_tool_item(item):
                if path not in pending_paths:
                    pending_paths.append(path)
            continue
        if pending_writes and _tool_item_is_failure(item):
            pending_failed.append(item)

    if not pending_writes:
        return {}
    reasons: list[str] = []
    if pending_failed:
        reasons.append("tool_failure_after_workspace_changes")
    if len(pending_writes) >= 2:
        reasons.append("multiple_workspace_writes")
    if not reasons:
        return {}
    result: dict[str, Any] = {
        "needed": True,
        "reasons": reasons,
        "workspace_write_count": len(pending_writes),
        "changed_paths": pending_paths[:16],
        "suggested_tools": ["bash", "git.status"],
    }
    if pending_failed:
        result["failed_count"] = len(pending_failed)
        result["failed_items"] = [_tool_failure_brief(item) for item in pending_failed[:OBSERVATION_CONTEXT_TOOL_ITEMS]]
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _tool_item_is_successful_verification_evidence(item: dict[str, Any]) -> bool:
    if _tool_item_is_failure(item):
        return False
    tool = str(item.get("tool") or "").strip()
    if tool in VERIFICATION_EVIDENCE_TOOLS:
        return True
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    action = str(data.get("action") or "").strip()
    return action in VERIFICATION_EVIDENCE_TOOLS


def _pending_verification_from_observations(observations: list[str] | tuple[str, ...]) -> dict[str, Any]:
    pending: dict[str, Any] = {}
    guarded = False
    for text in observations:
        payload = _load_json_object(str(text or "").strip())
        if not isinstance(payload, dict):
            continue
        kind = str(payload.get("type") or "")
        if kind == COMPLETION_GUARD_TYPE:
            guarded = True
            guard_pending = payload.get("pending_verification")
            if isinstance(guard_pending, dict):
                pending = dict(guard_pending)
            continue
        if pending and _observation_has_successful_verification_evidence(payload):
            pending = {}
        health = _health_from_observation_payload(payload)
        verification = health.get("verification") if isinstance(health.get("verification"), dict) else {}
        if verification:
            pending = _pending_verification_from_health(health)
    if pending:
        pending["guarded"] = guarded
    return {key: value for key, value in pending.items() if value not in (None, "", [], {})}


def _pending_verification_from_health(health: dict[str, Any]) -> dict[str, Any]:
    verification = health.get("verification") if isinstance(health.get("verification"), dict) else {}
    if not verification:
        return {}
    result = dict(verification)
    for key in ("status_counts", "workspace_write_count", "changed_paths", "failed_count", "failed_items"):
        value = health.get(key)
        if value not in (None, "", [], {}):
            result.setdefault(key, value)
    return result


def _observation_has_successful_verification_evidence(payload: dict[str, Any]) -> bool:
    kind = str(payload.get("type") or "")
    if kind == "fractal_tool_observation":
        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        return any(_tool_item_is_successful_verification_evidence(item) for item in items if isinstance(item, dict))
    if kind == "fractal_observation_context":
        rows = payload.get("observations") if isinstance(payload.get("observations"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            tool_observation = row.get("tool_observation") if isinstance(row.get("tool_observation"), dict) else {}
            items = tool_observation.get("items") if isinstance(tool_observation.get("items"), list) else []
            if any(_tool_item_is_successful_verification_evidence(item) for item in items if isinstance(item, dict)):
                return True
    return False


def _should_guard_completion(call: FractalCall, pending_verification: dict[str, Any]) -> bool:
    if not pending_verification:
        return False
    if pending_verification.get("guarded"):
        return False
    if call.spawn_kind == "final":
        return False
    suggested = [
        str(item)
        for item in pending_verification.get("suggested_tools", [])
        if str(item).strip()
    ]
    if suggested and any(tool in call.tools for tool in suggested):
        return True
    return any(tool in call.tools for tool in VERIFICATION_EVIDENCE_TOOLS)


def _auto_verification_tool_calls(
    call: FractalCall,
    pending_verification: dict[str, Any],
) -> list[dict[str, Any]]:
    if not pending_verification:
        return []
    if pending_verification.get("guarded"):
        return []
    if call.spawn_kind in {"auto_verification", "completion_guard", "final"}:
        return []
    tool_name = "bash" if "bash" in call.tools else "shell" if "shell" in call.tools else ""
    if not tool_name:
        return []
    python_paths = _python_paths_for_auto_verification(pending_verification.get("changed_paths") or [])
    if not python_paths:
        return []
    return [{
        "tool": tool_name,
        "arguments": {
            "command": ["python", "-m", "py_compile", *python_paths[:24]],
            "timeout_ms": 120000,
            "max_chars": 40000,
            "max_output_chars": 80000,
        },
    }]


def _python_paths_for_auto_verification(paths: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(paths, list):
        return result
    for raw in paths:
        path = _safe_relative_python_path(raw)
        if path and path not in result:
            result.append(path)
        if len(result) >= 24:
            break
    return result


def _safe_relative_python_path(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", text) or text.startswith("/"):
        return ""
    if text.startswith(".geist_state/") or text.startswith(".geist_runs/"):
        return ""
    if text == ".." or text.startswith("../") or "/../" in text:
        return ""
    if not text.endswith(".py"):
        return ""
    return text


def _build_completion_guard_observation(
    pending_verification: dict[str, Any],
    *,
    attempted_response: str,
) -> str:
    guard_pending = {
        key: value
        for key, value in pending_verification.items()
        if key != "guarded" and value not in (None, "", [], {})
    }
    payload = {
        "type": COMPLETION_GUARD_TYPE,
        "schema": "geist.fractal.completion_guard.v1",
        "reason": "pending_verification",
        "pending_verification": guard_pending,
        "attempted_final_response": _truncate_for_trace(attempted_response, max_chars=1600),
        "runtime_effect": (
            "The previous call tried to complete while recent workspace writes "
            "or failed post-write tools still had no later verification evidence."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _should_guard_empty_completion(call: FractalCall, response: str) -> bool:
    if str(response or "").strip():
        return False
    if not call.observations:
        return False
    if call.spawn_kind == "final":
        return False
    if _completion_guard_seen(call.observations, reason="empty_final_response"):
        return False
    return True


def _empty_completion_error(call: FractalCall, response: str) -> bool:
    if str(response or "").strip():
        return False
    if not call.observations:
        return False
    return _completion_guard_seen(call.observations, reason="empty_final_response")


def _completion_guard_seen(observations: list[str] | tuple[str, ...], *, reason: str) -> bool:
    for text in observations:
        payload = _load_json_object(str(text or "").strip())
        if not isinstance(payload, dict):
            continue
        if str(payload.get("type") or "") != COMPLETION_GUARD_TYPE:
            continue
        if str(payload.get("reason") or "") == reason:
            return True
    return False


def _build_empty_completion_guard_observation() -> str:
    payload = {
        "type": COMPLETION_GUARD_TYPE,
        "schema": "geist.fractal.completion_guard.v1",
        "reason": "empty_final_response",
        "runtime_effect": (
            "The previous call tried to complete after receiving observations "
            "but left final_response empty."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


REALITY_ANCHOR_PATH_KEYS = {
    "cwd",
    "package_path",
    "path",
    "relative_path",
    "source_path",
    "stdout_log",
    "stderr_log",
    "target_dir",
    "target_path",
}
REALITY_ANCHOR_MAX_ITEMS = 40


def _collect_reality_anchors(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in results:
        tool = str(row.get("tool") or "tool")
        data = row.get("data")
        _collect_reality_anchors_from_value(data, tool=tool, anchors=anchors, seen=seen)
        if len(anchors) >= REALITY_ANCHOR_MAX_ITEMS:
            break
    return anchors[:REALITY_ANCHOR_MAX_ITEMS]


def _collect_reality_anchors_from_value(
    value: Any,
    *,
    tool: str,
    anchors: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    key: str = "",
    depth: int = 0,
) -> None:
    if len(anchors) >= REALITY_ANCHOR_MAX_ITEMS or depth > 5:
        return
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key_text = str(raw_key)
            _maybe_add_reality_anchor(key_text, item, tool=tool, anchors=anchors, seen=seen)
            _collect_reality_anchors_from_value(
                item,
                tool=tool,
                anchors=anchors,
                seen=seen,
                key=key_text,
                depth=depth + 1,
            )
            if len(anchors) >= REALITY_ANCHOR_MAX_ITEMS:
                return
    elif isinstance(value, list):
        for item in value[:80]:
            _collect_reality_anchors_from_value(
                item,
                tool=tool,
                anchors=anchors,
                seen=seen,
                key=key,
                depth=depth + 1,
            )
            if len(anchors) >= REALITY_ANCHOR_MAX_ITEMS:
                return


def _maybe_add_reality_anchor(
    key: str,
    value: Any,
    *,
    tool: str,
    anchors: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
) -> None:
    if not isinstance(value, (str, int, float)):
        return
    text = str(value).strip()
    if not text:
        return
    key_text = key.lower()
    anchor: dict[str, Any] | None = None
    if key_text in {"change_id", "package_id"} and text.startswith("cp_"):
        anchor = {
            "type": "change",
            "value": text,
            "tool": tool,
            "retrieve": {"tool": "trace.read", "arguments": {"change_id": text}},
        }
    elif key_text in {"trace_id", "object_id"} or text.startswith("tr_"):
        anchor = {"type": "trace", "value": text, "tool": tool}
    elif key_text == "url" or text.startswith(("http://", "https://")):
        anchor = {"type": "url", "value": text, "tool": tool}
    elif key_text in REALITY_ANCHOR_PATH_KEYS:
        anchor = {"type": "path", "value": text, "tool": tool}
    elif key_text in {"id", "process_id"} and tool in {"process", "serve"}:
        anchor = {"type": "process", "value": text, "tool": tool}
    elif key_text == "pid" and tool in {"process", "serve"}:
        anchor = {"type": "pid", "value": text, "tool": tool}
    elif key_text == "name" and tool == "sandbox":
        anchor = {"type": "sandbox", "value": text, "tool": tool}
    if anchor is None:
        return
    dedupe_key = (str(anchor["type"]), str(anchor["value"]), tool)
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    anchors.append(anchor)


TRACE_WRITE_SUCCESS_WORDS = (
    "complete",
    "completed",
    "done",
    "passed",
    "success",
    "succeeded",
    "完成",
    "成功",
    "通过",
    "验证通过",
    "已完成",
)
TRACE_WRITE_FAILURE_WORDS = (
    "blocked",
    "conflict",
    "error",
    "failed",
    "failure",
    "incomplete",
    "partial",
    "冲突",
    "错误",
    "失败",
    "未完成",
    "部分",
    "不通过",
    "有问题",
)


def _blocked_trace_write_result(
    tool_call: dict[str, Any],
    prior_failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not prior_failures:
        return None
    if str(tool_call.get("tool") or "") != "trace.write":
        return None
    args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
    if not _trace_write_claims_success(args):
        return None
    return {
        "ok": False,
        "action": "trace.write",
        "blocked": True,
        "error": (
            "trace.write blocked: previous tool results in this runtime call "
            "contain failures or conflicts. Write a failure/partial summary, "
            "repair the issue first, or include the conflict explicitly."
        ),
        "requested_title": str(args.get("title") or ""),
        "related_failures": prior_failures[-8:],
    }


def _trace_write_claims_success(args: dict[str, Any]) -> bool:
    text = _trace_write_claim_text(args)
    if not text:
        return False
    lowered = text.lower()
    has_success = any(word in lowered for word in TRACE_WRITE_SUCCESS_WORDS)
    has_failure = any(word in lowered for word in TRACE_WRITE_FAILURE_WORDS)
    return has_success and not has_failure


def _trace_write_claim_text(args: dict[str, Any]) -> str:
    parts = [
        str(args.get("title") or ""),
        str(args.get("text") or args.get("content") or ""),
    ]
    data = args.get("data")
    if isinstance(data, dict):
        for key in ("status", "result", "summary", "title", "text", "note"):
            value = data.get(key)
            if value is not None:
                parts.append(str(value))
    return "\n".join(part for part in parts if part)


def _tool_failure_for_trace_guard(row: dict[str, Any]) -> dict[str, Any]:
    tool = str(row.get("tool") or "")
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    if data.get("ok") is True and not _result_has_conflict(data):
        return {}
    summary: dict[str, Any] = {"tool": tool}
    for key in (
        "action",
        "error",
        "exception_type",
        "path",
        "relative_path",
        "returncode",
        "conflict_count",
        "edit_count",
    ):
        value = data.get(key)
        if value not in (None, "", []):
            summary[key] = value
    conflicts = data.get("conflicts")
    if isinstance(conflicts, list) and conflicts:
        summary["conflicts"] = _summarize_guard_conflicts(conflicts)
    return summary


def _result_has_conflict(data: dict[str, Any]) -> bool:
    conflict_count = data.get("conflict_count")
    try:
        if int(conflict_count) > 0:
            return True
    except Exception:
        pass
    conflicts = data.get("conflicts")
    return isinstance(conflicts, list) and bool(conflicts)


def _summarize_guard_conflicts(conflicts: list[Any]) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for item in conflicts[:4]:
        if not isinstance(item, dict):
            summarized.append({"error": str(item)})
            continue
        entry: dict[str, Any] = {}
        for key in ("index", "path", "error", "expected_sha256", "current_sha256"):
            value = item.get(key)
            if value not in (None, "", []):
                entry[key] = value
        summarized.append(entry)
    if len(conflicts) > 4:
        summarized.append({"omitted_conflicts": len(conflicts) - 4})
    return summarized


def _summarize_observation_value(
    value: Any,
    *,
    key: str = "",
    parent: dict[str, Any] | None = None,
    depth: int = 0,
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> Any:
    if depth >= 8:
        return _materialize_observation_text(value, key=key, parent=parent, material_store=material_store)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key_text = str(raw_key)
            result[key_text] = _summarize_observation_value(
                item,
                key=key_text,
                parent=value,
                depth=depth + 1,
                material_store=material_store,
            )
        return result
    if isinstance(value, list):
        if len(value) > 60:
            return {
                "items": [
                    _summarize_observation_value(
                        item,
                        key=key,
                        parent=parent,
                        depth=depth + 1,
                        material_store=material_store,
                    )
                    for item in value[:60]
                ],
                "omitted_items": len(value) - 60,
            }
        return [
            _summarize_observation_value(
                item,
                key=key,
                parent=parent,
                depth=depth + 1,
                material_store=material_store,
            )
            for item in value
        ]
    if isinstance(value, str):
        return _materialize_observation_text(value, key=key, parent=parent, material_store=material_store)
    return value


def _materialize_observation_text(
    value: Any,
    *,
    key: str = "",
    parent: dict[str, Any] | None = None,
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> Any:
    text = str(value or "")
    key_text = str(key or "").lower()
    if key_text in {"stdout", "stderr", "output"}:
        max_inline = OBSERVATION_INLINE_OUTPUT_CHARS
    elif key_text in OBSERVATION_TEXT_KEYS:
        max_inline = OBSERVATION_INLINE_TEXT_CHARS
    else:
        max_inline = OBSERVATION_INLINE_TEXT_CHARS * 2
    if len(text) <= max_inline:
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    material: dict[str, Any] = {
        "material_ref": "mat_" + digest[:16],
        "kind": _infer_observation_material_kind(key_text, parent),
        "chars": len(text),
        "sha256": digest,
        "inline": False,
    }
    lines = text.count("\n") + 1 if text else 0
    if lines:
        material["lines"] = lines
    first_line = _first_nonempty_line(text)
    if first_line:
        material["first_line"] = _truncate_for_trace(first_line, max_chars=OBSERVATION_FIRST_LINE_CHARS)
    html_title = _html_title(text)
    if html_title:
        material["html_title"] = _truncate_for_trace(html_title, max_chars=OBSERVATION_FIRST_LINE_CHARS)
    source = _observation_material_source(parent)
    if source:
        material["source"] = source
    retrieve = _observation_retrieve_hint(key_text, parent)
    if retrieve:
        material["retrieve"] = retrieve
    if material_store is not None:
        try:
            stored = material_store({
                "material_ref": material["material_ref"],
                "kind": material["kind"],
                "text": text,
                "sha256": digest,
                "source": source,
            })
        except Exception as exc:
            material["store_error"] = str(exc)
        else:
            if isinstance(stored, dict):
                material["artifact"] = {
                    key: stored.get(key)
                    for key in ("ref", "path", "relative_path")
                    if stored.get(key)
                }
                material["retrieve"] = {
                    "tool": "artifact.read",
                    "arguments": {"ref": stored.get("ref") or material["material_ref"]},
                }
    return material


def _infer_observation_material_kind(key: str, parent: dict[str, Any] | None) -> str:
    content_type = _header_value(parent, "content-type").lower()
    if key in {"stdout", "stderr", "output"}:
        return "process_output"
    if "html" in content_type or key == "html":
        return "html"
    if "json" in content_type:
        return "json"
    path = _parent_text(parent, "path") or _parent_text(parent, "relative_path")
    if path:
        suffix = _path_like_suffix(path)
        if suffix:
            return f"file:{suffix}"
        return "file"
    if key in {"body", "response"} and _parent_text(parent, "url"):
        return "http_body"
    return "text"


def _observation_material_source(parent: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parent, dict):
        return {}
    source: dict[str, Any] = {}
    for key in ("action", "path", "relative_path", "url", "method", "status", "id", "pid"):
        value = parent.get(key)
        if isinstance(value, (str, int, float, bool)) and value != "":
            source[key] = value
    content_type = _header_value(parent, "content-type")
    if content_type:
        source["content_type"] = content_type
    return source


def _observation_retrieve_hint(key: str, parent: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parent, dict):
        return {}
    path = _parent_text(parent, "path") or _parent_text(parent, "relative_path")
    if path:
        return {"tool": "read", "arguments": {"path": path, "max_chars": 20000}}
    url = _parent_text(parent, "url")
    if url and key in {"body", "response", "html"}:
        hint: dict[str, Any] = {"tool": "http.request", "arguments": {"url": url}}
        method = _parent_text(parent, "method")
        if method:
            hint["arguments"]["method"] = method
        return hint
    log_path = _parent_text(parent, "stdout_log") if key in {"stdout", "output"} else _parent_text(parent, "stderr_log")
    if log_path:
        return {"tool": "read", "arguments": {"path": log_path, "max_chars": 20000}}
    return {}


def _parent_text(parent: dict[str, Any] | None, key: str) -> str:
    if not isinstance(parent, dict):
        return ""
    value = parent.get(key)
    return str(value) if isinstance(value, (str, int, float)) and value != "" else ""


def _header_value(parent: dict[str, Any] | None, name: str) -> str:
    if not isinstance(parent, dict):
        return ""
    headers = parent.get("headers")
    if not isinstance(headers, dict):
        return ""
    for key, value in headers.items():
        if str(key).lower() == name.lower() and isinstance(value, (str, int, float)):
            return str(value)
    return ""


def _path_like_suffix(path: str) -> str:
    normalized = path.replace("\\", "/").rstrip("/")
    name = normalized.rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()[:24]


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _html_title(text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _observation_fingerprint(observation: str) -> str:
    text = str(observation or "").strip()
    if not text:
        return ""
    payload = _load_json_object(text)
    if isinstance(payload, dict) and payload.get("type") == "fractal_context_projection":
        sha = str(payload.get("sha256") or "").strip()
        if sha:
            return "sha256:" + sha
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _contains_provider_tool_syntax(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "<minimax:tool_call",
            "<invoke ",
            "</invoke>",
            "[tool_call]",
            "[/tool_call]",
            "tool =>",
            "args =>",
        )
    )


def _normalize_runtime_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    tool_calls: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool = DecisionParser._extract_tool_name(item)
        if not tool:
            continue
        tool_calls.append({
            "tool": tool,
            "arguments": DecisionParser._extract_arguments(item),
        })
    return tool_calls


def _looks_like_runtime_json_attempt(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if "```runtime" in lowered:
        return True
    if not value.startswith("{"):
        return False
    runtime_keys = (
        '"final_response"',
        '"tool_calls"',
        '"fractals"',
        '"continuation_context"',
        '"continuation_content"',
        '"clear_continuation_context"',
    )
    return any(key in lowered for key in runtime_keys)


def _extract_partial_runtime_tool_calls(
    text: str,
    allowed_tools: dict[str, ToolSpec] | None = None,
) -> list[dict[str, Any]]:
    value = str(text or "")
    if '"tool_calls"' not in value.lower() and "'tool_calls'" not in value.lower():
        return []
    allowed = set(allowed_tools or {})
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"\{\s*\"(?:tool|name)\"\s*:", value):
        try:
            item, _end = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        normalized = _normalize_runtime_tool_calls(item)
        if not normalized:
            continue
        call = normalized[0]
        if allowed and str(call.get("tool") or "") not in allowed:
            continue
        calls.append(call)
    return calls


def _extract_provider_tool_calls(
    text: str,
    allowed_tools: dict[str, ToolSpec] | None = None,
) -> list[dict[str, Any]]:
    allowed = set(allowed_tools or {})
    if not allowed:
        return []
    calls: list[dict[str, Any]] = []

    for match in re.finditer(r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]", text, flags=re.IGNORECASE | re.DOTALL):
        call = _parse_bracket_provider_tool_call(match.group(1))
        if call and call["tool"] in allowed:
            calls.append(call)

    for match in re.finditer(
        r"<invoke\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</invoke>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        tool = match.group(1).strip()
        args = {
            param.group(1).strip(): _coerce_provider_arg(param.group(2).strip())
            for param in re.finditer(
                r"<parameter\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</parameter>",
                match.group(2),
                flags=re.IGNORECASE | re.DOTALL,
            )
        }
        call = {"tool": tool, "arguments": _normalize_provider_arguments(tool, args)}
        if tool and tool in allowed:
            calls.append(call)

    return calls[:8]


def _parse_bracket_provider_tool_call(block: str) -> dict[str, Any] | None:
    tool_match = re.search(r"\btool\s*=>\s*([\"'])(.*?)\1", block, flags=re.IGNORECASE | re.DOTALL)
    if not tool_match:
        return None
    tool = tool_match.group(2).strip()
    args: dict[str, Any] = {}
    for arg_match in re.finditer(
        r"--([A-Za-z_][\w.-]*)\s+(\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^\r\n}]+)",
        block,
        flags=re.DOTALL,
    ):
        key = arg_match.group(1).strip().replace("-", "_")
        args[key] = _coerce_provider_arg(arg_match.group(2).strip())
    return {"tool": tool, "arguments": _normalize_provider_arguments(tool, args)}


def _normalize_provider_arguments(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(args)
    if tool == "web_search":
        if "limit" in normalized and "max_results" not in normalized:
            normalized["max_results"] = normalized.pop("limit")
        if "q" in normalized and "query" not in normalized:
            normalized["query"] = normalized.pop("q")
    return normalized


def _coerce_provider_arg(raw: str) -> Any:
    value = raw.strip().rstrip(",")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _text_before_provider_tool_calls(text: str) -> str:
    indexes = [
        index
        for index in (
            text.lower().find("[tool_call]"),
            text.lower().find("<minimax:tool_call"),
            text.lower().find("<invoke "),
        )
        if index >= 0
    ]
    if not indexes:
        return text.strip()
    return text[:min(indexes)].strip()


def _plain_text_fallback(text: str) -> str:
    cleaned = re.sub(
        r"<minimax:tool_call>.*?</minimax:tool_call>",
        "",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    cleaned = re.sub(
        r"\[TOOL_CALL\].*?\[/TOOL_CALL\]",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    cleaned = re.sub(
        r"<invoke\b.*?</invoke>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    parts = [part.strip() for part in re.split(r"\n{3,}", cleaned) if part.strip()]
    candidate = parts[-1] if parts else cleaned
    if candidate.lower().startswith("```"):
        fenced = re.match(r"```[^\r\n]*\r?\n(.*?)```", candidate, flags=re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
    return candidate.strip() or "geist produced no terminal response."


def render_observations_for_context(
    observations: list[str] | tuple[str, ...],
    *,
    call: FractalCall | None = None,
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    rows = [str(item or "").strip() for item in observations or [] if str(item or "").strip()]
    if not rows:
        return [], {}

    total_chars = sum(len(item) for item in rows)
    largest_chars = max(len(item) for item in rows)
    needs_projection = any(_observation_needs_projection(item) for item in rows)
    if (
        not needs_projection
        and total_chars <= OBSERVATION_CONTEXT_INLINE_TOTAL_CHARS
        and largest_chars <= OBSERVATION_CONTEXT_INLINE_ITEM_CHARS
    ):
        return rows, {}

    projected = _build_observation_context_packet(rows, call=call, material_store=material_store)
    policy = {
        "embedded": "projected",
        "reason": "context_projection",
        "original_count": len(rows),
        "original_chars": total_chars,
        "largest_observation_chars": largest_chars,
        "projected_count": min(len(rows), OBSERVATION_CONTEXT_MAX_ITEMS),
        "full_material": "stored_when_artifact_ref_present",
    }
    return [json.dumps(projected, ensure_ascii=False, separators=(",", ":"), default=str)], policy


def _observation_needs_projection(text: str) -> bool:
    payload = _load_json_object(str(text or "").strip())
    if not isinstance(payload, dict):
        return False
    return str(payload.get("type") or "") in {
        "fractal_tool_observation",
        "fractal_cli_recent_history",
        "fractal_observation_group",
    }


def _build_observation_context_packet(
    observations: list[str],
    *,
    call: FractalCall | None,
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, text in enumerate(observations[:OBSERVATION_CONTEXT_MAX_ITEMS]):
        rows.append(_observation_context_brief(
            text,
            index=index,
            call=call,
            material_store=material_store,
        ))
    payload: dict[str, Any] = {
        "type": "fractal_observation_context",
        "schema": "geist.fractal.observation_context.v1",
        "body_policy": {
            "embedded": "projected",
            "full_material": "stored_when_artifact_ref_present",
        },
        "original": {
            "count": len(observations),
            "chars": sum(len(item) for item in observations),
            "largest_chars": max((len(item) for item in observations), default=0),
        },
        "observations": rows,
    }
    if len(observations) > OBSERVATION_CONTEXT_MAX_ITEMS:
        payload["omitted_observations"] = len(observations) - OBSERVATION_CONTEXT_MAX_ITEMS
    if call is not None:
        payload["call"] = {
            "run_id": getattr(call, "run_id", "") or "",
            "call_id": call.call_id,
            "parent_call_id": call.parent_call_id,
            "branch_path": call.branch_path,
            "spawn_kind": call.spawn_kind,
            "expansion_round": call.expansion_round,
            "tool_round": call.tool_round,
        }
        payload["call"] = {key: value for key, value in payload["call"].items() if value not in (None, "", [], {})}
    return payload


def _observation_context_brief(
    text: str,
    *,
    index: int,
    call: FractalCall | None,
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None,
) -> dict[str, Any]:
    value = str(text or "")
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
    payload = _load_json_object(value)
    row: dict[str, Any] = {
        "index": index,
        "chars": len(value),
        "sha256": digest,
    }
    if isinstance(payload, dict):
        kind = str(payload.get("type") or "json").strip() or "json"
        row["kind"] = kind
        schema = str(payload.get("schema") or "").strip()
        if schema:
            row["schema"] = schema
        if kind == "fractal_tool_observation":
            row.update(_tool_observation_context_brief(payload))
        elif kind == COMPLETION_GUARD_TYPE:
            row["completion_guard"] = {
                key: payload.get(key)
                for key in ("reason", "pending_verification", "attempted_final_response")
                if payload.get(key) not in (None, "", [], {})
            }
        elif kind == "fractal_context_projection":
            row["projection"] = {
                key: payload.get(key)
                for key in ("projection_kind", "reason", "original_chars", "first_line", "retrieve")
                if payload.get(key) not in (None, "", [], {})
            }
        elif kind == "fractal_cli_recent_history":
            turns = payload.get("turns") if isinstance(payload.get("turns"), list) else []
            row["history"] = {
                "turn_count": len(turns),
                "turns": [
                    {
                        "turn": item.get("turn"),
                        "user": _truncate_for_trace(item.get("user"), max_chars=240),
                        "assistant": _truncate_for_trace(item.get("assistant"), max_chars=320),
                        "trace_path": item.get("trace_path"),
                        "expansion_count": item.get("expansion_count"),
                        "tool_names": item.get("tool_names") or [],
                    }
                    for item in turns[:8]
                    if isinstance(item, dict)
                ],
            }
        else:
            row["keys"] = sorted(str(key) for key in payload.keys())[:32]
            row["preview"] = _context_value_brief(payload)
    else:
        row["kind"] = "text"
        first_line = _first_nonempty_line(value)
        if first_line:
            row["first_line"] = _truncate_for_trace(first_line, max_chars=OBSERVATION_FIRST_LINE_CHARS)
        row["preview"] = _truncate_for_trace(value, max_chars=OBSERVATION_CONTEXT_VALUE_TEXT_CHARS)

    stored = _store_context_observation_material(
        material_store,
        value,
        digest=digest,
        index=index,
        call=call,
    )
    if isinstance(stored, dict):
        row["artifact"] = {
            key: stored.get(key)
            for key in ("ref", "path", "relative_path")
            if stored.get(key)
        }
        ref = stored.get("ref")
        if ref:
            row["retrieve"] = {"tool": "artifact.read", "arguments": {"ref": ref}}
    return {key: value for key, value in row.items() if value not in (None, "", [], {})}


def _tool_observation_context_brief(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    tools: list[str] = []
    status_counts: dict[str, int] = {}
    projected_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "")
        status = str(item.get("status") or "")
        if tool and tool not in tools:
            tools.append(tool)
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
    selected_items, omitted_middle = _select_tool_observation_items(items)
    for item in selected_items:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "")
        status = str(item.get("status") or "")
        projected: dict[str, Any] = {
            "call_id": item.get("call_id"),
            "tool": tool,
            "status": status,
        }
        if "ok" in item:
            projected["ok"] = item.get("ok") is True
        data = item.get("data")
        if data not in (None, "", [], {}):
            projected["data"] = _context_value_brief(data)
        projected_items.append({key: value for key, value in projected.items() if value not in (None, "", [], {})})
    result: dict[str, Any] = {
        "tool_observation": {
            "item_count": len(items),
            "projected_item_count": len(projected_items),
            "tools": tools[:32],
            "status_counts": status_counts,
            "items": projected_items,
        }
    }
    if omitted_middle:
        result["tool_observation"]["omitted_middle_items"] = omitted_middle
    health = payload.get("health")
    if isinstance(health, dict) and health:
        result["tool_observation"]["health"] = health
    anchors = payload.get("anchors")
    if isinstance(anchors, list) and anchors:
        result["tool_observation"]["anchors"] = anchors[:REALITY_ANCHOR_MAX_ITEMS]
    if len(items) > len(projected_items):
        result["tool_observation"]["omitted_items"] = len(items) - len(projected_items)
    return result


def _select_tool_observation_items(items: list[Any]) -> tuple[list[Any], int]:
    if not any(isinstance(item, dict) and _tool_item_is_failure(item) for item in items):
        return _select_observation_items(items, head=6, tail=2)
    selected: list[Any] = []
    seen: set[int] = set()

    def add(index: int, item: Any) -> None:
        if index in seen or len(selected) >= OBSERVATION_CONTEXT_TOOL_ITEMS:
            return
        seen.add(index)
        selected.append(item)

    for index, item in enumerate(items):
        if isinstance(item, dict) and _tool_item_is_failure(item):
            add(index, item)
    for index, item in enumerate(items[:4]):
        add(index, item)
    tail_start = max(len(items) - 2, 0)
    for offset, item in enumerate(items[tail_start:], start=tail_start):
        add(offset, item)
    return selected, max(len(items) - len(selected), 0)


def _select_observation_items(items: list[Any], *, head: int, tail: int) -> tuple[list[Any], int]:
    if len(items) <= head + tail:
        return items, 0
    return [*items[:head], *items[-tail:]], len(items) - head - tail


def _context_value_brief(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return _context_scalar_brief(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        priority = [
            "ok",
            "action",
            "status",
            "error",
            "message",
            "path",
            "relative_path",
            "url",
            "method",
            "id",
            "pid",
            "name",
            "title",
            "count",
            "total_count",
            "visible_count",
            "running",
            "returncode",
            "artifact",
            "retrieve",
            "anchors",
        ]
        keys = [key for key in priority if key in value]
        keys.extend(key for key in value.keys() if key not in keys)
        for key in keys[:24]:
            result[str(key)] = _context_value_brief(value.get(key), depth=depth + 1)
        if len(value) > len(keys[:24]):
            result["omitted_keys"] = len(value) - len(keys[:24])
        return {key: item for key, item in result.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        items = [_context_value_brief(item, depth=depth + 1) for item in value[:12]]
        if len(value) > 12:
            items.append({"omitted_items": len(value) - 12})
        return items
    return _context_scalar_brief(value)


def _context_scalar_brief(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= OBSERVATION_CONTEXT_VALUE_TEXT_CHARS:
            return value
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        brief: dict[str, Any] = {
            "chars": len(value),
            "sha256": digest,
        }
        first_line = _first_nonempty_line(value)
        if first_line:
            brief["first_line"] = _truncate_for_trace(
                first_line,
                max_chars=OBSERVATION_CONTEXT_VALUE_TEXT_CHARS,
            )
        return brief
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = str(value)
    return text if len(text) <= OBSERVATION_CONTEXT_VALUE_TEXT_CHARS else {
        "type": type(value).__name__,
        "chars": len(text),
    }


def _store_context_observation_material(
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None,
    text: str,
    *,
    digest: str,
    index: int,
    call: FractalCall | None,
) -> dict[str, Any] | None:
    if material_store is None:
        return None
    source: dict[str, Any] = {"observation_index": index, "reason": "context_projection"}
    if call is not None:
        source.update({
            "call_id": call.call_id,
            "parent_call_id": call.parent_call_id,
            "branch_path": call.branch_path,
            "spawn_kind": call.spawn_kind,
            "expansion_round": call.expansion_round,
            "tool_round": call.tool_round,
        })
    try:
        stored = material_store({
            "material_ref": "obs_" + digest[:16],
            "kind": "fractal_context_observation",
            "text": text,
            "sha256": digest,
            "source": source,
        })
    except Exception:
        return None
    return stored if isinstance(stored, dict) else None


def _observations_health(observations: list[str] | tuple[str, ...]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    failed_items: list[dict[str, Any]] = []
    changed_paths: list[str] = []
    workspace_write_count = 0
    observation_count = 0
    for text in observations or []:
        payload = _load_json_object(str(text or "").strip())
        if not isinstance(payload, dict):
            continue
        observation_count += 1
        health = _health_from_observation_payload(payload)
        if not health:
            continue
        for status, count in (health.get("status_counts") or {}).items():
            status_text = str(status)
            try:
                status_count = int(count)
            except Exception:
                status_count = 0
            status_counts[status_text] = status_counts.get(status_text, 0) + status_count
        failed = health.get("failed_items") if isinstance(health.get("failed_items"), list) else []
        failed_items.extend(item for item in failed if isinstance(item, dict))
        for path in health.get("changed_paths") or []:
            path_text = str(path).strip()
            if path_text and path_text not in changed_paths:
                changed_paths.append(path_text)
        try:
            workspace_write_count += int(health.get("workspace_write_count") or 0)
        except Exception:
            pass

    if not (status_counts or failed_items or changed_paths or workspace_write_count):
        return {}
    result: dict[str, Any] = {
        "observation_count": observation_count,
        "status_counts": status_counts,
    }
    if failed_items:
        result["has_failed_tools"] = True
        result["failed_count"] = len(failed_items)
        result["failed_items"] = failed_items[:OBSERVATION_CONTEXT_TOOL_ITEMS]
    if workspace_write_count:
        result["workspace_write_count"] = workspace_write_count
    if changed_paths:
        result["changed_paths"] = changed_paths[:24]
    verification = _verification_cue_from_tool_items(
        [],
        failed_count=len(failed_items),
        write_count=workspace_write_count,
        changed_paths=changed_paths,
    )
    if verification:
        result["verification"] = verification
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _health_from_observation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    kind = str(payload.get("type") or "")
    if kind == "fractal_tool_observation":
        health = payload.get("health")
        if isinstance(health, dict) and health:
            return health
        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        return _tool_observation_health_from_items([item for item in items if isinstance(item, dict)])
    if kind == "fractal_observation_context":
        return _health_from_observation_context(payload)
    return {}


def _health_from_observation_context(payload: dict[str, Any]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    failed_items: list[dict[str, Any]] = []
    changed_paths: list[str] = []
    workspace_write_count = 0
    rows = payload.get("observations") if isinstance(payload.get("observations"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tool_observation = row.get("tool_observation") if isinstance(row.get("tool_observation"), dict) else {}
        health = tool_observation.get("health") if isinstance(tool_observation.get("health"), dict) else {}
        if not health:
            continue
        for status, count in (health.get("status_counts") or {}).items():
            try:
                status_counts[str(status)] = status_counts.get(str(status), 0) + int(count)
            except Exception:
                pass
        failed = health.get("failed_items") if isinstance(health.get("failed_items"), list) else []
        failed_items.extend(item for item in failed if isinstance(item, dict))
        for path in health.get("changed_paths") or []:
            path_text = str(path).strip()
            if path_text and path_text not in changed_paths:
                changed_paths.append(path_text)
        try:
            workspace_write_count += int(health.get("workspace_write_count") or 0)
        except Exception:
            pass
    result: dict[str, Any] = {"status_counts": status_counts}
    if failed_items:
        result["failed_items"] = failed_items
    if changed_paths:
        result["changed_paths"] = changed_paths
    if workspace_write_count:
        result["workspace_write_count"] = workspace_write_count
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _render_call_packet(
    call: FractalCall,
    *,
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> str:
    observations, observation_policy = render_observations_for_context(
        call.observations,
        call=call,
        material_store=material_store,
    )
    payload: dict[str, Any] = {
        "type": "fractal_api_call",
        "schema": "geist.fractal.api_call.v1",
        "root_task": call.root_task,
        "call": {
            "call_id": call.call_id,
            "parent_call_id": call.parent_call_id,
            "spawn_kind": call.spawn_kind,
            "spawn_sequence": call.spawn_sequence,
            "branch_path": call.branch_path,
            "sibling_index": call.sibling_index,
            "sibling_count": call.sibling_count,
            "expansion_round": call.expansion_round,
            "tool_round": call.tool_round,
        },
        "instruction": call.instruction or call.root_task,
        "continuation_context": call.continuation_context,
        "observations": observations,
    }
    observation_health = _observations_health(call.observations)
    if observation_health:
        payload["observation_health"] = observation_health
    if observation_policy:
        payload["observation_policy"] = observation_policy
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _merge_continuation(*parts: str, max_chars: int | None = 5000) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = str(part or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return _truncate_for_trace("\n\n".join(merged), max_chars=max_chars)


def _child_continuation(parent: str, child: str) -> str:
    value = str(child or "").strip()
    if value:
        return value
    return str(parent or "").strip()


def _truncate_for_trace(text: str, max_chars: int | None = 1000) -> str:
    value = str(text or "").strip()
    if not _limit_enabled(max_chars):
        return value
    return value[:max_chars]


def _render_trace_synthesis_observation(
    *,
    trace: tuple[dict[str, Any], ...],
    results: tuple[FractalCompleted, ...],
    expansion_count: int,
    max_chars: int | None,
    material_store: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> str:
    raw_payload = {
        "expansion_count": expansion_count,
        "completed_results": [
            _completed_result_projection(item)
            for item in results
        ],
        "trace": list(trace),
    }
    full_trace_artifact: dict[str, Any] | None = None
    if material_store is not None:
        raw_text = json.dumps(raw_payload, ensure_ascii=False, indent=2, default=str)
        digest = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()
        try:
            stored = material_store({
                "material_ref": "trace_" + digest[:16],
                "kind": "fractal_trace",
                "text": raw_text,
                "sha256": digest,
                "source": {"event": "trace_synthesis_observation"},
            })
        except Exception:
            stored = None
        if isinstance(stored, dict):
            full_trace_artifact = {
                key: stored.get(key)
                for key in ("ref", "path", "relative_path")
                if stored.get(key)
            }
    payload = {
        "expansion_count": expansion_count,
        "completed_results": [
            _completed_result_projection(item)
            for item in results
        ],
        "trace_projection": _compact_trace_projection(trace),
    }
    if full_trace_artifact:
        payload["full_trace_artifact"] = full_trace_artifact
        payload["full_trace_retrieve"] = {
            "tool": "artifact.read",
            "arguments": {"ref": full_trace_artifact.get("ref")},
        }
    text = json.dumps(
        {
            "type": "fractal_trace_synthesis_observation",
            "schema": "geist.fractal.trace_synthesis_observation.v1",
            **payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return _truncate_for_trace(text, max_chars=max_chars)


def _completed_result_projection(item: FractalCompleted) -> dict[str, Any]:
    payload = {
        "ok": item.ok,
        "error": item.error,
        "response": item.response,
        "continuation_context": item.continuation_context,
        "observation_count": len(item.observations),
        "pending_verification": item.pending_verification,
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _compact_trace_projection(trace: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    events = list(trace)
    omitted = 0
    if len(events) > TRACE_SYNTHESIS_HEAD_EVENTS + TRACE_SYNTHESIS_TAIL_EVENTS:
        omitted = len(events) - TRACE_SYNTHESIS_HEAD_EVENTS - TRACE_SYNTHESIS_TAIL_EVENTS
        selected = [
            *events[:TRACE_SYNTHESIS_HEAD_EVENTS],
            *events[-TRACE_SYNTHESIS_TAIL_EVENTS:],
        ]
    else:
        selected = events
    return {
        "event_count": len(events),
        "projected_count": len(selected),
        "omitted_middle_events": omitted,
        "events": [_compact_trace_event(event) for event in selected],
    }


def _compact_trace_event(event: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "event",
        "sequence",
        "call_id",
        "parent_call_id",
        "spawn_kind",
        "branch_path",
        "sibling_index",
        "sibling_count",
        "expansion_round",
        "tool_round",
        "reason",
        "error",
    ):
        value = event.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    for key in ("tools", "instructions", "added", "removed"):
        value = event.get(key)
        if isinstance(value, list) and value:
            compact[key] = value[:40]
            if len(value) > 40:
                compact[f"{key}_omitted"] = len(value) - 40
    for key in ("response", "continuation_context", "instruction"):
        value = str(event.get(key) or "").strip()
        if value:
            compact[key] = _truncate_for_trace(value, max_chars=1200)
    observation = str(event.get("observation") or "")
    if observation:
        compact["observation_chars"] = len(observation)
        compact["observation_preview"] = _truncate_for_trace(observation, max_chars=1200)
    for key in ("observation_chars", "expansion_count", "tool_count"):
        value = event.get(key)
        if value not in (None, "", [], {}) and key not in compact:
            compact[key] = value
    return compact


def _limit_enabled(limit: int | None) -> bool:
    return limit is not None and limit > 0


def _limit_reached(current: int, limit: int | None) -> bool:
    return _limit_enabled(limit) and current >= limit
