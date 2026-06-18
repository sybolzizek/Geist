"""High-level Geist agent wiring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from geist.context import ContextBundle, build_runtime_context, load_context_bundle
from geist.core.fractal import FractalCall, FractalLimits, FractalRun, FractalRuntime
from geist.local import LocalToolDispatcher
from geist.provider import OpenAICompatibleProvider, apply_dotenv, build_provider_from_env
from geist.session import SessionRef, SessionStore
from geist.trust import TrustStore


@dataclass
class AgentResult:
    response: str
    ok: bool
    error: str = ""
    run: FractalRun | None = None
    session: SessionRef | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "response": self.response,
            "error": self.error,
            "session_id": self.session.session_id if self.session else None,
            "session_path": str(self.session.path) if self.session else None,
            "expansion_count": self.run.expansion_count if self.run else 0,
            "trace": list(self.run.trace) if self.run else [],
        }


class GeistAgent:
    """Runnable local Geist agent."""

    def __init__(
        self,
        workspace: str | Path = ".",
        *,
        provider: OpenAICompatibleProvider | None = None,
        dispatcher: LocalToolDispatcher | None = None,
        session_store: SessionStore | None = None,
        trust_store: TrustStore | None = None,
        trusted: bool | None = None,
        session_id: str | None = None,
        continue_latest: bool = False,
        use_session: bool = True,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
        limits: FractalLimits | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        apply_dotenv(self.workspace)
        self.provider = provider or build_provider_from_env()
        self.dispatcher = dispatcher or LocalToolDispatcher(self.workspace)
        self.session_store = session_store or SessionStore()
        self.trust_store = trust_store or TrustStore()
        self.trusted = self.trust_store.is_trusted(self.workspace) if trusted is None else trusted
        self.session = (
            self.session_store.open(self.workspace, session_id=session_id, continue_latest=continue_latest)
            if use_session
            else None
        )
        self.trace_sink = trace_sink
        self.limits = limits or FractalLimits()
        self.bundle = load_context_bundle(self.workspace, trusted=self.trusted)

    async def run_turn(self, prompt: str) -> AgentResult:
        text = str(prompt or "").strip()
        if not text:
            return AgentResult(response="", ok=False, error="empty prompt", session=self.session)
        tools = self.dispatcher.get_tools()
        recent = self.session_store.recent_history(self.session) if self.session is not None else None
        if self.session is not None:
            self.session_store.append(self.session, {"event": "user", "text": text})

        def context_builder(call: FractalCall) -> list[dict[str, Any]]:
            return build_runtime_context(call, bundle=self.bundle, tools=call.tools or tools, recent_history=recent)

        runtime = FractalRuntime(
            self.provider,
            self.dispatcher,
            context_builder=context_builder,
            trace_sink=self._trace_sink,
            material_store=self._material_store,
            limits=self.limits,
        )
        run = await runtime.run(
            FractalCall(
                root_task=text,
                tools=tools,
                tool_state={"workspace": str(self.workspace), "session_id": self.session.session_id if self.session else None},
                workspace_label=self.workspace.name or str(self.workspace),
            )
        )
        result = run.results[0] if run.results else None
        response = result.response if result else ""
        ok = bool(result.ok) if result else False
        error = str(result.error or "") if result else "no result"
        if self.session is not None:
            self.session_store.append(self.session, {
                "event": "assistant",
                "ok": ok,
                "response": response,
                "error": error,
                "expansion_count": run.expansion_count,
            })
        return AgentResult(response=response, ok=ok, error=error, run=run, session=self.session)

    def _trace_sink(self, event: dict[str, Any]) -> None:
        self.dispatcher.trace.record_runtime_event(event)
        if self.session is not None:
            self.session_store.append(self.session, {"event": "trace", "trace": event})
        if self.trace_sink is not None:
            self.trace_sink(event)

    def _material_store(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        text = str(payload.get("text") or "")
        if not text:
            return None
        return self.dispatcher.artifacts.put_text(
            text,
            material_ref=str(payload.get("material_ref") or ""),
            kind=str(payload.get("kind") or "text"),
            source=payload.get("source") if isinstance(payload.get("source"), dict) else {},
            sha256=str(payload.get("sha256") or ""),
        )


def result_to_json_text(result: AgentResult) -> str:
    return json.dumps(result.as_json(), ensure_ascii=False, indent=2, default=str)
