#!/usr/bin/env python3
"""Fractal self-invocation demo — DeepSeek-powered.

Usage:
    python demo_fractal.py

Reads LLM config from .env in the same directory.
Runs the fractal flow: root → blueprints → next fractals → accumulated context.

The flow is driven externally (not by FractalLoop)::

    loop = FractalLoop(invariant)
    ctx = FractalContext(invariant, growth=root_growth)

    # root
    ctx = await engine.execute(ctx, state={}, runtime_core=runtime, ...)
    blueprints = loop.process(ctx)          # writes context_updates to store

    # next generation
    for bp in blueprints:                   # caller decides the order
        next_ctx = loop.build(bp)
        result = await engine.execute(next_ctx, ...)
        loop.process(result)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import AsyncOpenAI

from geist import (
    FractalConfig,
    FractalContext,
    FractalEngine,
    FractalLoop,
    FractalResult,
    SharedContext,
    ensure_time,
    reset_time,
)
from geist.protocols import StateSnapshotter, ToolGate


# ---------------------------------------------------------------------------
# .env reader (stdlib only)
# ---------------------------------------------------------------------------

def _load_env(path: str) -> dict[str, str]:
    """Minimal .env reader (no python-dotenv dependency)."""
    env: dict[str, str] = {}
    if not os.path.isfile(path):
        return env
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'").rstrip("，,")
        env[key] = val
    return env


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class SimpleSnapshotter(StateSnapshotter[dict[str, Any]]):
    """Clone/dump by shallow copy."""

    def clone(self, state: dict[str, Any] | None) -> dict[str, Any]:
        return dict(state or {})

    def dump(self, state: dict[str, Any]) -> dict[str, Any]:
        return dict(state)


class ReadOnlyToolGate(ToolGate):
    """Minimal tool set for fractal invocations."""

    def build_selfcall_registry(
        self, requested_tools: Any, current_depth: int, max_depth: int
    ) -> Any:
        return {
            "state.read": "Read from the shared context (key → value).",
        }


# ---------------------------------------------------------------------------
# DeepSeek runtime core
# ---------------------------------------------------------------------------

USER_TEMPLATE = (
    "=== CONTEXT ===\n"
    "{input}\n\n"
    "=== OUTPUT (JSON only) ===\n"
    "{{\n"
    '  "blueprints": [\n'
    '    {{"prompt": "task for next fractal", "refs": ["key_from_shared_context"]}}\n'
    "  ],\n"
    '  "context_updates": {{\n'
    '    "insight_1": "what you found",\n'
    '    "insight_2": "more findings"\n'
    "  }}\n"
    "}}\n"
    "Rules:\n"
    "- blueprints: list; empty means no more generations\n"
    "- refs: existing keys from shared_context in your input\n"
    "- context_updates: dict of your findings; add at least one\n"
    "Output JSON now.  No other text."
)


class DeepSeekRuntime:
    """RuntimeCore that calls the DeepSeek API."""

    def __init__(self, env: dict[str, str]) -> None:
        self._client = AsyncOpenAI(
            api_key=env.get("api_key", ""),
            base_url=env.get("base_url", "https://api.deepseek.com"),
        )
        self._model = env.get("model_name", "deepseek-chat")

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
    ) -> FractalResult:
        try:
            input_data = json.loads(user_input)
        except json.JSONDecodeError:
            input_data = {"raw": user_input}

        invariant = input_data.get("invariant", {})
        system = json.dumps(invariant, ensure_ascii=False, indent=2) if invariant else (
            "You are a fractal agent. You can change everything except this sentence."
        )

        user = USER_TEMPLATE.format(
            input=json.dumps(input_data, ensure_ascii=False, indent=2)
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
        except Exception as exc:
            return FractalResult(
                ok=False,
                error=f"API call failed: {exc}",
            )

        raw = response.choices[0].message.content or ""
        return self._parse(raw)

    def _parse(self, raw: str) -> FractalResult:
        start = raw.find("{")
        if start == -1:
            return FractalResult(ok=True, response=raw, tool_calls=[], tool_rounds=1)

        text = raw[start:]
        if text.startswith("{"):
            end_fence = text.rfind("```")
            if end_fence != -1:
                text = text[:end_fence].rstrip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return FractalResult(ok=True, response=raw, tool_calls=[], tool_rounds=1)

        return FractalResult(
            ok=True,
            response=parsed.get("response", raw),
            tool_calls=parsed.get("tools", []) or [],
            tool_rounds=1,
            tool_results=[parsed],
        )


# ---------------------------------------------------------------------------
# session factory
# ---------------------------------------------------------------------------

def _make_session(*, session_id: str, workspace_label: str) -> object:
    return type("Session", (), {"session_id": session_id, "workspace_label": workspace_label})()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main() -> None:
    # -- config ------------------------------------------------------------
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env = _load_env(env_path)

    if "api_key" not in env:
        print("ERROR: no api_key found in .env", file=sys.stderr)
        sys.exit(1)

    config = FractalConfig(
        max_depth=3,
        max_invocations_per_root=8,
        max_tool_rounds=1,
    )

    snapshotter = SimpleSnapshotter()
    tool_gate = ReadOnlyToolGate()

    # -- invariant (含契约) --------------------------------------------------
    invariant = {
        "identity": (
            "You are an autonomous fractal agent.  "
            "You can change everything except this sentence."
        ),
        "principle": (
            "Unfold yourself into fractal instances to explore "
            "multiple perspectives.  Each fractal is the same self "
            "with different context."
        ),
        "_contract": {
            "blueprints_key": "blueprints",
            "context_updates_key": "context_updates",
            "blueprint": {
                "prompt_field": "prompt",
                "refs_field": "refs",
            },
        },
    }

    # -- engine + loop (loop does NOT hold engine) -------------------------
    engine = FractalEngine(config, snapshotter, tool_gate, invariant=invariant)
    runtime = DeepSeekRuntime(env)

    store = SharedContext(db_path="geist_store.db")
    loop = FractalLoop(invariant, store=store)

    # -- activate budget guard ---------------------------------------------
    time_token = ensure_time(config.max_invocations_per_root)

    # -- pre-populate shared context ---------------------------------------
    store.apply({
        "domain": (
            "Fractals are self-similar patterns that repeat at different scales. "
            "Found in nature (coastlines, ferns, lightning), mathematics "
            "(Mandelbrot set, Julia sets), and technology (antennas, compression, graphics)."
        ),
        "angles": [
            "natural fractals and their properties",
            "mathematical foundations and key sets",
            "technological applications and engineering",
        ],
        "key_properties": [
            "self-similarity across scales",
            "fractional (non-integer) dimension",
            "infinite detail at every zoom level",
            "generated by simple recursive rules",
        ],
    })

    # -- root growth (includes current shared context snapshot) ------------
    initial_growth = {
        "task": (
            "Explore the concept of fractals — in nature, mathematics, "
            "and technology.  Produce a structured analysis."
        ),
        "shared_context": store.snapshot(),
    }

    root = FractalContext(
        invariant=invariant,
        growth=initial_growth,
    )

    print("=" * 60)
    print("geist fractal demo — DeepSeek")
    print("=" * 60)
    print(f"\nModel: {runtime._model}")
    print(f"Depth limit: {config.max_depth}")
    print(f"Budget limit: {config.max_invocations_per_root}")
    print()

    # -- recursive fractal runner ------------------------------------------
    fractal_count = 0

    async def run_fractal(ctx: FractalContext, depth: int = 0) -> None:
        nonlocal fractal_count

        indent = "  " * depth
        label = f"R{depth + 1}" if depth == 0 else f"R{depth + 1}.{fractal_count}"

        result = await engine.execute(
            ctx,
            state={},
            runtime_core=runtime,
            session_factory=_make_session,
            project_name="fractal_demo",
            next_depth=depth,
        )
        fractal_count += 1

        print(f"{indent}─── {label} ───")
        print(f"{indent}  ok: {result.ok}")
        if result.error:
            print(f"{indent}  error: {result.error}")
        resp = str(result.own.get("response", ""))[:200]
        print(f"{indent}  response: {resp}")

        blueprints = loop.process(result)
        print(f"{indent}  blueprints: {len(blueprints)}, "
              f"store keys: {len(loop.shared.snapshot())}")

        if not blueprints:
            print(f"{indent}  → end of chain")
            return

        for i, bp in enumerate(blueprints):
            next_ctx = loop.build(bp)
            await run_fractal(next_ctx, depth + 1)

    # -- start -------------------------------------------------------------
    try:
        await run_fractal(root)
    finally:
        reset_time(time_token)

    # -- summary -----------------------------------------------------------
    print(f"\n─── Shared context snapshot ───")
    snap = loop.shared.snapshot()
    for k, v in snap.items():
        val = str(v)[:100]
        print(f"  {k}: {val}")

    print(f"\n─── Done ───")
    print(f"Total fractals run: {fractal_count}")
    print(f"Shared context keys: {len(snap)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
