from __future__ import annotations

import json

import pytest

from geist.core.agent import ToolSpec
from geist.core.fractal import FractalCall, FractalRuntime, NATIVE_FRACTAL_PROTOCOL


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    async def generate(self, messages: list[dict[str, str]], **_: object) -> str:
        self.messages.append(messages)
        return self.responses.pop(0)


class NoopDispatcher:
    async def execute(self, tool_call: dict, state: object, project_name: str) -> dict:
        return {"ok": True, "tool_call": tool_call, "project_name": project_name}


def runtime_payload(**overrides: object) -> str:
    payload = {
        "final_response": "done",
        "continuation_context": "",
        "clear_continuation_context": False,
        "tool_calls": [],
        "fractals": [],
    }
    payload.update(overrides)
    return "```runtime\n" + json.dumps(payload) + "\n```"


@pytest.mark.asyncio
async def test_extracted_runtime_completes_one_plain_call() -> None:
    llm = FakeLLM([runtime_payload(final_response="hello from geist")])
    runtime = FractalRuntime(llm, NoopDispatcher())

    run = await runtime.run(FractalCall(root_task="say hello"))

    assert run.expansion_count == 0
    assert len(run.results) == 1
    assert run.results[0].ok is True
    assert run.results[0].response == "hello from geist"
    assert llm.messages[0][0]["content"] == NATIVE_FRACTAL_PROTOCOL


@pytest.mark.asyncio
async def test_extracted_runtime_executes_a_registered_tool() -> None:
    llm = FakeLLM([
        runtime_payload(
            final_response="",
            tool_calls=[{"tool": "echo", "arguments": {"text": "hi"}}],
        ),
        runtime_payload(final_response="tool observed"),
    ])
    runtime = FractalRuntime(llm, NoopDispatcher())

    run = await runtime.run(
        FractalCall(
            root_task="use echo",
            tools={"echo": ToolSpec(name="echo", description="Echo text", reads=set(), writes=set())},
            workspace_label="geist-test",
        )
    )

    assert run.results[0].ok is True
    assert run.results[0].response == "tool observed"
    assert any(event.get("event") == "tools" for event in run.trace)
