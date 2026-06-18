from __future__ import annotations

from pathlib import Path

import pytest

from geist.local import LocalArtifactStore, LocalToolApi, LocalToolDispatcher, LocalToolRegistry, LocalTraceStore


def test_local_artifact_store_put_read_and_search(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    entry = store.put_text("alpha\nbeta\nfitness plan", kind="note")
    read = store.read(entry["ref"], line_start=2, line_count=1)
    search = store.search("fitness")

    assert entry["ref"].startswith("mat_")
    assert read["ok"] is True
    assert read["content"] == "beta\n"
    assert search["ok"] is True
    assert search["count"] == 1
    assert search["matches"][0]["ref"] == entry["ref"]


def test_local_trace_store_records_and_reads_runtime_events(tmp_path: Path) -> None:
    store = LocalTraceStore(tmp_path / "trace")

    row = store.record_runtime_event({
        "event": "tools",
        "run_id": "run-1",
        "call_id": "call-1",
        "branch_path": "1.2",
        "tools": ["read", "write"],
    })
    result = store.read(event="tools", run_id="run-1", include_data=True)

    assert row["id"].startswith("tr_")
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["objects"][0]["call_id"] == "call-1"
    assert "geist.fractal.trace_rows.v1" in result["text"]


class EchoDispatcher:
    def get_tools(self) -> dict[str, object]:
        return {"echo": object()}

    async def execute(self, call: dict, state: object, project_name: str) -> dict:
        return {
            "ok": True,
            "tool": call.get("tool"),
            "arguments": call.get("arguments"),
            "project_name": project_name,
        }


@pytest.mark.asyncio
async def test_generated_local_tool_registry_executes_tool(tmp_path: Path) -> None:
    registry = LocalToolRegistry(tmp_path / "local_tools")
    result = registry.register(
        name="local.upper",
        description="Uppercase text",
        arguments={"text": "Text to uppercase"},
        overwrite=False,
        reserved_names=set(),
        code="""
async def execute(arguments, state, workspace, tool_api=None):
    text = str(arguments.get("text", ""))
    return {"text": text.upper(), "workspace_name": workspace.name}
""",
    )

    executed = await registry.execute(
        "local.upper",
        {"text": "geist"},
        state={},
        workspace=tmp_path,
    )

    assert result["ok"] is True
    assert "local.upper" in registry.specs()
    assert executed == {"ok": True, "text": "GEIST", "workspace_name": tmp_path.name}


@pytest.mark.asyncio
async def test_local_tool_api_can_call_dispatcher(tmp_path: Path) -> None:
    api = LocalToolApi(
        EchoDispatcher(),
        state={"s": 1},
        project_name="geist-test",
        workspace=tmp_path,
        stack=("local.composed",),
    )

    result = await api.call("echo", {"text": "hi"})

    assert result["ok"] is True
    assert result["tool"] == "echo"
    assert result["arguments"] == {"text": "hi"}
    assert result["project_name"] == "geist-test"


@pytest.mark.asyncio
async def test_local_tool_dispatcher_exposes_workspace_trace_artifacts_and_generated_tools(tmp_path: Path) -> None:
    dispatcher = LocalToolDispatcher(tmp_path, state_root=tmp_path / ".geist_state")

    written = await dispatcher.execute({"tool": "write", "arguments": {"path": "hello.txt", "content": "hello"}})
    read = await dispatcher.execute({"tool": "read", "arguments": {"path": "hello.txt"}})
    traced = await dispatcher.execute({"tool": "trace.write", "arguments": {"title": "note", "text": "trace text"}})
    trace = await dispatcher.execute({"tool": "trace.read", "arguments": {"query": "trace text"}})
    artifact_entry = dispatcher.artifacts.put_text("large material")
    artifact = await dispatcher.execute({"tool": "artifact.read", "arguments": {"ref": artifact_entry["ref"]}})
    scaffolded = await dispatcher.execute({
        "tool": "tool.scaffold",
        "arguments": {
            "name": "local.reverse",
            "description": "Reverse text",
            "arguments": {"text": "Text to reverse"},
            "code": """
async def execute(arguments, state, workspace, tool_api=None):
    return {"text": str(arguments.get("text", ""))[::-1]}
""",
        },
    })
    local_result = await dispatcher.execute({"tool": "local.reverse", "arguments": {"text": "geist"}})

    assert written["ok"] is True
    assert read["content"] == "hello"
    assert traced["ok"] is True
    assert trace["ok"] is True
    assert trace["count"] == 1
    assert artifact["ok"] is True
    assert artifact["content"] == "large material"
    assert scaffolded["ok"] is True
    assert local_result == {"ok": True, "text": "tsieg"}
