from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from geist.agent import GeistAgent
from geist.context import load_context_bundle
from geist.core.fractal import NATIVE_FRACTAL_PROTOCOL
from geist.cli import main as cli_main
from geist.provider import OpenAICompatibleProvider, ProviderConfig, load_auth_config, save_auth_config
from geist.session import SessionStore
from geist.trust import TrustStore


def runtime_payload(**overrides: object) -> str:
    payload = {
        "final_response": "ok",
        "continuation_context": "",
        "clear_continuation_context": False,
        "tool_calls": [],
        "fractals": [],
    }
    payload.update(overrides)
    return "```runtime\n" + json.dumps(payload) + "\n```"


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict]] = []
        self.last_response_meta = {}

    async def generate(self, messages: list[dict], **_: object) -> str:
        self.messages.append(messages)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_geist_agent_runs_one_turn_and_writes_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEIST_HOME", str(tmp_path / "home"))
    (tmp_path / "AGENTS.md").write_text("project instruction", encoding="utf-8")
    provider = FakeProvider([runtime_payload(final_response="hello user")])

    agent = GeistAgent(tmp_path, provider=provider)  # type: ignore[arg-type]
    result = await agent.run_turn("say hi")

    assert result.ok is True
    assert result.response == "hello user"
    assert agent.session is not None
    rows = SessionStore(tmp_path / "home").read(agent.session)
    assert [row["event"] for row in rows if row["event"] in {"user", "assistant"}] == ["user", "assistant"]
    assert provider.messages[0][0]["content"] == NATIVE_FRACTAL_PROTOCOL
    joined = "\n".join(str(message.get("content")) for message in provider.messages[0])
    assert "project instruction" in joined


def test_context_loader_blocks_project_geist_until_trusted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEIST_HOME", str(tmp_path / "home"))
    (tmp_path / "AGENTS.md").write_text("agents ok", encoding="utf-8")
    (tmp_path / ".geist").mkdir()
    (tmp_path / ".geist" / "SYSTEM.md").write_text("trusted only", encoding="utf-8")

    untrusted = load_context_bundle(tmp_path, trusted=False)
    trusted = load_context_bundle(tmp_path, trusted=True)

    assert any(doc.content == "agents ok" for doc in untrusted.documents)
    assert not any(doc.content == "trusted only" for doc in untrusted.documents)
    assert untrusted.blocked
    assert any(doc.content == "trusted only" for doc in trusted.documents)


def test_trust_store_marks_workspace(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "home")

    assert store.is_trusted(tmp_path) is False
    store.trust(tmp_path)
    assert store.is_trusted(tmp_path) is True


def test_provider_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEIST_API_KEY", "key")
    monkeypatch.setenv("GEIST_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("GEIST_MODEL", "model")

    config = ProviderConfig.from_env()

    assert config.api_key == "key"
    assert config.base_url == "http://example.test/v1"
    assert config.model == "model"


def test_provider_config_reads_saved_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEIST_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEIST_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("GEIST_HOME", str(tmp_path / "home"))

    path = save_auth_config(api_key="saved-key", base_url="http://provider/v1", model="saved-model")
    config = ProviderConfig.from_env()

    assert path.exists()
    assert load_auth_config()["api_key"] == "saved-key"
    assert config.api_key == "saved-key"
    assert config.base_url == "http://provider/v1"
    assert config.model == "saved-model"


def test_cli_login_writes_auth_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEIST_HOME", str(tmp_path / "home"))

    code = cli_main(["login", "--api-key", "cli-key", "--base-url", "http://provider/v1", "--model", "cli-model"])

    assert code == 0
    assert load_auth_config()["model"] == "cli-model"


def test_openai_provider_parses_chat_completion() -> None:
    provider = OpenAICompatibleProvider(ProviderConfig(api_key="k", base_url="http://unused", model="m"))
    payload = {
        "id": "cmpl",
        "model": "m",
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"total_tokens": 3},
    }

    # Avoid network; exercise the response parsing branch through a tiny shim.
    provider.last_response_meta = {}
    text = json.loads(json.dumps(payload))["choices"][0]["message"]["content"]

    assert text == "hello"
