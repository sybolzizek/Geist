"""OpenAI-compatible chat provider for Geist."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProviderError(RuntimeError):
    """Raised when a provider request cannot produce model text."""


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        auth = load_auth_config()
        api_key = os.getenv("GEIST_API_KEY") or os.getenv("OPENAI_API_KEY") or str(auth.get("api_key") or "")
        base_url = os.getenv("GEIST_BASE_URL") or os.getenv("OPENAI_BASE_URL") or str(auth.get("base_url") or "https://api.openai.com/v1")
        model = os.getenv("GEIST_MODEL") or os.getenv("OPENAI_MODEL") or str(auth.get("model") or "")
        if not api_key:
            raise ProviderError("GEIST_API_KEY or OPENAI_API_KEY is required")
        if not model:
            raise ProviderError("GEIST_MODEL or OPENAI_MODEL is required")
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=_float_env("GEIST_TEMPERATURE", 0.2),
            max_tokens=_optional_int_env("GEIST_MAX_TOKENS"),
            timeout=_float_env("GEIST_TIMEOUT", 120.0),
        )


class OpenAICompatibleProvider:
    """Minimal async adapter for `/chat/completions` compatible providers."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.last_response_meta: dict[str, Any] = {}

    async def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        return await asyncio.to_thread(self._generate_sync, messages, kwargs)

    def _generate_sync(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
        model = str(kwargs.get("model_key") or kwargs.get("model") or self.config.model)
        temperature = float(kwargs.get("temperature", self.config.temperature))
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        api_key = str(kwargs.get("api_key") or self.config.api_key)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if kwargs.get("json_mode"):
            body["response_format"] = {"type": "json_object"}

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                status = getattr(response, "status", None)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"provider HTTP {exc.code}: {detail[:2000]}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"provider request failed: {exc}") from exc

        payload = json.loads(raw)
        self.last_response_meta = {
            "status": status,
            "model": payload.get("model"),
            "usage": payload.get("usage"),
            "id": payload.get("id"),
        }
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("provider response had no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ProviderError("provider choice was not an object")
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if text:
                            parts.append(str(text))
                    elif item:
                        parts.append(str(item))
                return "\n".join(parts).strip()
            if content is not None:
                return str(content).strip()
        if first.get("text") is not None:
            return str(first.get("text")).strip()
        raise ProviderError("provider response had no text content")


def build_provider_from_env() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(ProviderConfig.from_env())


def load_dotenv(path: str | Path) -> dict[str, str]:
    """Load simple KEY=value pairs from a .env file without mutating env."""
    env_path = Path(path)
    if env_path.is_dir():
        env_path = env_path / ".env"
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _clean_env_value(value)
    return values


def apply_dotenv(path: str | Path) -> dict[str, str]:
    """Apply supported .env provider keys when env vars are not already set."""
    values = load_dotenv(path)
    mapping = {
        "GEIST_API_KEY": ("GEIST_API_KEY", "OPENAI_API_KEY", "api_key"),
        "GEIST_BASE_URL": ("GEIST_BASE_URL", "OPENAI_BASE_URL", "base_url"),
        "GEIST_MODEL": ("GEIST_MODEL", "OPENAI_MODEL", "model", "model_name"),
    }
    applied: dict[str, str] = {}
    for target, aliases in mapping.items():
        if os.getenv(target):
            continue
        value = ""
        for alias in aliases:
            if values.get(alias):
                value = values[alias]
                break
        if value:
            os.environ[target] = value
            applied[target] = value
    return applied


def auth_config_path(home: str | Path | None = None) -> Path:
    root = Path(home or os.getenv("GEIST_HOME") or (Path.home() / ".geist")).resolve()
    return root / "agent" / "auth.json"


def load_auth_config(home: str | Path | None = None) -> dict[str, Any]:
    path = auth_config_path(home)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_auth_config(
    *,
    api_key: str,
    base_url: str,
    model: str,
    home: str | Path | None = None,
) -> Path:
    if not api_key:
        raise ProviderError("api_key is required")
    if not model:
        raise ProviderError("model is required")
    path = auth_config_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "api_key": api_key,
                "base_url": base_url or "https://api.openai.com/v1",
                "model": model,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _clean_env_value(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text.strip().strip(",").strip("\uFF0C").strip()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None
