"""LLM response parser for tool-call protocol.

Domain-agnostic.  No manga, VinEnd, or ShortTermMemory coupling.
"""

from __future__ import annotations

import json
import re
from typing import Any

DEFAULT_ACTION_PROTOCOL = """
你通过普通文字回答用户；只有确实需要使用真实工具时，才在回复末尾附加 runtime JSON。

输出格式：
```runtime
{"final_response":"给用户看的回答","tool_calls":[],"self_intent":"本轮意图摘要","continuation_context":"下一轮要原样带入的自由上下文，可为空"}
```

tool_calls 里的 tool 必须是当前上下文真实工具清单里的精确名称。
arguments 必须按对应工具 schema 填写。
不要输出语义 intent、能力卡片 ID、编号或中间名称。
不要发明工具；不确定时说明缺少什么。
""".strip()


class DecisionParser:
    """Parse ```runtime JSON blocks from LLM responses.

    Extracts final_response, tool_calls, self_intent, continuation_context.
    No domain knowledge — pure JSON extraction.
    """

    def __init__(self, action_protocol: str = DEFAULT_ACTION_PROTOCOL) -> None:
        self.action_protocol = action_protocol

    def parse(self, text: str) -> tuple[str, list[dict[str, Any]], str, str | None]:
        payload = self._extract_payload(text)
        parsed = self._load_json(payload)
        if not isinstance(parsed, dict):
            return text.strip(), [], "", None

        final_response = str(parsed.get("final_response") or "").strip()
        if not final_response:
            final_response = self._text_before_runtime_payload(text)
        self_intent = str(parsed.get("self_intent") or "").strip()
        continuation_context = parsed.get("continuation_context")
        continuation_text = str(continuation_context).strip() if continuation_context is not None else None
        tool_calls = self._normalize_tool_calls(parsed.get("tool_calls"))
        return final_response, tool_calls, self_intent, continuation_text

    @staticmethod
    def _extract_payload(text: str) -> str:
        match = re.search(
            r"```runtime[^\S\r\n]*\r?\n(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _text_before_runtime_block(text: str) -> str:
        match = re.search(r"```runtime[^\S\r\n]*\r?\n", text, flags=re.IGNORECASE)
        if not match:
            return text.strip()
        return text[:match.start()].strip()

    @staticmethod
    def _text_before_runtime_payload(text: str) -> str:
        match = re.search(r"```runtime[^\S\r\n]*\r?\n", text, flags=re.IGNORECASE)
        if match:
            return text[:match.start()].strip()

        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            return text[:first].strip()
        return text.strip()

    @staticmethod
    def _load_json(payload: str) -> Any:
        try:
            return json.loads(payload)
        except Exception:
            first = payload.find("{")
            last = payload.rfind("}")
            if first >= 0 and last > first:
                try:
                    return json.loads(payload[first:last + 1])
                except Exception:
                    return None
            return None

    @staticmethod
    def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
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
        return tool_calls[:8]

    @staticmethod
    def _extract_tool_name(item: dict[str, Any]) -> str:
        direct = item.get("tool") or item.get("name")
        if direct:
            return str(direct).strip()
        function = item.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "").strip()
        return ""

    @staticmethod
    def _extract_arguments(item: dict[str, Any]) -> dict[str, Any]:
        raw = item.get("args")
        if raw is None:
            raw = item.get("arguments")
        if raw is None:
            raw = item.get("input")
        function = item.get("function")
        if raw is None and isinstance(function, dict):
            raw = function.get("arguments")
        return DecisionParser._coerce_arguments(raw)

    @staticmethod
    def _coerce_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}
