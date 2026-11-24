"""Parsers for Pi's JSON event stream (``pi --mode json``)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ductor_bot.cli.stream_events import (
    AssistantTextDelta,
    ResultEvent,
    StreamEvent,
    SystemInitEvent,
    SystemStatusEvent,
    ThinkingEvent,
    ToolUseEvent,
)

logger = logging.getLogger(__name__)


def load_pi_event(line: str) -> dict[str, Any] | None:
    """Decode one JSONL record from Pi, returning ``None`` for blank/bad lines."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.debug("Unparseable Pi stream line: %.200s", stripped)
        return None
    return data if isinstance(data, dict) else None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def _usage_dict(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    input_tokens = int(usage.get("input", 0) or 0)
    output_tokens = int(usage.get("output", 0) or 0)
    total_tokens = int(usage.get("totalTokens", 0) or 0)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": int(usage.get("cacheRead", 0) or 0),
        "cache_creation_input_tokens": int(usage.get("cacheWrite", 0) or 0),
        "total_tokens": total_tokens,
    }


def _cost_from_usage(usage: Any) -> float:
    if not isinstance(usage, dict):
        return 0.0
    cost = usage.get("cost")
    if not isinstance(cost, dict):
        return 0.0
    try:
        return float(cost.get("total", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(slots=True)
class PiResultState:
    """Accumulates the authoritative final message and usage for one Pi run."""

    session_id: str | None = None
    final_text: str = ""
    accumulated: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    model_name: str | None = None
    error: str | None = None
    turns: int = 0

    @property
    def result_text(self) -> str:
        return self.final_text or "".join(self.accumulated)

    def track(self, data: dict[str, Any]) -> None:
        """Update state from one decoded JSONL record."""
        event_type = data.get("type")

        if event_type == "session":
            session_id = data.get("id")
            if isinstance(session_id, str) and session_id:
                self.session_id = session_id
            return

        if event_type == "message_update":
            self._track_update(data)
            return

        if event_type in ("message_end", "turn_end"):
            self._track_message(data)
            return

        if event_type == "agent_settled":
            if data.get("aborted") is True:
                self.error = self.error or "Pi run aborted."
            return

        if event_type == "agent_end":
            if data.get("willRetry") is True:
                self.error = self.error or "Pi run failed and will retry."
            return

        if event_type in ("error", "extension_error"):
            self.error = str(data.get("error") or data.get("message") or "Pi run failed.")

    def _track_update(self, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        parsed_usage = _usage_dict(usage)
        if parsed_usage:
            self.usage = parsed_usage
            self.cost_usd = _cost_from_usage(usage) or self.cost_usd

        event = data.get("assistantMessageEvent")
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        if event_type == "text_delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                self.accumulated.append(delta)
        elif event_type == "error":
            self.error = str(event.get("reason") or event.get("error") or "Pi stream error.")

    def _track_message(self, data: dict[str, Any]) -> None:
        message = data.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return

        if data.get("type") == "turn_end":
            self.turns += 1

        text = _message_text(message)
        if text:
            self.final_text = text

        model = message.get("model")
        if isinstance(model, str) and model:
            self.model_name = model

        usage = message.get("usage")
        parsed_usage = _usage_dict(usage)
        if parsed_usage:
            self.usage = parsed_usage
            self.cost_usd = _cost_from_usage(usage) or self.cost_usd

        stop_reason = message.get("stopReason")
        if stop_reason in ("error", "aborted"):
            self.error = str(message.get("errorMessage") or f"Pi stopped: {stop_reason}")


def pi_events_from_data(  # noqa: PLR0911
    data: dict[str, Any],
) -> list[StreamEvent]:
    """Translate one decoded Pi record into ductor stream events."""
    event_type = data.get("type")

    if event_type == "session":
        session_id = data.get("id")
        return [
            SystemInitEvent(
                type="system",
                subtype="init",
                session_id=session_id if isinstance(session_id, str) else None,
            )
        ]

    if event_type == "message_update":
        event = data.get("assistantMessageEvent")
        if not isinstance(event, dict):
            return []
        event_type = event.get("type")
        if event_type == "text_delta":
            delta = event.get("delta")
            return [AssistantTextDelta(type="assistant", text=delta)] if delta else []
        if event_type == "thinking_delta":
            delta = event.get("delta")
            return [ThinkingEvent(type="assistant", text=delta)] if delta else []
        return []

    if event_type == "tool_execution_start":
        name = data.get("toolName")
        if not isinstance(name, str) or not name:
            return []
        args = data.get("args")
        return [
            ToolUseEvent(
                type="assistant",
                tool_name=name,
                tool_id=data.get("toolCallId") if isinstance(data.get("toolCallId"), str) else None,
                parameters=args if isinstance(args, dict) else None,
            )
        ]

    if event_type == "compaction_start":
        return [SystemStatusEvent(type="system", subtype="status", status="compacting")]

    if event_type == "compaction_end":
        return [SystemStatusEvent(type="system", subtype="status", status=None)]

    return []


def parse_pi_stream_line(line: str) -> list[StreamEvent]:
    """Parse one JSONL line into normalized stream events."""
    data = load_pi_event(line)
    if data is None:
        return []
    return pi_events_from_data(data)


def parse_pi_json(
    raw: str,
) -> tuple[str, str | None, dict[str, int], int, bool, float]:
    """Parse a complete ``--mode json`` stdout blob.

    Returns ``(text, session_id, usage, turns, is_error, cost_usd)``.
    """
    state = PiResultState()
    saw_record = False
    for line in raw.splitlines():
        data = load_pi_event(line)
        if data is None:
            continue
        saw_record = True
        state.track(data)

    is_error = bool(state.error) or (saw_record and not state.result_text)
    return (
        state.result_text,
        state.session_id,
        state.usage,
        state.turns,
        is_error,
        state.cost_usd,
    )


def pi_result_event(
    state: PiResultState,
    *,
    returncode: int | None,
    stderr: str,
) -> ResultEvent:
    """Build the terminal ``ResultEvent`` for a completed Pi process."""
    result = state.result_text
    is_error = bool(state.error) or (returncode not in (0, None) and not result)
    if is_error and not result:
        result = state.error or stderr.strip()[:500] or "Pi failed before producing a response."
    return ResultEvent(
        type="result",
        session_id=state.session_id,
        result=result,
        is_error=is_error,
        returncode=returncode,
        total_cost_usd=state.cost_usd or None,
        usage=state.usage,
        num_turns=state.turns,
    )
