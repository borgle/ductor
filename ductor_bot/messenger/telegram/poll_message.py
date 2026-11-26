"""Render Telegram polls and quizzes as text for the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiogram.types import Message


def _field(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def _poll_kind(poll: Any) -> str:
    raw = _field(poll, "type", "")
    value = getattr(raw, "value", raw)
    return str(value or "").lower()


def poll_to_text(message: Message) -> str | None:
    """Return a poll/quiz as readable text, or ``None`` when absent."""
    poll = _field(message, "poll")
    if poll is None:
        return None

    question = str(_field(poll, "question", "") or "").strip()
    options = _field(poll, "options") or []
    is_quiz = _poll_kind(poll) == "quiz"
    correct = _field(poll, "correct_option_id")

    lines = ["[Telegram quiz]" if is_quiz else "[Telegram poll]"]
    if question:
        lines += ["", f"Question: {question}"]

    rendered_options: list[str] = []
    correct_text = ""
    for index, option in enumerate(options):
        text = str(_field(option, "text", "") or "").strip()
        if not text:
            continue
        marker = "  ✅ correct" if is_quiz and correct == index else ""
        if marker:
            correct_text = text
        rendered_options.append(f"{index + 1}. {text}{marker}")

    if rendered_options:
        lines += ["", "Options:", *rendered_options]

    if is_quiz and correct_text:
        lines += ["", f"Correct answer: {correct_text}"]

    explanation = str(_field(poll, "explanation", "") or "").strip()
    if explanation:
        lines += ["", f"Explanation: {explanation}"]

    return "\n".join(lines).strip()
