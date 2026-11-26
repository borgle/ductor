"""Tests for rendering Telegram polls/quizzes as text."""

from __future__ import annotations

from types import SimpleNamespace

from ductor_bot.messenger.telegram.poll_message import poll_to_text


def _message(poll: object | None) -> SimpleNamespace:
    return SimpleNamespace(poll=poll)


def test_quiz_includes_question_options_and_answer() -> None:
    poll = SimpleNamespace(
        type="quiz",
        question="What is 2 + 2?",
        options=[
            SimpleNamespace(text="3"),
            SimpleNamespace(text="4"),
            SimpleNamespace(text="5"),
        ],
        correct_option_id=1,
        explanation="Basic arithmetic.",
    )
    text = poll_to_text(_message(poll))
    assert text is not None
    assert "[Telegram quiz]" in text
    assert "Question: What is 2 + 2?" in text
    assert "2. 4  ✅ correct" in text
    assert "Correct answer: 4" in text
    assert "Explanation: Basic arithmetic." in text


def test_regular_poll_has_no_answer_marker() -> None:
    poll = SimpleNamespace(
        type="regular",
        question="Pick one",
        options=[SimpleNamespace(text="A"), SimpleNamespace(text="B")],
        correct_option_id=None,
        explanation="",
    )
    text = poll_to_text(_message(poll))
    assert text is not None
    assert "[Telegram poll]" in text
    assert "✅" not in text
    assert "Correct answer:" not in text


def test_no_poll_returns_none() -> None:
    assert poll_to_text(_message(None)) is None


def test_quiz_without_correct_option_still_renders() -> None:
    poll = SimpleNamespace(
        type="quiz",
        question="Unknown",
        options=[SimpleNamespace(text="A")],
        correct_option_id=None,
    )
    text = poll_to_text(_message(poll))
    assert text is not None
    assert "1. A" in text
    assert "Correct answer:" not in text
