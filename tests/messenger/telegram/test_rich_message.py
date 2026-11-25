"""Tests for Telegram rich_message rendering (`rich_message_to_text`)."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiogram.types import Message

from ductor_bot.messenger.telegram.rich_message import rich_message_to_text


def _rich_message(blocks: list[dict], is_rtl: bool | None = None) -> MagicMock:
    """Build a message mock carrying a rich payload (dict form, version-safe)."""
    msg = MagicMock(spec=Message)
    msg.rich_message = {"blocks": blocks, "is_rtl": is_rtl}
    return msg


def _resolve_message(blocks: list[dict], *, text: str | None = None) -> MagicMock:
    """Message mock compatible with `TelegramBot._resolve_text`."""
    from tests.messenger.telegram.test_app import _make_message

    msg = _make_message(text=text)
    msg.rich_message = {"blocks": blocks}
    return msg


class TestRichMessageToText:
    def test_returns_none_without_rich_message(self) -> None:
        msg = MagicMock(spec=Message)
        msg.rich_message = None
        assert rich_message_to_text(msg) is None

    def test_returns_none_for_empty_blocks(self) -> None:
        assert rich_message_to_text(_rich_message([])) is None

    def test_heading_and_paragraph(self) -> None:
        msg = _rich_message(
            [
                {"type": "section_heading", "size": 3, "text": "DON'T CUT CORNERS!"},
                {"type": "paragraph", "text": "Imagine you're building a house"},
            ]
        )

        assert rich_message_to_text(msg) == (
            "## DON'T CUT CORNERS!\n\nImagine you're building a house"
        )

    def test_styled_text_nodes(self) -> None:
        msg = _rich_message(
            [
                {
                    "type": "paragraph",
                    "text": [
                        {"type": "bold", "text": "Meaning: "},
                        {"type": "italic", "text": "skipping steps"},
                        " and ",
                        {"type": "hashtag", "text": "#idioms"},
                    ],
                }
            ]
        )

        assert rich_message_to_text(msg) == "**Meaning: ***skipping steps* and #idioms"

    def test_custom_emoji_uses_alternative_text(self) -> None:
        msg = _rich_message(
            [
                {
                    "type": "paragraph",
                    "text": [
                        {
                            "type": "custom_emoji",
                            "custom_emoji_id": "5368",
                            "alternative_text": "🧠",
                        },
                        " Meaning",
                    ],
                }
            ]
        )

        assert rich_message_to_text(msg) == "🧠 Meaning"

    def test_list_and_divider(self) -> None:
        msg = _rich_message(
            [
                {"type": "divider"},
                {
                    "type": "list",
                    "items": [
                        {
                            "type": "list_item",
                            "blocks": [
                                {"type": "paragraph", "text": "first"},
                            ],
                        },
                        {
                            "type": "list_item",
                            "blocks": [
                                {"type": "paragraph", "text": "second"},
                            ],
                        },
                    ],
                },
            ]
        )

        assert rich_message_to_text(msg) == "---\n\n- first\n- second"

    def test_block_quotation_is_marked(self) -> None:
        msg = _rich_message(
            [
                {
                    "type": "block_quotation",
                    "blocks": [{"type": "paragraph", "text": "nobody will notice"}],
                }
            ]
        )

        assert rich_message_to_text(msg) == "> nobody will notice"

    async def test_resolve_text_reads_rich_message(self) -> None:
        """A rich-only message must reach the orchestrator instead of being dropped."""
        from tests.messenger.telegram.test_app import _make_orchestrator, _make_tg_bot

        tg_bot, _ = _make_tg_bot()
        tg_bot._orchestrator = _make_orchestrator()
        msg = _resolve_message(
            [
                {"type": "section_heading", "size": 3, "text": "DON'T CUT CORNERS!"},
                {"type": "paragraph", "text": "To do something more cheaply or quickly."},
            ]
        )

        result = await tg_bot._resolve_text(msg)

        assert result is not None
        assert "DON'T CUT CORNERS!" in result
        assert "more cheaply or quickly" in result

    async def test_plain_text_message_still_wins(self) -> None:
        from tests.messenger.telegram.test_app import _make_orchestrator, _make_tg_bot

        tg_bot, _ = _make_tg_bot()
        tg_bot._orchestrator = _make_orchestrator()
        msg = _resolve_message([{"type": "paragraph", "text": "rich body"}], text="plain body")

        assert await tg_bot._resolve_text(msg) == "plain body"
