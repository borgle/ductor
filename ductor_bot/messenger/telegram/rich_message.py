"""Plain-text rendering for Telegram "rich messages".

Bot API 9.x can deliver channel posts as `rich_message` blocks instead of a
plain `text` / `caption` string. Such updates carry no `Message.text` at all,
so any consumer that only looks at the text fields silently drops them.

`rich_message_to_text` flattens the block tree into Markdown-ish text that the
orchestrator can hand to the provider CLI like any other user message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiogram.types import Message

_BOLD = "bold"
_ITALIC = "italic"
_CODE = "code"
_PRE = "pre"
_BLOCKQUOTE = "blockquote"

_TEXT_CONTAINER_TYPES = (
    _BOLD,
    _ITALIC,
    _CODE,
    _PRE,
    _BLOCKQUOTE,
    "spoiler",
    "strikethrough",
    "underline",
    "subscript",
    "superscript",
    "marked",
)


def rich_message_to_text(message: Message) -> str | None:
    """Return *message*'s rich content as Markdown-ish text.

    Returns ``None`` when the message carries no rich content, so callers can
    keep their existing "nothing to process" branch.
    """
    rich = getattr(message, "rich_message", None)
    if rich is None:
        return None

    blocks = _field(rich, "blocks")
    if not blocks:
        return None

    rendered = [_block_to_text(block) for block in blocks]
    text = "\n\n".join(part for part in rendered if part)
    return text or None


def _field(node: Any, name: str) -> Any:
    """Read attribute *name* from an aiogram model or a plain mapping."""
    if isinstance(node, dict):
        return node.get(name)
    return getattr(node, name, None)


def _block_to_text(block: Any) -> str:
    """Render a single rich block (heading, paragraph, list, ...)."""
    kind = _field(block, "type")

    if kind == "divider":
        return "---"
    if kind in ("list", "ordered_list"):
        items = _field(block, "items") or []
        return "\n".join(_list_item_to_text(item) for item in items)
    if kind == "section_heading":
        size = _field(block, "size") or 3
        try:
            level = max(1, min(6, int(size) - 1))
        except (TypeError, ValueError):
            level = 3
        return f"{'#' * level} {_text_to_markdown(_field(block, 'text'))}".rstrip()
    if kind == "block_quotation":
        body = _blocks_to_text(_field(block, "blocks"))
        return _quote(body)

    # paragraph, footer, details summary, ...: render the text payload.
    return _text_to_markdown(_field(block, "text"))


def _list_item_to_text(item: Any) -> str:
    """Render one list item, keeping nested blocks under the bullet."""
    body = _blocks_to_text(_field(item, "blocks"))
    if not body:
        body = _text_to_markdown(_field(item, "text"))
    label = _field(item, "label")
    if label:
        body = f"{_text_to_markdown(label)} {body}".strip()
    return f"- {body}".rstrip()


def _blocks_to_text(blocks: Any) -> str:
    """Render a nested block list."""
    if not blocks:
        return ""
    rendered = [_block_to_text(block) for block in blocks]
    return "\n\n".join(part for part in rendered if part)


def _text_to_markdown(node: Any) -> str:
    """Recursively render rich text nodes (str, list or styled entity)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, (list, tuple)):
        return "".join(_text_to_markdown(item) for item in node)

    kind = _field(node, "type")
    if kind == "hashtag":
        return str(_field(node, "text") or "")
    if kind in ("custom_emoji", "emoji"):
        return str(
            _field(node, "emoji") or _field(node, "alternative_text") or _field(node, "text") or ""
        )
    return _apply_text_style(kind, _node_body(node))


def _node_body(node: Any) -> str:
    """Render the payload carried by a styled text node."""
    kind = _field(node, "type")
    inner = _field(node, "text")
    if inner is None and kind == "mathematical_expression":
        inner = _field(node, "expression")
    rendered = _text_to_markdown(inner)
    if not rendered and kind in _TEXT_CONTAINER_TYPES:
        rendered = _blocks_to_text(_field(node, "blocks"))
    return rendered


def _apply_text_style(kind: Any, rendered: str) -> str:
    """Wrap *rendered* text in the Markdown marker of its node type."""
    if kind == _BOLD:
        return f"**{rendered}**"
    if kind == _ITALIC:
        return f"*{rendered}*"
    if kind == _CODE:
        return f"`{rendered}`"
    if kind == _PRE:
        return f"\n```\n{rendered}\n```\n"
    if kind == _BLOCKQUOTE:
        return _quote(rendered)
    return rendered


def _quote(text: str) -> str:
    """Render *text* as a Markdown block quote."""
    if not text:
        return ""
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
