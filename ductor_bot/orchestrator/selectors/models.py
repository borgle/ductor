"""Transport-agnostic button and selector response types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Button:
    """A single interactive button."""

    text: str
    callback_data: str


@dataclass(frozen=True, slots=True)
class ButtonGrid:
    """Grid of buttons (list of rows, each row is a list of buttons)."""

    rows: list[list[Button]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SelectorResponse:
    """Result from a selector function: display text + optional buttons."""

    text: str
    buttons: ButtonGrid | None = None


@dataclass(slots=True)
class ModelBrowserState:
    """Ephemeral per-session state for the paginated model browser.

    Kept on the orchestrator so concurrent chats/topics never share a page or
    a pending filter prompt. Purely UI state: it is never persisted.
    """

    provider: str
    query: str = ""
    page: int = 0
    awaiting_query: bool = False
    updated_at: float = 0.0
