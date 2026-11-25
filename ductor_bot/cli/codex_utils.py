"""Helpers for Codex model IDs that carry an underlying provider."""

from __future__ import annotations

CODEX_MODEL_PREFIX = "codex/"


def format_codex_model_id(provider: str, model: str) -> str:
    """Build the internal ID for a model served by a custom Codex provider."""
    return f"{CODEX_MODEL_PREFIX}{provider}/{model}"


def split_codex_model_id(model_id: str) -> tuple[str, str] | None:
    """Split ``codex/<provider>/<model>`` into ``(provider, model)``.

    Model slugs may themselves contain slashes, so only the first separator
    after the prefix is treated as the provider boundary.
    """
    rest = model_id.removeprefix(CODEX_MODEL_PREFIX)
    if rest == model_id:
        return None
    provider, sep, model = rest.partition("/")
    if not sep or not provider or not model:
        return None
    return provider, model
