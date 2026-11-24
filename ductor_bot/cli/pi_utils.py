"""Helpers for Pi model IDs and CLI option mapping."""

from __future__ import annotations

from ductor_bot.config import PI_MODEL_PREFIX

# ductor uses "none" for providers that expose "off"; Pi calls the level "off".
PI_EFFORT_MAP: dict[str, str] = {"none": "off"}


def pi_effort(effort: str | None) -> str:
    """Translate a ductor reasoning level to Pi's ``--thinking`` value."""
    if not effort or effort == "default":
        return ""
    return PI_EFFORT_MAP.get(effort, effort)


def split_pi_model_id(model_id: str) -> tuple[str, str] | None:
    """Split ``pi/<provider>/<model>`` (or ``<provider>/<model>``) for Pi.

    Model IDs may themselves contain slashes (for example OpenRouter model
    names), so only the first separator is treated as the provider boundary.
    """
    rest = model_id.removeprefix(PI_MODEL_PREFIX)
    provider, sep, model = rest.partition("/")
    if not sep or not provider or not model:
        return None
    return provider, model
