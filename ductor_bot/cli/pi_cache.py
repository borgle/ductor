"""Persistent cache for Pi catalog models with periodic refresh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from ductor_bot.cli.model_cache import BaseModelCache
from ductor_bot.cli.pi_discovery import discover_pi_models


@dataclass(frozen=True)
class PiModelCache(BaseModelCache):
    """Immutable cache of Pi model IDs (``pi/<provider>/<model>``)."""

    last_updated: str  # ISO 8601 timestamp
    models: tuple[str, ...]

    @classmethod
    def _provider_name(cls) -> str:
        return "Pi"

    @classmethod
    async def _discover(cls) -> tuple[str, ...]:
        return await discover_pi_models()

    @classmethod
    def _empty_models(cls) -> tuple[str, ...]:
        return ()

    def validate_model(self, model_id: str) -> bool:
        """Check if a Pi model exists in the cache."""
        return model_id in self.models

    def to_json(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "last_updated": self.last_updated,
            "models": list(self.models),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Deserialize from JSON."""
        return cls(
            last_updated=data["last_updated"],
            models=tuple(data["models"]),
        )
