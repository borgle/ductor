"""Background observer for periodic Pi catalog refresh."""

from __future__ import annotations

from ductor_bot.cli.model_cache import BaseModelCacheObserver
from ductor_bot.cli.pi_cache import PiModelCache


class PiCacheObserver(BaseModelCacheObserver[PiModelCache]):
    """Refreshes the Pi model catalog periodically."""

    def _provider_name(self) -> str:
        return "Pi"

    async def _load_cache(self, *, initial: bool) -> PiModelCache:
        return await PiModelCache.load_or_refresh(self._cache_path, force_refresh=initial)
