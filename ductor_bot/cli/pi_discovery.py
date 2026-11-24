"""Model discovery for the Pi coding-agent CLI (``pi --list-models``)."""

from __future__ import annotations

import asyncio
import logging
from shutil import which

from ductor_bot.config import PI_MODEL_PREFIX
from ductor_bot.infra.platform import CREATION_FLAGS as _CREATION_FLAGS
from ductor_bot.infra.process_tree import force_kill_process_tree

logger = logging.getLogger(__name__)

_DISCOVERY_TIMEOUT = 20.0
_HEADER_PREFIXES = ("provider", "usage:", "flags:", "available")


async def discover_pi_models() -> tuple[str, ...]:
    """Return Pi catalog models as ``pi/<provider>/<model>`` IDs.

    ``--offline`` keeps startup discovery local and uses Pi's cached catalog.
    Returns an empty tuple when the CLI is missing, unauthenticated, times out,
    or errors — callers then fall back to the cached list.
    """
    binary = which("pi")
    if not binary:
        logger.debug("pi not available for model discovery")
        return ()

    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "--offline",
            "--list-models",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=_CREATION_FLAGS,
        )
    except (OSError, ValueError):
        logger.debug("pi --list-models spawn failed", exc_info=True)
        return ()

    try:
        async with asyncio.timeout(_DISCOVERY_TIMEOUT):
            stdout_bytes, stderr_bytes = await proc.communicate()
    except TimeoutError:
        logger.warning("pi --list-models discovery timed out")
        force_kill_process_tree(proc.pid)
        await proc.communicate()
        return ()

    stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
    if proc.returncode not in (0, None):
        logger.debug("pi --list-models exited with code %s: %s", proc.returncode, stderr[:300])
        return ()

    return _parse_models(stdout_bytes.decode(errors="replace"))


def _parse_models(output: str) -> tuple[str, ...]:
    """Parse the fixed-width ``pi --list-models`` table into internal IDs."""
    models: list[str] = []
    seen: set[str] = set()

    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("usage:", "flags:", "available subcommands:")):
            return ()
        if lowered.startswith(_HEADER_PREFIXES):
            continue

        columns = line.split()
        if len(columns) < 2:
            continue
        provider, model = columns[0], columns[1]
        if not provider or not model or "/" in provider:
            continue

        model_id = f"{PI_MODEL_PREFIX}{provider}/{model}"
        if model_id in seen:
            continue
        seen.add(model_id)
        models.append(model_id)

    return tuple(models)
