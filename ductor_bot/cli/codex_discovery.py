"""Dynamic Codex model discovery via ``codex app-server`` JSON-RPC."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from shutil import which

import aiohttp

from ductor_bot.infra.platform import CREATION_FLAGS as _CREATION_FLAGS

logger = logging.getLogger(__name__)

_INIT_MSG = json.dumps(
    {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {"clientInfo": {"name": "ductor", "version": "1.0"}},
    }
)
_LIST_MSG = json.dumps(
    {
        "jsonrpc": "2.0",
        "method": "model/list",
        "id": 2,
        "params": {},
    }
)
_INPUT = f"{_INIT_MSG}\n{_LIST_MSG}\n"


@dataclass(frozen=True, slots=True)
class CodexModelInfo:
    """A model discovered from the Codex app-server."""

    id: str
    display_name: str
    description: str
    supported_efforts: tuple[str, ...]
    default_effort: str
    is_default: bool


DISCOVERY_TIMEOUT = 30.0


async def discover_codex_models(  # noqa: C901
    *, deadline: float = DISCOVERY_TIMEOUT
) -> list[CodexModelInfo]:
    """Query ``codex app-server`` for available models.

    Returns an empty list on timeout, missing CLI, or parse error.
    Never raises -- all errors are logged and swallowed.
    """
    codex_path = which("codex")
    if not codex_path:
        logger.debug("codex CLI not found, skipping model discovery")
        return []

    process: asyncio.subprocess.Process | None = None
    lines: list[str] = []

    try:
        process = await asyncio.create_subprocess_exec(
            codex_path,
            "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=_CREATION_FLAGS,
        )
        if process.stdin is None or process.stdout is None:
            logger.warning("Codex app-server spawned without pipes")
            return []

        process.stdin.write(_INPUT.encode())
        await process.stdin.drain()
        # Keep stdin open while reading; closing early can terminate app-server
        # before it sends the model/list response.

        async with asyncio.timeout(deadline):
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace")
                lines.append(line)
                with contextlib.suppress(json.JSONDecodeError):
                    if json.loads(line).get("id") == 2:
                        break
    except TimeoutError:
        logger.warning("Codex discovery timeout after %.0fs", deadline)
        return []
    except OSError:
        logger.warning("Failed to spawn codex app-server", exc_info=True)
        return []
    finally:
        if process is not None:
            if process.stdin is not None and not process.stdin.is_closing():
                with contextlib.suppress(Exception):
                    process.stdin.close()
            await _kill_process(process)

    models = _parse_response("".join(lines))
    if not models:
        models = await _discover_catalog_models()
    logger.info("Codex discovery found %d models", len(models))
    return models


@dataclass(frozen=True, slots=True)
class _CodexCatalogConfig:
    """Active Codex provider catalog endpoint and credentials."""

    model: str
    catalog_url: str
    token: str


def _codex_home() -> Path:
    default = Path.home() / ".codex"
    return Path(os.environ.get("CODEX_HOME", str(default)))


def _load_codex_catalog_config() -> _CodexCatalogConfig | None:
    """Read the active provider's ``model_catalog_url`` from CODEX_HOME.

    Custom Codex providers (Ollama, local gateways, proxy catalogs) are not
    returned by the app-server ``model/list`` RPC in every Codex version. Their
    ``config.toml`` exposes the same catalog over HTTP, so fall back to that.
    """
    try:
        data = tomllib.loads((_codex_home() / "config.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None

    provider_id = str(data.get("model_provider") or "openai")
    providers = data.get("model_providers")
    if not isinstance(providers, dict):
        return None
    provider = providers.get(provider_id)
    if not isinstance(provider, dict):
        return None

    catalog_url = provider.get("model_catalog_url")
    if not isinstance(catalog_url, str) or not catalog_url.startswith(("http://", "https://")):
        return None

    token = provider.get("experimental_bearer_token")
    if not isinstance(token, str) or not token:
        env_key = provider.get("env_key")
        token = os.environ.get(str(env_key), "") if env_key else ""

    return _CodexCatalogConfig(
        model=str(data.get("model") or ""),
        catalog_url=catalog_url,
        token=token if isinstance(token, str) else "",
    )


async def _discover_catalog_models() -> list[CodexModelInfo]:
    """Query the active Codex provider's HTTP model catalog when available."""
    config = await asyncio.to_thread(_load_codex_catalog_config)
    if config is None:
        return []

    # Some local proxy catalogs mishandle gzip; ask for the raw body.
    headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"

    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session,
            session.get(config.catalog_url, headers=headers) as resp,
        ):
            if resp.status != 200:
                logger.debug("Codex model catalog returned HTTP %s", resp.status)
                return []
            data = await resp.json()
    except (aiohttp.ClientError, TimeoutError, ValueError):
        logger.debug("Codex model catalog fetch failed", exc_info=True)
        return []

    return _parse_catalog(data, default_model=config.model)


def _parse_catalog(data: object, *, default_model: str) -> list[CodexModelInfo]:
    """Parse the provider catalog shape used by ``codex debug models``."""
    if not isinstance(data, dict):
        return []
    entries = data.get("models")
    if not isinstance(entries, list):
        return []

    models: list[CodexModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("visibility") not in (None, "list"):
            continue
        if entry.get("supported_in_api") is False:
            continue

        model_id = str(entry.get("slug") or entry.get("id") or "")
        if not model_id:
            continue

        levels = entry.get("supported_reasoning_levels")
        efforts = tuple(
            str(level["effort"])
            for level in (levels if isinstance(levels, list) else [])
            if isinstance(level, dict) and level.get("effort")
        )
        models.append(
            CodexModelInfo(
                id=model_id,
                display_name=str(entry.get("display_name") or entry.get("displayName") or model_id),
                description=str(entry.get("description") or ""),
                supported_efforts=efforts or ("medium",),
                default_effort=str(
                    entry.get("default_reasoning_level")
                    or entry.get("defaultReasoningEffort")
                    or "medium"
                ),
                is_default=False,
            )
        )

    if not models:
        return []

    default_index = next(
        (i for i, model in enumerate(models) if model.id == default_model),
        0,
    )
    models[default_index] = CodexModelInfo(
        id=models[default_index].id,
        display_name=models[default_index].display_name,
        description=models[default_index].description,
        supported_efforts=models[default_index].supported_efforts,
        default_effort=models[default_index].default_effort,
        is_default=True,
    )
    return models


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    """Best-effort kill of a hung process."""
    with contextlib.suppress(OSError):
        process.kill()
    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
        await asyncio.wait_for(process.wait(), timeout=0.2)


def _parse_response(raw: str) -> list[CodexModelInfo]:
    """Parse JSON-RPC stdout lines for the model/list response."""
    for line in raw.strip().splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") != 2:
            continue
        data = msg.get("result", {}).get("data", [])
        return [_parse_model(m) for m in data if isinstance(m, dict)]

    logger.warning("No model/list response found in codex app-server output")
    return []


def _parse_model(entry: dict[str, object]) -> CodexModelInfo:
    """Parse a single model entry from the JSON-RPC response."""
    efforts_raw = entry.get("supportedReasoningEfforts", [])
    efforts = tuple(
        e["reasoningEffort"]
        for e in (efforts_raw if isinstance(efforts_raw, list) else [])
        if isinstance(e, dict) and "reasoningEffort" in e
    )
    return CodexModelInfo(
        id=str(entry.get("id", "")),
        display_name=str(entry.get("displayName", "")),
        description=str(entry.get("description", "")),
        supported_efforts=efforts or ("medium",),
        default_effort=str(entry.get("defaultReasoningEffort", "medium")),
        is_default=bool(entry.get("isDefault", False)),
    )
