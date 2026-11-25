"""Dynamic Codex model discovery via ``codex app-server`` JSON-RPC."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import which

import aiohttp

from ductor_bot.cli.codex_utils import format_codex_model_id
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

    catalog_models = await _discover_catalog_models()
    if catalog_models:
        logger.info("Codex discovery found %d models from provider catalogs", len(catalog_models))
        return catalog_models

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
    logger.info("Codex discovery found %d models via app-server", len(models))
    return models


@dataclass(frozen=True, slots=True)
class _CodexCatalogConfig:
    """One Codex provider's catalog endpoint and credentials."""

    provider: str
    model: str
    catalog_url: str
    token: str
    active: bool


def _codex_home() -> Path:
    default = Path.home() / ".codex"
    return Path(os.environ.get("CODEX_HOME", str(default)))


def _load_codex_catalog_configs() -> list[_CodexCatalogConfig]:
    """Read every provider with a ``model_catalog_url`` from CODEX_HOME.

    Codex itself uses one active provider per invocation, but its config can
    define several custom providers. ductor lists all of their catalogs and
    rewrites the selected model back to ``-c model_provider=<provider>``.
    """
    try:
        data = tomllib.loads((_codex_home() / "config.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []

    active_provider = str(data.get("model_provider") or "openai")
    active_model = str(data.get("model") or "")
    providers = data.get("model_providers")
    if not isinstance(providers, dict):
        return []

    provider_ids = [active_provider, *(pid for pid in providers if pid != active_provider)]
    configs: list[_CodexCatalogConfig] = []
    for provider_id in provider_ids:
        provider = providers.get(provider_id)
        if not isinstance(provider, dict):
            continue

        catalog_url = provider.get("model_catalog_url")
        if not isinstance(catalog_url, str) or not catalog_url.startswith(("http://", "https://")):
            continue

        token = provider.get("experimental_bearer_token")
        if not isinstance(token, str) or not token:
            env_key = provider.get("env_key")
            token = os.environ.get(str(env_key), "") if env_key else ""

        configs.append(
            _CodexCatalogConfig(
                provider=str(provider_id),
                model=active_model if provider_id == active_provider else "",
                catalog_url=catalog_url,
                token=token if isinstance(token, str) else "",
                active=provider_id == active_provider,
            )
        )
    return configs


async def _discover_catalog_models() -> list[CodexModelInfo]:
    """Fetch all configured Codex provider catalogs."""
    configs = await asyncio.to_thread(_load_codex_catalog_configs)
    return await _discover_catalog_models_for(configs)


async def _discover_catalog_models_for(
    configs: list[_CodexCatalogConfig],
) -> list[CodexModelInfo]:
    if not configs:
        return []

    groups = await asyncio.gather(*(_fetch_catalog(config) for config in configs))
    models: list[CodexModelInfo] = []
    seen: set[str] = set()
    for group in groups:
        for model in group:
            if model.id in seen:
                continue
            seen.add(model.id)
            models.append(model)

    if models and not any(model.is_default for model in models):
        models[0] = replace(models[0], is_default=True)
    return models


async def _fetch_catalog(config: _CodexCatalogConfig) -> list[CodexModelInfo]:
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
                logger.debug("Codex catalog for %s returned HTTP %s", config.provider, resp.status)
                return []
            data = await resp.json()
    except (aiohttp.ClientError, TimeoutError, ValueError):
        logger.debug("Codex catalog fetch failed for %s", config.provider, exc_info=True)
        return []

    return _parse_catalog(data, provider=config.provider, default_model=config.model)


def _parse_catalog(
    data: object,
    *,
    provider: str,
    default_model: str = "",
) -> list[CodexModelInfo]:
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
                id=format_codex_model_id(provider, model_id),
                display_name=str(entry.get("display_name") or entry.get("displayName") or model_id),
                description=str(entry.get("description") or ""),
                supported_efforts=efforts or ("medium",),
                default_effort=str(
                    entry.get("default_reasoning_level")
                    or entry.get("defaultReasoningEffort")
                    or "medium"
                ),
                is_default=bool(default_model) and model_id == default_model,
            )
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
