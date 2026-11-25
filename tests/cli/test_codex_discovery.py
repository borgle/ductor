"""Tests for dynamic Codex model discovery."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ductor_bot.cli import codex_discovery
from ductor_bot.cli.codex_discovery import (
    CodexModelInfo,
    _CodexCatalogConfig,
    _parse_catalog,
    discover_codex_models,
)

_INIT_RESPONSE = json.dumps(
    {
        "id": 1,
        "result": {"userAgent": "ductor/0.98.0"},
    }
)

_MODEL_LIST_RESPONSE = json.dumps(
    {
        "id": 2,
        "result": {
            "data": [
                {
                    "id": "gpt-5.2-codex",
                    "displayName": "gpt-5.2-codex",
                    "description": "Frontier agentic coding model.",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low", "description": "Fast"},
                        {"reasoningEffort": "medium", "description": "Balanced"},
                        {"reasoningEffort": "high", "description": "Deep"},
                        {"reasoningEffort": "xhigh", "description": "Extra deep"},
                    ],
                    "defaultReasoningEffort": "medium",
                    "isDefault": True,
                },
                {
                    "id": "gpt-5.1-codex-mini",
                    "displayName": "gpt-5.1-codex-mini",
                    "description": "Cheaper, faster.",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium", "description": "Default"},
                        {"reasoningEffort": "high", "description": "Deep"},
                    ],
                    "defaultReasoningEffort": "medium",
                    "isDefault": False,
                },
            ],
            "nextCursor": None,
        },
    }
)

_STDOUT = f"{_INIT_RESPONSE}\n{_MODEL_LIST_RESPONSE}\n"


@pytest.fixture(autouse=True)
def _disable_catalog_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests hermetic: no real ~/.codex/config.toml or HTTP fetch."""
    monkeypatch.setattr(
        codex_discovery,
        "_discover_catalog_models",
        AsyncMock(return_value=[]),
    )


def _mock_process(stdout: str = _STDOUT, returncode: int = 0) -> AsyncMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()
    proc.stdout = MagicMock()
    lines = [f"{line}\n".encode() for line in stdout.splitlines()] + [b""]
    proc.stdout.readline = AsyncMock(side_effect=lines)
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    return proc


async def test_discover_models_parses_response() -> None:
    proc = _mock_process()
    with (
        patch("ductor_bot.cli.codex_discovery.which", return_value="/usr/bin/codex"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        models = await discover_codex_models()

    assert len(models) == 2

    first = models[0]
    assert first.id == "gpt-5.2-codex"
    assert first.display_name == "gpt-5.2-codex"
    assert first.supported_efforts == ("low", "medium", "high", "xhigh")
    assert first.default_effort == "medium"
    assert first.is_default is True

    second = models[1]
    assert second.id == "gpt-5.1-codex-mini"
    assert second.supported_efforts == ("medium", "high")
    assert second.is_default is False
    proc.stdin.close.assert_not_called()


async def test_discover_models_codex_not_installed() -> None:
    with patch("ductor_bot.cli.codex_discovery.which", return_value=None):
        models = await discover_codex_models()
    assert models == []


async def test_discover_models_timeout() -> None:
    async def _hang() -> bytes:
        await asyncio.sleep(1)
        return b""

    proc = _mock_process(stdout="")
    proc.stdout.readline = AsyncMock(side_effect=_hang)
    proc.kill = MagicMock()

    with (
        patch("ductor_bot.cli.codex_discovery.which", return_value="/usr/bin/codex"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        models = await discover_codex_models(deadline=0.1)

    assert models == []
    proc.kill.assert_called()


async def test_discover_models_invalid_json() -> None:
    proc = _mock_process(stdout="not json at all\n")
    with (
        patch("ductor_bot.cli.codex_discovery.which", return_value="/usr/bin/codex"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        models = await discover_codex_models()
    assert models == []


async def test_discover_models_spawn_error() -> None:
    with (
        patch("ductor_bot.cli.codex_discovery.which", return_value="/usr/bin/codex"),
        patch("asyncio.create_subprocess_exec", side_effect=OSError("exec failed")),
    ):
        models = await discover_codex_models()
    assert models == []


def test_parse_catalog_shapes() -> None:
    data = {
        "models": [
            {
                "slug": "deepseek-flash",
                "display_name": "DeepSeek-V4.1-Flash",
                "description": "Fast",
                "visibility": "list",
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "Fast"},
                    {"effort": "high", "description": "Deep"},
                ],
                "default_reasoning_level": "high",
            },
            {
                "slug": "hidden-model",
                "display_name": "Hidden",
                "visibility": "hide",
            },
            {
                "slug": "deepseek-v4-pro",
                "display_name": "DeepSeek-V4-Pro",
                "supported_reasoning_levels": [{"effort": "low"}],
            },
        ]
    }
    models = _parse_catalog(data, default_model="deepseek-flash")
    assert [model.id for model in models] == ["deepseek-flash", "deepseek-v4-pro"]
    assert models[0].is_default is True
    assert models[0].supported_efforts == ("low", "high")
    assert models[0].default_effort == "high"
    assert models[1].is_default is False


def test_load_catalog_config_reads_active_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        """model = "deepseek-flash"
model_provider = "deepseek"

[model_providers.deepseek]
model_catalog_url = "http://127.0.0.1:8787/deepseek/v1/models"
experimental_bearer_token = "secret-token"
"""
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    config = codex_discovery._load_codex_catalog_config()
    assert config == _CodexCatalogConfig(
        model="deepseek-flash",
        catalog_url="http://127.0.0.1:8787/deepseek/v1/models",
        token="secret-token",
    )


async def test_discover_models_falls_back_to_catalog() -> None:
    proc = _mock_process(stdout="not json at all\n")
    catalog_models = [
        CodexModelInfo(
            id="deepseek-flash",
            display_name="DeepSeek-V4.1-Flash",
            description="Fast",
            supported_efforts=("low", "high"),
            default_effort="high",
            is_default=True,
        )
    ]
    with (
        patch("ductor_bot.cli.codex_discovery.which", return_value="/usr/bin/codex"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
        patch(
            "ductor_bot.cli.codex_discovery._discover_catalog_models",
            AsyncMock(return_value=catalog_models),
        ) as catalog_mock,
    ):
        models = await discover_codex_models()

    catalog_mock.assert_awaited_once()
    assert models == catalog_models


async def test_codex_model_info_is_frozen() -> None:
    info = CodexModelInfo(
        id="test",
        display_name="Test",
        description="A test model",
        supported_efforts=("medium", "high"),
        default_effort="medium",
        is_default=False,
    )
    with pytest.raises(AttributeError):
        info.id = "changed"  # type: ignore[misc]
