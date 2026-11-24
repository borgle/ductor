"""Tests for underlying provider-CLI version discovery and upgrades."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from ductor_bot.cli import cli_versions
from ductor_bot.cli.cli_versions import (
    CLIStatus,
    CLIUpgradeResult,
    _build_upgrade_command,
    get_all_cli_statuses,
    get_cli_status,
    parse_version,
    upgrade_cli,
)

if TYPE_CHECKING:
    import pytest


def test_parse_version_variants() -> None:
    assert parse_version("2.1.331 (Claude Code)") == "2.1.331"
    assert parse_version("codex-cli 0.172.0") == "0.172.0"
    assert parse_version("pi 1.6.0-beta.1") == "1.6.0-beta.1"
    assert parse_version("no version here") is None


def test_build_npm_upgrade_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_versions.shutil, "which", lambda _name: "/usr/bin/npm")
    spec = cli_versions.cli_spec("claude")
    assert spec is not None
    cmd, error = _build_upgrade_command(spec)
    assert error == ""
    assert cmd == ["/usr/bin/npm", "install", "-g", "@anthropic-ai/claude-code@latest"]


def test_build_pi_self_upgrade_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_versions.shutil,
        "which",
        lambda name: "/usr/bin/pi" if name == "pi" else None,
    )
    spec = cli_versions.cli_spec("pi")
    assert spec is not None
    cmd, error = _build_upgrade_command(spec)
    assert error == ""
    assert cmd == ["/usr/bin/pi", "update"]


def test_build_manual_upgrade_is_unsupported() -> None:
    spec = cli_versions.cli_spec("grok")
    assert spec is not None
    cmd, error = _build_upgrade_command(spec)
    assert cmd is None
    assert "Grok" in error


async def test_get_cli_status_update_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_versions,
        "_probe_version",
        AsyncMock(return_value=(True, "1.2.3", "")),
    )
    monkeypatch.setattr(
        cli_versions,
        "fetch_npm_latest",
        AsyncMock(return_value="1.3.0"),
    )
    status = await get_cli_status("codex")
    assert status is not None
    assert status.installed is True
    assert status.version == "1.2.3"
    assert status.latest == "1.3.0"
    assert status.update_available is True
    assert status.upgrade_supported is True


async def test_get_cli_status_without_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_versions,
        "_probe_version",
        AsyncMock(return_value=(True, "1.2.3", "")),
    )
    fetch = AsyncMock()
    monkeypatch.setattr(cli_versions, "fetch_npm_latest", fetch)
    status = await get_cli_status("codex", include_latest=False)
    assert status is not None
    assert status.latest is None
    fetch.assert_not_awaited()


async def test_get_all_statuses_preserves_provider_order(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_status(provider: str, *, include_latest: bool = True) -> CLIStatus:
        del include_latest
        return CLIStatus(
            provider=provider,
            label=provider.title(),
            installed=True,
            version="1.0.0",
            latest="1.0.0",
            update_available=False,
            upgrade_supported=True,
        )

    monkeypatch.setattr(cli_versions, "get_cli_status", fake_status)
    statuses = await get_all_cli_statuses()
    assert [status.provider for status in statuses] == list(cli_versions.PROVIDER_ORDER)


async def test_upgrade_cli_unknown_provider() -> None:
    result = await upgrade_cli("does-not-exist")
    assert result.ok is False
    assert "Unknown provider" in result.output


async def test_upgrade_cli_manual_provider() -> None:
    result = await upgrade_cli("grok")
    assert result.ok is False
    assert "Grok" in result.output


def test_upgrade_result_summary() -> None:
    ok = cli_versions.upgrade_result_summary(CLIUpgradeResult(provider="codex", ok=True, output=""))
    assert "codex" in ok
    failed = cli_versions.upgrade_result_summary(
        CLIUpgradeResult(provider="codex", ok=False, output="boom")
    )
    assert "boom" in failed
