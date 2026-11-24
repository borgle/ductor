"""Tests for the provider CLI version/upgrade selector."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from ductor_bot.cli.cli_versions import CLIStatus, CLIUpgradeResult
from ductor_bot.orchestrator.selectors.cli_selector import (
    cli_selector_start,
    handle_cli_callback,
    is_cli_selector_callback,
)
from ductor_bot.session.key import SessionKey


def _status(
    provider: str,
    *,
    installed: bool = True,
    version: str | None = "1.0.0",
    latest: str | None = "1.0.0",
    update_available: bool = False,
    upgrade_supported: bool = True,
) -> CLIStatus:
    return CLIStatus(
        provider=provider,
        label=provider.title(),
        installed=installed,
        version=version,
        latest=latest,
        update_available=update_available,
        upgrade_supported=upgrade_supported,
    )


def test_prefix_detection() -> None:
    assert is_cli_selector_callback("cli:r") is True
    assert is_cli_selector_callback("cli:u:claude") is True
    assert is_cli_selector_callback("ms:r") is False


async def test_selector_lists_versions_and_buttons() -> None:
    statuses = [
        _status("claude", version="2.1.0", latest="2.2.0", update_available=True),
        _status("codex", version="0.10.0", latest="0.10.0"),
        _status("pi", installed=False, version=None, latest="1.0.0"),
    ]
    with patch(
        "ductor_bot.orchestrator.selectors.cli_selector.get_all_cli_statuses",
        AsyncMock(return_value=statuses),
    ):
        resp = await cli_selector_start(None, SessionKey(chat_id=1))  # type: ignore[arg-type]

    assert resp.buttons is not None
    assert "CLI" in resp.text
    callbacks = [btn.callback_data for row in resp.buttons.rows for btn in row]
    labels = [btn.text for row in resp.buttons.rows for btn in row]
    assert "cli:u:claude" in callbacks
    assert "cli:u:codex" in callbacks
    assert "cli:r" in callbacks
    assert any("2.1.0" in label and "2.2.0" in label for label in labels)


async def test_refresh_callback_rebuilds_list() -> None:
    with patch(
        "ductor_bot.orchestrator.selectors.cli_selector.get_all_cli_statuses",
        AsyncMock(return_value=[_status("claude")]),
    ):
        resp = await handle_cli_callback(
            None,  # type: ignore[arg-type]
            SessionKey(chat_id=1),
            "cli:r",
        )
    assert resp.buttons is not None


async def test_upgrade_callback_runs_upgrade_and_refreshes() -> None:
    current = _status("claude", version="2.1.0", latest="2.2.0", update_available=True)
    refreshed = _status("claude", version="2.2.0", latest="2.2.0")
    with (
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.get_cli_status",
            AsyncMock(side_effect=[current, refreshed]),
        ),
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.upgrade_cli",
            AsyncMock(return_value=CLIUpgradeResult("claude", True, "ok")),
        ) as upgrade_mock,
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.get_all_cli_statuses",
            AsyncMock(return_value=[refreshed]),
        ),
    ):
        resp = await handle_cli_callback(
            None,  # type: ignore[arg-type]
            SessionKey(chat_id=1),
            "cli:u:claude",
        )

    upgrade_mock.assert_awaited_once_with("claude")
    assert "upgraded" in resp.text.lower()


async def test_install_callback_installs_missing_cli() -> None:
    missing = _status("gemini", installed=False, version=None, latest="0.71.1")
    installed = _status("gemini", version="0.71.1", latest="0.71.1")
    with (
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.get_cli_status",
            AsyncMock(side_effect=[missing, installed]),
        ),
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.upgrade_cli",
            AsyncMock(return_value=CLIUpgradeResult("gemini", True, "ok")),
        ) as install_mock,
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.get_all_cli_statuses",
            AsyncMock(return_value=[installed]),
        ),
    ):
        resp = await handle_cli_callback(
            None,  # type: ignore[arg-type]
            SessionKey(chat_id=1),
            "cli:u:gemini",
        )

    install_mock.assert_awaited_once_with("gemini")
    assert "installed" in resp.text.lower()


async def test_upgrade_callback_skips_when_current() -> None:
    current = _status("claude", version="2.2.0", latest="2.2.0")
    with (
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.get_cli_status",
            AsyncMock(return_value=current),
        ),
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.upgrade_cli",
            AsyncMock(),
        ) as upgrade_mock,
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.get_all_cli_statuses",
            AsyncMock(return_value=[current]),
        ),
    ):
        resp = await handle_cli_callback(
            None,  # type: ignore[arg-type]
            SessionKey(chat_id=1),
            "cli:u:claude",
        )

    upgrade_mock.assert_not_awaited()
    assert "already" in resp.text.lower()


async def test_upgrade_callback_reports_manual_provider() -> None:
    current = _status(
        "grok",
        version="1.0.0",
        latest=None,
        upgrade_supported=False,
    )
    with (
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.get_cli_status",
            AsyncMock(return_value=current),
        ),
        patch(
            "ductor_bot.orchestrator.selectors.cli_selector.get_all_cli_statuses",
            AsyncMock(return_value=[current]),
        ),
    ):
        resp = await handle_cli_callback(
            None,  # type: ignore[arg-type]
            SessionKey(chat_id=1),
            "cli:u:grok",
        )

    assert "not supported" in resp.text.lower()
