"""Interactive version/upgrade picker for underlying provider CLIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ductor_bot.cli.cli_versions import (
    CLIStatus,
    cli_spec,
    format_status_line,
    get_all_cli_statuses,
    get_cli_status,
    upgrade_cli,
    upgrade_result_summary,
)
from ductor_bot.i18n import t
from ductor_bot.orchestrator.selectors.models import Button, ButtonGrid, SelectorResponse

if TYPE_CHECKING:
    from ductor_bot.orchestrator.core import Orchestrator
    from ductor_bot.session.key import SessionKey

logger = logging.getLogger(__name__)

CLI_PREFIX = "cli:"


def is_cli_selector_callback(data: str) -> bool:
    """Return True when *data* belongs to the provider-CLI version picker."""
    return data.startswith(CLI_PREFIX)


def _button_label(status: CLIStatus) -> str:
    if not status.installed:
        if status.upgrade_supported:
            return t("cli.btn_install", label=status.label)
        return t("cli.btn_not_installed", label=status.label)
    if status.latest and status.update_available:
        return t(
            "cli.btn_update",
            label=status.label,
            current=status.version_display,
            latest=status.latest_display,
        )
    return t("cli.btn_current", label=status.label, version=status.version_display)


def _status_line(status: CLIStatus) -> str:
    if not status.installed:
        key = "cli.line_installable" if status.upgrade_supported else "cli.line_not_installed"
        return t(key, label=status.label)
    if status.latest and status.update_available:
        return t(
            "cli.line_update",
            label=status.label,
            current=status.version_display,
            latest=status.latest_display,
        )
    if status.latest:
        return t(
            "cli.line_current",
            label=status.label,
            version=status.version_display,
        )
    return t("cli.line_unknown_latest", label=status.label, version=status.version_display)


async def cli_selector_start(
    orch: Orchestrator,
    key: SessionKey,
    *,
    note: str = "",
) -> SelectorResponse:
    """Build the CLI versions overview keyboard."""
    del orch, key
    statuses = await get_all_cli_statuses(include_latest=True)

    lines = [t("cli.header"), ""]
    if note:
        lines += [note, ""]
    lines += [_status_line(status) for status in statuses]
    lines += ["", t("cli.tap_hint")]

    rows = [
        [Button(text=_button_label(status), callback_data=f"cli:u:{status.provider}")]
        for status in statuses
    ]
    rows.append([Button(text=t("cli.btn_refresh"), callback_data="cli:r")])
    return SelectorResponse(text="\n".join(lines), buttons=ButtonGrid(rows=rows))


async def _upgrade_one(provider: str) -> str:
    status = await get_cli_status(provider, include_latest=True)
    if status is None:
        return t("cli.unknown_provider", provider=provider)
    if not status.upgrade_supported:
        spec = cli_spec(provider)
        hint = spec.manual_hint if spec and spec.manual_hint else ""
        return t("cli.manual", label=status.label, hint=hint).strip()

    installing = not status.installed
    if not installing and status.latest and not status.update_available:
        return t("cli.no_update", label=status.label, version=status.version_display)

    result = await upgrade_cli(provider)
    if not result.ok:
        return upgrade_result_summary(result, action="install" if installing else "upgrade")

    refreshed = await get_cli_status(provider, include_latest=True)
    if refreshed and refreshed.version:
        key = "cli.installed" if installing else "cli.upgraded"
        return t(
            key,
            label=refreshed.label,
            version=refreshed.version_display,
        )
    return upgrade_result_summary(result, action="install" if installing else "upgrade")


async def handle_cli_callback(
    orch: Orchestrator,
    key: SessionKey,
    data: str,
) -> SelectorResponse:
    """Route ``cli:*`` callbacks to refresh or upgrade a provider CLI."""
    logger.debug("CLI selector step=%s", data[:40])
    parts = data[len(CLI_PREFIX) :].split(":", 2)
    action = parts[0] if parts else ""
    payload = parts[1] if len(parts) > 1 else ""

    if action == "r":
        return await cli_selector_start(orch, key)

    if action == "u":
        note = await _upgrade_one(payload)
        return await cli_selector_start(orch, key, note=note)

    logger.warning("Unknown CLI selector callback: %s", data)
    return SelectorResponse(text=t("cli.unknown_action"))


__all__ = [
    "CLI_PREFIX",
    "cli_selector_start",
    "format_status_line",
    "handle_cli_callback",
    "is_cli_selector_callback",
]
