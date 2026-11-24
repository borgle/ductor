"""Version discovery and upgrades for underlying provider CLIs."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiohttp

from ductor_bot.infra.platform import CREATION_FLAGS as _CREATION_FLAGS

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\b")
_NPM_TIMEOUT = aiohttp.ClientTimeout(total=10)
_PROBE_TIMEOUT_S = 15.0
_UPGRADE_TIMEOUT_S = 240.0


@dataclass(frozen=True, slots=True)
class CLISpec:
    """Static description of one provider CLI."""

    provider: str
    label: str
    binary: str
    npm_package: str | None = None
    upgrade_kind: str = "npm"  # "npm", "self", or "manual"
    manual_hint: str = ""


@dataclass(frozen=True, slots=True)
class CLIStatus:
    """Installed/latest version state for one provider CLI."""

    provider: str
    label: str
    installed: bool
    version: str | None
    latest: str | None
    update_available: bool
    upgrade_supported: bool
    error: str = ""

    @property
    def version_display(self) -> str:
        return self.version or "?"

    @property
    def latest_display(self) -> str:
        return self.latest or "?"


@dataclass(frozen=True, slots=True)
class CLIUpgradeResult:
    """Outcome of one provider CLI upgrade attempt."""

    provider: str
    ok: bool
    output: str


_SPECS: dict[str, CLISpec] = {
    "claude": CLISpec(
        provider="claude",
        label="Claude Code",
        binary="claude",
        npm_package="@anthropic-ai/claude-code",
    ),
    "codex": CLISpec(
        provider="codex",
        label="Codex",
        binary="codex",
        npm_package="@openai/codex",
    ),
    "gemini": CLISpec(
        provider="gemini",
        label="Gemini",
        binary="gemini",
        npm_package="@google/gemini-cli",
    ),
    "pi": CLISpec(
        provider="pi",
        label="Pi",
        binary="pi",
        npm_package="@earendil-works/pi-coding-agent",
        upgrade_kind="self",
    ),
    "grok": CLISpec(
        provider="grok",
        label="Grok Build",
        binary="grok",
        upgrade_kind="manual",
        manual_hint="Reinstall with the xAI Grok CLI installer.",
    ),
    "antigravity": CLISpec(
        provider="antigravity",
        label="Antigravity",
        binary="agy",
        upgrade_kind="manual",
        manual_hint="Update Antigravity with its own installer.",
    ),
}

PROVIDER_ORDER: tuple[str, ...] = (
    "claude",
    "codex",
    "gemini",
    "antigravity",
    "grok",
    "pi",
)


def parse_version(text: str) -> str | None:
    """Extract the first semantic version from CLI output."""
    match = _VERSION_RE.search(text or "")
    return match.group(1) if match else None


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    core = value.split("-", 1)[0].split("+", 1)[0]
    parts: list[int] = []
    for segment in core.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            break
    return tuple(parts)


def _is_newer(candidate: str | None, current: str | None) -> bool:
    return bool(candidate and current and _version_tuple(candidate) > _version_tuple(current))


async def _probe_version(spec: CLISpec) -> tuple[bool, str | None, str]:
    """Return ``(installed, version, error)`` for one CLI."""
    binary = shutil.which(spec.binary)
    if not binary:
        return False, None, ""

    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=_CREATION_FLAGS,
        )
    except (OSError, ValueError) as exc:
        logger.debug("Version probe failed for %s: %s", spec.provider, exc)
        return True, None, str(exc)

    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_S):
            stdout, _ = await proc.communicate()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return True, None, "version probe timed out"

    output = stdout.decode(errors="replace").strip() if stdout else ""
    return True, parse_version(output), "" if output else "no version output"


async def fetch_npm_latest(package: str) -> str | None:
    """Fetch the latest published version of an npm package."""
    url = f"https://registry.npmjs.org/{package}/latest"
    try:
        async with (
            aiohttp.ClientSession(timeout=_NPM_TIMEOUT) as session,
            session.get(url) as resp,
        ):
            if resp.status != 200:
                return None
            data = await resp.json()
    except (aiohttp.ClientError, TimeoutError, ValueError):
        logger.debug("npm version check failed for %s", package, exc_info=True)
        return None

    version = data.get("version") if isinstance(data, dict) else None
    return version if isinstance(version, str) and version else None


async def get_cli_status(provider: str, *, include_latest: bool = True) -> CLIStatus | None:
    """Return version state for one provider CLI."""
    spec = _SPECS.get(provider)
    if spec is None:
        return None

    installed, version, error = await _probe_version(spec)
    latest: str | None = None
    if include_latest and spec.npm_package:
        latest = await fetch_npm_latest(spec.npm_package)

    return CLIStatus(
        provider=spec.provider,
        label=spec.label,
        installed=installed,
        version=version,
        latest=latest,
        update_available=_is_newer(latest, version),
        upgrade_supported=spec.upgrade_kind in {"npm", "self"},
        error=error,
    )


async def get_all_cli_statuses(*, include_latest: bool = True) -> list[CLIStatus]:
    """Return version state for every known provider CLI in display order."""
    results = await asyncio.gather(
        *(get_cli_status(provider, include_latest=include_latest) for provider in PROVIDER_ORDER)
    )
    return [status for status in results if status is not None]


def _build_upgrade_command(spec: CLISpec) -> tuple[list[str] | None, str]:
    """Return ``(argv, error)`` for a provider CLI upgrade."""
    if spec.upgrade_kind == "self":
        binary = shutil.which(spec.binary)
        if not binary:
            return None, f"{spec.binary} is not installed."
        return [binary, "update"], ""

    if spec.upgrade_kind == "npm":
        npm = shutil.which("npm")
        if not npm:
            return None, "npm is not available on PATH."
        if not spec.npm_package:
            return None, f"No npm package configured for {spec.label}."
        return [npm, "install", "-g", f"{spec.npm_package}@latest"], ""

    return None, spec.manual_hint or f"Automatic upgrade is not supported for {spec.label}."


async def upgrade_cli(provider: str) -> CLIUpgradeResult:
    """Upgrade one provider CLI through npm or its self-updater."""
    spec = _SPECS.get(provider)
    if spec is None:
        return CLIUpgradeResult(provider=provider, ok=False, output="Unknown provider CLI.")

    cmd, error = _build_upgrade_command(spec)
    if cmd is None:
        return CLIUpgradeResult(provider=provider, ok=False, output=error)

    logger.info("Upgrading provider CLI %s: %s", provider, " ".join(cmd))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=_CREATION_FLAGS,
        )
    except (OSError, ValueError) as exc:
        return CLIUpgradeResult(provider=provider, ok=False, output=str(exc))

    try:
        async with asyncio.timeout(_UPGRADE_TIMEOUT_S):
            stdout, _ = await proc.communicate()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return CLIUpgradeResult(
            provider=provider,
            ok=False,
            output=f"Upgrade timed out after {int(_UPGRADE_TIMEOUT_S)}s.",
        )

    output = stdout.decode(errors="replace").strip() if stdout else ""
    return CLIUpgradeResult(provider=provider, ok=(proc.returncode or 0) == 0, output=output)


def cli_spec(provider: str) -> CLISpec | None:
    """Return the static spec for *provider*."""
    return _SPECS.get(provider)


def format_status_line(status: CLIStatus) -> str:
    """Render one compact version line for diagnostics."""
    if not status.installed:
        return f"{status.label}: not installed"
    if status.latest and status.update_available:
        return f"{status.label}: {status.version_display} -> {status.latest_display} available"
    return f"{status.label}: {status.version_display}"


def _join_output(output: str, limit: int = 1200) -> str:
    text = output.strip()
    if len(text) <= limit:
        return text
    return f"...{text[-limit:]}"


def upgrade_result_summary(result: CLIUpgradeResult) -> str:
    """Build a short user-facing upgrade result string."""
    if result.ok:
        return f"{result.provider}: upgrade command completed."
    output = _join_output(result.output)
    return f"{result.provider}: upgrade failed." + (f"\n{output}" if output else "")


def statuses_for_diagnose(statuses: Sequence[CLIStatus]) -> str:
    """Render the installed-version block used by ``/diagnose``."""
    lines = [format_status_line(status) for status in statuses]
    return "\n".join(lines)
