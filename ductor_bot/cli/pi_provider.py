"""Async wrapper around the Pi coding-agent CLI."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

from ductor_bot.cli.base import (
    BaseCLI,
    CLIConfig,
    add_cli_opt,
    docker_wrap,
    format_cli_cmd,
)
from ductor_bot.cli.executor import (
    SubprocessResult,
    SubprocessSpec,
    run_oneshot_subprocess,
    run_streaming_subprocess,
)
from ductor_bot.cli.pi_events import (
    PiResultState,
    load_pi_event,
    pi_events_from_data,
    pi_result_event,
)
from ductor_bot.cli.pi_utils import pi_effort, split_pi_model_id
from ductor_bot.cli.stream_events import StreamEvent
from ductor_bot.cli.types import CLIResponse

if TYPE_CHECKING:
    from ductor_bot.cli.timeout_controller import TimeoutController

logger = logging.getLogger(__name__)

_PI_MISSING = (
    "pi CLI not found on PATH. "
    "Install via: npm install -g --ignore-scripts @earendil-works/pi-coding-agent"
)


class PiCLI(BaseCLI):
    """Async wrapper around the Pi coding-agent CLI."""

    def __init__(self, config: CLIConfig) -> None:
        self._config = config
        self._working_dir = Path(config.working_dir).resolve()
        self._cli = "pi" if config.docker_container else self._find_cli()
        logger.info("Pi CLI wrapper: cwd=%s, model=%s", self._working_dir, config.model)

    @staticmethod
    def _find_cli() -> str:
        path = which("pi")
        if not path:
            raise FileNotFoundError(_PI_MISSING)
        return path

    def _build_command(
        self,
        *,
        resume_session: str | None = None,
        continue_session: bool = False,
    ) -> list[str]:
        """Build the non-interactive JSON-mode command.

        The prompt itself is fed through stdin so long prompts never hit the
        per-argument limit.
        """
        cfg = self._config
        cmd = [self._cli, "--print", "--mode", "json"]

        if cfg.permission_mode == "bypassPermissions":
            cmd.append("--approve")

        if cfg.model:
            parts = split_pi_model_id(cfg.model)
            if parts is None:
                add_cli_opt(cmd, "--model", cfg.model)
            else:
                cmd += ["--provider", parts[0], "--model", parts[1]]

        effort = pi_effort(cfg.reasoning_effort)
        if effort:
            cmd += ["--thinking", effort]

        add_cli_opt(cmd, "--system-prompt", cfg.system_prompt)
        add_cli_opt(cmd, "--append-system-prompt", cfg.append_system_prompt)
        if cfg.allowed_tools:
            cmd += ["--tools", ",".join(cfg.allowed_tools)]
        if cfg.disallowed_tools:
            cmd += ["--exclude-tools", ",".join(cfg.disallowed_tools)]

        if resume_session:
            cmd += ["--session-id", resume_session]
        elif continue_session:
            cmd.append("--continue")

        cmd.extend(cfg.cli_parameters)
        return cmd

    def _docker_wrap(self, cmd: list[str]) -> tuple[list[str], str | None]:
        """Keep stdin open for Dockerized Pi so the piped prompt reaches the CLI."""
        return docker_wrap(cmd, self._config, interactive=True)

    async def send(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
        timeout_controller: TimeoutController | None = None,
    ) -> CLIResponse:
        """Send a prompt and return the final result."""
        cmd = self._build_command(
            resume_session=resume_session,
            continue_session=continue_session,
        )
        exec_cmd, use_cwd = self._docker_wrap(cmd)
        _log_cmd(exec_cmd)
        return await run_oneshot_subprocess(
            config=self._config,
            spec=SubprocessSpec(
                exec_cmd,
                use_cwd,
                prompt,
                timeout_seconds,
                timeout_controller,
                stdin_text=prompt,
            ),
            parse_output=self._parse_output,
            provider_label="Pi",
        )

    async def send_streaming(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
        timeout_controller: TimeoutController | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send a prompt and yield stream events as they arrive."""
        cmd = self._build_command(
            resume_session=resume_session,
            continue_session=continue_session,
        )
        exec_cmd, use_cwd = self._docker_wrap(cmd)
        _log_cmd(exec_cmd, streaming=True)

        state = PiResultState()

        async def line_handler(line: str) -> AsyncGenerator[StreamEvent, None]:
            data = load_pi_event(line)
            if data is None:
                return
            state.track(data)
            for event in pi_events_from_data(data):
                yield event

        async def post_handler(result: SubprocessResult) -> AsyncGenerator[StreamEvent, None]:
            stderr = result.stderr_bytes.decode(errors="replace") if result.stderr_bytes else ""
            yield pi_result_event(
                state,
                returncode=result.process.returncode,
                stderr=stderr,
            )

        async for event in run_streaming_subprocess(
            config=self._config,
            spec=SubprocessSpec(
                exec_cmd,
                use_cwd,
                prompt,
                timeout_seconds,
                timeout_controller,
                stdin_text=prompt,
            ),
            line_handler=line_handler,
            provider_label="Pi",
            post_handler=post_handler,
        ):
            yield event

    @staticmethod
    def _parse_output(
        stdout: bytes,
        stderr: bytes,
        returncode: int | None,
    ) -> CLIResponse:
        """Parse Pi JSONL output into a CLIResponse."""
        stderr_text = stderr.decode(errors="replace")[:2000] if stderr else ""
        if stderr_text:
            logger.warning("Pi stderr (exit=%s): %s", returncode, stderr_text[:500])

        raw = stdout.decode(errors="replace").strip()
        if not raw:
            logger.error("Pi returned empty output (exit=%s)", returncode)
            return CLIResponse(
                result=stderr_text.strip(),
                is_error=True,
                returncode=returncode,
                stderr=stderr_text,
            )

        state = PiResultState()
        saw_record = False
        for line in raw.splitlines():
            data = load_pi_event(line)
            if data is None:
                continue
            saw_record = True
            state.track(data)

        if not saw_record:
            return CLIResponse(
                result=raw,
                is_error=returncode not in (0, None),
                returncode=returncode,
                stderr=stderr_text,
            )

        result = state.result_text
        is_error = bool(state.error) or (returncode not in (0, None) and not result)
        if not result and is_error:
            result = state.error or stderr_text.strip()[:500] or "Pi produced no response."

        return CLIResponse(
            session_id=state.session_id,
            result=result,
            is_error=is_error,
            returncode=returncode,
            stderr=stderr_text,
            total_cost_usd=state.cost_usd or None,
            usage=state.usage,
            num_turns=state.turns,
        )


def _log_cmd(cmd: list[str], *, streaming: bool = False) -> None:
    """Log the Pi CLI command with redacted, truncated long values."""
    kind = "stream cmd" if streaming else "cmd"
    logger.info("Pi %s: %s", kind, format_cli_cmd(cmd))
