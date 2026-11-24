"""Unit tests for Pi discovery, event parsing, command building, and auth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ductor_bot.cli.auth import _PI_ENV_KEYS, AuthStatus, check_pi_auth
from ductor_bot.cli.base import CLIConfig
from ductor_bot.cli.factory import create_cli
from ductor_bot.cli.param_resolver import TaskExecutionConfig
from ductor_bot.cli.pi_discovery import _parse_models
from ductor_bot.cli.pi_events import (
    PiResultState,
    parse_pi_json,
    parse_pi_stream_line,
    pi_events_from_data,
)
from ductor_bot.cli.pi_provider import PiCLI
from ductor_bot.cli.pi_utils import pi_effort, split_pi_model_id
from ductor_bot.cli.stream_events import (
    AssistantTextDelta,
    SystemInitEvent,
    SystemStatusEvent,
    ThinkingEvent,
    ToolUseEvent,
)
from ductor_bot.config import ModelRegistry, reset_pi_models, set_pi_models
from ductor_bot.cron.execution import _build_pi_cmd, parse_pi_result


def _assistant_message(text: str = "OK") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "deepseek-v4.1-flash",
        "usage": {
            "input": 10,
            "output": 2,
            "totalTokens": 12,
            "cost": {"total": 0.01},
        },
        "stopReason": "stop",
    }


def _jsonl(*records: dict[str, Any]) -> str:
    return "\n".join(json.dumps(record) for record in records)


class TestPiDiscovery:
    def test_parse_models_prefixes_provider(self) -> None:
        output = """\
provider     model                       context  max-out  thinking  images
opencode-go  deepseek-v4.1-flash         1M       384K     yes       yes
openrouter   google/gemini-flash-image   1M       32K      no        yes
"""
        assert _parse_models(output) == (
            "pi/opencode-go/deepseek-v4.1-flash",
            "pi/openrouter/google/gemini-flash-image",
        )

    def test_parse_models_skips_blank_and_help_output(self) -> None:
        assert _parse_models("") == ()
        assert _parse_models("Usage:\n  pi --list-models") == ()


class TestPiEvents:
    def test_parse_stream_line_session(self) -> None:
        events = parse_pi_stream_line('{"type":"session","id":"sess-1"}')
        assert len(events) == 1
        assert isinstance(events[0], SystemInitEvent)
        assert events[0].session_id == "sess-1"

    def test_parse_stream_line_text_and_thinking(self) -> None:
        text = parse_pi_stream_line(
            '{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"hi"}}'
        )
        thinking = parse_pi_stream_line(
            '{"type":"message_update","assistantMessageEvent":'
            '{"type":"thinking_delta","delta":"plan"}}'
        )
        assert isinstance(text[0], AssistantTextDelta)
        assert text[0].text == "hi"
        assert isinstance(thinking[0], ThinkingEvent)
        assert thinking[0].text == "plan"

    def test_parse_stream_line_tool_and_compaction(self) -> None:
        tool = parse_pi_stream_line(
            '{"type":"tool_execution_start","toolCallId":"t1",'
            '"toolName":"bash","args":{"command":"ls"}}'
        )
        compact = parse_pi_stream_line('{"type":"compaction_start","reason":"threshold"}')
        assert isinstance(tool[0], ToolUseEvent)
        assert tool[0].tool_name == "bash"
        assert tool[0].parameters == {"command": "ls"}
        assert isinstance(compact[0], SystemStatusEvent)
        assert compact[0].status == "compacting"

    def test_pi_events_from_data_ignores_unknown(self) -> None:
        assert pi_events_from_data({"type": "agent_start"}) == []

    def test_parse_json_collects_final_message_and_usage(self) -> None:
        raw = _jsonl(
            {"type": "session", "id": "sess-1"},
            {
                "type": "message_update",
                "usage": {"input": 10, "output": 2, "totalTokens": 12},
                "assistantMessageEvent": {"type": "text_delta", "delta": "O"},
            },
            {"type": "message_end", "message": _assistant_message("OK")},
            {"type": "turn_end", "message": _assistant_message("OK"), "toolResults": []},
            {"type": "agent_settled", "aborted": False},
        )
        text, session_id, usage, turns, is_error, cost = parse_pi_json(raw)
        assert text == "OK"
        assert session_id == "sess-1"
        assert usage["total_tokens"] == 12
        assert turns == 1
        assert is_error is False
        assert cost == 0.01

    def test_parse_json_error(self) -> None:
        raw = _jsonl(
            {"type": "session", "id": "sess-1"},
            {"type": "error", "message": "auth failed"},
        )
        text, _sid, _usage, _turns, is_error, _cost = parse_pi_json(raw)
        assert text == ""
        assert is_error is True

    def test_result_state_aborted(self) -> None:
        state = PiResultState()
        state.track({"type": "agent_settled", "aborted": True})
        assert state.error == "Pi run aborted."


class TestPiProvider:
    def test_find_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ductor_bot.cli.pi_provider.which", lambda _: None)
        with pytest.raises(FileNotFoundError, match="pi CLI not found"):
            PiCLI(CLIConfig(provider="pi"))

    def test_build_command_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ductor_bot.cli.pi_provider.which", lambda _: "/usr/bin/pi")
        cli = PiCLI(
            CLIConfig(
                provider="pi",
                model="pi/opencode-go/deepseek-v4.1-flash",
                permission_mode="bypassPermissions",
                reasoning_effort="none",
                system_prompt="SYS",
                append_system_prompt="RULES",
                allowed_tools=["read", "grep"],
                disallowed_tools=["bash"],
                cli_parameters=["--no-mcp"],
            )
        )
        cmd = cli._build_command()
        assert cmd[0] == "/usr/bin/pi"
        assert "--print" in cmd
        assert cmd[cmd.index("--mode") + 1] == "json"
        assert "--approve" in cmd
        assert cmd[cmd.index("--provider") + 1] == "opencode-go"
        assert cmd[cmd.index("--model") + 1] == "deepseek-v4.1-flash"
        assert cmd[cmd.index("--thinking") + 1] == "off"
        assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
        assert cmd[cmd.index("--append-system-prompt") + 1] == "RULES"
        assert cmd[cmd.index("--tools") + 1] == "read,grep"
        assert cmd[cmd.index("--exclude-tools") + 1] == "bash"
        assert "--no-mcp" in cmd

    def test_build_command_resume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ductor_bot.cli.pi_provider.which", lambda _: "/usr/bin/pi")
        cli = PiCLI(CLIConfig(provider="pi", model="opencode-go/deepseek-v4.1-flash"))
        cmd = cli._build_command(resume_session="sess-1")
        assert cmd[cmd.index("--session-id") + 1] == "sess-1"
        assert "--continue" not in cmd

        cmd_continue = cli._build_command(continue_session=True)
        assert "--continue" in cmd_continue

    def test_parse_output_jsonl(self) -> None:
        raw = _jsonl(
            {"type": "session", "id": "sess-1"},
            {"type": "message_end", "message": _assistant_message("OK")},
            {"type": "turn_end", "message": _assistant_message("OK"), "toolResults": []},
        ).encode()
        response = PiCLI._parse_output(raw, b"", 0)
        assert response.result == "OK"
        assert response.session_id == "sess-1"
        assert response.is_error is False
        assert response.total_tokens == 12
        assert response.total_cost_usd == 0.01
        assert response.num_turns == 1

    def test_parse_output_plain_text_fallback(self) -> None:
        response = PiCLI._parse_output(b"plain output", b"", 0)
        assert response.result == "plain output"
        assert response.is_error is False

    def test_parse_output_empty(self) -> None:
        response = PiCLI._parse_output(b"", b"boom", 1)
        assert response.result == "boom"
        assert response.is_error is True


class TestPiHelpersAndRouting:
    def test_split_model_id(self) -> None:
        assert split_pi_model_id("pi/opencode-go/deepseek-v4.1-flash") == (
            "opencode-go",
            "deepseek-v4.1-flash",
        )
        assert split_pi_model_id("openrouter/google/gemini-flash") == (
            "openrouter",
            "google/gemini-flash",
        )
        assert split_pi_model_id("plain-model") is None

    def test_effort_mapping(self) -> None:
        assert pi_effort("none") == "off"
        assert pi_effort("high") == "high"
        assert pi_effort("default") == ""
        assert pi_effort(None) == ""

    def test_factory_returns_pi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ductor_bot.cli.pi_provider.which", lambda _: "/usr/bin/pi")
        cli = create_cli(CLIConfig(provider="pi", model="opencode-go/deepseek-v4.1-flash"))
        assert isinstance(cli, PiCLI)

    def test_model_registry_routes_pi(self) -> None:
        reset_pi_models()
        try:
            set_pi_models(("pi/opencode-go/deepseek-v4.1-flash",))
            assert ModelRegistry.provider_for("pi/opencode-go/deepseek-v4.1-flash") == "pi"
            assert ModelRegistry.provider_for("opencode-go/deepseek-v4.1-flash") == "pi"
            assert ModelRegistry.provider_for("sonnet") == "claude"
        finally:
            reset_pi_models()


class TestPiCron:
    def _cfg(self, **kwargs: Any) -> TaskExecutionConfig:
        base: dict[str, Any] = {
            "provider": "pi",
            "model": "pi/opencode-go/deepseek-v4.1-flash",
            "reasoning_effort": "medium",
            "permission_mode": "bypassPermissions",
            "cli_parameters": [],
            "working_dir": "/tmp",
            "file_access": "all",
        }
        base.update(kwargs)
        return TaskExecutionConfig(**base)

    def test_build_cmd_uses_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ductor_bot.cron.execution.which", lambda _: "/usr/bin/pi")
        one = _build_pi_cmd(self._cfg(), "hello")
        assert one is not None
        assert one.stdin_input == b"hello"
        assert "--provider" in one.cmd
        assert "--model" in one.cmd
        assert "--thinking" in one.cmd

    def test_parse_result(self) -> None:
        raw = _jsonl(
            {"type": "session", "id": "s1"},
            {"type": "message_end", "message": _assistant_message("hello")},
        ).encode()
        assert parse_pi_result(raw) == "hello"


class TestPiAuth:
    def test_missing_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ductor_bot.cli.auth.shutil.which", lambda _: None)
        assert check_pi_auth().status is AuthStatus.NOT_FOUND

    def test_auth_json_authenticates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "auth.json").write_text('{"opencode-go": {"type": "api_key"}}')
        monkeypatch.setattr("ductor_bot.cli.auth._pi_agent_dir", lambda: agent_dir)
        monkeypatch.setattr("ductor_bot.cli.auth.shutil.which", lambda _: "/usr/bin/pi")
        for key in _PI_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        result = check_pi_auth()
        assert result.status is AuthStatus.AUTHENTICATED
        assert result.auth_file == agent_dir / "auth.json"

    def test_installed_without_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        monkeypatch.setattr("ductor_bot.cli.auth._pi_agent_dir", lambda: agent_dir)
        monkeypatch.setattr("ductor_bot.cli.auth.shutil.which", lambda _: "/usr/bin/pi")
        for key in _PI_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        assert check_pi_auth().status is AuthStatus.INSTALLED
