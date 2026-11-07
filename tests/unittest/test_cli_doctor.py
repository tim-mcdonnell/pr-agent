"""Preflight checks for `pr-agent doctor` (#4320)."""


import io
import sys

import pytest

import pr_agent.cli as cli
import pr_agent.doctor as doctor
from pr_agent.algo.ai_handlers.claude_agent_sdk_handler import ClaudeAgentSDKHandler
from pr_agent.config_loader import get_settings
from tests.unittest._settings_helpers import restore_settings, snapshot_settings

FAKE_CLI_PATH = "/opt/claude-agent-sdk/_bundled/claude"


@pytest.fixture
def cli_mode_settings(monkeypatch):
    snapshot = snapshot_settings(("config.model", "config.cli_mode", "push_outputs"))
    get_settings().set("config.model", "opus")
    get_settings().set("push_outputs", {"enable": False})
    for variable in doctor.CLAUDE_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-oauth-token")
    monkeypatch.setattr(doctor, "claude_code_cli_path", lambda: FAKE_CLI_PATH)
    try:
        yield get_settings()
    finally:
        restore_settings(snapshot)


class _Ping:
    """Record ClaudeAgentSDKHandler.chat_completion calls and optionally fail them."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []


def _install_ping(monkeypatch, error=None):
    ping = _Ping(error)

    async def chat_completion(self, model, system, user, temperature=0.2, img_path=None):
        assert isinstance(self, ClaudeAgentSDKHandler)
        ping.calls.append((model, system, user))
        if ping.error:
            raise ping.error
        return "ok", "stop"

    monkeypatch.setattr(ClaudeAgentSDKHandler, "chat_completion", chat_completion)
    return ping


def test_doctor_passes_and_exits_zero(monkeypatch, cli_mode_settings, capsys):
    ping = _install_ping(monkeypatch)

    assert cli.run(inargs=["doctor"]) == 0
    out = capsys.readouterr().out
    assert out == ("✓ config.model = opus\n"
                   f"✓ Claude Code binary found ({FAKE_CLI_PATH})\n"
                   "✓ credential found (env CLAUDE_CODE_OAUTH_TOKEN)\n"
                   "✓ model completion succeeded\n"
                   "4 of 4 checks passed\n")
    assert ping.calls == [("opus", doctor._PING_SYSTEM, doctor._PING_USER)]


@pytest.mark.parametrize("model", ["sonnet", "sonnet[1m]", "anthropic/haiku", "claude-opus-5-5", "best"])
def test_doctor_accepts_claude_code_model_slugs(monkeypatch, cli_mode_settings, capsys, model):
    ping = _install_ping(monkeypatch)
    cli_mode_settings.set("config.model", model)

    assert cli.run(inargs=["doctor"]) == 0
    assert capsys.readouterr().out.startswith(f"✓ config.model = {model}\n")
    assert [call[0] for call in ping.calls] == [model]


@pytest.mark.parametrize("model", ["gpt-4o", "openai/gpt-4o", "bedrock/us.anthropic.claude-sonnet-4-5"])
def test_non_claude_model_fails_without_a_ping(monkeypatch, cli_mode_settings, capsys, model):
    ping = _install_ping(monkeypatch)
    cli_mode_settings.set("config.model", model)

    assert cli.run(inargs=["doctor"]) == 1
    out = capsys.readouterr().out
    assert f"✗ config.model = {model} is not a Claude Code model slug" in out
    assert out.endswith("2 of 3 checks passed\n")
    assert ping.calls == []


def test_empty_model_fails_without_a_ping(monkeypatch, cli_mode_settings, capsys):
    ping = _install_ping(monkeypatch)
    cli_mode_settings.set("config.model", "")

    assert cli.run(inargs=["doctor"]) == 1
    assert "✗ config.model is empty; set it in configuration.toml or CONFIG__MODEL" in capsys.readouterr().out
    assert ping.calls == []


def test_missing_claude_code_binary_fails_without_a_ping(monkeypatch, cli_mode_settings, capsys):
    ping = _install_ping(monkeypatch)
    monkeypatch.setattr(doctor, "claude_code_cli_path", lambda: None)

    assert cli.run(inargs=["doctor"]) == 1
    out = capsys.readouterr().out
    assert "✗ Claude Code binary not found; reinstall claude-agent-sdk for this platform" in out
    assert out.endswith("2 of 3 checks passed\n")
    assert ping.calls == []


def test_missing_credential_defers_to_the_ping(monkeypatch, cli_mode_settings, capsys):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    ping = _install_ping(monkeypatch)

    assert cli.run(inargs=["doctor"]) == 0
    out = capsys.readouterr().out
    assert "✓ no credential in the environment; relying on an existing Claude Code login" in out
    assert ping.calls


def test_missing_credential_still_fails_when_the_ping_fails(monkeypatch, cli_mode_settings, capsys):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    _install_ping(monkeypatch, error=RuntimeError("Not logged in · Please run /login"))

    assert cli.run(inargs=["doctor"]) == 1
    out = capsys.readouterr().out
    assert "✗ model completion failed: Not logged in · Please run /login" in out
    assert out.endswith("3 of 4 checks passed\n")


@pytest.mark.parametrize("present, reported", [
    (("ANTHROPIC_API_KEY",), "ANTHROPIC_API_KEY"),
    (("ANTHROPIC_AUTH_TOKEN",), "ANTHROPIC_AUTH_TOKEN"),
    (("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"), "CLAUDE_CODE_OAUTH_TOKEN"),
])
def test_credential_check_names_the_first_variable_in_documented_order(
    monkeypatch, cli_mode_settings, capsys, present, reported
):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    for variable in present:
        monkeypatch.setenv(variable, "synthetic-credential")
    _install_ping(monkeypatch)

    assert cli.run(inargs=["doctor"]) == 0
    assert f"✓ credential found (env {reported})\n" in capsys.readouterr().out


def test_missing_provider_key_env_vars_shared_with_the_runner(monkeypatch, cli_mode_settings):
    assert doctor.missing_provider_key_env_vars() is None

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    assert doctor.missing_provider_key_env_vars() == ("claude", doctor.CLAUDE_CREDENTIAL_ENV_VARS)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-api-key")
    assert doctor.missing_provider_key_env_vars() is None


def test_empty_credential_variable_counts_as_missing(monkeypatch, cli_mode_settings):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "")

    assert doctor.missing_provider_key_env_vars() == ("claude", doctor.CLAUDE_CREDENTIAL_ENV_VARS)


@pytest.mark.parametrize("channels, expected_exit, summary", [
    (["stdout"], 0, "5 of 5 checks passed"),
    (["webhook"], 1, "4 of 5 checks passed"),
])
def test_doctor_checks_enabled_outputs_after_ping(monkeypatch, cli_mode_settings, capsys,
                                                channels, expected_exit, summary):
    cli_mode_settings.set("push_outputs", {"enable": True, "channels": channels})
    ping = _install_ping(monkeypatch)
    check = doctor._check_output_sinks

    def after_ping(cfg):
        assert [call[0] for call in ping.calls] == ["opus"]
        return check(cfg)

    monkeypatch.setattr(doctor, "_check_output_sinks", after_ping)
    assert cli.run(inargs=["doctor"]) == expected_exit
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6
    assert lines[3] == "✓ model completion succeeded"
    assert "push_outputs" in lines[4]
    assert lines[-1] == summary


def test_doctor_contains_settings_operation_errors_after_ping(monkeypatch, cli_mode_settings, capsys):
    ping = _install_ping(monkeypatch)
    config = cli_mode_settings.config

    class BrokenSettings:
        def __init__(self):
            self.config = config

        def get(self, key, default=None):
            assert ping.calls
            raise ValueError("synthetic-token https://secret.example/hook /private/path")

    monkeypatch.setattr(doctor, "get_settings", lambda: BrokenSettings())
    assert cli.run(inargs=["doctor"]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[-2:] == ["✗ push_outputs configuration could not be checked", "4 of 5 checks passed"]
    assert all(value not in out for value in ("synthetic-token", "secret.example", "/private/path", "ValueError"))


def test_doctor_report_survives_stdout_that_cannot_encode(monkeypatch, cli_mode_settings):
    # A redirected Windows stdout uses cp1252, which has no check mark characters.
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    monkeypatch.setattr(doctor, "claude_code_cli_path", lambda: None)
    _install_ping(monkeypatch)

    assert cli.run(inargs=["doctor"]) == 1
    sys.stdout.flush()
    out = raw.getvalue().decode("utf-8")
    assert "✓ config.model = opus\n" in out
    assert "✗ Claude Code binary not found" in out
    assert out.endswith("2 of 3 checks passed\n")


def test_bad_ping_reports_the_first_line_of_the_reason(monkeypatch, cli_mode_settings, capsys):
    _install_ping(monkeypatch, error=RuntimeError("Connection error to https://api.example\ntraceback details"))

    assert cli.run(inargs=["doctor"]) == 1
    out = capsys.readouterr().out
    assert "✗ model completion failed: Connection error to https://api.example\n" in out
    assert "traceback details" not in out
    assert "3 of 4 checks passed" in out


def test_ping_failure_reason_is_truncated(monkeypatch, cli_mode_settings, capsys):
    _install_ping(monkeypatch, error=RuntimeError("x" * 500))

    assert cli.run(inargs=["doctor"]) == 1
    failed = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("✗"))
    assert failed == "✗ model completion failed: " + "x" * 200


def test_ping_failure_redacts_credentials_quoted_by_the_provider(monkeypatch, cli_mode_settings, capsys):
    secret = "sk-ant-api03-synthetic-key-value-for-tests-0123456789abcdef"
    _install_ping(monkeypatch, error=RuntimeError(f"401 invalid x-api-key {secret} at https://api.example"))

    assert cli.run(inargs=["doctor"]) == 1
    out = capsys.readouterr().out
    assert secret not in out
    assert "<redacted>" in out
    assert "401 invalid x-api-key" in out


def test_doctor_needs_no_pr_url_and_skips_provider_setup(monkeypatch, cli_mode_settings):
    constructed = []

    def fail_agent(*_args, **_kwargs):
        constructed.append("PRAgent")
        raise AssertionError("doctor must not dispatch to PRAgent")

    monkeypatch.setattr(cli, "PRAgent", fail_agent)
    _install_ping(monkeypatch)

    assert cli.run(inargs=["doctor"]) == 0
    assert constructed == []


def test_credential_value_never_appears_in_output(monkeypatch, cli_mode_settings, capsys):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-secret-value")
    _install_ping(monkeypatch)

    cli.run(inargs=["doctor"])
    assert "synthetic-secret-value" not in capsys.readouterr().out
