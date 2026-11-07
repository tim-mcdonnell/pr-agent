import asyncio
import os
import subprocess
from pathlib import Path

import claude_agent_sdk
import pytest
from claude_agent_sdk import AssistantMessage, ResultError, ResultMessage, TextBlock

from pr_agent.algo import run_details
from pr_agent.algo.ai_handlers import claude_agent_sdk_handler as handler_module
from pr_agent.algo.ai_handlers.claude_agent_sdk_handler import (
    ClaudeAgentSDKError,
    ClaudeAgentSDKHandler,
    current_command,
    exploration_root,
    make_read_only_guard,
    snapshot_checkout,
)
from pr_agent.algo.cli_args import CliArgs
from pr_agent.algo.token_budget import FallbackEligibleError
from pr_agent.config_loader import get_settings
from pr_agent.config_security import is_repo_host_only_key
from tests.unittest._settings_helpers import restore_settings, snapshot_settings

SETTINGS_KEYS = (
    "config.reasoning_effort", "config.ai_timeout", "config.output_run_cost", "config.max_output_tokens",
    "claude_code.explore_repo", "claude_code.explore_commands", "claude_code.repo_path", "claude_code.max_turns",
)


@pytest.fixture(autouse=True)
def restore_handler_settings(monkeypatch):
    snapshot = snapshot_settings(SETTINGS_KEYS)
    # Keep the GitHub Actions workspace of a CI run from enabling exploration in unrelated tests.
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    get_settings().set("claude_code.repo_path", "")
    yield
    restore_settings(snapshot)


def _assistant(text="answer: 4", stop_reason="end_turn", error=None):
    return AssistantMessage(content=[TextBlock(text=text)], model="claude-opus-5-5", error=error,
                            stop_reason=stop_reason)


def _result(text="answer: 4", stop_reason="end_turn", is_error=False, subtype="success", cost=0.25):
    return ResultMessage(
        subtype=subtype, duration_ms=10, duration_api_ms=8, is_error=is_error, num_turns=1, session_id="s",
        stop_reason=stop_reason, total_cost_usd=cost, result=text,
        usage={"input_tokens": 5, "output_tokens": 7, "cache_read_input_tokens": 100,
               "cache_creation_input_tokens": 20},
    )


def _fake_query(monkeypatch, *messages, raises=None, calls=None):
    async def fake(*, prompt, options):
        if calls is not None:
            with open(options.system_prompt["path"], encoding="utf-8") as f:
                calls.append({"prompt": prompt, "options": options, "system": f.read()})
        for message in messages:
            yield message
        if raises is not None:
            raise raises

    monkeypatch.setattr(claude_agent_sdk, "query", fake)


async def test_runs_a_toolless_single_turn_claude_code_session(monkeypatch):
    calls = []
    _fake_query(monkeypatch, _assistant(), _result(), calls=calls)

    resp, finish_reason = await ClaudeAgentSDKHandler().chat_completion(
        model="anthropic/claude-opus-5-5", system="SYSTEM PROMPT", user="USER PROMPT")

    assert (resp, finish_reason) == ("answer: 4", "stop")
    call = calls[0]
    options = call["options"]
    assert call["prompt"] == "USER PROMPT"
    assert call["system"] == "SYSTEM PROMPT"
    assert options.model == "claude-opus-5-5"
    assert options.tools == []
    assert options.setting_sources == []
    assert options.max_turns == 1
    assert options.cwd and options.system_prompt["path"].startswith(str(options.cwd))
    assert options.extra_args == {"strict-mcp-config": None}


async def test_records_usage_including_cache_tokens_and_optional_cost(monkeypatch):
    get_settings().set("config.output_run_cost", True)
    _fake_query(monkeypatch, _assistant(), _result(cost=0.25))
    token = run_details._run_details.set(run_details.RunDetails())
    try:
        await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")
        details = run_details.get_run_details()
        assert details.num_ai_calls == 1
        assert (details.prompt_tokens, details.completion_tokens) == (125, 7)
        assert (details.cache_read_tokens, details.cache_creation_tokens) == (100, 20)
        assert details.known_cost_call_count == 1
    finally:
        run_details._run_details.reset(token)


async def test_max_tokens_stop_reason_maps_to_length(monkeypatch):
    _fake_query(monkeypatch, _assistant(stop_reason="max_tokens"), _result(stop_reason="max_tokens"))

    _, finish_reason = await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")

    assert finish_reason == "length"


@pytest.mark.parametrize("oauth_token", ["oauth-token", ""])
async def test_oauth_token_keeps_inherited_api_keys_from_outranking_it(monkeypatch, oauth_token):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", oauth_token)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-inherited")
    calls = []
    _fake_query(monkeypatch, _assistant(), _result(), calls=calls)

    await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")

    env = calls[0]["options"].env
    assert env["CLAUDE_AGENT_SDK_CLIENT_APP"] == "pr-agent"
    if oauth_token:
        assert env["ANTHROPIC_API_KEY"] == "" and env["ANTHROPIC_AUTH_TOKEN"] == ""
    else:
        assert "ANTHROPIC_API_KEY" not in env


@pytest.mark.parametrize(("effort", "expected"), [
    ("none", {"thinking": {"type": "disabled"}}),
    ("minimal", {"effort": "low"}),
    ("medium", {"effort": "medium"}),
    ("xhigh", {"effort": "xhigh"}),
    ("", {}),
])
async def test_reasoning_effort_maps_to_claude_code_effort(monkeypatch, effort, expected):
    get_settings().set("config.reasoning_effort", effort)
    calls = []
    _fake_query(monkeypatch, _assistant(), _result(), calls=calls)

    await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")

    options = calls[0]["options"]
    assert {k: v for k in ("thinking", "effort") if (v := getattr(options, k)) is not None} == expected


async def test_non_claude_model_is_fallback_eligible_without_launching_claude_code(monkeypatch):
    calls = []
    _fake_query(monkeypatch, _assistant(), _result(), calls=calls)

    with pytest.raises(FallbackEligibleError, match="not a Claude Code model slug"):
        await ClaudeAgentSDKHandler().chat_completion(model="gpt-6.1-sol", system="s", user="u")
    assert calls == []


async def test_authentication_failure_is_not_retried_on_fallback_models(monkeypatch):
    error_result = _result(text="Invalid API key", stop_reason="stop_sequence", is_error=True)
    _fake_query(monkeypatch, _assistant(text="Invalid API key", error="authentication_failed"), error_result,
                raises=ResultError("Claude Code returned an error result", data={"subtype": "success"}))

    with pytest.raises(ClaudeAgentSDKError, match="authentication_failed.*setup-token"):
        await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")


async def test_rate_limit_is_fallback_eligible(monkeypatch):
    _fake_query(monkeypatch, _assistant(text="limit reached", error="rate_limit"),
                _result(text="limit reached", is_error=True),
                raises=ResultError("Claude Code returned an error result", data={"subtype": "success"}))

    with pytest.raises(FallbackEligibleError):
        await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")


async def test_refusal_and_empty_responses_are_fallback_eligible(monkeypatch):
    _fake_query(monkeypatch, _assistant(text="no"), _result(text="no", stop_reason="refusal"))
    with pytest.raises(FallbackEligibleError, match="declined"):
        await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")

    _fake_query(monkeypatch, _result(text=""))
    with pytest.raises(FallbackEligibleError, match="empty response"):
        await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")


async def test_ai_timeout_bounds_the_claude_code_session(monkeypatch):
    get_settings().set("config.ai_timeout", 0.01)

    closed = []

    async def slow(*, prompt, options):
        try:
            await asyncio.sleep(5)
            yield _result()
        finally:
            closed.append(True)

    monkeypatch.setattr(claude_agent_sdk, "query", slow)

    with pytest.raises(asyncio.TimeoutError):
        await ClaudeAgentSDKHandler().chat_completion(model="claude-opus-5-5", system="s", user="u")
    assert closed == [True]  # the stream, and with it the Claude Code process, is closed before returning


@pytest.mark.parametrize("slug", [
    "opus", "sonnet", "haiku", "fable", "best", "default", "opus[1m]", "sonnet[1m]", "claude-opus-5-5",
    "anthropic/claude-sonnet-5-5", "claude-fable-5-1",
])
async def test_claude_code_slugs_reach_claude_code_unchanged(monkeypatch, slug):
    calls = []
    _fake_query(monkeypatch, _assistant(), _result(), calls=calls)

    await ClaudeAgentSDKHandler().chat_completion(model=slug, system="s", user="u")

    assert calls[0]["options"].model == slug.removeprefix("anthropic/")


def test_claude_code_section_is_host_only():
    for key in ("explore_repo", "explore_commands", "repo_path", "max_turns"):
        assert is_repo_host_only_key("claude_code", key)
    assert CliArgs._host_only_setting_arg("--claude_code.repo_path=/") == ".claude_code.repo_path"


def test_max_output_tokens_and_secrets_reach_the_claude_code_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_secret")
    get_settings().set("config.max_output_tokens", 8000)

    env = ClaudeAgentSDKHandler._child_env()

    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8000"
    assert env["GITHUB_TOKEN"] == ""


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("x = 1\n")
    (tmp_path / "outside.txt").write_text("secret\n")
    os.symlink(tmp_path / "outside.txt", repo / "link.txt")  # tracked link leaving the tree
    os.symlink("src/app.py", repo / "alias.py")  # tracked link inside the tree
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    # What actions/checkout and earlier CI steps leave behind: a persisted token and an untracked credential.
    _git(repo, "config", "http.https://github.com/.extraheader", "AUTHORIZATION: basic eC1hY2Nlc3MtdG9rZW46")
    (repo / "gha-creds-1.json").write_text('{"private_key": "secret"}\n')
    get_settings().set("claude_code.repo_path", str(repo))
    get_settings().set("claude_code.explore_repo", True)
    get_settings().set("claude_code.explore_commands", ["review"])
    token = current_command.set("review")
    yield repo
    current_command.reset(token)


def test_exploration_root_requires_an_enabled_command_and_a_git_checkout(checkout, tmp_path):
    assert exploration_root() == os.path.realpath(checkout)

    token = current_command.set("describe")
    assert exploration_root() is None
    current_command.reset(token)

    get_settings().set("claude_code.explore_repo", False)
    assert exploration_root() is None
    get_settings().set("claude_code.explore_repo", True)

    get_settings().set("claude_code.repo_path", str(tmp_path))  # not a git checkout
    assert exploration_root() is None


def test_snapshot_holds_only_tracked_files_without_git_or_escaping_links(checkout):
    snapshot = snapshot_checkout(str(checkout))

    files = sorted(str(p.relative_to(snapshot)) for p in Path(snapshot).rglob("*"))
    assert files == ["alias.py", "src", "src/app.py"]
    assert Path(snapshot, "alias.py").read_text() == "x = 1\n"
    assert snapshot_checkout(str(checkout)) == snapshot  # one copy per HEAD

    (checkout / "src" / "app.py").write_text("x = 2\n")
    _git(checkout, "commit", "-qam", "change")
    newer = snapshot_checkout(str(checkout))
    assert newer != snapshot and Path(newer, "src", "app.py").read_text() == "x = 2\n"


def test_exploration_root_defaults_to_the_actions_workspace(checkout, monkeypatch):
    get_settings().set("claude_code.repo_path", "")
    assert exploration_root() is None

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(checkout))
    assert exploration_root() == os.path.realpath(checkout)


async def test_exploring_session_offers_only_read_tools_under_dont_ask(monkeypatch, checkout):
    get_settings().set("claude_code.max_turns", 7)
    calls = []
    _fake_query(monkeypatch, _assistant(), _result(), calls=calls)

    await ClaudeAgentSDKHandler().chat_completion(model="opus", system="SYSTEM", user="u")

    options = calls[0]["options"]
    assert options.tools == ["Read", "Grep", "Glob"]
    assert options.allowed_tools == []  # a bare allow rule would approve reads outside the checkout
    assert options.permission_mode == "dontAsk"
    assert options.cwd == snapshot_checkout(str(checkout))  # a snapshot, never the checkout itself
    assert Path(options.cwd, "src", "app.py").is_file() and not Path(options.cwd, ".git").exists()
    assert options.max_turns == 7
    assert "PreToolUse" in options.hooks
    assert calls[0]["system"].startswith("SYSTEM") and "at most 7 turns" in calls[0]["system"]


async def test_non_exploring_command_stays_tool_less(monkeypatch, checkout):
    calls = []
    _fake_query(monkeypatch, _assistant(), _result(), calls=calls)
    token = current_command.set("describe")
    try:
        await ClaudeAgentSDKHandler().chat_completion(model="opus", system="SYSTEM", user="u")
    finally:
        current_command.reset(token)

    options = calls[0]["options"]
    assert (options.tools, options.max_turns, options.hooks) == ([], 1, None)
    assert calls[0]["system"] == "SYSTEM"


@pytest.mark.parametrize(("tool", "tool_input", "allowed"), [
    ("Read", {"file_path": "src/app.py"}, True),
    ("Read", {"file_path": "{repo}/src/app.py"}, True),
    ("Grep", {"pattern": "x"}, True),
    ("Glob", {"pattern": "src/**/*.py"}, True),
    ("Read", {"file_path": "link.txt"}, False),  # symlink to a file outside the checkout
    ("Read", {"file_path": "../outside.txt"}, False),
    ("Read", {"file_path": "/etc/hosts"}, False),
    ("Read", {"file_path": "~/.ssh/id_rsa"}, False),
    ("Read", {"file_path": ".git/config"}, False),
    ("Read", {"file_path": ".GIT/config"}, False),  # case-insensitive filesystems
    ("Glob", {"pattern": ".GIT/*"}, False),
    ("Read", {"file_path": "src/a\x00b"}, False),  # malformed input fails closed
    ("Grep", {"pattern": "token", "path": ".git"}, False),
    ("Grep", {"pattern": "x", "path": "/"}, False),
    ("Glob", {"pattern": "../*"}, False),
    ("Glob", {"pattern": "/etc/*"}, False),
    ("Glob", {"pattern": "src/../../*"}, False),
    ("Glob", {"pattern": "**/*.{py,toml}"}, True),
    ("Glob", {"pattern": "**/.github/**"}, True),
    ("Glob", {"pattern": "{/etc/*,x}"}, False),  # brace group with an absolute path
    ("Glob", {"pattern": "{..,src}/*"}, False),
    ("Glob", {"pattern": "**/.git/**"}, False),
    ("Glob", {"pattern": "{src,.git}/*"}, False),
    ("Grep", {"pattern": "token", "glob": ".git/**"}, False),
    ("Grep", {"pattern": "token", "glob": "*.py"}, True),
    ("Glob", {"pattern": "x" * 2000}, False),
    ("Bash", {"command": "cat /etc/hosts"}, False),
])
async def test_read_only_guard_confines_tools_to_the_checkout(checkout, tool, tool_input, allowed):
    tool_input = {k: v.replace("{repo}", str(checkout)) for k, v in tool_input.items()}
    guard = make_read_only_guard(str(checkout))

    decision = await guard({"tool_name": tool, "tool_input": tool_input}, "id", {})

    denied = decision.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
    assert denied is not allowed


def test_cli_path_finds_the_bundled_claude_code_binary():
    path = handler_module.claude_code_cli_path()
    assert path and os.path.basename(path).startswith("claude")
