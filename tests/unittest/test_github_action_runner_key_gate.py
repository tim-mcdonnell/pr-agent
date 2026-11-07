"""Verify the runner's Claude-credential gate while allowing bare static commands (#4324)."""

import copy
import json
from unittest.mock import Mock

import pytest
from starlette_context import request_cycle_context

from pr_agent import doctor, git_providers
from pr_agent.agent import pr_agent as pr_agent_module
from pr_agent.algo.ai_handlers.claude_agent_sdk_handler import ClaudeAgentSDKHandler
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.servers import github_action_runner
from tests.unittest._settings_helpers import restore_settings, snapshot_settings

ACTION_INPUTS = (*github_action_runner.ACTION_CREDENTIAL_INPUTS, "INPUT_MODEL", "INPUT_FALLBACK_MODELS")
MISSING_CREDENTIAL = ("claude", doctor.CLAUDE_CREDENTIAL_ENV_VARS)
GATE_MESSAGE = (
    "Stopping before any tool runs: no Claude credential. "
    "Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN "
    "(or the action's claude_code_oauth_token input). "
    "'pr-agent doctor' preflights the same check by hand."
)


@pytest.fixture
def action_env_factory(monkeypatch, tmp_path):
    def _factory(event, event_name):
        event_path = tmp_path / f"{event_name}.json"
        event_path.write_text(json.dumps(event), encoding="utf-8")
        monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
        monkeypatch.setenv("GITHUB_TOKEN", "token")
        # setenv first so monkeypatch restores (removes) a value _apply_action_inputs() may export.
        for name in (*doctor.CLAUDE_CREDENTIAL_ENV_VARS, *ACTION_INPUTS):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)

    return _factory


@pytest.fixture
def action_env(action_env_factory):
    action_env_factory({
        "action": "opened",
        "pull_request": {
            "html_url": "https://github.com/owner/repo/pull/1",
            "url": "https://api.github.com/repos/owner/repo/pulls/1",
        },
    }, "pull_request")


@pytest.fixture
def pr_comment_env(action_env_factory):
    def _factory(body, event_name):
        pr = {
            "html_url": "https://github.com/owner/repo/pull/1",
            "url": "https://api.github.com/repos/owner/repo/pulls/1",
        }
        event = {
            "action": "created", "sender": {"type": "User"},
            "comment": {"body": body, "id": 123, "pull_request_url": pr["url"]},
        }
        if event_name == "issue_comment":
            event["issue"] = {"pull_request": pr}
        else:
            event["pull_request"] = pr
        action_env_factory(event, event_name)

    return _factory


@pytest.mark.parametrize("event_name", ["issue_comment", "pull_request_review_comment"])
@pytest.mark.parametrize("body", ["/help", "/config", "/settings", " \t/HELP \n", " /CoNfIg \t", "\n/SETTINGS "])
async def test_static_pr_comments_render_without_claude_credential(monkeypatch, pr_comment_env, event_name, body):
    class StaticProvider(GithubProvider):
        def __init__(self, pr_url=None):
            super().__init__(None)
            self.pr_url = pr_url

        publish_comment = Mock()
        remove_initial_comment = Mock()
        add_eyes_reaction = Mock()

        def get_request_policy_metadata(self, required_fields):
            return {"title": "Test PR"}

    gate = Mock(return_value=MISSING_CREDENTIAL)
    ai_init = Mock(side_effect=AssertionError("Static commands must not initialize an AI handler"))
    monkeypatch.setattr(github_action_runner, "missing_provider_key_env_vars", gate)
    monkeypatch.setattr(github_action_runner, "apply_repo_settings", lambda url: None)
    monkeypatch.setattr(pr_agent_module, "apply_repo_settings", lambda url: None)
    monkeypatch.setattr(ClaudeAgentSDKHandler, "__init__", ai_init)
    monkeypatch.setitem(git_providers._GIT_PROVIDERS, "github", StaticProvider)
    monkeypatch.setitem(get_settings().config, "git_provider", "github")
    monkeypatch.setitem(get_settings().config, "publish_output", True)
    pr_comment_env(body, event_name)

    with request_cycle_context({"settings": copy.deepcopy(get_settings())}):
        await github_action_runner.run_action()

    gate.assert_not_called()
    ai_init.assert_not_called()
    StaticProvider.publish_comment.assert_called_once()
    expected = "PR Agent Walkthrough" if body.strip().lower() == "/help" else "config.model ="
    assert expected in StaticProvider.publish_comment.call_args.args[0]


@pytest.mark.parametrize("event_name", ["issue_comment", "pull_request_review_comment"])
@pytest.mark.parametrize("body", ["/help How do I configure reviews?", "/review", "/configuration", "/config typo"])
async def test_other_pr_commands_still_require_claude_credential(monkeypatch, pr_comment_env, event_name, body):
    monkeypatch.setattr(github_action_runner, "apply_repo_settings", lambda url: None)
    monkeypatch.setattr(
        github_action_runner, "missing_provider_key_env_vars",
        lambda: MISSING_CREDENTIAL,
    )
    pr_comment_env(body, event_name)

    with pytest.raises(SystemExit, match="Stopping before any tool runs.*CLAUDE_CODE_OAUTH_TOKEN"):
        await github_action_runner.run_action()


@pytest.mark.parametrize("event_name", ["issue_comment", "pull_request_review_comment"])
async def test_static_pr_comments_still_require_github_token(monkeypatch, pr_comment_env, event_name):
    pr_comment_env("/help", event_name)
    monkeypatch.delenv("GITHUB_TOKEN")

    with pytest.raises(SystemExit, match="GITHUB_TOKEN not set"):
        await github_action_runner.run_action()


async def test_missing_credential_stops_before_tools_naming_the_env_variables(
    monkeypatch, action_env
):
    monkeypatch.setattr(
        github_action_runner, "missing_provider_key_env_vars",
        lambda: MISSING_CREDENTIAL,
    )

    with pytest.raises(SystemExit, match="Stopping before any tool runs.*CLAUDE_CODE_OAUTH_TOKEN"):
        await github_action_runner.run_action()


async def test_real_gate_stops_without_any_claude_credential(monkeypatch, action_env, capsys):
    dispatches = []

    async def run_tool(*args):
        dispatches.append(args)

    monkeypatch.setattr(github_action_runner, "apply_repo_settings", lambda url: None)
    monkeypatch.setattr(github_action_runner, "enforce_request_policy", lambda *args, **kwargs: True)
    monkeypatch.setattr(github_action_runner, "_run_auto_tool", run_tool)

    with pytest.raises(SystemExit) as error:
        await github_action_runner.run_action()

    assert str(error.value) == GATE_MESSAGE
    assert dispatches == []


@pytest.mark.parametrize(
    "credential", [*doctor.CLAUDE_CREDENTIAL_ENV_VARS, *github_action_runner.ACTION_CREDENTIAL_INPUTS],
)
async def test_any_claude_credential_or_action_input_passes_the_real_gate(monkeypatch, action_env, credential):
    secret = "synthetic-claude-credential-not-for-output"
    dispatches = []

    async def run_tool(*args):
        dispatches.append(args)

    monkeypatch.setenv(credential, secret)
    monkeypatch.setattr(github_action_runner, "apply_repo_settings", lambda url: None)
    monkeypatch.setattr(github_action_runner, "enforce_request_policy", lambda *args, **kwargs: True)
    monkeypatch.setattr(github_action_runner, "_run_auto_tool", run_tool)
    monkeypatch.setattr(github_action_runner, "_inject_artifact_context", lambda: None)

    with request_cycle_context({"settings": copy.deepcopy(get_settings())}):
        await github_action_runner.run_action()

    assert doctor.missing_provider_key_env_vars() is None
    assert dispatches


async def test_repository_settings_are_applied_before_key_gate(monkeypatch, action_env):
    order = []
    monkeypatch.setattr(
        github_action_runner, "apply_repo_settings",
        lambda pr_url: order.append(("repo_settings", pr_url)),
    )
    monkeypatch.setattr(
        github_action_runner, "missing_provider_key_env_vars",
        lambda: order.append("gate") or MISSING_CREDENTIAL,
    )
    monkeypatch.setattr(
        github_action_runner, "get_setting_or_env", lambda key, default=None: default,
    )

    # A repo-level .pr_agent.toml can select a different provider, so its settings must
    # be applied before the gate judges the active provider's key.
    with pytest.raises(SystemExit):
        await github_action_runner.run_action()
    assert order == [("repo_settings", "https://api.github.com/repos/owner/repo/pulls/1"), "gate"]


@pytest.mark.parametrize(
    "event_name, event, expected_url",
    [
        (
            "issue_comment",
            {
                "action": "created",
                "comment": {"body": "/review"},
                "issue": {"number": 1, "pull_request": {
                    "html_url": "https://github.com/owner/repo/pull/1",
                    "url": "https://api.github.com/repos/owner/repo/pulls/1",
                }},
            },
            "https://api.github.com/repos/owner/repo/pulls/1",
        ),
        (
            "workflow_run",
            {
                "action": "workflow_run",
                "workflow_run": {
                    "event": "pull_request",
                    "conclusion": "success",
                    "pull_requests": [{"url": "https://api.github.com/repos/owner/repo/pulls/1"}],
                },
            },
            "https://api.github.com/repos/owner/repo/pulls/1",
        ),
    ],
)
async def test_repository_settings_are_applied_before_key_gate_on_comment_and_workflow_run_events(
    monkeypatch, action_env_factory, event_name, event, expected_url
):
    order = []
    monkeypatch.setattr(
        github_action_runner, "apply_repo_settings",
        lambda pr_url: order.append(("repo_settings", pr_url)),
    )
    monkeypatch.setattr(
        github_action_runner, "missing_provider_key_env_vars",
        lambda: order.append("gate") or MISSING_CREDENTIAL,
    )
    monkeypatch.setattr(
        github_action_runner, "get_setting_or_env", lambda key, default=None: default,
    )
    action_env_factory(event, event_name)

    # The gate must judge the provider the repo-level settings select, whatever event
    # shape carried the PR.
    with pytest.raises(SystemExit):
        await github_action_runner.run_action()
    assert order == [("repo_settings", expected_url), "gate"]


@pytest.mark.parametrize(
    "event_name, event",
    [
        (
            "issue_comment",
            {
                "action": "created",
                "issue": {"number": 1},
                "comment": {"body": "not a command"},
            },
        ),
        (
            "workflow_run",
            {
                "action": "workflow_run",
                "workflow_run": {"event": "push", "conclusion": "success"},
            },
        ),
    ],
)
async def test_events_without_a_pr_skip_the_gate(
    monkeypatch, action_env_factory, event_name, event
):
    calls = []
    monkeypatch.setattr(
        github_action_runner, "apply_repo_settings",
        lambda pr_url: calls.append(("repo_settings", pr_url)),
    )
    monkeypatch.setattr(
        github_action_runner, "missing_provider_key_env_vars",
        lambda: calls.append("gate") or MISSING_CREDENTIAL,
    )
    monkeypatch.setattr(
        github_action_runner, "get_setting_or_env", lambda key, default=None: default,
    )
    action_env_factory(event, event_name)

    # No PR, no repo settings: the run decides, as on main.
    await github_action_runner.run_action()
    assert calls == []


@pytest.mark.parametrize(
    "event_name, action, sender_type, body",
    [
        ("issue_comment", "created", "User", "Thanks, this looks good."),
        ("pull_request_review_comment", "created", "User", "Thanks!"),
        ("issue_comment", "edited", "User", "  "),
        ("issue_comment", "created", "User", ""),
        ("pull_request_review_comment", "created", "User", None),
        ("issue_comment", "created", "Bot", "/review"),
        ("pull_request_review_comment", "deleted", "User", "/review"),
    ],
)
async def test_ignored_pr_comments_skip_repository_settings_and_key_gate(
    monkeypatch, action_env_factory, event_name, action, sender_type, body
):
    calls = []

    async def handle_request(url, command, notify=None):
        calls.append("dispatch")

    monkeypatch.setattr(github_action_runner, "_handle_request", handle_request)
    monkeypatch.setattr(github_action_runner, "apply_repo_settings", lambda url: calls.append("repo_settings"))
    monkeypatch.setattr(
        github_action_runner, "missing_provider_key_env_vars",
        lambda: calls.append("gate") or MISSING_CREDENTIAL,
    )
    pr = {"html_url": "https://github.com/owner/repo/pull/1", "url": "https://api.github.com/repos/owner/repo/pulls/1"}
    event = {"action": action, "sender": {"type": sender_type}, "comment": {"body": body}}
    if event_name == "issue_comment":
        event["issue"] = {"pull_request": pr}
    else:
        event["pull_request"] = pr
    action_env_factory(event, event_name)

    await github_action_runner.run_action()

    assert calls == []


@pytest.mark.parametrize("body", ["/ask Why?", "/unknown"])
async def test_inline_command_comments_keep_setup_before_line_conversion(monkeypatch, action_env_factory, body):
    order = []
    converted = ["/ask_line", "Why?"]
    monkeypatch.setattr(github_action_runner, "apply_repo_settings", lambda url: order.append("repo_settings"))
    monkeypatch.setattr(github_action_runner, "missing_provider_key_env_vars", lambda: order.append("gate"))
    monkeypatch.setattr(github_action_runner, "_inject_artifact_context", lambda: None)

    def convert_line_comment(event, comment_body):
        assert order == ["repo_settings", "gate"]
        assert comment_body == "/ask Why?"
        order.append("line_conversion")
        return converted

    async def handle_request(url, command, notify=None):
        order.append(("dispatch", command))

    monkeypatch.setattr(github_action_runner, "handle_line_comments", convert_line_comment)
    monkeypatch.setattr(github_action_runner, "_handle_request", handle_request)
    pr = {"html_url": "https://github.com/owner/repo/pull/1", "url": "https://api.github.com/repos/owner/repo/pulls/1"}
    action_env_factory({
        "action": "created",
        "pull_request": pr,
        "comment": {"body": body, "pull_request_url": pr["url"], "id": 123},
        "sender": {"type": "User"},
    }, "pull_request_review_comment")

    await github_action_runner.run_action()

    expected = ["repo_settings", "gate"]
    if body.startswith("/ask"):
        expected += ["line_conversion", ("dispatch", converted)]
    else:
        expected.append(("dispatch", body))
    assert order == expected


class TestApplyActionInputs:
    @pytest.fixture(autouse=True)
    def clean_inputs(self, monkeypatch):
        snapshot = snapshot_settings(("config.model", "config.fallback_models"))
        for name in (*doctor.CLAUDE_CREDENTIAL_ENV_VARS, *ACTION_INPUTS):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
        yield
        restore_settings(snapshot)

    @pytest.mark.parametrize("input_name, env_name", github_action_runner.ACTION_CREDENTIAL_INPUTS.items())
    def test_credential_input_is_exported_stripped(self, monkeypatch, input_name, env_name):
        import os

        monkeypatch.setenv(input_name, "  synthetic-credential \n")
        github_action_runner._apply_action_inputs()
        assert os.environ[env_name] == "synthetic-credential"

    @pytest.mark.parametrize("input_name, env_name", github_action_runner.ACTION_CREDENTIAL_INPUTS.items())
    def test_explicit_env_var_wins_over_the_input(self, monkeypatch, input_name, env_name):
        import os

        monkeypatch.setenv(env_name, "explicit-env-credential")
        monkeypatch.setenv(input_name, "input-credential")
        github_action_runner._apply_action_inputs()
        assert os.environ[env_name] == "explicit-env-credential"

    @pytest.mark.parametrize("value", ["", "   ", "\n"])
    def test_empty_inputs_change_nothing(self, monkeypatch, value):
        import os

        get_settings().set("config.model", "opus")
        get_settings().set("config.fallback_models", ["sonnet"])
        for name in ACTION_INPUTS:
            monkeypatch.setenv(name, value)
        github_action_runner._apply_action_inputs()
        assert all(name not in os.environ for name in doctor.CLAUDE_CREDENTIAL_ENV_VARS)
        assert get_settings().config.model == "opus"
        assert list(get_settings().config.fallback_models) == ["sonnet"]

    def test_model_inputs_set_model_and_fallbacks(self, monkeypatch):
        monkeypatch.setenv("INPUT_MODEL", " sonnet[1m] ")
        monkeypatch.setenv("INPUT_FALLBACK_MODELS", " haiku, ,claude-opus-5-5 ,")
        github_action_runner._apply_action_inputs()
        assert get_settings().config.model == "sonnet[1m]"
        assert list(get_settings().config.fallback_models) == ["haiku", "claude-opus-5-5"]

    async def test_run_action_applies_inputs_before_the_gate(self, monkeypatch, action_env):
        seen = []

        def gate():
            seen.append((get_settings().config.model, list(get_settings().config.fallback_models)))
            return MISSING_CREDENTIAL

        monkeypatch.setenv("INPUT_MODEL", "haiku")
        monkeypatch.setenv("INPUT_FALLBACK_MODELS", "sonnet")
        monkeypatch.setattr(github_action_runner, "apply_repo_settings", lambda url: None)
        monkeypatch.setattr(github_action_runner, "missing_provider_key_env_vars", gate)
        monkeypatch.setattr(github_action_runner, "enforce_request_policy", lambda *args, **kwargs: True)

        with request_cycle_context({"settings": copy.deepcopy(get_settings())}):
            with pytest.raises(SystemExit, match="no Claude credential"):
                await github_action_runner.run_action()

        assert seen == [("haiku", ["sonnet"])]
