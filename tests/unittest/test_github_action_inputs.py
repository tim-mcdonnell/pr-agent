import pytest

from pr_agent.config_loader import get_settings
from pr_agent.servers import github_action_runner
from tests.unittest._settings_helpers import restore_settings, snapshot_settings


@pytest.fixture(autouse=True)
def isolated_inputs(monkeypatch):
    snapshot = snapshot_settings(("config.model", "config.fallback_models"))
    for name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "INPUT_CLAUDE_CODE_OAUTH_TOKEN",
                 "INPUT_ANTHROPIC_API_KEY", "INPUT_MODEL", "INPUT_FALLBACK_MODELS"):
        # setenv first so teardown removes what _apply_action_inputs() writes to os.environ.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    yield
    restore_settings(snapshot)


def test_inputs_supply_the_credential_and_model_slugs(monkeypatch):
    monkeypatch.setenv("INPUT_CLAUDE_CODE_OAUTH_TOKEN", " sk-ant-oat-token ")
    monkeypatch.setenv("INPUT_MODEL", "sonnet[1m]")
    monkeypatch.setenv("INPUT_FALLBACK_MODELS", "opus, haiku,")

    github_action_runner._apply_action_inputs()

    assert github_action_runner.os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-token"
    assert get_settings().config.model == "sonnet[1m]"
    assert list(get_settings().config.fallback_models) == ["opus", "haiku"]


def test_empty_inputs_and_explicit_env_take_precedence(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "from-env")
    monkeypatch.setenv("INPUT_CLAUDE_CODE_OAUTH_TOKEN", "from-input")
    monkeypatch.setenv("INPUT_MODEL", "  ")
    model = get_settings().config.model

    github_action_runner._apply_action_inputs()

    assert github_action_runner.os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == "from-env"
    assert "ANTHROPIC_API_KEY" not in github_action_runner.os.environ
    assert get_settings().config.model == model
