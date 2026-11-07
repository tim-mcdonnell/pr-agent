import pytest


@pytest.fixture(autouse=True)
def isolate_run_details():
    """Start each test from a clean run-details ContextVar and restore it.

    The collector lives in a module-level ContextVar, so a test that leaves
    details behind would otherwise be visible to whichever test runs next.
    """
    from pr_agent.algo import run_details

    token = run_details._run_details.set(None)
    yield
    run_details._run_details.reset(token)


@pytest.fixture(autouse=True)
def block_real_claude_code(monkeypatch):
    """Fail any unit test that would launch the Claude Code binary instead of stubbing claude_agent_sdk.query."""
    import claude_agent_sdk

    def refuse(*args, **kwargs):
        raise AssertionError("unit tests must not launch Claude Code; patch claude_agent_sdk.query")

    monkeypatch.setattr(claude_agent_sdk, "query", refuse)
