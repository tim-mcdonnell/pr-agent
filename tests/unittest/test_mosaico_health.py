"""HTTP /health route tests.

Exercise build_app() directly through an in-process ASGI transport and monkeypatch
health_check (the no-retry behavior itself was proven in 2c). Verify the route and
200/503 response shape without relying on Starlette's thread-backed TestClient.

Also exercises the REAL health_check(), which sends one Claude Code probe through
ClaudeAgentSDKHandler.chat_completion (stubbed here, or via claude_agent_sdk.query)."""
import asyncio
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import claude_agent_sdk
import httpx
import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from pr_agent.algo.ai_handlers.claude_agent_sdk_handler import ClaudeAgentSDKHandler
from pr_agent.config_loader import get_settings
from pr_agent.mosaico import executor as executor_mod
from pr_agent.mosaico import server as server_mod
from pr_agent.mosaico.executor import health_check
from pr_agent.mosaico.server import build_app
from tests.unittest._settings_helpers import restore_settings, snapshot_settings


def _app(monkeypatch, health_value):
    async def fake_health_check():
        return health_value

    # health_check is imported into server_mod's namespace and called by _HealthApp._health.
    monkeypatch.setattr(server_mod, "health_check", fake_health_check)
    return build_app()


async def _get_health(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get("/health")


class TestHealthRoute:
    @pytest.mark.asyncio
    async def test_healthy_returns_200(self, monkeypatch):
        resp = await _get_health(_app(monkeypatch, "OK"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_healthy"] is True
        assert body["status"] == "OK"

    @pytest.mark.asyncio
    async def test_unhealthy_returns_503(self, monkeypatch):
        resp = await _get_health(_app(monkeypatch, "Unhealthy: connection refused"))
        assert resp.status_code == 503
        body = resp.json()
        assert body["is_healthy"] is False
        assert "Unhealthy" in body["status"]
        assert "Unhealthy" in body["detail"]


_PROBE_SYSTEM = "Reply with exactly: ok"
_PROBE_USER = "ping"


@pytest.fixture
def restore_config_model(monkeypatch):
    """Restore the probe settings exactly, including originally-absent state."""
    snapshot = snapshot_settings(
        ["CONFIG.MODEL", "MOSAICO.HEALTH_TIMEOUT_SECONDS", "CLAUDE_CODE.REPO_PATH", "CLAUDE_CODE.EXPLORE_REPO"]
    )
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    get_settings().set("CONFIG.MODEL", "sonnet")
    get_settings().set("CLAUDE_CODE.REPO_PATH", "")
    yield get_settings()
    restore_settings(snapshot)


def _stub_completion(monkeypatch, side_effect=None):
    completion = AsyncMock(side_effect=side_effect, return_value=("ok", "stop"))

    async def chat_completion(self, model, system, user, temperature=0.2, img_path=None):
        assert isinstance(self, ClaudeAgentSDKHandler)
        return await completion(model, system, user)

    monkeypatch.setattr(ClaudeAgentSDKHandler, "chat_completion", chat_completion)
    return completion


class TestHealthCheckProbe:
    """Exercise the REAL health_check() (not the monkeypatched stub)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["sonnet", "opus[1m]", "claude-haiku-5-5"])
    async def test_probe_sends_one_completion_for_the_configured_model(
        self, monkeypatch, restore_config_model, model
    ):
        restore_config_model.set("CONFIG.MODEL", model)
        completion = _stub_completion(monkeypatch)

        assert await health_check() == "OK"
        assert completion.await_args_list == [call(model, _PROBE_SYSTEM, _PROBE_USER)]

    @pytest.mark.asyncio
    async def test_probe_runs_through_claude_code_query(self, monkeypatch, restore_config_model):
        """Without a chat_completion stub, the probe reaches claude_agent_sdk.query with the MOSAICO model."""
        calls = []

        async def fake_query(*, prompt, options):
            calls.append((prompt, options.model))
            yield AssistantMessage(content=[TextBlock(text="ok")], model="claude-sonnet-5-5", stop_reason="end_turn")
            yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1,
                                session_id="s", stop_reason="end_turn", result="ok")

        monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

        assert await health_check() == "OK"
        assert calls == [(_PROBE_USER, "sonnet")]

    @pytest.mark.asyncio
    async def test_non_claude_model_is_unhealthy_without_launching_claude_code(
        self, monkeypatch, restore_config_model
    ):
        restore_config_model.set("CONFIG.MODEL", "openai/gpt-4o")
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)

        # The autouse conftest fixture makes claude_agent_sdk.query raise if the probe got that far.
        assert await health_check() == "Unhealthy: LLM probe failed"
        assert logger.mock_calls == [call.warning("MOSAICO health_check unhealthy: FallbackEligibleError")]

    @pytest.mark.asyncio
    async def test_live_probe_failure_returns_unhealthy(self, monkeypatch, restore_config_model):
        _stub_completion(monkeypatch, side_effect=RuntimeError("connection refused"))

        assert await health_check() == "Unhealthy: LLM probe failed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_stage", ["construction", "completion"])
    async def test_failure_details_do_not_reach_http_body_or_health_log(
        self, monkeypatch, restore_config_model, failure_stage
    ):
        error = RuntimeError("private-provider.example: credential=health-test-secret")
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
        completion = _stub_completion(monkeypatch, side_effect=error)
        if failure_stage == "construction":
            monkeypatch.setattr(ClaudeAgentSDKHandler, "__init__", MagicMock(side_effect=error))

        response = await _get_health(build_app())

        assert response.status_code == 503
        assert response.json() == {
            "is_healthy": False,
            "status": "Unhealthy: LLM probe failed",
            "detail": "Unhealthy: LLM probe failed",
        }
        assert logger.mock_calls == [call.warning("MOSAICO health_check unhealthy: RuntimeError")]
        assert completion.await_count == (failure_stage == "completion")

    @pytest.mark.asyncio
    async def test_deadline_cancels_cooperative_probe_work(self, monkeypatch, restore_config_model):
        deadline = asyncio.timeout(None)
        seen_timeouts = []

        def controlled_timeout(seconds):
            seen_timeouts.append(seconds)
            return deadline

        monkeypatch.setattr(executor_mod, "asyncio", SimpleNamespace(timeout=controlled_timeout))
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
        cancelled = asyncio.Event()

        async def wait_forever(*args, **kwargs):
            # Expire only after reaching the probe, without a timing race.
            deadline.reschedule(asyncio.get_running_loop().time())
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        completion = _stub_completion(monkeypatch, side_effect=wait_forever)

        result = await asyncio.wait_for(health_check(), timeout=1)

        assert seen_timeouts == [restore_config_model.get("MOSAICO.HEALTH_TIMEOUT_SECONDS")]
        assert result == "Unhealthy: LLM probe failed"
        assert cancelled.is_set()
        logger.warning.assert_called_once_with("MOSAICO health_check unhealthy: TimeoutError")
        completion.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout", [10, 25, 0.5])
    async def test_configured_timeout_bounds_the_probe(self, monkeypatch, restore_config_model, timeout):
        restore_config_model.set("MOSAICO.HEALTH_TIMEOUT_SECONDS", timeout)
        seen_timeouts = []

        def controlled_timeout(seconds):
            seen_timeouts.append(seconds)
            # Record the configured value without racing the probe on a slow runner.
            return asyncio.timeout(None)

        monkeypatch.setattr(executor_mod, "asyncio", SimpleNamespace(timeout=controlled_timeout))
        completion = _stub_completion(monkeypatch)

        assert await health_check() == "OK"
        assert seen_timeouts == [timeout]
        completion.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout", [None, True, False, 0, -1, float("nan"), float("inf"), "secret-timeout", []])
    async def test_invalid_timeout_fails_without_dispatch(self, monkeypatch, restore_config_model, timeout):
        restore_config_model.set("MOSAICO.HEALTH_TIMEOUT_SECONDS", timeout)
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
        completion = _stub_completion(monkeypatch)

        assert await health_check() == "Unhealthy: LLM probe failed"
        completion.assert_not_awaited()
        assert logger.mock_calls == [call.warning("MOSAICO health_check unhealthy: ValueError")]

    @pytest.mark.asyncio
    async def test_caller_cancellation_propagates_without_retry_or_failure_log(
        self, monkeypatch, restore_config_model
    ):
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
        started = asyncio.Event()

        async def wait_forever(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        completion = _stub_completion(monkeypatch, side_effect=wait_forever)
        task = asyncio.create_task(health_check())
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        completion.assert_awaited_once()
        logger.warning.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["", None])
    async def test_no_model_configured_returns_unhealthy(self, monkeypatch, restore_config_model, model):
        restore_config_model.set("CONFIG.MODEL", model)
        completion = _stub_completion(monkeypatch)

        assert await health_check() == "Unhealthy: no model configured"
        completion.assert_not_awaited()
