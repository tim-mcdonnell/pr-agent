"""Tests for pr_agent/mosaico/env_bridge.py and observability.py.

These tests mutate the module-global ``global_settings`` (no request context is active),
so the ``restore_settings`` fixture snapshots and restores the settings keys the
env-bridge touches.

asyncio_mode=auto; monkeypatch is the repo convention for env vars."""
import pytest

from pr_agent.config_loader import get_settings, global_settings
from pr_agent.mosaico.env_bridge import apply_mosaico_env, langfuse_env_present
from pr_agent.mosaico.observability import mosaico_log_context, parse_observability_metadata

_SNAPSHOT_KEYS = [
    "OPENAI.API_BASE",
    "OPENAI.KEY",
    "CONFIG.MODEL",
    "CONFIG.FALLBACK_MODELS",
    "CONFIG.CUSTOM_MODEL_MAX_TOKENS",
]

_MOSAICO_ENV_VARS = [
    "API_BASE", "API_KEY", "MODEL_NAME", "MODEL_MAX_TOKENS",
    "LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
]

_SENTINEL = object()


@pytest.fixture
def restore_settings():
    """Snapshot the settings keys the bridge may touch and restore them after."""
    snapshot = {k: global_settings.get(k, _SENTINEL) for k in _SNAPSHOT_KEYS}
    yield
    for k, v in snapshot.items():
        if v is _SENTINEL:
            # Key was absent before; best-effort reset to empty so the global is not polluted.
            global_settings.set(k, [] if k == "CONFIG.FALLBACK_MODELS" else None)
        else:
            global_settings.set(k, v)


@pytest.fixture
def clear_mosaico_env(monkeypatch):
    for var in _MOSAICO_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def warnings(monkeypatch):
    import pr_agent.mosaico.env_bridge as env_bridge

    messages = []

    class _Logger:
        def warning(self, message, *args, **kwargs):
            messages.append(message)

    monkeypatch.setattr(env_bridge, "get_logger", lambda: _Logger())
    return messages


def _snapshot():
    return {k: global_settings.get(k, _SENTINEL) for k in _SNAPSHOT_KEYS}


class TestApplyMosaicoEnv:
    def test_apply_mosaico_env_noop_when_absent(self, restore_settings, clear_mosaico_env, warnings):
        """THE load-bearing regression guard: with no MOSAICO env, nothing changes."""
        before = _snapshot()
        apply_mosaico_env()
        assert _snapshot() == before
        assert warnings == []

    @pytest.mark.parametrize("env", [
        {"API_BASE": "https://mosaico.example/v1"},
        {"API_KEY": "secret-key"},
        {"API_BASE": "https://mosaico.example/v1", "API_KEY": "secret-key"},
    ])
    def test_api_base_and_key_are_ignored_with_a_warning(
            self, restore_settings, clear_mosaico_env, warnings, monkeypatch, env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        before = _snapshot()

        apply_mosaico_env()

        assert _snapshot() == before
        assert warnings == ["MOSAICO: API_BASE/API_KEY are ignored; models run through Claude Code."]
        assert all("secret-key" not in message for message in warnings)

    @pytest.mark.parametrize("model_name", ["opus", "sonnet[1m]", "anthropic/claude-haiku-5-5", "my-model"])
    def test_model_name_is_set_verbatim_and_clears_fallbacks(
            self, restore_settings, clear_mosaico_env, monkeypatch, model_name):
        global_settings.set("CONFIG.FALLBACK_MODELS", ["sonnet"])
        monkeypatch.setenv("MODEL_NAME", model_name)

        apply_mosaico_env()

        s = get_settings()
        assert s.get("CONFIG.MODEL") == model_name
        assert s.get("CONFIG.FALLBACK_MODELS") == []

    def test_api_settings_do_not_reach_openai_settings(self, restore_settings, clear_mosaico_env, monkeypatch):
        monkeypatch.setenv("API_BASE", "https://mosaico.example/v1")
        monkeypatch.setenv("API_KEY", "secret-key")
        monkeypatch.setenv("MODEL_NAME", "sonnet")
        openai_before = (global_settings.get("OPENAI.API_BASE", None), global_settings.get("OPENAI.KEY", None))

        apply_mosaico_env()

        assert (get_settings().get("OPENAI.API_BASE", None), get_settings().get("OPENAI.KEY", None)) == openai_before
        assert get_settings().get("CONFIG.MODEL") == "sonnet"

    def test_apply_mosaico_env_is_idempotent(self, restore_settings, clear_mosaico_env, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "opus")
        monkeypatch.setenv("MODEL_MAX_TOKENS", "8192")
        apply_mosaico_env()
        first = _snapshot()
        apply_mosaico_env()
        assert _snapshot() == first

    def test_langfuse_env_present_requires_all_three_variables(self, clear_mosaico_env, monkeypatch):
        assert langfuse_env_present() is False
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.example")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        assert langfuse_env_present() is False
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        assert langfuse_env_present() is False
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        assert langfuse_env_present() is True

    def test_langfuse_env_alone_changes_no_settings(self, restore_settings, clear_mosaico_env, warnings, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.example")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        before = _snapshot()
        apply_mosaico_env()
        assert _snapshot() == before
        assert warnings == []

    def test_model_max_tokens_valid_positive_value(self, restore_settings, clear_mosaico_env, monkeypatch):
        """A valid positive MODEL_MAX_TOKENS is written through unchanged."""
        monkeypatch.setenv("MODEL_NAME", "my-model")
        monkeypatch.setenv("MODEL_MAX_TOKENS", "8192")
        apply_mosaico_env()
        assert get_settings().get("CONFIG.CUSTOM_MODEL_MAX_TOKENS") == 8192

    def test_model_max_tokens_applies_without_model_name(self, restore_settings, clear_mosaico_env, monkeypatch):
        global_settings.set("CONFIG.MODEL", "opus")
        monkeypatch.setenv("MODEL_MAX_TOKENS", "8192")
        apply_mosaico_env()
        assert get_settings().get("CONFIG.CUSTOM_MODEL_MAX_TOKENS") == 8192
        assert get_settings().get("CONFIG.MODEL") == "opus"

    def test_model_name_without_max_tokens_keeps_custom_limit(self, restore_settings, clear_mosaico_env, monkeypatch):
        """There is no default custom limit any more: Claude models resolve their own window."""
        global_settings.set("CONFIG.CUSTOM_MODEL_MAX_TOKENS", 0)
        monkeypatch.setenv("MODEL_NAME", "opus")
        apply_mosaico_env()
        assert get_settings().get("CONFIG.CUSTOM_MODEL_MAX_TOKENS") == 0

    @pytest.mark.parametrize("value", ["0", "-5", "not-a-number", "8192.5"])
    def test_unusable_model_max_tokens_is_ignored_with_a_warning(
            self, restore_settings, clear_mosaico_env, warnings, monkeypatch, value):
        global_settings.set("CONFIG.CUSTOM_MODEL_MAX_TOKENS", 1234)
        monkeypatch.setenv("MODEL_NAME", "my-model")
        monkeypatch.setenv("MODEL_MAX_TOKENS", value)
        apply_mosaico_env()
        assert get_settings().get("CONFIG.CUSTOM_MODEL_MAX_TOKENS") == 1234
        assert warnings == [f"MOSAICO: ignoring MODEL_MAX_TOKENS={value!r}; expected a positive integer."]


class TestParseObservabilityMetadata:
    def test_all_three_keys(self):
        raw = {
            "mosaico-root-task-id": "123e4567-e89b-12d3-a456-426614174000",
            "mosaico-super-task-id": "00112233-4455-6677-8899-aabbccddeeff",
            "mosaico-root-task-name": "name1",
        }
        assert parse_observability_metadata(raw) == raw

    def test_missing_one_key_returns_partial_dict(self):
        raw = {"mosaico-root-task-id": "123e4567-e89b-12d3-a456-426614174000", "mosaico-root-task-name": "name1"}
        out = parse_observability_metadata(raw)
        assert out == {
            "mosaico-root-task-id": "123e4567-e89b-12d3-a456-426614174000", "mosaico-root-task-name": "name1",
        }
        assert out != {}

    def test_non_string_value_key_omitted(self):
        raw = {"mosaico-root-task-id": "123e4567-e89b-12d3-a456-426614174000", "mosaico-super-task-id": 12345}
        assert parse_observability_metadata(raw) == {"mosaico-root-task-id": "123e4567-e89b-12d3-a456-426614174000"}

    def test_non_mapping_returns_empty_and_never_raises(self):
        for bad in (None, [], "string", 42):
            assert parse_observability_metadata(bad) == {}

    def test_extra_keys_ignored(self):
        raw = {"mosaico-root-task-id": "123e4567-e89b-12d3-a456-426614174000", "unrelated": "x"}
        assert parse_observability_metadata(raw) == {"mosaico-root-task-id": "123e4567-e89b-12d3-a456-426614174000"}


class TestMosaicoLogContext:
    def test_binds_ids(self):
        from pr_agent.log import get_logger
        captured = {}

        def sink(message):
            captured.update(message.record["extra"])

        logger = get_logger()
        handler_id = logger.add(sink, format="{message}")
        try:
            meta = {"mosaico-root-task-id": "r1", "mosaico-super-task-id": "s1"}
            with mosaico_log_context(meta, "ctx-123"):
                logger.info("inside")
            assert captured.get("mosaico-root-task-id") == "r1"
            assert captured.get("mosaico-super-task-id") == "s1"
            assert captured.get("context_id") == "ctx-123"
            # After exit, a fresh log record must not carry the bindings.
            captured.clear()
            logger.info("outside")
            assert "mosaico-root-task-id" not in captured
        finally:
            logger.remove(handler_id)

    def test_empty_meta_is_clean_passthrough(self):
        # Must not raise and must yield even with empty meta and no context_id.
        executed = False
        with mosaico_log_context({}, None):
            executed = True

        assert executed is True
