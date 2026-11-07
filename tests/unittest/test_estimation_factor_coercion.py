import pytest

from pr_agent.algo import token_handler
from pr_agent.algo.token_handler import DEFAULT_CLAUDE_TOKEN_FACTOR, TokenEncoder, estimate_tokens
from pr_agent.config_loader import get_settings

TEXT = "x" * 100  # 100 raw tokens with the one-token-per-character encoder below
DEFAULT_ESTIMATE = 170


@pytest.fixture
def restore_config(monkeypatch):
    import copy
    settings = get_settings(use_context=False)
    original = copy.deepcopy(settings.get("CONFIG", None))

    class CharacterEncoder:
        @staticmethod
        def encode(text, disallowed_special=()):
            return list(text)

    monkeypatch.setattr(TokenEncoder, "get_token_encoder", lambda model=None: CharacterEncoder())
    monkeypatch.setattr(token_handler, "_warned_factors", set())
    yield settings
    if original is None:
        settings.set("CONFIG", {})
    else:
        settings.set("CONFIG", original)


def test_default_factor_is_the_measured_claude_ratio(restore_config):
    assert DEFAULT_CLAUDE_TOKEN_FACTOR == 0.7
    assert restore_config.get("config.model_token_count_estimate_factor") == DEFAULT_CLAUDE_TOKEN_FACTOR

    assert estimate_tokens(TEXT) == DEFAULT_ESTIMATE


def test_accept_a_quoted_factor(restore_config):
    """Accept a quoted number, which is what TOML yields for
    `model_token_count_estimate_factor = "0.3"`."""
    restore_config.set("config.model_token_count_estimate_factor", "0.3")

    assert estimate_tokens(TEXT) == 130


@pytest.mark.parametrize("value, expected", [(0.5, 150), (0, 100), (1, 200), ("0", 100)])
def test_numeric_factor_scales_the_raw_count(restore_config, value, expected):
    restore_config.set("config.model_token_count_estimate_factor", value)

    assert estimate_tokens(TEXT) == expected


def test_estimate_rounds_up(restore_config):
    restore_config.set("config.model_token_count_estimate_factor", 0.7)

    assert estimate_tokens("abc") == 6  # ceil(3 * 1.7) = ceil(5.1)


def test_fall_back_to_the_default_factor_for_an_unreadable_factor(restore_config):
    """Fall back to the default Claude factor rather than raising when the value cannot be read."""
    restore_config.set("config.model_token_count_estimate_factor", "abc")

    assert estimate_tokens(TEXT) == DEFAULT_ESTIMATE


@pytest.mark.parametrize("value", ["inf", "-infinity", "nan", float("inf"), float("nan")])
def test_fall_back_for_a_non_finite_factor(restore_config, value):
    """Fall back to the default factor for a non-finite value, which float() happily accepts."""
    restore_config.set("config.model_token_count_estimate_factor", value)

    assert estimate_tokens(TEXT) == DEFAULT_ESTIMATE


@pytest.mark.parametrize("value", [True, False])
def test_fall_back_for_a_boolean_factor(restore_config, value):
    """Treat a boolean as unusable rather than letting float(True) become a factor of 2."""
    restore_config.set("config.model_token_count_estimate_factor", value)

    assert estimate_tokens(TEXT) == DEFAULT_ESTIMATE


@pytest.mark.parametrize("value", [1e308, "1e308"])
def test_fall_back_when_the_factor_overflows_the_estimate(restore_config, value):
    """Fall back to the raw count when the multiplication overflows to infinity."""
    restore_config.set("config.model_token_count_estimate_factor", value)

    assert estimate_tokens(TEXT) == 100


@pytest.mark.parametrize("value", [-1, -2, "-3.5"])
def test_fall_back_for_a_factor_that_would_zero_or_negate_the_count(restore_config, value):
    """Fall back to the default factor rather than reporting a zero or negative token count."""
    restore_config.set("config.model_token_count_estimate_factor", value)

    assert estimate_tokens(TEXT) == DEFAULT_ESTIMATE


def test_unusable_factor_warns_once_per_value(restore_config, monkeypatch):
    warnings = []

    class Logger:
        def warning(self, message, *args, **kwargs):
            warnings.append(message)

    monkeypatch.setattr(token_handler, "get_logger", lambda: Logger())
    restore_config.set("config.model_token_count_estimate_factor", "abc")
    estimate_tokens(TEXT)
    estimate_tokens(TEXT)
    restore_config.set("config.model_token_count_estimate_factor", "nan")
    estimate_tokens(TEXT)

    assert warnings == [
        "model_token_count_estimate_factor is not a usable number ('abc'), using 0.7",
        "model_token_count_estimate_factor is not a usable number ('nan'), using 0.7",
    ]


def test_usable_factor_does_not_warn(restore_config, monkeypatch):
    warnings = []

    class Logger:
        def warning(self, message, *args, **kwargs):
            warnings.append(message)

    monkeypatch.setattr(token_handler, "get_logger", lambda: Logger())
    restore_config.set("config.model_token_count_estimate_factor", 0.4)
    estimate_tokens(TEXT)

    assert warnings == []
