import threading
from math import ceil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tiktoken import get_encoding

from pr_agent.algo import token_handler


def _settings(model="primary-model", estimate_factor=0):
    return SimpleNamespace(
        config=SimpleNamespace(model=model),
        get=lambda key, default=None: {
            "config.model_token_count_estimate_factor": estimate_factor,
        }.get(key, default),
    )


class CharacterEncoder:
    """One token per character, so estimates are easy to compute by hand."""

    def __init__(self, model=None):
        self.model = model

    @staticmethod
    def encode(text, disallowed_special=()):
        return list(text)


@pytest.fixture
def character_encoder(monkeypatch):
    encoder = CharacterEncoder()
    monkeypatch.setattr(token_handler.TokenEncoder, "get_token_encoder", lambda model=None: encoder)
    return encoder


@pytest.mark.parametrize("factor, expected", [(0, 10), (0.7, 17), (0.25, 13), (1, 20)])
def test_estimate_tokens_scales_the_o200k_count_by_one_plus_factor(monkeypatch, character_encoder, factor, expected):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings(estimate_factor=factor))

    assert token_handler.estimate_tokens("a" * 10) == expected


@pytest.mark.parametrize("text", ["", None])
def test_estimate_tokens_of_empty_text_is_zero(monkeypatch, text):
    encoder = MagicMock()
    monkeypatch.setattr(token_handler.TokenEncoder, "get_token_encoder", lambda model=None: encoder)

    assert token_handler.estimate_tokens(text) == 0
    encoder.encode.assert_not_called()


def test_estimate_tokens_uses_the_real_o200k_encoder_with_default_factor(monkeypatch):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings(estimate_factor=0.7))
    text = "def add(a, b):\n    return a + b  # <|endoftext|> is plain text here\n"

    raw = len(get_encoding("o200k_base").encode(text, disallowed_special=()))

    assert token_handler.estimate_tokens(text) == ceil(raw * 1.7)


@pytest.mark.parametrize("model", [None, "opus", "claude-sonnet-5-5", "gpt-4o", "unknown/model"])
def test_token_encoder_is_always_o200k_base(model):
    encoder = token_handler.TokenEncoder.get_token_encoder(model)

    assert encoder.name == "o200k_base"
    assert encoder is token_handler.TokenEncoder.get_token_encoder()


def test_token_encoder_builds_the_shared_encoder_once_under_concurrency(monkeypatch):
    monkeypatch.setattr(token_handler.TokenEncoder, "_encoder_instance", None)
    created = []
    release = threading.Event()

    def build(name):
        created.append(name)
        assert release.wait(5), "test did not release the encoder build"
        return MagicMock(name=name)

    monkeypatch.setattr(token_handler, "get_encoding", build)
    results = []
    threads = [
        threading.Thread(target=lambda model=model: results.append(token_handler.TokenEncoder.get_token_encoder(model)))
        for model in ("opus", "sonnet", None)
    ]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(5)

    assert created == ["o200k_base"]
    assert len(results) == 3
    assert all(result is results[0] for result in results)


@pytest.mark.parametrize("force_accurate", [False, True])
def test_count_tokens_is_the_estimate_whatever_force_accurate_says(monkeypatch, character_encoder, force_accurate):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings(estimate_factor=0.3))
    handler = token_handler.TokenHandler(model="opus")

    assert handler.count_tokens("patch", force_accurate=force_accurate) == 7  # ceil(5 * 1.3)


def test_no_pr_handler_initializes_zero_prompt_tokens(monkeypatch):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings())
    monkeypatch.setattr(
        token_handler.TokenEncoder,
        "get_token_encoder",
        lambda _model=None: MagicMock(),
    )

    handler = token_handler.TokenHandler(model="fallback-model")

    assert handler.model == "fallback-model"
    assert handler.prompt_tokens == 0


def test_handler_defaults_to_the_configured_model(monkeypatch, character_encoder):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings(model="sonnet"))

    assert token_handler.TokenHandler().model == "sonnet"


@pytest.mark.parametrize("factor, expected", [(0, 13), (0.7, 16 + 7)])
def test_prompt_tokens_estimate_each_rendered_prompt(monkeypatch, character_encoder, factor, expected):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings(estimate_factor=factor))

    handler = token_handler.TokenHandler(object(), {"title": "PR"}, "system {{ title }}", "user")

    # "system PR" is 9 characters and "user" is 4; each prompt is estimated and rounded up on its own.
    assert handler.prompt_tokens == expected


def test_prompt_rendering_error_counts_zero(monkeypatch, character_encoder):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings())

    handler = token_handler.TokenHandler(object(), {}, "system {{ missing }}", "user")

    assert handler.prompt_tokens == 0


def test_count_prompt_tokens_false_skips_rendering(monkeypatch, character_encoder):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings())

    handler = token_handler.TokenHandler(object(), {}, "system {{ missing }}", "user", count_prompt_tokens=False)

    assert handler.prompt_tokens == 0


def test_for_model_rebinds_rendered_prompt_without_mutating_source(monkeypatch, character_encoder):
    monkeypatch.setattr(token_handler, "get_settings", lambda use_context=True: _settings())
    variables = {"title": "PR"}
    handler = token_handler.TokenHandler(object(), variables, "system {{ title }}", "user")

    fallback_handler = handler.for_model("fallback-model")

    assert handler.for_model("primary-model") is handler
    assert handler.model == "primary-model"
    assert fallback_handler is not handler
    assert fallback_handler.model == "fallback-model"
    assert fallback_handler.vars is variables
    assert fallback_handler.encoder is handler.encoder
    assert fallback_handler.prompt_tokens == len("system PR") + len("user")
