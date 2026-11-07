import pytest

import pr_agent.algo.token_budget as token_budget
from pr_agent.algo import (
    _CLAUDE_MODEL_FAMILIES,
    MAX_TOKENS,
    _generate_claude_registries,
    _validate_claude_model_family,
)
from pr_agent.algo.token_budget import get_max_tokens

CLAUDE_EXTENDED_THINKING_MODELS = _generate_claude_registries()[1]


def _settings(custom_model_max_tokens=0, max_model_tokens=0):
    return type("", (), {
        "config": type("", (), {
            "custom_model_max_tokens": custom_model_max_tokens, "max_model_tokens": max_model_tokens,
        })(),
    })()


class TestGetMaxTokens:

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol", "gpt-5.6"])
    def test_registered_models_keep_registry_priority(self, monkeypatch, model):
        settings = _settings(custom_model_max_tokens=16000)
        monkeypatch.setattr(token_budget, "get_settings", lambda: settings)

        assert get_max_tokens(model) == 1050000

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol", "gpt-5.6"])
    @pytest.mark.parametrize("custom_limit", [0, -1])
    def test_without_override_keeps_registered_limit(self, monkeypatch, model, custom_limit):
        settings = _settings(custom_model_max_tokens=custom_limit)
        monkeypatch.setattr(token_budget, "get_settings", lambda: settings)

        assert get_max_tokens(model) == 1050000

    # Test if the file is in MAX_TOKENS
    def test_model_max_tokens(self, monkeypatch):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        model = "gpt-3.5-turbo"
        expected = MAX_TOKENS[model]

        assert get_max_tokens(model) == expected

    @pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.4-2026-03-05"])
    def test_gpt54_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 272000

    @pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5.4-mini-2026-03-17"])
    def test_gpt54_mini_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 400000

    @pytest.mark.parametrize("model", ["gpt-5.4-nano", "gpt-5.4-nano-2026-03-17"])
    def test_gpt54_nano_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 400000

    @pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    def test_gpt56_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1050000

    @pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
    @pytest.mark.parametrize("prefix", [
        "", "openai/", "azure/", "azure/openai/", "openrouter/", "openrouter/openai/",
    ])
    @pytest.mark.parametrize("suffix", ["", "_thinking"])
    @pytest.mark.parametrize("cap", [0, 32000])
    def test_gpt6_model_max_tokens(self, monkeypatch, model, prefix, suffix, cap):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": cap,
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(f"{prefix}{model}{suffix}") == (cap or 1050000)

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
    @pytest.mark.parametrize("suffix", ["", "_thinking"])
    @pytest.mark.parametrize("custom_limit, expected", [(0, 1050000), (128000, 128000)])
    def test_azure_ai_gpt6_max_tokens(self, monkeypatch, model, suffix, custom_limit, expected):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": custom_limit,
                "max_model_tokens": 0,
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(f"azure_ai/{model}{suffix}") == expected

    @pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
    @pytest.mark.parametrize("alias", [
        "openai/{}", "azure/{}", "azure/openai/{}_thinking", "{}_thinking",
        "openrouter/{}", "openrouter/openai/{}_thinking",
    ])
    @pytest.mark.parametrize("cap, expected", [(0, 128000), (32000, 32000)])
    def test_gpt6_alias_preserves_custom_limit(self, monkeypatch, model, alias, cap, expected):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 128000,
                "max_model_tokens": cap,
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(alias.format(model)) == expected

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
    @pytest.mark.parametrize("route", [":nitro", ":floor", ":online", ":exacto"])
    @pytest.mark.parametrize("custom_limit, expected", [(0, 1050000), (128000, 128000)])
    def test_gpt6_thinking_route_alias_max_tokens(self, monkeypatch, model, route, custom_limit, expected):
        fake_settings = _settings(custom_model_max_tokens=custom_limit)
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(f"openrouter/openai/{model}_thinking{route}") == expected

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
    @pytest.mark.parametrize("route", [":nitro", ":floor", ":online", ":exacto"])
    def test_non_openrouter_gpt6_variant_needs_custom_limit(self, monkeypatch, model, route):
        """A routing suffix outside an openrouter/ model is not a GPT-6 alias, so it needs a custom limit."""
        variant = f"{model}{route}"
        monkeypatch.setattr(token_budget, "get_settings", lambda: _settings())
        with pytest.raises(Exception, match="Unable to resolve a positive input-token limit"):
            get_max_tokens(variant)

        monkeypatch.setattr(token_budget, "get_settings", lambda: _settings(custom_model_max_tokens=16384))
        assert get_max_tokens(variant) == 16384

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
    @pytest.mark.parametrize("route", [":batch", ":free"])
    def test_gpt6_thinking_nonrouting_variant_uses_provider_metadata(self, monkeypatch, model, route):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0,
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        with pytest.raises(Exception, match="Unable to resolve a positive input-token limit"):
            get_max_tokens(f"openrouter/openai/{model}_thinking{route}")

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
    @pytest.mark.parametrize("alias", ["{}_thinking", "openai/{}", "openai/{}_thinking"])
    def test_gpt6_alias_preserves_registered_limit(self, monkeypatch, model, alias):
        monkeypatch.setattr(token_budget, "get_settings", lambda: _settings())

        assert get_max_tokens(alias.format(model)) == 1050000

    @pytest.mark.parametrize("model", [
        "gpt-5.4-mini_thinking",
        "gpt-5.6_thinking",
        "gpt-5.6-sol_thinking",
        "gpt-5.6-terra_thinking",
        "gpt-5.6-luna_thinking",
    ])
    def test_gpt5_thinking_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)
        assert get_max_tokens(model) == MAX_TOKENS[model.removesuffix("_thinking")]

    @pytest.mark.parametrize("model", [
        "gpt-5.6_thinking",
        "openai/gpt-5.6_thinking",
        "azure/gpt-5.6_thinking",
    ])
    def test_gpt5_thinking_alias_preserves_custom_limit(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 7000,
                "max_model_tokens": 0
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 7000

    @pytest.mark.parametrize("custom_limit", [
        pytest.param(0, id="normalized-registry"),
        pytest.param(9000, id="custom-limit"),
    ])
    def test_gpt5_thinking_alias_respects_max_model_tokens_cap(self, monkeypatch, custom_limit):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": custom_limit,
                "max_model_tokens": 7000
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens("openai/gpt-5.6_thinking") == 7000

    @pytest.mark.parametrize("model", [
        "azure/gpt-5.6_thinking",
        "openai/gpt-5.6-sol_thinking",
        "azure/gpt-5.6-terra_thinking",
        "openai/gpt-5.6-luna_thinking",
        "azure/openai/gpt-5.6_thinking",
    ])
    def test_gpt5_thinking_prefixed_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)
        # Strip provider prefixes then _thinking suffix to get base key
        tmp = model
        while tmp.startswith(("openai/", "azure/")):
            tmp = tmp.removeprefix("openai/").removeprefix("azure/")
        base = tmp.removesuffix("_thinking")
        assert get_max_tokens(model) == MAX_TOKENS[base]

    def test_non_gpt5_thinking_model_max_tokens_not_stripped(self, monkeypatch):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)
        with pytest.raises(Exception, match="Unable to resolve a positive input-token limit"):
            get_max_tokens("gpt-4o_thinking")

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openrouter/auto", 2000000),
            ("openrouter/free", 200000),
            ("openrouter/fusion", 1000000),
            ("openrouter/pareto-code", 2000000),
        ],
    )
    @pytest.mark.parametrize("custom_limit", [0, 20000])
    @pytest.mark.parametrize("cap", [0, 32000])
    def test_openrouter_router_model_max_tokens(self, monkeypatch, model, expected, custom_limit, cap):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": custom_limit,
                "max_model_tokens": cap,
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == (min(expected, cap) if cap else expected)
        assert get_max_tokens(model, ignore_max_model_tokens=True) == expected

    # Test situations where the model is not registered and exists as a custom model
    def test_model_has_custom(self, monkeypatch):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 5000,
                'max_model_tokens': 0  # 제한 없음
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        model = "custom-model"
        expected = 5000

        assert get_max_tokens(model) == expected

    @pytest.mark.parametrize("model", [
        "gpt-5.2-codex",
        "gpt-5.3-codex",
    ])
    def test_gpt_codex_models_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == MAX_TOKENS[model]

    def test_model_not_max_tokens_and_not_has_custom(self, monkeypatch):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        model = "custom-model"

        with pytest.raises(Exception, match="Unable to resolve a positive input-token limit"):
            get_max_tokens(model)

    def test_model_max_tokens_with__limit(self, monkeypatch):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 10000
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        model = "gpt-3.5-turbo"  # this model setting is 160000
        expected = 10000

        assert get_max_tokens(model) == expected

    @pytest.mark.parametrize("model", [
        "gemini/gemini-3.1-flash",
        "vertex_ai/gemini-3.1-flash",
        "gemini/gemini-3.1-pro",
        "vertex_ai/gemini-3.1-pro",
        "gemini/gemini-3.5-pro",
        "vertex_ai/gemini-3.5-pro",
        "gemini/gemini-3.8-flash",
        "vertex_ai/gemini-3.8-flash",
    ])
    def test_gemini_3_x_models_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0,
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)
        assert get_max_tokens(model) == 1048576

    @pytest.mark.parametrize("model", [
        "gemini/gemini-4-argon",
        "vertex_ai/gemini-4-argon",
    ])
    def test_gemini_4_x_models_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0,
            })()
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)
        assert get_max_tokens(model) == 1048576

    def test_bedrock_mantle_grok_4_3_model_max_tokens(self, monkeypatch):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0,
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens("bedrock_mantle/xai.grok-4.3") == 1000000

    @pytest.mark.parametrize("model", [
        "bedrock/moonshotai.kimi-k3",
        "bedrock/us.moonshotai.kimi-k3",
        "bedrock/global.moonshotai.kimi-k3",
        "bedrock/converse/moonshotai.kimi-k3",
        "bedrock/converse/us.moonshotai.kimi-k3",
        "bedrock/converse/global.moonshotai.kimi-k3",
    ])
    def test_bedrock_kimi_k3_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0,
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize("model", [
        "xai/grok-build-latest",
        "openrouter/x-ai/grok-4.5",
        "openrouter/x-ai/grok-4.6",
    ])
    def test_xai_grok_build_and_openrouter_grok_4_5_and_4_6_models_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0,
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 500000

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-4-8",
            "claude-opus-4-8",
            "vertex_ai/claude-opus-4-8",
            "bedrock/anthropic.claude-opus-4-8",
            "bedrock/global.anthropic.claude-opus-4-8",
            "bedrock/us.anthropic.claude-opus-4-8",
            "bedrock/eu.anthropic.claude-opus-4-8",
            "bedrock/au.anthropic.claude-opus-4-8",
            "bedrock/jp.anthropic.claude-opus-4-8",
        ],
    )
    def test_claude_opus_4_8_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-5",
            "claude-opus-5",
            "vertex_ai/claude-opus-5",
            "bedrock/anthropic.claude-opus-5",
            "bedrock/global.anthropic.claude-opus-5",
            "bedrock/us.anthropic.claude-opus-5",
            "bedrock/eu.anthropic.claude-opus-5",
            "bedrock/au.anthropic.claude-opus-5",
            "bedrock/jp.anthropic.claude-opus-5",
            "anthropic/claude-opus-5-5",
            "claude-opus-5-5",
            "vertex_ai/claude-opus-5-5",
            "bedrock/anthropic.claude-opus-5-5",
            "bedrock/global.anthropic.claude-opus-5-5",
            "bedrock/us.anthropic.claude-opus-5-5",
            "bedrock/eu.anthropic.claude-opus-5-5",
            "bedrock/au.anthropic.claude-opus-5-5",
            "bedrock/jp.anthropic.claude-opus-5-5",
        ],
    )
    def test_claude_opus_5_family_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-4-7",
            "claude-opus-4-7",
            "vertex_ai/claude-opus-4-7",
            "bedrock/anthropic.claude-opus-4-7",
            "bedrock/global.anthropic.claude-opus-4-7",
            "bedrock/us.anthropic.claude-opus-4-7",
        ],
    )
    def test_claude_opus_4_7_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-4-6",
            "claude-opus-4-6",
            "vertex_ai/claude-opus-4-6",
            "bedrock/anthropic.claude-opus-4-6-v1:0",
            "bedrock/global.anthropic.claude-opus-4-6-v1:0",
            "bedrock/us.anthropic.claude-opus-4-6-v1:0",
        ],
    )
    def test_claude_opus_4_6_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-5",
            "claude-sonnet-5",
            "vertex_ai/claude-sonnet-5",
            "bedrock/anthropic.claude-sonnet-5",
            "bedrock/global.anthropic.claude-sonnet-5",
            "bedrock/us.anthropic.claude-sonnet-5",
            "bedrock/au.anthropic.claude-sonnet-5",
            "bedrock/eu.anthropic.claude-sonnet-5",
            "bedrock/jp.anthropic.claude-sonnet-5",
            "anthropic/claude-sonnet-5-5",
            "claude-sonnet-5-5",
            "vertex_ai/claude-sonnet-5-5",
            "bedrock/anthropic.claude-sonnet-5-5",
            "bedrock/global.anthropic.claude-sonnet-5-5",
            "bedrock/us.anthropic.claude-sonnet-5-5",
            "bedrock/au.anthropic.claude-sonnet-5-5",
            "bedrock/eu.anthropic.claude-sonnet-5-5",
        ],
    )
    def test_claude_sonnet_5_family_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-haiku-5-5",
            "claude-haiku-5-5",
            "vertex_ai/claude-haiku-5-5",
            "bedrock/anthropic.claude-haiku-5-5",
            "bedrock/global.anthropic.claude-haiku-5-5",
            "bedrock/us.anthropic.claude-haiku-5-5",
            "bedrock/au.anthropic.claude-haiku-5-5",
            "bedrock/eu.anthropic.claude-haiku-5-5",
            "bedrock/jp.anthropic.claude-haiku-5-5",
        ],
    )
    def test_claude_haiku_5_5_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-fable-5-1",
            "claude-fable-5-1",
            "vertex_ai/claude-fable-5-1",
            "bedrock/anthropic.claude-fable-5-1",
            "bedrock/global.anthropic.claude-fable-5-1",
            "bedrock/us.anthropic.claude-fable-5-1",
        ],
    )
    def test_claude_fable_5_1_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0,
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4-6",
            "claude-sonnet-4-6",
            "vertex_ai/claude-sonnet-4-6",
            "bedrock/anthropic.claude-sonnet-4-6",
            "bedrock/global.anthropic.claude-sonnet-4-6",
            "bedrock/us.anthropic.claude-sonnet-4-6",
            "bedrock/au.anthropic.claude-sonnet-4-6",
            "bedrock/eu.anthropic.claude-sonnet-4-6",
            "bedrock/jp.anthropic.claude-sonnet-4-6",
        ],
    )
    def test_claude_sonnet_4_6_model_max_tokens(self, monkeypatch, model):
        fake_settings = type('', (), {
            'config': type('', (), {
                'custom_model_max_tokens': 0,
                'max_model_tokens': 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "zai/glm-5.2",
        ],
    )
    def test_zai_glm_5_2_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        # 1M per LiteLLM (issue #3196): zai/glm-5.2 was understated at 200000.
        assert get_max_tokens(model) == 1000000

    @pytest.mark.parametrize(
        "model",
        [
            "dashscope/qwen3.8-max",
        ],
    )
    def test_qwen_3_8_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1000000



    @pytest.mark.parametrize(
        "model",
        [
            "xiaomi_mimo/mimo-v2.5",
        ],
    )
    def test_xiaomi_mimo_v2_5_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1048576

    @pytest.mark.parametrize(
        "model",
        [
            "xiaomi_mimo/mimo-v2.5-pro",
        ],
    )
    def test_xiaomi_mimo_v2_5_pro_model_max_tokens(self, monkeypatch, model):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 0,
                "max_model_tokens": 0
            })()
        })()

        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 1048576

    @pytest.mark.parametrize("metadata_model", ["fake-provider/fake-unknown-model", "anthropic/not-a-claude-model"])
    def test_unknown_model_failure_explains_input_limit(self, monkeypatch, metadata_model):
        """Explain that an unregistered model needs config.custom_model_max_tokens."""
        monkeypatch.setattr(token_budget, "get_settings", lambda: _settings())

        with pytest.raises(Exception, match="Unable to resolve a positive input-token limit") as error:
            get_max_tokens(metadata_model)
        message = str(error.value)
        assert metadata_model in message
        assert "built-in limits" in message
        assert "LiteLLM" not in message
        assert "config.custom_model_max_tokens to a verified positive input-token limit" in message
        assert "config.max_model_tokens only caps a resolved limit" in message

    def test_claude_opus_4_8_family_expansion_and_capabilities(self):
        """Independent verification of Opus 4.8 generated aliases and capabilities."""
        expected_aliases = {
            "claude-opus-4-8",
            "anthropic/claude-opus-4-8",
            "vertex_ai/claude-opus-4-8",
            "bedrock/anthropic.claude-opus-4-8",
            "bedrock/global.anthropic.claude-opus-4-8",
            "bedrock/us.anthropic.claude-opus-4-8",
            "bedrock/eu.anthropic.claude-opus-4-8",
            "bedrock/au.anthropic.claude-opus-4-8",
            "bedrock/jp.anthropic.claude-opus-4-8",
        }

        # Exact set equality against filtered MAX_TOKENS
        actual_aliases = {k for k in MAX_TOKENS if "claude-opus-4-8" in k}
        assert actual_aliases == expected_aliases

        # Context window tokens
        for alias in expected_aliases:
            assert MAX_TOKENS[alias] == 1000000

        # Capability lists
        for alias in expected_aliases:
            assert alias not in CLAUDE_EXTENDED_THINKING_MODELS

        # Negative checks to ensure no over-generation
        assert "bedrock/apac.anthropic.claude-opus-4-8" not in MAX_TOKENS

    def test_claude_sonnet_5_5_family_expansion_and_capabilities(self):
        """Independent verification of Sonnet 5.5 generated aliases and capabilities."""
        expected_aliases = {
            "claude-sonnet-5-5",
            "anthropic/claude-sonnet-5-5",
            "vertex_ai/claude-sonnet-5-5",
            "bedrock/anthropic.claude-sonnet-5-5",
            "bedrock/global.anthropic.claude-sonnet-5-5",
            "bedrock/us.anthropic.claude-sonnet-5-5",
            "bedrock/eu.anthropic.claude-sonnet-5-5",
            "bedrock/au.anthropic.claude-sonnet-5-5",
        }

        # Exact set equality against filtered MAX_TOKENS
        actual_aliases = {k for k in MAX_TOKENS if "claude-sonnet-5-5" in k}
        assert actual_aliases == expected_aliases

        # Context window tokens
        for alias in expected_aliases:
            assert MAX_TOKENS[alias] == 1000000

        # Capability lists
        for alias in expected_aliases:
            assert alias not in CLAUDE_EXTENDED_THINKING_MODELS

        # Negative checks to ensure no over-generation
        assert "bedrock/jp.anthropic.claude-sonnet-5-5" not in MAX_TOKENS
        assert "bedrock/apac.anthropic.claude-sonnet-5-5" not in MAX_TOKENS

    _OPUS_4_6_MAX_TOKENS_ALIASES = {
        "claude-opus-4-6",
        "anthropic/claude-opus-4-6",
        "vertex_ai/claude-opus-4-6",
        "bedrock/anthropic.claude-opus-4-6-v1:0",
        "bedrock/global.anthropic.claude-opus-4-6-v1:0",
        "bedrock/eu.anthropic.claude-opus-4-6-v1:0",
        "bedrock/au.anthropic.claude-opus-4-6-v1:0",
        "bedrock/jp.anthropic.claude-opus-4-6-v1:0",
        "bedrock/apac.anthropic.claude-opus-4-6-v1:0",
        "bedrock/us.anthropic.claude-opus-4-6-v1:0",
        "claude-opus-4-6-20260120",
        "anthropic/claude-opus-4-6-20260120",
        "vertex_ai/claude-opus-4-6@20260120",
        "bedrock/anthropic.claude-opus-4-6-20260120-v1:0",
        "bedrock/us.anthropic.claude-opus-4-6-20260120-v1:0",
    }
    # Extended thinking: 9 aliases
    _OPUS_4_6_THINKING_ALIASES = {
        "anthropic/claude-opus-4-6",
        "claude-opus-4-6",
        "vertex_ai/claude-opus-4-6",
        "bedrock/anthropic.claude-opus-4-6-v1:0",
        "bedrock/us.anthropic.claude-opus-4-6-v1:0",
        "bedrock/au.anthropic.claude-opus-4-6-v1:0",
        "bedrock/eu.anthropic.claude-opus-4-6-v1:0",
        "bedrock/jp.anthropic.claude-opus-4-6-v1:0",
        "bedrock/global.anthropic.claude-opus-4-6-v1:0",
    }

    def test_claude_opus_4_6_family_registers_expected_aliases(self):
        """Independent verification of Opus 4.6 generated aliases and capabilities."""
        # Exact set equality against filtered MAX_TOKENS
        actual_aliases = {k for k in MAX_TOKENS if "claude-opus-4-6" in k}
        assert actual_aliases == self._OPUS_4_6_MAX_TOKENS_ALIASES

    def test_claude_opus_4_6_aliases_use_1m_context_window(self):
        # Context window tokens
        for alias in self._OPUS_4_6_MAX_TOKENS_ALIASES:
            assert MAX_TOKENS[alias] == 1000000

    def test_claude_opus_4_6_extended_thinking_aliases(self):
        actual_thinking_aliases = {k for k in CLAUDE_EXTENDED_THINKING_MODELS if "claude-opus-4-6" in k}
        assert actual_thinking_aliases == self._OPUS_4_6_THINKING_ALIASES

    def test_claude_opus_4_6_apac_and_dated_variants_lack_extended_thinking(self):
        # Negative checks: apac and dated variants NOT in extended thinking
        assert "bedrock/apac.anthropic.claude-opus-4-6-v1:0" not in CLAUDE_EXTENDED_THINKING_MODELS
        assert "claude-opus-4-6-20260120" not in CLAUDE_EXTENDED_THINKING_MODELS
        assert "anthropic/claude-opus-4-6-20260120" not in CLAUDE_EXTENDED_THINKING_MODELS
        assert "vertex_ai/claude-opus-4-6@20260120" not in CLAUDE_EXTENDED_THINKING_MODELS

    @pytest.fixture
    def family_with_unknown_top_level_key(self):
        bad_fam = dict(_CLAUDE_MODEL_FAMILIES[0])
        bad_fam["bedrock_region"] = ("global", "us")
        return bad_fam

    @pytest.fixture
    def family_with_unknown_nested_alias_key(self):
        return {
            "model_id": "claude-opus-4-6",
            "extra_aliases": {
                "some/alias": {
                    "max_tokens": 200000,
                    "extended_thinkin": True,
                }
            },
        }

    def test_shipped_claude_model_families_pass_validation(self):
        # All current shipped families must pass validation
        for fam in _CLAUDE_MODEL_FAMILIES:
            _validate_claude_model_family(fam)

    def test_validator_rejects_unknown_top_level_family_key(self, family_with_unknown_top_level_key):
        # 1. Copied family with unknown top-level key must raise ValueError
        bad_fam = family_with_unknown_top_level_key

        with pytest.raises(ValueError) as exc_info:
            _validate_claude_model_family(bad_fam)

        err_msg = str(exc_info.value)
        assert "bedrock_region" in err_msg
        assert bad_fam["model_id"] in err_msg

    def test_generator_rejects_unknown_top_level_family_key(self, family_with_unknown_top_level_key):
        # Generator must also reject the unknown top-level key when expanding
        bad_fam = family_with_unknown_top_level_key

        with pytest.raises(ValueError) as exc_info_gen:
            _generate_claude_registries(families=[bad_fam])
        assert "bedrock_region" in str(exc_info_gen.value)
        assert bad_fam["model_id"] in str(exc_info_gen.value)

        # Ensure global families list was not mutated
        assert "bedrock_region" not in _CLAUDE_MODEL_FAMILIES[0]

    def test_validator_rejects_unknown_nested_alias_key(self, family_with_unknown_nested_alias_key):
        # 2. Family with unknown nested extra_aliases key must raise ValueError
        bad_nested_fam = family_with_unknown_nested_alias_key

        with pytest.raises(ValueError) as exc_info_nested:
            _validate_claude_model_family(bad_nested_fam)

        err_nested = str(exc_info_nested.value)
        assert "extended_thinkin" in err_nested
        assert "some/alias" in err_nested
        assert "claude-opus-4-6" in err_nested

    def test_generator_rejects_unknown_nested_alias_key(self, family_with_unknown_nested_alias_key):
        # Generator must also reject the unknown nested key
        bad_nested_fam = family_with_unknown_nested_alias_key

        with pytest.raises(ValueError) as exc_info_gen_nested:
            _generate_claude_registries(families=[bad_nested_fam])
        assert "extended_thinkin" in str(exc_info_gen_nested.value)
        assert "some/alias" in str(exc_info_gen_nested.value)
        assert "claude-opus-4-6" in str(exc_info_gen_nested.value)

    def test_claude_exceptional_extra_alias_capabilities(self):
        """Prove exceptional extra-alias capability routing.

        Proves:
        - scalar int extra_aliases only populate token counts.
        - structured extra_aliases route to extended_thinking when flagged.
        - capability flags do not bleed to unrelated aliases.
        """
        synthetic_family = {
            "model_id": "test-claude-synthetic",
            "max_tokens": 500000,
            "bedrock": False,
            "vertex": False,
            "extra_aliases": {
                "test/extra-token-only": 500000,
                "test/extra-thinking": {
                    "max_tokens": 500000,
                    "extended_thinking": True,
                },
            },
        }

        tokens, thinking = _generate_claude_registries(families=[synthetic_family])

        # Token verification
        assert tokens["test-claude-synthetic"] == 500000
        assert tokens["anthropic/test-claude-synthetic"] == 500000
        assert tokens["test/extra-token-only"] == 500000
        assert tokens["test/extra-thinking"] == 500000

        # Extended-thinking capability routing (no bleed)
        assert "test/extra-thinking" in thinking
        assert "test/extra-token-only" not in thinking

    def test_claude_registries_baseline_parity_and_no_duplicates(self):
        """Verify baseline capability parity and that no duplicate entries exist."""
        # Capability lists have no duplicates
        assert len(CLAUDE_EXTENDED_THINKING_MODELS) == len(set(CLAUDE_EXTENDED_THINKING_MODELS))

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("opus", 1000000),
            ("claude-sonnet-5-5", 1000000),
            ("custom-model", 128000),
        ],
    )
    def test_ignore_max_model_tokens_returns_unreduced_value(self, monkeypatch, model, expected):
        """Sites that bypass the max_model_tokens clamp (pr_help_message, pr_help_docs,
        pr_code_suggestions) get the raw model context, while the default path keeps
        clamping to config.max_model_tokens."""
        fake_settings = _settings(custom_model_max_tokens=128000, max_model_tokens=32000)
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)

        assert get_max_tokens(model) == 32000
        assert get_max_tokens(model, ignore_max_model_tokens=True) == expected


class TestCorrectedClaudeWindows:

    @pytest.mark.parametrize("model, window", [
        ("anthropic/claude-sonnet-4-6", 1000000),
        ("bedrock/global.anthropic.claude-sonnet-4-6", 1000000),
        ("vertex_ai/claude-opus-4-6", 1000000),
        ("claude-opus-4-6-20260120", 1000000),
    ])
    @pytest.mark.parametrize("cap", [0, 150000, 1100000])
    def test_static_window_precedence_and_cap(self, monkeypatch, model, window, cap):
        fake_settings = type("", (), {
            "config": type("", (), {
                "custom_model_max_tokens": 12345,
                "max_model_tokens": cap,
            })(),
        })()
        monkeypatch.setattr(token_budget, "get_settings", lambda: fake_settings)
        assert MAX_TOKENS[model] == window
        assert get_max_tokens(model) == (min(window, cap) if cap else window)
        assert get_max_tokens(model, ignore_max_model_tokens=True) == window


class TestClaudeCodeAliases:

    @pytest.mark.parametrize("alias, resolved", [
        ("opus", "claude-opus-5-5"),
        ("default", "claude-opus-5-5"),
        ("sonnet", "claude-sonnet-5-5"),
        ("haiku", "claude-haiku-5-5"),
        ("fable", "claude-fable-5-1"),
        ("best", "claude-fable-5-1"),
        ("opus[1m]", "claude-opus-5-5"),
        ("anthropic/sonnet", "claude-sonnet-5-5"),
        ("anthropic/claude-haiku-5-5", "claude-haiku-5-5"),
        ("claude-opus-5-5[1m]", "claude-opus-5-5"),
    ])
    @pytest.mark.parametrize("cap", [0, 64000])
    def test_alias_uses_the_selected_models_window(self, monkeypatch, alias, resolved, cap):
        monkeypatch.setattr(token_budget, "get_settings", lambda: _settings(max_model_tokens=cap))

        assert get_max_tokens(alias) == (min(MAX_TOKENS[resolved], cap) if cap else MAX_TOKENS[resolved])
        assert get_max_tokens(alias, ignore_max_model_tokens=True) == MAX_TOKENS[resolved]

    def test_registered_alias_ignores_custom_limit(self, monkeypatch):
        monkeypatch.setattr(token_budget, "get_settings", lambda: _settings(custom_model_max_tokens=5000))

        assert get_max_tokens("sonnet") == MAX_TOKENS["claude-sonnet-5-5"]

    @pytest.mark.parametrize("model", ["claude-unreleased-9", "anthropic/claude-unreleased-9[1m]"])
    def test_unregistered_claude_model_needs_custom_limit(self, monkeypatch, model):
        monkeypatch.setattr(token_budget, "get_settings", lambda: _settings())
        with pytest.raises(Exception, match="Unable to resolve a positive input-token limit"):
            get_max_tokens(model)

        monkeypatch.setattr(token_budget, "get_settings", lambda: _settings(custom_model_max_tokens=200000))
        assert get_max_tokens(model) == 200000
