"""Maps MOSAICO's env-var contract onto pr-agent's Dynaconf settings. Every function is a no-op
unless the corresponding MOSAICO env var is set: importing or calling apply_mosaico_env() with no
MOSAICO env present changes nothing.

Model calls run through Claude Code, so MOSAICO's API_BASE and API_KEY (an OpenAI-compatible
gateway) are ignored; MODEL_NAME selects a Claude Code model slug."""
import os

from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

ENV_API_BASE = "API_BASE"
ENV_API_KEY = "API_KEY"
ENV_MODEL_NAME = "MODEL_NAME"
ENV_MODEL_MAX_TOKENS = "MODEL_MAX_TOKENS"
ENV_LANGFUSE_HOST = "LANGFUSE_HOST"
ENV_LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"


def langfuse_env_present() -> bool:
    return bool(
        os.getenv(ENV_LANGFUSE_HOST) and os.getenv(ENV_LANGFUSE_PUBLIC_KEY) and os.getenv(ENV_LANGFUSE_SECRET_KEY))


def apply_mosaico_env() -> None:
    """Idempotent. Call once at MOSAICO-server startup. Does nothing when MOSAICO env is absent."""
    settings = get_settings()
    if os.getenv(ENV_API_BASE) or os.getenv(ENV_API_KEY):
        get_logger().warning("MOSAICO: API_BASE/API_KEY are ignored; models run through Claude Code.")
    model_name = os.getenv(ENV_MODEL_NAME)
    if model_name:
        settings.set("CONFIG.MODEL", model_name)
        settings.set("CONFIG.FALLBACK_MODELS", [])
    max_tokens_env = os.getenv(ENV_MODEL_MAX_TOKENS)
    if max_tokens_env:
        try:
            custom_max_tokens = int(max_tokens_env)
        except ValueError:
            custom_max_tokens = 0
        if custom_max_tokens > 0:
            settings.set("CONFIG.CUSTOM_MODEL_MAX_TOKENS", custom_max_tokens)
        else:
            get_logger().warning(f"MOSAICO: ignoring MODEL_MAX_TOKENS={max_tokens_env!r}; expected a positive integer.")
