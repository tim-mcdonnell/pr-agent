"""Preflight checks that catch a broken deployment before a full run (#4320).

`pr-agent doctor` reports, without touching any git provider, whether the active
configuration can reach Claude through Claude Code: the model is a Claude model,
the bundled Claude Code binary is present, a credential is available, and a
completion succeeds. The exit code is 0 only when every check passes, so CI
setups can gate on it. Enabled output sinks receive an additional basic settings
check, without constructing sinks or probing their destinations.
"""

import os
import sys

from pr_agent.algo.ai_handlers.claude_agent_sdk_handler import ClaudeAgentSDKHandler, claude_code_cli_path
from pr_agent.algo.output_sinks import _check_output_sinks
from pr_agent.algo.token_budget import FallbackEligibleError
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.git_provider import redact_credentials
from pr_agent.log import get_logger

DOCTOR_COMMANDS = ("doctor",)
# Credentials Claude Code reads from the environment, in the order the docs recommend them for CI.
CLAUDE_CREDENTIAL_ENV_VARS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

_PING_SYSTEM = "You are a health check probe. Reply with exactly: ok"
_PING_USER = "ping"


def _check_config(model: str) -> tuple[bool, str]:
    if not model:
        return False, "config.model is empty; set it in configuration.toml or CONFIG__MODEL"
    try:
        ClaudeAgentSDKHandler.resolve_model(model)
    except FallbackEligibleError:
        return False, f"config.model = {model} is not a Claude Code model slug (e.g. 'opus', 'claude-sonnet-5-5')"
    return True, f"config.model = {model}"


def _check_cli() -> tuple[bool, str]:
    path = claude_code_cli_path()
    if path:
        return True, f"Claude Code binary found ({path})"
    return False, "Claude Code binary not found; reinstall claude-agent-sdk for this platform"


def _credential_source() -> str | None:
    return next((name for name in CLAUDE_CREDENTIAL_ENV_VARS if os.environ.get(name)), None)


def _check_credentials() -> tuple[bool, str]:
    source = _credential_source()
    if source:
        return True, f"credential found (env {source})"
    # A `claude` login on the host is also valid; the ping below decides.
    return True, "no credential in the environment; relying on an existing Claude Code login (ping decides)"


def missing_provider_key_env_vars() -> tuple[str, tuple[str, ...]] | None:
    """Return the credential env var names when none is set, else None.

    The GitHub Action gates on this before running a tool, since a runner has no `claude` login.
    """
    if _credential_source():
        return None
    return "claude", CLAUDE_CREDENTIAL_ENV_VARS


async def _check_model_ping(handler: ClaudeAgentSDKHandler, model: str) -> tuple[bool, str]:
    try:
        await handler.chat_completion(model, _PING_SYSTEM, _PING_USER)
    except Exception as error:
        # Errors can quote request details; redact before the text reaches stdout or a shareable log.
        reason = redact_credentials(str(error).split("\n")[0])[:200]
        get_logger().debug(f"Doctor model ping failed: {redact_credentials(repr(error))}")
        return False, f"model completion failed: {reason}"
    return True, "model completion succeeded"


def _print_report(report: str) -> None:
    try:
        print(report)
    except UnicodeEncodeError:
        # A redirected stdout uses the locale encoding (cp1252 on Windows), which cannot
        # encode the check marks; emit UTF-8 bytes instead of losing the report.
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is None:
            raise
        sys.stdout.flush()
        buffer.write((report + "\n").encode("utf-8"))
        buffer.flush()


async def run_doctor_checks() -> int:
    """Run every check, print one line per check, and return the process exit code."""
    model = str(get_settings().config.get("model", "") or "")
    results = [_check_config(model), _check_cli(), _check_credentials()]
    if all(ok for ok, _ in results):
        results.append(await _check_model_ping(ClaudeAgentSDKHandler(), model))
    try:
        sink_result = _check_output_sinks(get_settings().get("push_outputs", {}) or {})
    except Exception:
        # Settings operations can include secret-bearing values in their exceptions.
        sink_result = False, "push_outputs configuration could not be checked"
    if sink_result is not None:
        results.append(sink_result)

    lines = [f"{'✓' if ok else '✗'} {detail}" for ok, detail in results]
    passed = sum(1 for ok, _ in results if ok)
    lines.append(f"{passed} of {len(results)} checks passed")
    _print_report("\n".join(lines))
    return 0 if passed == len(results) else 1
