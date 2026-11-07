from math import ceil, isfinite
from threading import Lock

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from tiktoken import get_encoding

from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

# Claude has no offline tokenizer, so counts are o200k_base tiktoken counts scaled by
# config.model_token_count_estimate_factor. Claude 5.5 models produced 1.40-1.68x the o200k count
# on this repository's diffs, docs and prompts (1.62-1.68x on anything over 1k tokens).
DEFAULT_CLAUDE_TOKEN_FACTOR = 0.7
_warned_factors: set = set()


def _claude_token_multiplier() -> float:
    raw_factor = get_settings().get("config.model_token_count_estimate_factor", DEFAULT_CLAUDE_TOKEN_FACTOR)
    try:
        multiplier = 1 + float(raw_factor)
    except (TypeError, ValueError, OverflowError):
        multiplier = None
    if multiplier is None or isinstance(raw_factor, bool) or not isfinite(multiplier) or not multiplier > 0:
        if repr(raw_factor) not in _warned_factors:
            _warned_factors.add(repr(raw_factor))
            get_logger().warning(
                f"model_token_count_estimate_factor is not a usable number ({raw_factor!r}), "
                f"using {DEFAULT_CLAUDE_TOKEN_FACTOR}")
        multiplier = 1 + DEFAULT_CLAUDE_TOKEN_FACTOR
    return multiplier


def estimate_tokens(text: str) -> int:
    """Estimate the Claude token count of ``text``."""
    if not text:
        return 0
    raw = len(TokenEncoder.get_token_encoder().encode(text, disallowed_special=()))
    try:
        return ceil(raw * _claude_token_multiplier())
    except (OverflowError, ValueError):
        return raw


class TokenEncoder:
    """Process-wide o200k_base tiktoken encoder used for every token estimate."""

    _encoder_instance = None
    _lock = Lock()

    @classmethod
    def get_token_encoder(cls, model=None):
        with cls._lock:
            if cls._encoder_instance is None:
                cls._encoder_instance = get_encoding("o200k_base")
            return cls._encoder_instance


class TokenHandler:
    """
    A class for handling tokens in the context of a pull request.

    Attributes:
    - encoder: The o200k_base tiktoken encoder that token estimates are based on.
    - limit: The maximum number of tokens allowed for the given model, as defined in the MAX_TOKENS dictionary in the
      pr_agent.algo module.
    - prompt_tokens: The number of tokens in the system and user strings, as calculated by the _get_system_user_tokens
      method.
    """

    def __init__(self, pr=None, vars: dict | None = None, system="", user="", model=None, *,
                 count_prompt_tokens: bool = True):
        """
        Initializes the TokenHandler object.

        Args:
        - pr: The pull request object.
        - vars: A dictionary of variables.
        - system: The system string.
        - user: The user string.
        - model: Optional model name whose tokenizer should be used.
        - count_prompt_tokens: Whether to render and count the initial prompt during construction.
        """
        if vars is None:
            vars = {}
        self.model = model or get_settings().config.model
        self.pr = pr
        self.vars = vars
        self.system = system
        self.user = user
        self.prompt_tokens = 0
        self.encoder = TokenEncoder.get_token_encoder(self.model)

        if pr is not None and count_prompt_tokens:
            self.prompt_tokens = self._get_system_user_tokens(pr, self.encoder, vars, system, user)

    def for_model(self, model: str):
        """Return a handler bound to ``model`` without mutating this handler."""
        if model == self.model:
            return self
        return TokenHandler(self.pr, self.vars, self.system, self.user, model=model)

    def _get_system_user_tokens(self, pr, encoder, vars: dict, system, user):
        """
        Calculates the number of tokens in the system and user strings.

        Args:
        - pr: The pull request object.
        - encoder: Unused; kept for callers that pass the handler's encoder.
        - vars: A dictionary of variables.
        - system: The system string.
        - user: The user string.

        Returns:
        The sum of the number of tokens in the system and user strings.
        """
        try:
            environment = SandboxedEnvironment(undefined=StrictUndefined)
            system_prompt = environment.from_string(system).render(vars)
            user_prompt = environment.from_string(user).render(vars)
            return estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        except Exception as e:
            get_logger().error(f"Error in _get_system_user_tokens: {e}")
            return 0

    def count_tokens(self, patch: str, force_accurate: bool = False) -> int:
        """Return the estimated Claude token count of ``patch``.

        ``force_accurate`` is accepted for callers that ask for a careful count near a limit; every
        estimate already includes the Claude correction factor, so it changes nothing.
        """
        return estimate_tokens(patch)
