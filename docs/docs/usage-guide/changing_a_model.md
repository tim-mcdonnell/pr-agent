---
title: "Changing a Model"
sidebar_position: 8
---

## Changing a model in PR-Agent

Every model call runs through Claude Code (see [Claude Code integration](./claude_agent_sdk.md)), so models are
named with Claude Code model slugs:

| Slug | Selects |
| --- | --- |
| `opus` | the latest Opus model (Opus 5.5 on the Anthropic API) |
| `sonnet` | the latest Sonnet model (Sonnet 5.5) |
| `haiku` | the latest Haiku model (Haiku 5.5) |
| `fable`, `best` | Fable 5.1 where your plan has it |
| `default` | your account's default model |
| `opus[1m]`, `sonnet[1m]` | the 1M-context variant, where the base model does not already have it |
| `claude-opus-5-5`, `claude-sonnet-5-5`, ... | a pinned model version |

Aliases follow Claude Code: when a new Opus ships, `opus` moves to it. Pin a full model id to stay on one version.
An `anthropic/` prefix is accepted and dropped. Any other name, such as an OpenAI model, fails that attempt with a
"not a Claude Code model slug" error and the next fallback model is tried.

```toml
[config]
model = "opus"
fallback_models = ["sonnet"]
#model_weak = "sonnet"       # optional: /describe, /ask and /update_changelog use it
#model_reasoning = "opus"    # optional: /improve's self-reflection uses it
```

In the GitHub Action, the `model` and `fallback_models` inputs set these, so you can keep the slug in a secret or
variable (see [GitHub Action](../installation/github.md#run-as-a-github-action)). Elsewhere use `CONFIG__MODEL=sonnet`.

Keep `fallback_models` different from `model`, or set it to `[]` for no backup model. Claude Code already retries
rate limits, overload and connection errors on the same model, so a fallback only runs after Claude Code gives up,
the model refuses or returns nothing, or the call times out. Authentication and billing errors stop at once.

## Reasoning effort

```toml
[config]
reasoning_effort = "medium" # passed to Claude Code as its effort level
```

`low`, `medium`, `high`, `xhigh` and `max` are passed through; `minimal` becomes `low`, and `none` turns thinking off.
`config.temperature` has no effect, because Claude Code does not expose it.

## Token budget

PR-Agent compresses each PR to fit `config.max_model_tokens` (default 64,000 Claude tokens) of prompt context,
whatever larger window the model has. Claude has no offline tokenizer, so PR-Agent counts with tiktoken's
`o200k_base` encoding and scales the result by `1 + config.model_token_count_estimate_factor` (default 0.7).
Claude 5.5 models produced 1.40 to 1.68 times the tiktoken count on this repository's code, docs and prompts.

## Output token limit

```toml
[config]
max_output_tokens = 0 # 0 = Claude Code's default
```

A positive value is passed to Claude Code as `CLAUDE_CODE_MAX_OUTPUT_TOKENS`.

## Routing small pull requests to a cheaper model

Without routing, regular-model calls use `config.model`, however small the pull request. In this example,
the first rule sends tiny pull requests to Haiku; the second sends small ones to Sonnet. Larger pull requests
keep `config.model`, and `config.fallback_models` still apply after the selected primary:

```toml
[model_routing]
enable = true

[[model_routing.rules]]
max_hunks = 3
max_patch_chars = 10000
model = "haiku"

[[model_routing.rules]]
max_hunks = 15
max_files = 6
model = "sonnet"
```

Rules are checked in order, and the first one whose limits the pull request fits selects the primary model for
the call. A pull request that fits no rule uses `config.model`. Size is measured by the number of diff hunks
(`max_hunks`) and changed files (`max_files`) left after the `[ignore]` rules. Every git provider reports both,
and neither depends on a model's tokenizer, so a threshold means the same thing whichever model it selects.

A rule can also set `max_patch_chars`, an inclusive ceiling on the sum of Python character counts in the supplied
patch strings. All limits named by a rule must match. This can distinguish a long single-line change from a
short change with the same file and hunk counts. Headers and line endings count too: LF is one character,
CRLF is two, and a combining character counts separately from its base character.

If any file has a missing or non-string patch, or its content could not be fetched, the character count is
unknown. Rules with `max_patch_chars` are then skipped; later rules without it can still match. An explicit
empty patch counts as zero. An incomplete head file alone does not invalidate its supplied patch string.
The limit measures the text the provider supplies, including any normalization or undetectable clipping;
it does not measure tokens, complete files, or a guaranteed complete diff. Rules without this field keep
using their existing file and hunk limits.

The retry chain starts as `[config.model] + config.fallback_models`, with a deployment paired to each model.
A matched rule replaces the first pair. If the routed `(model, deployment)` pair also appears among the
configured fallbacks, the effective chain removes duplicate pairs while preserving their order and adds
the displaced configured primary pair at the end, unless it is already present. This avoids retrying the
same endpoint while retaining the configured primary in this collision case. With
`config.model = "opus"`, no deployment configured, and the first rule above:

| `config.fallback_models` | retry order if earlier attempts fail |
| --- | --- |
| `["haiku"]` | `haiku`, `opus` |
| `["haiku", "sonnet"]` | `haiku`, `sonnet`, `opus` |

The first successful attempt returns immediately; later entries are attempted only after fallback-eligible
failures. An empty fallback list adds no backup, and a noncolliding route still replaces the primary without
adding it back. The same model with distinct deployment IDs is treated as distinct endpoints. `/improve`
uses this effective chain to recover failed chunks without rerunning successful chunks. Its defensive
uniqueness checks remain: repeated complete pairs in unchanged, noncolliding or unrouted chains can still
disable chunk recovery; those chains are not deduplicated by this routing-specific repair.

Routing applies only to calls that ask for the regular model: `/review`, `/improve`, `/generate_labels` and
`/add_docs`. Tools that already use `config.model_weak` (`/describe`, `/ask`, `/update_changelog`) are left
alone, and a dedicated `config.model_reasoning` is still used for self-reflection. With `config.output_run_details`
enabled, the run details show which model a routed pull request ended up on.
