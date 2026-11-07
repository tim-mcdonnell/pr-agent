---
title: "Claude Code integration"
sidebar_position: 14
---

## How PR-Agent calls Claude

Every model call runs through the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), which
launches the Claude Code binary bundled in the `claude-agent-sdk` wheel. Node.js is not required.

PR-Agent's pipeline is otherwise its own: it fetches the PR, compresses the diff to the token budget, renders its
prompts and parses the answer. Each call is a Claude Code session that:

- uses PR-Agent's system prompt in place of Claude Code's;
- loads no filesystem settings, `CLAUDE.md`, skills, hooks or MCP servers, from the host or from the reviewed
  repository;
- is bounded by `config.ai_timeout` (default 300 seconds).

Model names are Claude Code slugs such as `opus` or `sonnet[1m]`; see [Changing a model](./changing_a_model.md).

## Authentication

Claude Code authenticates the request itself. Supply one of:

- `CLAUDE_CODE_OAUTH_TOKEN`: a one-year token from `claude setup-token`, billed to your Claude subscription. When
  it is set, PR-Agent blanks `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` for the Claude Code process, because
  Claude Code would otherwise prefer them and bill the API instead.
- `ANTHROPIC_API_KEY`: a Claude Console API key.
- An existing `claude` login on the host, for local CLI runs.

The GitHub Action takes these as its `claude_code_oauth_token` and `anthropic_api_key` inputs, and stops before
running any tool when neither is available. `pr-agent doctor` checks the model slug, the binary and the credential,
then sends one ping.

:::note
Subscription credentials are for your own use of Claude Code. Anthropic's
[terms](https://code.claude.com/docs/en/legal-and-compliance#authentication-and-credential-use) do not allow
routing requests through Free, Pro, Max or Ultra plan credentials on behalf of other users. Use an API key for a
PR-Agent deployment that serves other people's repositories.
:::

## Read-only repository exploration

When a checkout of the PR is available, calls for `/review`, `/improve` and `/ask` let the model use Claude Code's
Read, Grep and Glob tools on it, so it can check callers, definitions and tests that the diff does not show.
Other commands, such as `/describe`, stay single-turn and tool-less.

```toml
[claude_code]
explore_repo = true
explore_commands = ["review", "auto_review", "improve", "ask", "ask_line"]
repo_path = ""   # empty: $GITHUB_WORKSPACE inside GitHub Actions; otherwise exploration is off
max_turns = 20   # per model call, including the final answer
```

In the GitHub Action, add an `actions/checkout` step of the PR before PR-Agent (see the
[workflow example](../installation/github.md#run-as-a-github-action)). Without a checkout, PR-Agent works from the
diff alone, as before. Every key in `[claude_code]` is host-only: a repository's `.pr_agent.toml` cannot change it.

The model reads the repository's tracked files and nothing else:

- It never sees the checkout itself. PR-Agent extracts `git archive HEAD` into a temporary directory, once per
  commit, and Claude Code runs there. The copy has no `.git` (so no token `actions/checkout` persisted), none of the
  untracked files earlier CI steps leave in the workspace (such as cloud credential files), and no symlinks that
  point outside it.
- Only Read, Grep and Glob are offered. There is no Bash, Edit, Write, web or MCP tool.
- Claude Code runs in `dontAsk` mode with no pre-approved tools, so reads outside the copy are denied rather than
  prompted.
- A PreToolUse hook resolves every requested path and pattern, following symlinks, and denies anything outside the
  copy, any `.git` path in any letter case, parent references and paths hidden in brace groups. It denies the call if
  it cannot parse the input.
- The Claude Code process gets blank `GITHUB_TOKEN`, `GH_TOKEN`, `GITLAB_TOKEN` and OpenAI key variables.

`persist-credentials: false` is still worth setting on the checkout, as defense in depth.

Repository content is still untrusted input. The model is told to treat it as data, but a pull request can try to
steer the review, so keep relying on human review for merge decisions.

## Token counting

Claude has no offline tokenizer. PR-Agent counts with tiktoken's `o200k_base` encoding and multiplies by
`1 + config.model_token_count_estimate_factor` (default 0.7). Claude 5.5 models produced 1.40 to 1.68 times the
tiktoken count on this repository's code, docs and prompts. Claude Code reports exact usage after each call, and
PR-Agent records it in the run details (`config.output_run_details`).

## Differences from upstream PR-Agent

- Only Claude models are supported, and there is no LiteLLM, OpenAI, Bedrock, Vertex or OpenRouter routing.
  `/similar_issue` still uses OpenAI embeddings and needs `uv sync --group similar-issue` and `openai.key`.
- `config.temperature` is ignored, because Claude Code does not expose it.
- Image input (`/ask` with an image) is ignored with a warning.
- Claude Code retries API errors itself, so PR-Agent has no `num_retries`. Authentication and billing errors fail the
  request at once. Errors Claude Code gives up on, refusals, empty answers and timeouts move to the next fallback
  model.
- With `config.output_run_cost = true`, the reported cost is Claude Code's list-price estimate, not a subscription
  charge.
- Each call starts a Claude Code process, which adds one to two seconds.
