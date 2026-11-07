---
title: "GitHub Integration"
sidebar_position: 4
---

In this page we will cover how to install and run PR-Agent as a GitHub Action or GitHub App, and how to configure it for your needs.

## Run as a GitHub Action

The action runs this fork's image, `ghcr.io/tim-mcdonnell/pr-agent:github_action`, which bundles Claude Code.
Every model call goes through Claude Code, authenticated with your Claude subscription or an API key. See
[Claude Code integration](../usage-guide/claude_agent_sdk.md) for how calls are made and what the model may read.

1. Create a token for your Claude subscription by running `claude setup-token` locally. Add it to the repository
   under `Settings > Secrets and variables > Actions` as `CLAUDE_CODE_OAUTH_TOKEN`. Optionally, add a
   `PR_AGENT_MODEL` secret (or variable) holding a Claude Code model slug such as `opus`, `sonnet` or `sonnet[1m]`.

2. Add `.github/workflows/pr_agent.yml`:

   ```yaml
   on:
     pull_request:
       types: [opened, reopened, ready_for_review]
     issue_comment:
   jobs:
     pr_agent_job:
       if: ${{ github.event.sender.type != 'Bot' }}
       runs-on: ubuntu-latest
       timeout-minutes: 30
       permissions:
         contents: read
         issues: write
         pull-requests: write
       steps:
         # Optional: a checkout lets the model read the repository beyond the diff (read-only).
         - uses: actions/checkout@v4
           with:
             ref: ${{ github.event.pull_request.head.sha || format('refs/pull/{0}/head', github.event.issue.number) }}
             persist-credentials: false
         - name: PR Agent action step
           uses: tim-mcdonnell/pr-agent@main
           with:
             claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
             model: ${{ secrets.PR_AGENT_MODEL }}
           env:
             GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
   ```

   The `GITHUB_TOKEN` secret is created by GitHub, but you must pass it to the action through `env` as shown.
   If it is missing or empty, events with a PR or issue target fail with a nonzero exit status and a
   `GITHUB_TOKEN not set` message.

3. Merge the workflow to your default branch. The next PR you open gets a review from the `github-actions` bot.

The action's inputs:

| Input | Meaning |
| --- | --- |
| `claude_code_oauth_token` | Token from `claude setup-token`, billed to your Claude subscription. |
| `anthropic_api_key` | A Claude Console API key, used instead when no OAuth token is given. |
| `model` | Claude Code model slug. Empty uses `config.model` (default `opus`). |
| `fallback_models` | Comma-separated slugs tried in order after a fallback-eligible failure. |
| `artifact_path`, `artifact_instructions` | A CI file to add as context; see [CI artifact context](#ci-artifact-context). |

Configure anything else through environment variables named after keys in the
[configuration file](https://github.com/tim-mcdonnell/pr-agent/blob/main/pr_agent/settings/configuration.toml), for
example:

```yaml
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        PR_REVIEWER__EXTRA_INSTRUCTIONS: "Focus on security issues and error handling."
        PR_CODE_SUGGESTIONS__NUM_CODE_SUGGESTIONS_PER_CHUNK: "6"
        GITHUB_ACTION_CONFIG__AUTO_IMPROVE: "true"
```

See detailed usage instructions in the [usage guide](../usage-guide/automations_and_usage.md#github-action).

When a PR opens with a reviewer already requested, GitHub can send both `opened` and `review_requested`. With both
actions enabled, the Action skips the initial `review_requested` event when the PR's creation and update timestamps
match, so the tools run once for the opening event stream. Later review requests still run the enabled tools when
`review_requested` is listed in `github_action_config.pr_actions`.

:::warning[Subscription credentials]
A subscription token is for your own use. Anthropic's
[terms](https://code.claude.com/docs/en/legal-and-compliance#authentication-and-credential-use) do not allow routing
requests through Free, Pro, Max or Ultra credentials on behalf of other users, so use `anthropic_api_key` for
repositories whose PRs you run PR-Agent for as a service.
:::

### Using with pull_request_target (fork/contribution support)

The `pull_request` event gets no repository secrets for PRs from forks, so PR-Agent cannot authenticate to Claude
for them. `pull_request_target` runs in the base repository's context with its secrets. Under it, a plain
`actions/checkout` fetches the default branch; check out the PR head only for read-only exploration, never to build,
install or run its content in the same job. PR-Agent's model only reads a copy of its tracked files. Read GitHub's
[guide to pull_request_target](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target)
before enabling it.

:::warning[Never use `uses: ./` here]
Reference the action as `tim-mcdonnell/pr-agent@<ref>`, not `./`. A local reference loads `action.yaml` from the
checked-out PR, which would let the PR choose the image that receives your token.
:::

### Configuration Examples

Prefer a `.pr_agent.toml` in the repository root for repository preferences:

```toml
[config]
model = "sonnet"
fallback_models = ["opus"]

[pr_reviewer]
extra_instructions = "Focus on security issues and code quality."

[pr_code_suggestions]
num_code_suggestions_per_chunk = 6
```

Host-only settings, such as `[claude_code]` exploration, are ignored in `.pr_agent.toml`; set them as workflow
environment variables (for example `CLAUDE_CODE__MAX_TURNS: "12"`).

### CI artifact context

A file produced by an earlier CI step — a test report, a coverage summary, a linter or SAST output — can be injected into the prompts of `/review`, `/describe` and `/improve`, so the model reviews the PR with your pipeline's own findings in hand.

Point the action at the file with the `artifact_path` input. The path is resolved relative to `GITHUB_WORKSPACE` (an absolute path also works), so the file must already exist in the workspace when PR-Agent runs — produce it in a previous step, or download it with `actions/download-artifact`:

```yaml
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      - name: Run tests
        run: pytest --junitxml=reports/pytest.xml || true
      - name: PR Agent action step
        uses: tim-mcdonnell/pr-agent@main
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
          artifact_path: reports/pytest.xml
          artifact_instructions: "These are the failing tests from this PR's CI run. Call out any suggestion that would not fix them."
```

Setting `artifact_path` turns the feature on by itself; there is no separate enable switch to flip in the workflow. Supported target tools receive a dedicated artifact section in their prompt. The label and file contents are explicitly marked as untrusted data, while `artifact_instructions` appears separately as subordinate analysis guidance. The supported targets are `pr_reviewer`, `pr_description`, `pr_code_suggestions` and `pr_questions`; unsupported names are skipped with a warning.

When overriding a supported tool's prompt, keep `artifact_context.instructions` in the system prompt and render
`artifact_context.label`, `artifact_context.content`, `artifact_context.start_marker`, and `artifact_context.end_marker` in
a clearly marked, untrusted section of the user prompt. This replaces the legacy `extra_instructions` behavior for CI
artifacts.

The remaining knobs live in the `[artifacts]` section of your configuration:

```toml
[artifacts]
enable = true                                               # required when configuring paths here
artifact_path = [
  { path = "reports/pytest.xml" },
  { path = "reports/coverage.xml" },
]
artifact_instructions = ""                                  # empty = a generic "treat this as CI context" instruction
artifact_label = ""                                         # shared label; empty = each file's name
target_tools = ["pr_reviewer", "pr_description", "pr_code_suggestions", "pr_questions"]
max_artifact_size = 50000                                   # characters; longer files are truncated with a marker
```

`artifact_path` may also be a single string. For multiple files, each entry can set
`path`, `artifact_label`, `artifact_instructions`, `target_tools`, and `truncate_from` on
its own line; a shared `artifact_label` applies only to a single file, otherwise each file falls
back to its name.

Automatic enabling applies to the action's `artifact_path` input or the `ARTIFACT_PATH` / `PR_AGENT_ARTIFACT_PATH` environment variables. A path set only in `[artifacts]` requires `enable = true`.

CI artifacts are checked for recognizable URL credentials, authorization headers, standard GitLab tokens, AWS access keys, and common credential assignments before model submission, with redaction matches logged only by type and count; this pattern-based check does not recognize every secret format, so use sanitized CI logs as input.

:::note
A path that resolves outside `GITHUB_WORKSPACE` is rejected, and a missing or unreadable file is skipped with a warning — in both cases the tools still run, just without the artifact context.
:::

The same settings apply when PR-Agent runs as a CLI from another CI system, with `ARTIFACT_PATH` set in the job's environment; see the [GitLab pipeline example](./gitlab.md#ci-artifact-context).

### Using a specific release

`tim-mcdonnell/pr-agent@main` always runs the image built from `main`. For a reproducible setup, pin the action by
commit SHA and run the image by digest:

```yaml
      - name: PR Agent action step
        uses: docker://ghcr.io/tim-mcdonnell/pr-agent@sha256:<digest>
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          CLAUDE_CODE_OAUTH_TOKEN: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
```

Resolve a digest with `docker buildx imagetools inspect ghcr.io/tim-mcdonnell/pr-agent:github_action --format
'{{.Manifest.Digest}}'`. Images carry GitHub artifact attestations, so you can check a digest's origin:

```sh
gh attestation verify "oci://ghcr.io/tim-mcdonnell/pr-agent@sha256:<digest>" --repo tim-mcdonnell/pr-agent
```

### Action for GitHub enterprise server or GHE.com

:::tip
To use the action with a GitHub enterprise server, add an environment variable `GITHUB__BASE_URL` with the API URL of
your GitHub server, for example `GITHUB__BASE_URL: "https://github.mycompany.com/api/v3"`. For GHE.com, use the
workflow's `${{ github.api_url }}`.
:::

---

## Run as a GitHub App

Allowing you to automate the review process on your private or public repositories.

1. Create a GitHub App from the [GitHub Developer Portal](https://docs.github.com/en/developers/apps/creating-a-github-app).

   - Set the following permissions:
     - Pull requests: Read & write
     - Issues: Read & write
     - Metadata: Read-only
     - Contents: Read-only (or Read & write if using `resolve_threads` — see note below)
   - Set the following events:
     - Issue comment
     - Pull request
     - Pull request review
     - Push (if you need to enable triggering on PR update)
     - Pull request review comment (required for `/ask` on review threads)

   > **Note:** If you enable `pr_questions.resolve_threads`, the GitHub App requires **Contents: Read & write** permission. GitHub's `resolveReviewThread` GraphQL mutation is gated behind the Contents permission, even though it only modifies PR thread metadata. See [GitHub community discussion](https://github.com/orgs/community/discussions/204269) for details.
   >
   > **Important:** When enabled, the LLM may resolve threads started by
   > human reviewers — not only bot-generated threads. Use this setting
   > only when your team is comfortable with AI-driven thread resolution.
   > The feature is opt-in and defaults to off.

2. Generate a random secret for your app, and save it for later. The webhook secret is required: if `GITHUB.WEBHOOK_SECRET` is not configured, the server rejects every incoming webhook with HTTP 403. For example, you can use:

   ```bash
   WEBHOOK_SECRET=$(python -c "import secrets; print(secrets.token_hex(10))")
   ```

3. Acquire the following pieces of information from your app's settings page:

   - App private key (click "Generate a private key" and save the file)
   - App ID

4. Clone this repository:

   ```bash
   git clone https://github.com/the-pr-agent/pr-agent.git
   cd pr-agent
   ```

5. Copy the secrets template file and fill in the following:

   ```bash
   cp pr_agent/settings/.secrets_template.toml pr_agent/settings/.secrets.toml
   # Edit .secrets.toml file
   ```

   - Your OpenAI key.
   - Copy your app's private key to the private_key field.
   - Copy your app's ID to the app_id field.
   - Copy your app's webhook secret to the webhook_secret field (required).
   - The GitHub App sets `deployment_type` to `app` automatically by default. No manual configuration is needed unless `github_app.override_deployment_type` is disabled.

    > The local `.secrets.toml` file is excluded from the Docker build context. Never bake secrets into a container image.
    > For container deployments, provide secrets at runtime through environment variables or a mounted secret volume.
    > For example, in order to inject a secrets file as a volume in a Kubernetes environment you can update your pod spec to include the following,
    > assuming you have a secret named `pr-agent-settings` with a key named `.secrets.toml`:

    ```
           volumes:
            - name: settings-volume
              secret:
                secretName: pr-agent-settings
    // ...
           containers:
    // ...
              volumeMounts:
                - mountPath: /app/pr_agent/settings_prod
                  name: settings-volume
    ```

    > Service images run as UID/GID `10001`: mounted secrets must be readable, and mounted data paths writable, by that user.
    > Home-relative mounts belong under `/home/pragent`.

    > Another option is to set the secrets as environment variables in your deployment environment, for example `OPENAI__KEY`, `GITHUB__APP_ID`, `GITHUB__PRIVATE_KEY`, and `GITHUB__WEBHOOK_SECRET`.

6. Build a Docker image for the app and optionally push it to a Docker repository. We'll use Dockerhub as an example:

    ```bash
    docker build . -t pr-agent:github_app --target github_app -f docker/Dockerfile

    # Optional, to push it to your own Docker repository:
    docker tag pr-agent:github_app <your-registry>/pr-agent:github_app
    docker push <your-registry>/pr-agent:github_app
    ```

7. Host the app using a server, serverless function, or container environment. Alternatively, for development and
   debugging, you may use tools like smee.io to forward webhooks to your local machine.
    You can check [Deploy as a Lambda Function](#deploy-as-a-lambda-function)

8. Go back to your app's settings, and set the following:

   - Webhook URL: `https://<PR_AGENT_HOSTNAME>/api/v1/github_webhooks`, or your smee.io channel URL with forwarding configured to that endpoint.
   - Webhook secret: The secret you generated earlier.

9. Install the app by navigating to the "Install App" tab and selecting your desired repositories.

10. The app runs under gunicorn with multiple worker processes. See [Sizing a self-hosted webhook server](./index.md#sizing-a-self-hosted-webhook-server) for the `GUNICORN_WORKERS` / `GUNICORN_MAX_WORKERS` knobs and memory guidance — worth reading before setting a memory limit.

> **Note:** When running PR-Agent from GitHub app, the default configuration file (configuration.toml) will be loaded.
> However, you can override the default tool parameters by uploading a local configuration file `.pr_agent.toml`
> To use organization-level global configuration, set `config.global_settings_repo = "pr-agent-settings"`, create `<organization>/pr-agent-settings` with a `.pr_agent.toml` file, and install the GitHub App on that repository too.
> The app needs read access to the settings repository as well as the pull request repositories. This applies to both GitHub.com and GitHub Enterprise Server.
> For more information please check out the [USAGE GUIDE](../usage-guide/automations_and_usage.md#github-app)
---

## Additional deployment methods

### Deploy as a Lambda Function

Note that since AWS Lambda env vars cannot have "." in the name, you can replace each "." in an env variable with "__".<br>
For example: `GITHUB.WEBHOOK_SECRET` --> `GITHUB__WEBHOOK_SECRET`

1. Follow steps 1-5 from [here](#run-as-a-github-app).
2. Build a docker image that can be used as a lambda function

    ```shell
    docker buildx build --platform=linux/amd64 . -t pr-agent:github_lambda --target github_lambda -f docker/Dockerfile.lambda
   ```
   (Note: --target github_lambda is optional as it's the default target)


3. Push image to ECR

    ```shell
    docker tag pr-agent:github_lambda <AWS_ACCOUNT>.dkr.ecr.<AWS_REGION>.amazonaws.com/pr-agent:github_lambda
    docker push <AWS_ACCOUNT>.dkr.ecr.<AWS_REGION>.amazonaws.com/pr-agent:github_lambda
    ```

4. Create a lambda function that uses the uploaded image. Set the lambda timeout to be at least 3m.
5. Configure the lambda function to have a Function URL.
6. In the environment variables of the Lambda function, specify `AZURE_DEVOPS_CACHE_DIR` to a writable location such as /tmp. (see [link](https://github.com/the-pr-agent/pr-agent/pull/450#issuecomment-1840242269))
7. Go back to steps 8-9 of [Method 5](#run-as-a-github-app) with the function url as your Webhook URL.
    The Webhook URL would look like `https://<LAMBDA_FUNCTION_URL>/api/v1/github_webhooks`

#### Using AWS Secrets Manager

For production Lambda deployments, use AWS Secrets Manager instead of environment variables:

1. Create a secret in AWS Secrets Manager with JSON format like this:

   ```json
   {
     "openai.key": "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
     "github.app_id": 123456,
     "github.webhook_secret": "your-webhook-secret-from-step-2",
     "github.private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
   }
   ```

2. Add IAM permission `secretsmanager:GetSecretValue` to your Lambda execution role
3. Set these environment variables in your Lambda:

   ```bash
   AWS_SECRETS_MANAGER__SECRET_ARN=arn:aws:secretsmanager:us-east-1:123456789012:secret:pr-agent-secrets-AbCdEf
   CONFIG__SECRET_PROVIDER=aws_secrets_manager
   ```

---

### AWS CodeCommit Setup

Not all features have been added to CodeCommit yet.  As of right now, CodeCommit has been implemented to run the PR-Agent CLI on the command line, using AWS credentials stored in environment variables.  CodeCommit pull requests with multiple targets are reviewed across every target repository and commit comparison; single-target pull requests keep the same behavior.  The following is a set of instructions to have PR-Agent do a review of your CodeCommit pull request from the command line:

1. Create an IAM user that you will use to read CodeCommit pull requests and post comments
    - Note: That user should have CLI access only, not Console access
2. Add IAM permissions to that user, to allow access to CodeCommit (see IAM Role example below)
3. Generate an Access Key for your IAM user
4. Set the Access Key and Secret using environment variables (see Access Key example below)
5. Set the `git_provider` value to `codecommit` in the `pr_agent/settings/configuration.toml` settings file
6. Set the `PYTHONPATH` to include your `pr-agent` project directory
    - Option A: Run `export PYTHONPATH="/PATH/TO/PROJECTS/pr-agent"` in your shell. PR-Agent does not automatically load `.env` files.
    - Option B: Set `PYTHONPATH` and run the CLI in one command, for example:
        - `PYTHONPATH="/PATH/TO/PROJECTS/pr-agent" python pr_agent/cli.py [--ARGS]`

---

#### AWS CodeCommit IAM Role Example

Example IAM permissions to that user to allow access to CodeCommit:

- Note: The following is a working example of IAM permissions that has read access to the repositories and write access to allow posting comments
- Note: If you only want pr-agent to review your pull requests, you can tighten the IAM permissions further, however this IAM example will work, and allow the pr-agent to post comments to the PR
- Note: You may want to replace the `"Resource": "*"` with your list of repos, to limit access to only those repos

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "codecommit:BatchDescribe*",
                "codecommit:BatchGet*",
                "codecommit:Describe*",
                "codecommit:EvaluatePullRequestApprovalRules",
                "codecommit:Get*",
                "codecommit:List*",
                "codecommit:PostComment*",
                "codecommit:PutCommentReaction",
                "codecommit:UpdateComment",
                "codecommit:UpdatePullRequestDescription",
                "codecommit:UpdatePullRequestTitle"
            ],
            "Resource": "*"
        }
    ]
}
```

#### AWS CodeCommit Access Key and Secret

Example setting the Access Key and Secret using environment variables

```sh
export AWS_ACCESS_KEY_ID="XXXXXXXXXXXXXXXX"
export AWS_SECRET_ACCESS_KEY="XXXXXXXXXXXXXXXX"
export AWS_DEFAULT_REGION="us-east-1"
```

#### AWS CodeCommit CLI Example

After you set up AWS CodeCommit using the instructions above, here is an example CLI run that tells pr-agent to **review** a given pull request.
(Replace your specific PYTHONPATH and PR URL in the example)

```sh
PYTHONPATH="/PATH/TO/PROJECTS/pr-agent" python pr_agent/cli.py \
  --pr_url https://us-east-1.console.aws.amazon.com/codesuite/codecommit/repositories/MY_REPO_NAME/pull-requests/321 \
  review
```
