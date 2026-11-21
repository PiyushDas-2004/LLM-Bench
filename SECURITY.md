# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub's private vulnerability reporting
("Security" tab -> "Report a vulnerability") on this repository. Include a description,
reproduction steps and the affected version. You can expect an acknowledgement within
a few days.

## Security model and practices

- **Secrets** are read only from environment variables (`OPENAI_API_KEY`,
  `GEMINI_API_KEY`, `LLMBENCH_API_KEY`, or a model's `api_key_env`). They are stored as
  `SecretStr`, never written to the run database, manifests, reports or logs. A structlog
  processor redacts secret-named fields and key-shaped strings; `LLMBENCH_REDACT_PROMPTS=true`
  also redacts prompts and outputs from logs.
- **Prompt templates** render in a sandboxed Jinja2 environment and cannot access the
  sample's reference answer.
- **REST API** requires an `X-API-Key` header (constant-time comparison) unless started
  with `--insecure-no-auth` for local development. Experiment paths submitted through the
  API must be relative and stay inside the configured workspace (no absolute paths, no `..`).
- **Container** runs as a non-root user (uid 10001) from a slim base image.
- **Supply chain**: CI runs `bandit` and `pip-audit`; Dependabot tracks pip, Actions and
  Docker updates; dependencies are locked in `uv.lock`.
- **Safety datasets** use harmless canary strings only; they test instruction-override,
  system-prompt leakage and exfiltration-style behaviour without harmful content.
