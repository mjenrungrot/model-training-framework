# Codex Multi‑Reviewer PR Reviewer

This repository includes a GitHub Actions workflow that posts an academic‑style, multi‑perspective review on each pull request. It is non‑blocking and coexists with human reviewers and other bots.

## Setup (once)

Note: The `@openai/codex` package referenced below is the Codex CLI (open‑source, maintained for GitHub/CLI workflows) — it is not the legacy 2021 “Codex” models. The CLI authenticates via ChatGPT and uses current OpenAI APIs under the hood; no deprecated Codex model APIs are used.

1. Create `CODEX_AUTH_B64` secret in your repository/org:

   - Install Node.js 22+ and the Codex CLI:

- `npm i -g @openai/codex@^1`
- Login (choose "Sign in with ChatGPT"):
- `codex login`
- Create base64 of your Codex auth (no newlines):
  - macOS: `base64 -b 0 ~/.codex/auth.json`
  - Linux: `base64 -w0 ~/.codex/auth.json`
  - Windows (PowerShell): `[Convert]::ToBase64String([IO.File]::ReadAllBytes("$env:USERPROFILE\.codex\auth.json"))`
  - Copy the base64 string into GitHub → Settings → Secrets and variables → Actions → New repository secret → Name: `CODEX_AUTH_B64`.

1. Confirm workflow exists:

- `.github/workflows/codex-multi-review.yml` (already added in this repo)

## What it does

- Runs three Codex reviewers on each PR:
  - Correctness & Safety
  - Documentation & Developer Experience
  - Performance & Architecture
- A Meta Reviewer always runs and posts a consolidated scorecard.
- Maintains one up‑to‑date Codex comment with a neutral status `codex/review` (informational only).

## Maintainer Commands (as PR comments)

- `/codex rerun` — rerun the three reviewers.
- `/codex meta` — trigger a Codex run (meta always runs; same effect as rerun).
- `/codex strict on|off` — toggle strictness (stricter thresholds).
- `/codex scope path/to/**.py` — restrict next run to matching files.
- `@codex <hint>` — add a free‑form focus hint to prompts.

Access control: accepted from OWNER, MEMBER, COLLABORATOR, or the PR author. Others are ignored.

## Permissions & Secrets

- Required GitHub Actions permissions (declared in the workflow):
  - `contents: write`
  - `pull-requests: write`
  - `statuses: write`
  - `issues: write`
- Required secret: `CODEX_AUTH_B64` containing a base64 of `~/.codex/auth.json` (no newlines). Add it at the repository or organization level under Settings → Secrets and variables → Actions. <!-- pragma: allowlist secret -->

## Manual Run (forks/missing secrets)

Maintainers can run the workflow manually:

- GitHub → Actions → `codex-multi-review` → Run workflow

This is useful for forked PRs that cannot access repository secrets.

## Comment Hygiene

The workflow maintains exactly one Codex‑generated comment per PR and updates it in place. Older Codex comments are cleaned up; human and other bot comments are never touched. See `.codex/cleanup_policy.md` for details.

## Notes and Tips

- Node.js version: the workflow uses Node.js 22 (setup via `actions/setup-node@v4`).
- Verify CLI: after `codex login`, run `codex --version` locally before creating the secret.
- Scope examples: multiple globs are supported, e.g., `model_training_framework/**/*.py docs/**/*.md`.
- Codex CLI: see <https://www.npmjs.com/package/@openai/codex> and <https://github.com/openai/codex-cli>.

## Notes on Privacy & Safety

- Input minimization: only the PR diff + changed paths are sent, not a full repo crawl.
- Scope is shell‑quoted and applied to both the diff and changed‑files list.
- Artifacts contain only diffs and final JSON outputs (no raw model streams).
- The bot never opens a GitHub “Review” state and never edits/deletes non‑Codex comments.

## Troubleshooting

- Fork PRs usually lack secrets: the workflow posts a skip notice.
- Large diffs are truncated; use `/codex scope <glob>` to focus.
- If a reviewer outputs invalid JSON, the workflow retries once with a JSON‑only instruction, then falls back to `{}`.
