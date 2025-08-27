# Codex Multi‑Reviewer PR Reviewer

This repository includes a GitHub Actions workflow that posts an academic‑style, multi‑perspective review on each pull request. It is non‑blocking and coexists with human reviewers and other bots.

## Setup (once)

1. Create `CODEX_AUTH_B64` secret in your repository/org:

   - Install Node.js 18+ and the Codex CLI:

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
- If all three decisions agree (or forced), a Meta Reviewer posts a consolidated scorecard.
- Maintains one up‑to‑date Codex comment with a neutral status `codex/review` (informational only).

## Maintainer Commands (as PR comments)

- `/codex rerun` — rerun the three reviewers.
- `/codex meta` — force the Meta Reviewer even if decisions differ.
- `/codex strict on|off` — toggle strictness (stricter thresholds).
- `/codex scope path/to/**.py` — restrict next run to matching files.
- `@codex <hint>` — add a free‑form focus hint to prompts.

Access control: accepted from OWNER, MEMBER, COLLABORATOR, or the PR author. Others are ignored.

## Notes on Privacy & Safety

- Input minimization: only the PR diff + changed paths are sent, not a full repo crawl.
- Scope is shell‑quoted and applied to both the diff and changed‑files list.
- Artifacts contain only diffs and final JSON outputs (no raw model streams).
- The bot never opens a GitHub “Review” state and never edits/deletes non‑Codex comments.

## Troubleshooting

- Fork PRs usually lack secrets: the workflow posts a skip notice.
- Large diffs are truncated; use `/codex scope <glob>` to focus.
- If a reviewer outputs invalid JSON, the workflow retries once with a JSON‑only instruction, then falls back to `{}`.
