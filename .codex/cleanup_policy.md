# Cleanup/Update Policy for Codex Review Comments

- Maintain exactly one Codex-generated comment per PR.
- Identify the Codex comment by the hidden marker:
  <!-- codex-review -->
- On each run, update the newest Codex comment in place.
- Delete older comments that both:
  1) were posted by github-actions[bot], and
  2) contain the codex marker.
- Never edit or delete human comments or comments from other bots.
- Do not open a formal GitHub “Review”; only post a single consolidated comment.
