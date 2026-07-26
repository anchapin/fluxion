# PR Body Conventions for Wave Orchestration

## The Problem

GitHub's `gh pr create --fill` auto-fills the PR body from recent commit
messages and PR activity. When the orchestrator's sub-agent template passes
both `--fill` AND a body containing `Closes #N`, GitHub's
`closingIssuesReferences` parser additionally pattern-matches every other
issue number it can find in the title and commits — not just the explicit
`Closes #N` mention.

### Symptom (observed in PR #1487)

> Commit message: "test: resolve #1425 — proptest FiveR1C steady_state_flux contract under query-only separation"
> Auto-closed by GitHub: `[1415, 1425, 1438, 1441, 1444, 1461, 1462, 1465]`

Seven of those were already closed by other PRs in the same wave; GitHub
recorded `closingReferences` for them on this PR anyway, polluting the
issue→PR linkage.

### Same noise on PR #1486 (Perez dedup), PR #1482 (solar constant)

The closing-references lists pull in 1461/1462/1465 (ThermalManifold /
GaugeSolver Phase 1/2/3 issues) just because the PR body mentions
`ARCHITECTURE.md` once.

## The Fix

**Use `--body`, never `--fill`.** The body must contain exactly one
`Closes #N` line.

```bash
gh pr create --base develop \
  --title "fix: resolve #N — <title>" \
  --body "$(cat <<'EOF'
Closes #N

<one-paragraph description>
EOF
)"
```

Rules:

1. **One `Closes #N` line.** No other `#NNNN` references in body or title.
2. **Single-paragraph description.** Avoid bullet lists — they sometimes
   contain `#NNN` references that the parser picks up.
3. **Title format:** `{fix|feat|docs|ci|test|refactor}: resolve #N — <title-slug>`.
   The slug must not contain `#NNN`.

## Verification

Run immediately after `gh pr create`:

```bash
gh pr view <PR> --json closingIssuesReferences --jq '.closingIssuesReferences | length'
# Expected: 1
```

If it returns more than 1:

```bash
gh pr edit <PR> --body "Closes #N\n\n<minimal description>"
```

And re-run the check.

## History

This convention was added as part of issue #1507 after the 2026-07-10/11
wave orchestration polluted closing-references on PRs #1482, #1486, #1487.
The corresponding skill update is at
`~/.agents/skills/github-wave-orchestrator/REFERENCE.md` (lines 22–41).
