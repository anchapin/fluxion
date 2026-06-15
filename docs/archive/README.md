# Archive

Historical session notes, planning artifacts, and superseded reports.
Files here are kept for historical context but are **not** authoritative
references for current Fluxion development.

## Contents

- `sessions/` — 37 numbered session prompt drafts (`session_1_prompt.md`
  through `session_38_prompt.md`, with #34 missing in the original set).
  These were per-session scratch files used during the M2 / ASHRAE 140
  improvement work and are preserved for traceability of the decision
  history.
- `planning/` — Stale CI / phase / completion summaries and superseded
  ASHRAE 140 improvement plans:
  - `CI_FIX_SUMMARY.md`, `CI_RETRIGGER.md`
  - `FAILING_TESTS_TRACKING.md`
  - `M2-07-COMPLETION.md`
  - `PHASE_44_WAVE_0_SUMMARY.md`, `PHASE_45_WAVE_0_SUMMARY.md`
  - `PLAN_ashrae140_improvement.md`, `PLAN_ashrae140_remainder.md`
  - `ashrae-140-prompt.md`
- `security/` — Superseded security reports. The current security
  advisory is at the repo root in
  `SECURITY_REPORT_CVE-2026-27448_GHSA-fv5p-p927-qmxr.md`.
  - `SECURITY_FINDINGS_2026-04-27.md` (general finding sweep, predates
    the CVE-specific report)

## Why these are at `docs/archive/` rather than at the repo root

The repo root previously held 56 markdown files, of which 47 were
session-specific or single-issue artifacts that obscured the canonical
documents (`README.md`, `CHANGELOG.md`, `AGENTS.md`, `RULES.md`,
`CLAUDE.md`, plus the live security advisory and three checked-in script
outputs: `SCORECARD.md`, `sensitivity_report.md`, `validation_report.md`).

This directory was created by [Issue #768](https://github.com/anchapin/fluxion/issues/768).

## Adding to the archive

If you produce a session-scoped or one-off report, place it directly here
under the appropriate subdirectory rather than at the repo root.
