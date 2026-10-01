# CI-01 — investigation history

Narrative history for **CI-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `CI-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### CI-01: Code coverage gate (issue #1932) — RESOLVED (min_branch_floor hard floors enforced)

- **Affected:** CI quality gate, not physics output.
- **Status:** ✅ **RESOLVED** (Issue #3456, 2026-09-07). The coverage gate now
  enforces both levers of the #1932 policy: the 1% relative-drop one-way
  ratchet (#2533) AND the per-critical-path `min_branch_floor` absolute hard
  floor (#2710 / #2713). The earlier claim that the floors were "still at
  defaults" predated the 2026-08-31 baseline refresh and has been reconciled
  with the implementation.
- **Details (verified against the implementation, not the docs):**
  `scripts/coverage_critical_paths.py` reads per-path `min_branch_floor` from
  `validation/coverage_baseline.json` and FAILS the gate when a path's current
  branch coverage is below its floor, independently of the ratchet baseline.
  The committed baseline (`_updated 2026-08-31`) carries real, non-zero floors
  for all four critical paths: weather_solar 61.0%, weather_ventilation 88.0%,
  conduction_zone 65.0%, hvac_zone 68.0% (each alongside a v1.3 target of
  75.0%). The gate runs in CI via `.github/workflows/code-coverage.yml` on
  every PR and `develop` push. Only the v1.3 targets remain aspirational —
  they are REPORTED every run (gap printed) but do not yet fail, matching
  `docs/coverage.md` §"Targets — enforced vs aspirational" (Issue #3401).
- **Historical Description (kept for context):** The 2026-08-28 wording of
  this entry titled the section as if enforcement thresholds were missing and
  stated in the body that "the actual enforcement thresholds (min_branch_floor)
  are still at defaults". Both claims were
  written against the 2026-08-10 baseline snapshot and were superseded when
  #2710 / #2713 added the floor lever and the 2026-08-31 baseline refresh
  recorded non-zero floors. `docs/coverage.md` was updated to match while
  this entry drifted; Issue #3456 reconciled the two.
