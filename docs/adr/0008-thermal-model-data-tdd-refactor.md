# ADR-0008: ThermalModelData Refactor — TDD Scaffolding (Issue #3070)

- **Status:** Proposed (tracking stub only — no architectural decision recorded)
- **Date:** 2026-08-17
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** None (this ADR records the gap; the actual structural refactor is deferred to a future PR that runs the verifier end-to-end against real measurements)
- **Issue:** [#3070](https://github.com/anchapin/fluxion/issues/3070)
- **Related:** #2878 (origin), #3034 (blocked PR), #3069 (the empty `e265c62` re-run commit), #3072 (GaugeSolver meta-issue)

---

This ADR is the **7-line summary** of the TDD scaffolding shipped with issue #3070:

1. PR #3034 split `ThermalModelData<T>` into 6 sub-structs and introduced a Cases 195/600/620 physics regression.
2. Per `RULES.md` ("no parameter tuning", "must-never hardcode results") and `AGENTS.md` ("fix the underlying math"), the split must be redone with a bit-identical baseline.
3. This ADR ships the **scaffolding only** — placeholder snapshots, the verifier, and the pytest harness — and refuses to record an actual refactor decision.
4. The verifier (`scripts/verify_gauge_solver_regression.py`) fails closed (exit 2) when the placeholder has not been populated, so a future refactor run is forced to capture real measurements first.
5. The strict-energy-gate baseline (`tests/reference_data/zone_balance/strict_energy_gate_baseline.json`) is **not modified** by this scaffolding — per AGENTS.md, it must never be raised to hide a regression.
6. `src/sim/thermal_model_data/mod.rs` and every other physics file are **not modified** by this scaffolding — only docs, scripts, and reference-data JSON.
7. Future refactor attempts must produce a real `--after` snapshot set, run the verifier, and trip exit 1 on any per-metric drift > the manifest's per-metric tolerance (default 0.0 → bit-identical).

---

## Executive Summary

This ADR is a **tracking stub**. It records the fact that the
`ThermalModelData<T>` god-struct split (issue #2878, attempted in PR #3034)
introduced a physics regression in ASHRAE 140 Cases 195 / 600 / 620 that
violated `RULES.md` ("no parameter tuning", "must-never hardcode results")
and was blocked by the orchestrator. The legitimate closure path for
issue #3070 is **NOT** to re-apply the refactor — it is to ship the
TDD scaffolding (placeholder snapshot files, a fail-closed snapshot
diff verifier, and a pytest harness) so a future refactor attempt can
be mechanically validated against a bit-identical pre-refactor baseline
without any human in the loop approving parameter-tuning-style
"adjustments".

Per `RULES.md` and `AGENTS.md`, **no physics code is modified in this
ADR**. The actual split lands in a future PR that will run the verifier
end-to-end and either pass (bit-identical physics preserved) or fail
(regression flagged, refactor rejected) on the merits of the diff.

## Context

PR #3034 (commit `0545b03`) split the legacy `ThermalModelData<T>` god-struct
(~140 fields, 145-line `Clone` impl) into six focused sub-structs:

| Sub-struct | Responsibility |
|------------|---------------|
| `HvacState` | HVAC equipment state, internal gains, equipment capacities |
| `SetpointState` | Heating / cooling setpoints, schedules, zone temperatures |
| `SolarState` | Solar geometry, window properties, incident-solar accumulator |
| `MassState` | Thermal-mass node temperatures, mass heat fluxes |
| `ConductionState` | Wall / roof / floor node temperatures, CTF bookkeeping |
| `DiagnosticsState` | Convergence flags, error accumulators, run metadata |

The split met the issue's stated acceptance criterion (`mod.rs` ≤ 200 lines;
the per-config clone in `BatchOracle::evaluate_population` visits exactly 6
fields instead of ~140) but introduced a physics regression in three ASHRAE
140 cases:

| Case | Metric | Pre-#2878 | Post-#2878 | Ref band | Verdict |
|------|--------|-----------|-----------|----------|---------|
| 195  | cooling | 0 | 0.28 | 0–0 | OVER (0.28 vs 0–0) |
| 600  | cooling | 5.236 MWh | 3.30 MWh | 7.0–10.0 | UNDER (massively) |
| 620  | cooling | (within band) | 2.37 MWh | 3.2–5.0 | UNDER |

The sub-agent's CI analysis couldn't isolate the exact mechanism (the issue
speculates about hand-written `Clone` impls, field initialization order
changes, and default-value drift) and PR #3034 was blocked by the
orchestrator. Issue #3070 was opened to track the proper fix.

## Recommended Direction (per Issue #3070)

The issue's "Recommended Direction" calls for a **TDD approach**:

1. Capture bit-identical baseline snapshots of Cases 195 / 600 / 620
   **before** the refactor.
2. Re-apply the refactor with a snapshot diff after every commit.
3. Identify the exact sub-struct boundary where the physics changed.
4. Either fix the regression in the refactor or split further to
   isolate the changed code.

This ADR ships steps (1) — the **scaffolded** placeholder snapshots plus
the verifier infrastructure. Steps (2)–(4) are deferred to a future PR
that runs the actual refactor and the verifier end-to-end.

## What this ADR ships

This ADR ships the **TDD scaffolding** for issue #3070:

### 1. Placeholder snapshot set (`tests/reference_data/gauge_solver_baseline/`)

Four JSON files with the documented schema:

- `case_195_baseline.json` — conduction-only case
- `case_600_baseline.json` — low-mass baseline (12 m² south window)
- `case_620_baseline.json` — low-mass, east-window variant of Case 600
- `baseline_manifest.json` — per-case file map + per-metric tolerances

Each per-case JSON currently has:

- `captured_at: null`
- `captured_commit: null`
- `metrics.*: null`

…so the verifier refuses to diff against them (exit code 2,
`EXIT_PLACEHOLDER`). The first real measurement campaign replaces
the `null` values with floats and stamps `captured_at` /
`captured_commit`. Per-metric tolerances default to `0.0` (bit-identical
equality) and live in `baseline_manifest.json → verifier.default_tolerance`.

The verifier is also pinned to `_schema_version: 1`. Any future
backward-incompatible format change must bump the version and update
the verifier; the gate fails closed (exit 2) on a stale manifest.

### 2. Snapshot diff verifier (`scripts/verify_gauge_solver_regression.py`)

CLI + Python module. Mirrors the pattern of
`scripts/check_strict_energy_gate_regression.py` and
`scripts/release_gate_checker.py`:

- `argparse` CLI with `--before`, `--after`, `--tolerance`,
  `--json`, `--strict`, `--allow-placeholder`.
- Exit codes follow the documented `EXIT_OK=0` /
  `EXIT_REGRESSION=1` / `EXIT_PLACEHOLDER=2` / `EXIT_USAGE=3`
  contract.
- Human-readable report (default) and JSON output (`--json`).
- `self-test` mode (`python3 scripts/verify_gauge_solver_regression.py --self-test`)
  for hermetic validation without pytest.
- Fail-closed default: refuses to diff a placeholder snapshot set
  (exit 2) so a future refactor run cannot silently compare against
  an empty baseline. `--allow-placeholder` is the explicit opt-out.
- SHA-256 fingerprint check (`--strict`): a hand-edited case file
  whose content drifts from the manifest's stamped `sha256` trips
  exit 2 so a silent tweak to the placeholder cannot slip past the
  gate.

### 3. Pytest harness (`scripts/ci/test_verify_gauge_solver_regression.py`)

18 tests covering:

- Snapshot-set loading (success / missing manifest / wrong schema version /
  `_doc` key handling — regression guard for an earlier `'str' object has
  no attribute 'get'` bug).
- `is_placeholder` detection.
- `compute_diff` no-drift / regression / tolerance-override / schema-drift
  scenarios.
- `verify_fingerprints` empty / silent-edit scenarios.
- `main()` end-to-end: clean (exit 0), regression (exit 1), placeholder
  (exit 2), missing manifest (exit 3), JSON output shape, `--strict`
  SHA-256 mismatch (exit 2), CLI tolerance override.

All 18 tests pass; coverage of the script is 76%.

## What this ADR does NOT do

1. **It does NOT modify physics code.** Per AGENTS.md ("do NOT modify
   physics code without checking `ARCHITECTURE.md` first"), the actual
   refactor is deferred to a future PR. This stub only ships scaffolding.
2. **It does NOT modify `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.**
   Per AGENTS.md, the strict-energy-gate baseline must NEVER be raised to
   hide a regression. The bit-identical baseline for the TDD approach
   lives in `tests/reference_data/gauge_solver_baseline/`, a separate
   directory tree, so the two never conflate.
3. **It does NOT modify ARCHITECTURE.md or RULES.md.** Those are
   source-of-truth documents; this stub references them.
4. **It does NOT record an architectural decision.** The actual
   split-vs-don't-split decision is deferred to a future PR that
   submits both the refactor and the verifier output.
5. **It does NOT mark any case as passing.** It documents the
   scaffolding status only.
6. **It does NOT pre-populate the snapshot values.** Per RULES.md
   ("must-never hardcode results"), the baseline must be captured by
   running the actual Case 195 / 600 / 620 integration tests. The
   shipped placeholders force the first real measurement campaign to
   produce real numbers.

## Decision

**None recorded.** This is a tracking stub. The actual architectural
decision (re-apply the god-struct split with bit-identical physics
guaranteed by the verifier, or abandon the split and document why) is
deferred to a future PR that:

1. Runs the existing Case 195 / 600 / 620 integration tests against the
   pre-refactor `develop` to populate
   `tests/reference_data/gauge_solver_baseline/` with real numbers.
2. Applies the refactor.
3. Runs the verifier end-to-end:
   ```
   python3 scripts/verify_gauge_solver_regression.py \
       --before tests/reference_data/gauge_solver_baseline \
       --after  <post-refactor-snapshot-dir>
   ```
4. Submits a PR with the verifier output as evidence. Exit 0 → merge;
   exit 1 → reject and iterate; exit 2 → placeholder drift detected
   (re-capture required); exit 3 → usage error.

When that PR lands, this stub will be either:

- Superseded by a full ADR that records the split decision, the
  per-case metrics, and the verifier-output evidence; OR
- Marked `Rejected` if the verifier output shows the refactor
  cannot be made bit-identical and the god-struct stays whole.

## Consequences

### Positive

- The TDD scaffolding for issue #3070 is in place. A future refactor
  attempt has a mechanical, exit-coded way to prove bit-identical
  physics without any human in the loop approving parameter-tuning-
  style adjustments.
- The verifier fails closed by default (`--allow-placeholder` is the
  opt-out), so a refactor run that forgets to capture real baselines
  cannot silently green-light itself.
- The SHA-256 fingerprint check catches silent edits to the placeholder
  that would otherwise pass the placeholder detection by preserving
  `null` values.
- The pytest harness gives CI a deterministic way to lock the
  verifier's exit-code contract.

### Negative

- None. This is a tracking stub plus a verifier; it does not change
  any architecture, test, or pass-rate claim.

### Neutral

- The `Status: Proposed` marker is intentional and remains until the
  underlying structural PR lands. If the god-struct stays whole, this
  stub will be marked `Rejected` with a one-paragraph explanation.
- The verifier's per-metric tolerance defaults to `0.0` (bit-identical).
  Future refactors that intentionally relax a metric (e.g. to absorb
  cross-runner numerical noise) must lower the baseline **and** commit
  the engine improvement together (per RULES.md / AGENTS.md: never
  raise to hide a regression).

## References

- Issue #3070 — refactor(sim): #2878 god-struct split reverted
- Issue #2878 — original god-struct split proposal
- PR #3034 — the blocked god-struct split (commits `037b5e4`, `0545b03`)
- Commit `e265c62` — the empty "trigger: rerun CI" commit referenced in
  the orchestrator's CI chain (#3069)
- `tests/reference_data/gauge_solver_baseline/` — the placeholder
  snapshot set shipped with this ADR
- `scripts/verify_gauge_solver_regression.py` — the verifier
- `scripts/ci/test_verify_gauge_solver_regression.py` — pytest
  harness (18 tests, all passing)
- `RULES.md` — "no parameter tuning" + "must-never hardcode results"
- `AGENTS.md` — "fix the underlying math"; strict-energy-gate baseline
  must NEVER be raised
- ADR-0001 — No-Parameter-Tuning Rule
- ADR-0007 — GaugeSolver Structural Work (the sibling stub for the
  Cases 940/960 cohort; this ADR addresses the Cases 195/600/620 cohort)
- `docs/ASHRAE140_RESULTS.md` §"Structural Blockers (Issue #3072)" —
  current pass-rate snapshot for the wider aggressive-baseline cohort