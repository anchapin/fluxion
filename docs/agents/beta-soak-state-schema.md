# `beta-soak-state.json` schema — β-phase 30-day soak window gate (Issue #3286)

The β-phase gate contract that gates removal of the `gauge-solver` cargo feature (Phase A8, Issue #3291, PR #3482) is the consecutive-green-streak counter persisted as `beta-soak-state.json`. This file documents the canonical schema, the gate contract, and the script that enforces both.

## Gate contract

30 consecutive successful nightly runs of the seven ADR-0007 acceptance criteria (`docs/adr/0007-gauge-solver-structural-work.md` §"Required CI gates green") must be GREEN before the gauge default-flip is unblocked. The streak is computed stateless from `.github/workflows/nightly-ashrae-140-gauge.yml`'s run history and persisted to the state file. The streak resets to zero on any failure. Until §LIMIT-21 closes (Issue #3297), the streak will sit at 0/30 — silently resetting it would violate `RULES.md` / `AGENTS.md` / ADR-0001.

## Schema (v1)

The canonical schema lives in `scripts/check_beta_soak_gate.py` — this document mirrors it; both must be updated together.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | optional | Always `"1"` for this schema. Absent in pre-#3286 artefacts; back-compat path accepts these as v1. |
| `streak` | int `[0, 10000]` | **required** | Consecutive-green-run count from the most recent nightly. |
| `target` | int `[1, 10000]` | **required** | Soak target. The Issue #3286 contract fixes `target=30`. |
| `unblocked` | bool | **required** | `true` iff `streak >= target`. Cross-checked with the two scalars. |
| `generated_at` | string (ISO-8601) | **required** | UTC timestamp of state computation. |
| `remaining` | int `[0, 10000]` | optional | `max(0, target - streak)`. Cross-checked. |
| `runs_considered` | int `[0, 10000]` | optional | Total runs scanned when the streak was last computed (for diagnostics). |
| `last_run_id` | string | optional | `databaseId` of the most recent run (for diagnostics). |
| `first_run_id` | string | optional | `databaseId` of the oldest run considered in the streak (for diagnostics). |
| `oldest_run_at` | string (ISO-8601) | optional | ISO-8601 timestamp of the oldest run in the streak (for diagnostics). |

## Validation rules

- The top-level value must be a JSON object.
- All required fields must be present.
- No unknown fields are allowed (drift detected at the gate, not silently green).
- `unblocked` must equal `streak >= target`.
- If `remaining` is supplied, it must equal `max(0, target - streak)`.
- `generated_at` must be ISO-8601 (`YYYY-MM-DDThh:mm:ss[.fff][Z|±hh:mm]`).

## Counting convention

Only completed runs on `develop` count. The workflow passes `--status completed` to `gh run list`, so the in-flight run is excluded — tonight's success counts from tomorrow (one-run-conservative, per Issue #3286). Any failure resets the streak to zero.

## Wire-up

1. `.github/workflows/nightly-ashrae-140-gauge.yml` writes the state file in its "Compute β-soak streak" step (back-compat: the inline construction omits `schema_version`, accepted by the validator).
2. The state file is uploaded as a 90-day artifact (`beta-soak-state`).
3. The same workflow's "ADR-0007 criteria + soak streak" job can be extended with a final step that runs `python3 scripts/check_beta_soak_gate.py --state beta-soak-state.json --json` for an additional consistency check.
4. PRs that touch the gauge default (e.g. modifying `src/sim/thermal_selector.rs`, `Cargo.toml` `[features] gauge-solver`, or `tests/zone_balance_eplus_isolation.rs`) must invoke `python3 scripts/check_beta_soak_gate.py --gate enforce` and have it exit `0`. Until 30 consecutive nightly runs are GREEN, the gate is closed and the PR fails CI.

## Related

- `scripts/check_beta_soak_gate.py` — single source of truth for the schema + gate logic.
- `scripts/ci/test_check_beta_soak_gate.py` — pytest coverage of the validator and CLI.
- `.github/workflows/nightly-ashrae-140-gauge.yml` — the nightly workflow that produces the state file.
- `docs/agents/beta-soak-gate-failure-no-reset.md` — the gate-failure protocol note (Issue #3354).
- `docs/adr/0007-gauge-solver-structural-work.md` — ADR-0007 acceptance criteria.
- Issue #3286 — β-phase 30-day soak window tracking (gate contract for default flip).
- Issue #3291 / PR #3482 — Phase A8 default flip (gated on this contract).
