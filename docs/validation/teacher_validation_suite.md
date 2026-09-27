# Teacher Validation Suite (Issue #3986 / PR-C #4116)

## 7-Line Summary

Umbrella nightly gate that aggregates three sub-suites of the #3986 teacher validation suite (ASHRAE 140 fabric + ASHRAE 1052-RP analytical conduction + PCM test-box skeleton) into a single pass/fail verdict via `scripts/check_teacher_validation_suite.py`. **Advisory only** — registered in `release_gates.yaml::ci.nightly_authority` per the user's PR-C scope decision; never blocks PRs. Sub-suites: `ashrae_140_fabric` (PR-A #4119, 7 tests), `conduction_1052rp` (#3981, 6 tests), `pcm_test_box` (PR-B #4120, 8 tests). Total: 21 tests; all PASS at PR-C landing; BLOCKER branch on the PCM sub-suite per `tests/reference_data/pcm_test_box/PROVENANCE.md` (PR-B+1 unlocks the real reference-data matching).

## What this gate aggregates

| Sub-suite | Cargo target | Source PR | Tests | Status |
|---|---|---|---|---|
| `ashrae_140_fabric` | `--test all_tests ashrae_140_validator_selector_parity::` | PR-A #4119 | 7 | PASS |
| `conduction_1052rp` | `--test all_tests conduction_1052rp_analytical::` | #3981 (pre-existing) | 6 | PASS |
| `pcm_test_box` | `--test all_tests teacher_validation_pcm_box::` | PR-B #4120 | 8 | PASS (BLOCKER branch — see below) |

Baseline counts recorded at `tests/reference_data/teacher_validation_suite/baseline.json`. Per `AGENTS.md` / `RULES.md`, the baseline must never be loosened to hide a regression.

## BLOCKER branch on the PCM sub-suite

Per `tests/reference_data/pcm_test_box/PROVENANCE.md` §"BLOCKER", the Mazzeo et al. RT27 experimental solid-fraction-vs-time curve is **not available** within the spike window. The PCM sub-suite currently validates the `PhaseChangeMaterial` + `PCMTestBox` skeleton (nominal Rubitherm RT27 properties, apparent-heat-capacity method, sentinel `solid_fraction_at_time` return). PR-B+1 unlocks the real reference-data matching once a licensable RT27 curve is identified (criteria in `PROVENANCE.md` §"Unlock criteria").

## Run locally

```bash
# 1. Capture each sub-suite's cargo test output to a log file.
cargo test -p fluxion --test all_tests ashrae_140_validator_selector_parity:: 2>&1 \
    | tee /tmp/ashrae_140_fabric.log
cargo test -p fluxion --test all_tests conduction_1052rp_analytical:: 2>&1 \
    | tee /tmp/conduction_1052rp.log
cargo test -p fluxion --test all_tests teacher_validation_pcm_box:: 2>&1 \
    | tee /tmp/pcm_test_box.log

# 2. Aggregate the verdict.
python3 scripts/check_teacher_validation_suite.py \
    --log ashrae_140_fabric=/tmp/ashrae_140_fabric.log \
    --log conduction_1052rp=/tmp/conduction_1052rp.log \
    --log pcm_test_box=/tmp/pcm_test_box.log

# Exit code: 0 if all PASS, 1 if any FAIL, 2 on argument errors.
```

To re-run a single sub-suite (nightly debugging):

```bash
python3 scripts/check_teacher_validation_suite.py \
    --sub-suite ashrae_140_fabric \
    --log ashrae_140_fabric=/tmp/ashrae_140_fabric.log
```

The other sub-suites are reported as `FILTERED` in the output and do not contribute to the verdict.

## How it's wired

- **Workflow**: `.github/workflows/teacher_validation_suite.yml` — cron `0 6 * * *` (06:00 UTC nightly, matching the existing nightly cadence per `AGENTS.md` / ADR-0016). `workflow_dispatch` accepts an optional `--sub-suite` filter for ad-hoc re-runs.
- **Script**: `scripts/check_teacher_validation_suite.py` — pure-function verdict logic + `argparse` CLI. Mirrors `scripts/check_strict_energy_gate_regression.py` shape.
- **Pytest tests**: `scripts/ci/test_check_teacher_validation_suite.py` — 10 behavior tests covering pure functions + the CLI. Wired into `scripts-tests.yml`.
- **Baseline**: `tests/reference_data/teacher_validation_suite/baseline.json` — initial 21-test PASS state.
- **Release-gate registration**: `release_gates.yaml::ci.nightly_authority` — `"Teacher Validation Suite (Issue #3986)"`.

## Verdict semantics

- `PASS`: every evaluated sub-suite's `cargo test --` output contains a `test result: ok.` line.
- `FAIL`: any evaluated sub-suite reports `test result: FAILED.` (or no `test result:` line at all, which is treated as a hard failure to surface broken pipelines).
- `FILTERED`: not evaluated (only seen with `--sub-suite`).

The aggregate verdict is **PASS iff every sub-suite PASSes**; a single FAIL flips the overall verdict to FAIL. This conservative policy ensures teacher-path regressions cannot be masked by other sub-suites passing.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Missing --log entries for sub-suites: <names>` | Forgot to pass `--log` for one or more sub-suites (and no `--sub-suite` filter applied) | Re-run with `--log NAME=PATH` for each sub-suite |
| `<sub-suite>: FAIL (pass=N, fail=M, ...)` | One or more tests in the sub-suite failed | Run the cargo test command directly to see the failing test names; this gate does NOT diagnose — it only aggregates |
| `ERROR: --log PATH does not exist for sub-suite <name>: <path>` | Path passed in `--log` doesn't exist on disk | Verify the path; usually a typo or the test step didn't run |
| `test result:` line not found | Cargo test output was empty (e.g. compilation failure); gate treats this as FAIL | Run `cargo test ... 2>&1 | head -50` to see compilation errors |
| Workflow failed but local `cargo test` passes | The nightly runner may have stale build artifacts; the workflow caches them | Push a no-op commit to invalidate the cache, or run `cargo clean` locally |

## Out of scope

- Promoting this gate to `ci.required_checks` — per the user's PR-C scope decision, this gate is **advisory only**. Lane-3 nightly authority per ADR-0016.
- Hardening the script with retry / rate-limiting / distributed execution — premature for a 21-test nightly.
- Real PCM physics — PR-B+1 territory.
- Coupling to ASHRAE 140 fabric tests that *use* the selector wiring — PR-A+1 territory.

## Related issues

- **#3986** — Meta-issue (teacher validation suite). Closes the three-PR split.
- **#4116** — This PR-C sub-issue.
- **#4117** — PR-A sub-issue (selector wiring through ASHRAE 140 validator; merged at #4119).
- **#4118** — PR-B sub-issue (PCM test-box skeleton; merged at #4120).
- **#3981** — Pre-existing 1052-RP wiring; one of the three sub-suites.
