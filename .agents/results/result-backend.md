# Backend — Issue #1847 OSimFlow pytest coverage

**Status:** ✅ Complete
**Date:** 2026-07-18
**Branch:** `fix/issue-1847-establish-pytest-coverage`
**PR (anchapin/fluxion#1847):** see `gh pr view` once the push completes.

## Summary

Established pytest infrastructure for the OSimFlow Python orchestration
layer (`scripts/`) and added baseline unit tests for the three largest
shipped-in-untested modules:

- `scripts/cloud_campaign_manager.py` (1,274 LoC; AWS/Nomad/DynamoDB)
- `scripts/autonomous_parameter_sweep.py` (653 LoC; ASHRAE 140 divergence harness)
- `scripts/ashrae_benchmark_harness.py` (551 LoC; benchmark runner + delta reporter)

## Files Changed

```
A  .github/workflows/python-tests.yml        # CI: py 3.10–3.13 matrix on Ubuntu, no cloud creds
A  scripts/README.md                         # Documents the scripts/ci layout + fixtures
A  scripts/ci/__init__.py                    # Test-package marker
A  scripts/ci/test_ashrae_benchmark_harness.py
A  scripts/ci/test_autonomous_parameter_sweep.py
A  scripts/ci/test_cloud_campaign_manager.py
A  scripts/conftest.py                       # Shared fixtures (fake_aws_clients, populated_state_store,
                                            # fake_urlopen, fake_subprocess, temp_trace_dir, ...)
A  scripts/pytest.ini                        # pytest-asyncio auto, coverage reporting, scripts/ci testpaths
A  scripts/requirements-test.txt             # pytest>=8, pytest-asyncio, pytest-cov, boto3
M  requirements-dev.txt                      # Added pytest>=8, pytest-asyncio, pytest-cov
M  scripts/cloud_campaign_manager.py        # Resolved 11 unresolved git-merge-conflict markers
                                            # (webhook HEAD vs email 1d0e1c8) so the file parses;
                                            # both implementations now coexist.
```

## Acceptance Criteria — checklist

- [x] **Task 1 — Pytest scaffolding**
  - [x] `scripts/pytest.ini` configuring pytest, pytest-asyncio, pytest-cov
  - [x] `scripts/conftest.py` with shared fixtures (temp campaign state, sample model spec, mocked subprocess)
  - [x] `scripts/ci/` directory populated; layout documented in `scripts/README.md`

- [x] **Task 2 — Baseline unit tests**
  - [x] `scripts/ci/test_cloud_campaign_manager.py` (51 tests):
    - campaign-state serialization round-trip (dataclass + JSON)
    - sweep-config validation (`generate_grid_points`, `generate_random_points`)
    - mocked S3 / DynamoDB / Lambda job-spec generation (`create_campaign`, `trigger_aggregator`)
    - state-store aggregation (`check_campaign_progress`)
    - notification paths (webhook / email / SNS), idempotency, transport errors
  - [x] `scripts/ci/test_autonomous_parameter_sweep.py` (23 tests):
    - parameter-space enumeration (grid + random)
    - early-termination logic (`tolerance_mae` abort)
    - result aggregation (JSONL + CSV logs, sweep_state persistence)
    - subprocess timeout / error handling
    - divergence-report markdown rendering
  - [x] `scripts/ci/test_ashrae_benchmark_harness.py` (30 tests):
    - benchmark-config parsing (regex group: `TestParseValidationOutput`)
    - **MAE + pass-rate edge cases**: `inf%`, leading whitespace, missing summary
    - per-case ValidationCase construction (including ref-range `inf`)
    - report rendering (`print_summary`, `print_delta`)
    - GitHub-Actions step summary writer

- [x] **Task 3 — CI wiring**
  - [x] `.github/workflows/python-tests.yml` on Ubuntu
  - [x] Python 3.10 / 3.11 / 3.12 / 3.13 matrix
  - [x] No `DWAVE_API_TOKEN`, no Redis, no Kubernetes, no cloud credentials;
        the workflow `unset`s them defensively and all externals are mocked via
        `scripts/conftest.py` fixtures.
  - [x] Enforces ≥60% line coverage on each of the three target files
        via inline `coverage.xml` post-check.

## Coverage Achieved

```
scripts/ashrae_benchmark_harness.py       278     45    84%   ✅
scripts/autonomous_parameter_sweep.py     263     43    84%   ✅
scripts/cloud_campaign_manager.py         449    144    68%   ✅
```

All three target files exceed the 60% acceptance threshold.

104 tests pass on Python 3.12 (local), pytest-asyncio in `auto` mode,
`pytest-cov` configured with `--cov=scripts --cov-report=term-missing`.

## How to run

```bash
pip install -r requirements-dev.txt -r scripts/requirements-test.txt
pytest scripts/ci/ -c scripts/pytest.ini
```

Or directly via the workflow on a PR:
`.github/workflows/python-tests.yml`.

## Blockers / Notes

- **`scripts/cloud_campaign_manager.py` had 11 unresolved git-merge-conflict
  markers** (`<<<<<<< HEAD` / `=======` / `>>>>>>> 1d0e1c8`) at the time
  this branch was created. The file was syntactically broken and could
  not be imported at all. Resolved by keeping both webhook (HEAD) AND
  email (1d0e1c8) implementations intact — `send_completion_notification`
  now handles both channels. The `create_campaign` signature also accepts
  `webhook_url` and `email_config` simultaneously.

- Coverage on `cloud_campaign_manager.py` lands at 68% — well above the
  60% requirement — but the `main()` argparse dispatcher (`if args.action
  == "..."`) is intentionally untested because it requires end-to-end
  fixture wiring beyond the issue scope. Future test expansions (CLI
  parametrization, end-to-end flows) could push the file to 80%+ without
  touching the merge resolution.

- `get_campaign_state` only suppresses `ClientError` when `Error.Code == "404"`
  (literal string match). Real S3 returns `"NoSuchKey"`. This is a latent bug
  in the source but was left untouched per issue scope ("Do not modify files
  outside the scope of this issue"). Tests use a `Code="404"` mock.
