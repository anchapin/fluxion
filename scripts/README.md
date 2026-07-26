# OSimFlow Python Orchestration

> Cloud- and local-orchestration Python scripts that drive Fluxion
> simulation campaigns. This README documents the test infrastructure
> added in Issue #1847.

## Layout

```
scripts/
├── cloud_campaign_manager.py        # AWS/Nomad campaign manager (Issue #1192)
├── autonomous_parameter_sweep.py    # Local MAE-divergence sweep (Issue #1450)
├── ashrae_benchmark_harness.py      # ASHRAE 140 benchmark harness (Issue #1488)
├── check_osimflow_coverage.py       # Per-file coverage gate (Issue #1864)
├── state_store.py                   # DynamoDB/Redis/in-memory backend (T7.3)
├── conftest.py                      # Shared pytest fixtures (see below)
├── pytest.ini                       # Pytest config scoped to this directory
├── requirements-test.txt            # Test-only Python dependencies
└── ci/
    ├── __init__.py                  # Marker for the test package
    ├── test_cloud_campaign_manager.py
    ├── test_autonomous_parameter_sweep.py
    ├── test_ashrae_benchmark_harness.py
    └── test_check_osimflow_coverage.py
```

## Running the tests

```bash
# Install test dependencies (idempotent).
pip install -r scripts/requirements-test.txt

# Run the full OSimFlow test suite with coverage.
pytest scripts/ci/ -c scripts/pytest.ini

# Targeted runs.
pytest scripts/ci/test_ashrae_benchmark_harness.py -c scripts/pytest.ini

# Coverage report only (text + html under .agents/coverage/python-osimflow/)
pytest scripts/ci/ -c scripts/pytest.ini --cov=scripts --cov-report=term-missing
```

Coverage targets (Issue #1847 acceptance criteria):

| File                                       | Line coverage target |
| ------------------------------------------ | -------------------- |
| `scripts/cloud_campaign_manager.py`        | ≥ 60 %               |
| `scripts/autonomous_parameter_sweep.py`    | ≥ 60 %               |
| `scripts/ashrae_benchmark_harness.py`      | ≥ 60 %               |

## Shared fixtures (`scripts/conftest.py`)

* `_scrub_cloud_env` (autouse) — strips AWS / DWAVE / state-store env vars
  so a stray credential never leaks into CI.
* `fake_aws_clients` — in-memory S3 / SNS / DynamoDB / STS / Lambda fakes
  keyed off `get_aws_clients()`.
* `in_memory_state_store` and `populated_state_store` —
  `state_store.InMemoryStateStore`, optionally pre-loaded with 5 work
  units in varied `TaskStatus` states.
* `fake_urlopen` — patches `urllib.request.urlopen` to return a fake
  HTTP response, raise, or invoke a capture callback.
* `fake_subprocess` — patches `subprocess.run` / `subprocess.check_output`
  so the parameter-sweep and benchmark-harness scripts run hermetically.
* `fluxion_model_spec` — deterministic `FluxionModel` parameter spec.
* `temp_trace_dir`, `temp_campaign_db` — `tmp_path`-backed scratch dirs.

## CI workflow

`.github/workflows/python-tests.yml` runs the suite on Ubuntu for
Python 3.10 / 3.11 / 3.12 / 3.13, requires no cloud credentials, no
D-Wave token, no Redis, and no Kubernetes context.

## Per-file coverage gate

`scripts/check_osimflow_coverage.py` enforces the ≥ 60 % line-coverage
threshold on each target file from the Cobertura XML produced by the
pytest run. It is invoked by the workflow's `Enforce per-file coverage
thresholds` step:

```bash
pytest scripts/ci/ -c scripts/pytest.ini \
  --cov-report=xml:scripts/ci/coverage.xml
python scripts/check_osimflow_coverage.py scripts/ci/coverage.xml
```

The checker normalizes the `filename` attribute emitted by coverage.py
(which is relative to `--cov=scripts`, e.g. `cloud_campaign_manager.py`)
onto the canonical `scripts/<name>` keys, and counts lines from both the
Cobertura summary attributes and the per-line `<line hits>` records that
coverage.py >= 6 emits. Tests live in `scripts/ci/test_check_osimflow_coverage.py`
(Issue #1864 — the previous inline-YAML gate mismatched paths and read
absent summary attributes, turning every OSimFlow pytest job red).

## Conventions

* Tests use **plain `pytest` fixtures** — no class-based unittest unless
  a group of tests needs shared state. (See
  `test_ashrae_benchmark_harness.py::TestParseValidationOutput` for the
  only class used here — it groups regex edge cases.)
* `pytest-asyncio` runs in `auto` mode (`asyncio_mode = auto`); mark a
  coroutine with `async def` only when the code under test is async.
* AWS / SNS / webhook targets are always mocked. Never hit a real
  endpoint from a test.
* New helpers belong in `conftest.py`, not in individual test modules.
