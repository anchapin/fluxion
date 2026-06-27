# ADR-0004: ONNX model versioning + golden-output regression harness

- **Status**: Accepted
- **Date**: 2026-06-27
- **Issue**: [#1335](https://github.com/anchapin/fluxion/issues/1335)
- **Owners**: Surrogate / ML track
- **Supersedes**: none
- **Superseded by**: none

## Context

`SurrogateManager::ModelMetadata::model_version` defaulted to the literal
string `"0.0.0"` (see `src/ai/surrogate.rs:393`). There was no SHA-256 of
the ONNX bytes, no ONNX opset version, no training-data hash, and no
frozen-output regression test. The post-#1323 retraining pipeline was
expected to ship a `v3.1` release; without pinned semantic versioning and a
golden-output harness, a silent regression in the retrained weights could
ship unnoticed.

The ONNX files themselves are **not committed to git** (binary artefacts,
hundreds of MB each). Therefore the version contract lives in a small
JSON registry, and the binary is delivered out-of-band (CI artifact store
or model registry).

## Decision

### 1. Versioning schema — strict semver

`ModelMetadata::model_version` follows strict semver
(`MAJOR.MINOR.PATCH[-prerelease][+build]`), with each numeric component in
`0..=999` and no leading `v`. The placeholder `"0.0.0"` is **rejected**
by `ModelMetadata::with_semver` so the default-constructed metadata
cannot be silently released.

```rust
ModelMetadata::with_semver("3.1.0").is_ok();
ModelMetadata::with_semver("0.0.0").is_err(); // VersionError::PlaceholderVersion
ModelMetadata::with_semver("3.1"  ).is_err(); // VersionError::InvalidSemver
ModelMetadata::with_semver("v3"   ).is_err(); // VersionError::InvalidSemver
```

Errors are typed (`VersionError`) so callers can distinguish a malformed
semver from a hash mismatch or an unsupported opset.

### 2. Registry — JSON file with hash-based identity

A small registry file enumerates pinned model versions:

```json
{
  "versions": [
    {
      "version": "3.1.0",
      "model_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "onnx_opset_version": 17,
      "training_data_hash": "0000000000000000000000000000000000000000000000000000000000000000",
      "trained_on": "2026-06-27",
      "training_data_summary": "ASHRAE 140 cases 600-960 + Denver TMY3 100k timesteps",
      "expected_accuracy": 0.0,
      "model_path": "models/surrogate_v3.1.0.onnx"
    }
  ]
}
```

The `.onnx` file is **never** in git; `model_path` points at a location
in the CI artifact store or model registry.

`ModelRegistry` parses this file with `ModelRegistry::from_json_str`,
validates each entry through `ModelVersion::new`, and exposes `lookup`,
`latest`, `len`, `is_empty`.

### 3. Load path — hash-checked `SurrogateManager::load_version`

```rust
let registry = ModelRegistry::from_json_str(&fs::read_to_string("registry.json")?)?;
let m = SurrogateManager::load_version("3.1.0", &registry)?;
```

`load_version`:

1. looks up the version in the registry (typed error if missing),
2. checks the file exists (typed error if not),
3. computes the file's SHA-256 with `compute_file_sha256` (sha2 crate),
4. compares it case-insensitively against `model_sha256`,
5. **only then** delegates to the existing `load_onnx`.

A build without the `ort` feature still validates the hash and returns a
typed `requires ort` error so the failure mode is obvious in CI logs.

### 4. Golden-output regression harness

`tests/surrogate_golden_output.rs` runs **100 fixed inputs** through
`SurrogateManager::deterministic_analytical_loads` (a new, time-free
analytical helper added in this change) and asserts per-tensor
element-equality against `tests/surrogate_models/golden/golden_v3_1_0.json`
with **max relative error ≤ 1e-6**.

The deterministic helper exists because the pre-existing
`analytical_loads(temps)` calls `SystemTime::now()` for the daily solar
cycle, which makes it unusable for frozen-output testing. The new helper
mirrors the shape of the legacy formula but is a pure function of its
inputs.

The harness does **not** require an ONNX model to run — the test stays
green even when `model_sha256` is the all-zero placeholder. The hash
machinery is exercised by a separate `load_version_rejects_hash_mismatch`
test that uses a synthetic temp file with a wrong hash and asserts that
`load_version` errors out.

### 5. Hash utilities

```rust
pub fn compute_file_sha256(path: &Path) -> Result<String, String>;
pub fn compute_bytes_sha256(bytes: &[u8]) -> String;
pub fn validate_hash(expected: &str, actual: &str) -> Result<(), String>;
pub fn validate_sha256_hex(hash: &str) -> Result<(), VersionError>;
```

The `sha2` crate is already a transitive dependency
(`src/validation/reference_loader.rs`).

### 6. Semver bump rules

| Change | Bump |
|--------|------|
| Weights re-trained on the same training set, no schema change | PATCH |
| Training data refresh (e.g. add ASHRAE 140 case 980) | MINOR |
| I/O tensor shape change, input/output rename | MAJOR |
| ONNX opset change | MINOR (warn: opset 17 is the upper bound per scope) |
| Float precision change (FP32 → FP16) | MAJOR (downstream tolerances change) |

### 7. When the golden file must be re-baselined

The golden file **must** be re-baselined when any of the following change:

- `SurrogateManager::deterministic_analytical_loads` (or any function it
  calls) is intentionally modified;
- the surrogate's *expected* behavior changes for valid inputs (e.g.
  retraining produces outputs that diverge from the analytical fallback
  by more than the agreed tolerance);
- the input schema (the `inputs` block of the JSON) is restructured.

Re-baseline procedure:

```bash
python3 scripts/gen_golden_outputs.py > tests/surrogate_models/golden/golden_v3_1_0.json
git add tests/surrogate_models/golden/golden_v3_1_0.json
# In the PR description, justify why each re-baselined value moved.
```

If only the *outputs* move but the *inputs* are unchanged, the diff is
expected to be a small numeric change; reviewers should require
engineering justification in the PR body. **The CI gate (item 8) will
fail automatically** when the JSON file changes between the base branch
and the PR head.

### 8. CI gate placement

A new CI step runs after the existing `cargo test --features ort`
matrix:

```yaml
- name: Surrogate golden-output regression
  run: |
    cargo test --features ort --test surrogate_golden_output -- --nocapture
    git diff --exit-code tests/surrogate_models/golden/golden_v3_1_0.json
```

The `git diff --exit-code` check fails the build when the committed
golden file differs from the freshly-generated output of the same inputs
— i.e. when a contributor changed either the inputs or the deterministic
helper without re-baselining. Combined with the in-test relative-error
check (`assert_close_envelope`), this catches both intentional
un-flagged changes (via the diff step) and unintentional numerical drift
(via the tolerance check).

## Consequences

### Positive

- Silent retraining regressions become visible before merge.
- The placeholder `"0.0.0"` can no longer ship as a release tag.
- `load_version` is a single audited entry point — no callers need to
  know about hash checking.
- Registry is small (kilobytes), reviewable in a PR diff, and doesn't
  pollute the repo with binary artefacts.

### Negative

- One new deterministic helper that must stay in sync with the legacy
  `analytical_loads` formula. Drift is caught by the golden test but
  may surprise contributors.
- CI must regenerate + diff the golden file; this adds ~5 seconds to
  the test matrix.
- Two `ModelMetadata` fields (sha256 / opset) are `Option` so legacy
  default-constructed metadata still compiles — see follow-up #1402
  for the migration plan to required fields.

## Out of scope (per issue body)

- Retraining the surrogate (owned by C#1, depends on #1323).
- Bindings / Python API surface (D#).
- ONNX opset upgrade beyond opset 17.
- CPU/CUDA backend parity (owned by C#3).

## Verification

```bash
cargo test --features ort --lib ai::surrogate
cargo test --features ort --test surrogate_golden_output
cargo test --features ort --test surrogate_config
cargo clippy --lib --features ort -- -D warnings
```

All four must pass on `fix/issue-1335-onnx-versioning`.