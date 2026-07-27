# Tensor Dataset Output Schema (FTDS v1)

> Versioned tensor format for ML surrogate training data. Produced by the
> Rust writer in `src/ai/tensor_dataset.rs` (Issue #1778, plan key T5.3) and
> consumed by the Phase 4 surrogate trainers.

This schema defines the on-disk layout for sharded tensor datasets that turn
physics-solver outputs (`BatchResults` / `SimulationOutput`) into stable
training tensors. It is the Rust-native counterpart to the Parquet-based
`docs/synthetic_data_schema.md` and exists to ship dense `f64` tensors without
a Python/Parquet runtime dependency. The format is named **FTDS** (Fluxion
Tensor Dataset Shard).

---

## Schema Version

Current version: **1.0.0** (semver; encoded into every shard header).

- **Major bump**: breaking layout change (magic bytes, header field order,
  dtype semantics, footer scheme). Readers refuse incompatible majors.
- **Minor bump**: additive fields in the JSON sidecar (readers ignore unknown
  keys).
- **Patch bump**: documentation/validation-only changes.

Constant: `fluxion::ai::tensor_dataset::TENSOR_DATASET_SCHEMA_VERSION`.

The reader's compatibility rule is **major == 1** (`is_compatible_version`).
A shard tagged `2.x.x` is rejected with `UnsupportedVersion`.

---

## Dataset Layout

A dataset is a directory:

```
my_dataset/
├── manifest.json          # TensorDatasetManifest (top-level index)
├── shard-000000.ftds      # FTDS binary shard (≤ shard_size samples)
├── shard-000001.ftds
└── ...
```

### `manifest.json`

```json
{
  "schema_version": "1.0.0",
  "created_at_utc": "2026-07-26T12:34:56Z",
  "dtype": "F64",
  "n_samples_total": 4096,
  "n_input_features": 3,
  "input_feature_names": ["window_u_value", "heating_setpoint", "cooling_setpoint"],
  "target_names": [
    "total_energy_kwh", "peak_heating_load_w", "peak_cooling_load_w",
    "annual_heating_kwh", "annual_cooling_kwh", "eui_kwh_m2"
  ],
  "has_timeseries": false,
  "timeseries_length": 0,
  "normalization": null,
  "shards": [
    { "path": "shard-000000.ftds", "n_samples": 1024,
      "sha256": "ab12…" }
  ]
}
```

---

## Shard Binary Layout (`.ftds`)

All multi-byte integers are **little-endian**. The file is the concatenation
of: fixed header, JSON sidecar, three payload arrays, sample IDs, and a
SHA-256 footer.

| Offset | Length | Field | Notes |
|--------|--------|-------|-------|
| 0   | 4  | `magic`            | ASCII `"FTDS"` = `[0x46, 0x54, 0x44, 0x53]` |
| 4   | 6  | `schema_version`  | Three `u16` LE: major, minor, patch |
| 10  | 1  | `dtype`           | `0 = F32` (reserved), `1 = F64` (only emitted value in v1.x) |
| 11  | 8  | `n_samples`       | `u64` LE; must be `> 0` |
| 19  | 8  | `n_input_features`| `u64` LE |
| 27  | 8  | `n_targets`       | `u64` LE |
| 35  | 1  | `has_timeseries`  | `0` or `1` |
| 36  | 8  | `timeseries_length` | `u64` LE (0 when `has_timeseries == 0`) |
| 44  | 4  | `sidecar_len`     | `u32` LE — byte length of JSON sidecar |
| 48  | `sidecar_len` | `sidecar_json` | `TensorShardHeader` JSON (see below) |
| …   | `n_samples * n_input_features * 8` | `inputs`    | Row-major `f64` LE `[N, F]` |
| …   | `n_samples * n_targets * 8`        | `targets`   | Row-major `f64` LE `[N, T]` |
| …   | `n_samples * timeseries_length * 8`| `timeseries`| Row-major `f64` LE `[N, L]` (omitted when `has_timeseries == 0`) |
| …   | `n_samples * 8` | `sample_ids`     | `u64` LE per sample |
| end - 32 | 32 | `sha256_footer` | SHA-256 of *everything preceding* |

**Total file size** is therefore fully determined by the header; a reader can
verify integrity and reject truncation without scanning the payload.

### `TensorShardHeader` JSON sidecar

```json
{
  "schema_version": "1.0.0",
  "dtype": "F64",
  "n_samples": 1024,
  "n_input_features": 3,
  "n_targets": 6,
  "has_timeseries": false,
  "timeseries_length": 0,
  "input_feature_names": ["window_u_value", "heating_setpoint", "cooling_setpoint"],
  "target_names": ["total_energy_kwh", "peak_heating_load_w", "peak_cooling_load_w",
                   "annual_heating_kwh", "annual_cooling_kwh", "eui_kwh_m2"]
}
```

The sidecar **must** agree with the fixed header fields (`n_samples`,
`n_input_features`, `n_targets`, `has_timeseries`, `timeseries_length`,
`dtype`); any mismatch is rejected by the parser.

---

## Column Semantics

### Default inputs (positional)

Derived from `ParameterSample::parameters` (see
`src/ai/batch_runner.rs::ParameterSpec`):

| Index | Name              | Unit     |
|-------|-------------------|----------|
| 0     | `window_u_value`  | W/m²K    |
| 1     | `heating_setpoint`| °C       |
| 2     | `cooling_setpoint`| °C       |

### Default targets (order is stable; append-only across versions)

| Index | Name                  | Unit |
|-------|-----------------------|------|
| 0     | `total_energy_kwh`    | kWh  |
| 1     | `peak_heating_load_w` | W    |
| 2     | `peak_cooling_load_w` | W    |
| 3     | `annual_heating_kwh`  | kWh  |
| 4     | `annual_cooling_kwh`  | kWh  |
| 5     | `eui_kwh_m2`          | kWh/m² |

### Optional timeseries

When `has_timeseries == true`, each sample carries an `L`-length zone
temperature trace (°C). `L` is fixed per dataset (`timeseries_length`).

---

## Validation Rules

`validate_shard(path, manifest)` and `validate_dataset_dir(dir)` enforce:

1. **Magic** bytes equal `b"FTDS"`.
2. **Schema major** is `1`.
3. **dtype** is `F64` (v1.x).
4. **n_samples > 0** (empty shards rejected).
5. **Sidecar / fixed-header consistency** for shapes, dtype, names.
6. **Payload length** matches the declared shapes (no truncation, no padding).
7. **Footer SHA-256** of the body matches the stored digest.
8. **Finiteness**: no `NaN`/`Inf` in inputs, targets, or timeseries.
9. **Manifest linkage**: every shard in `manifest.json` exists on disk, its
   stored `sha256` matches the file, and sample counts sum to
   `n_samples_total`.

Any violation produces a `ValidationReport { ok: false, errors: [...] }`.

---

## Rust API

```rust
use fluxion::ai::tensor_dataset::{
    TensorDatasetWriter, TensorSample, TensorFeatureSpec,
    validate_dataset_dir,
};

let feature_spec = TensorFeatureSpec::defaults();
let target_names: Vec<String> = TensorSample::default_target_names()
    .iter().map(|s| s.to_string()).collect();

let mut writer = TensorDatasetWriter::new(
    std::path::Path::new("data/dataset"),
    feature_spec,
    target_names,
    0,        // timeseries_length (0 disables)
    1024,     // shard_size
)?;

writer.push(sample)?;            // streams + flushes shards
let manifest = writer.finish()?; // writes manifest.json

let report = validate_dataset_dir(std::path::Path::new("data/dataset"))?;
assert!(report.ok);
```

### Formatting from physics outputs

```rust
use fluxion::ai::tensor_dataset::{TensorSample, batch_results_to_samples};

// single sample
let ts = TensorSample::from_simulation_output(
    &sim_output, &params, &feature_names, /* timeseries_length */ 0)?;

// whole batch (skips failed simulations)
let extracted = batch_results_to_samples(&batch, &manifest, &feature_names, 0);
for sample in extracted.samples { /* ... */ }
```

---

## Design Notes

- **Why a custom binary format?** HDF5 requires a system C library and
  `ndarray`-to-HDF5 glue that would inflate the published crate beyond the
  10 MB crates.io cap. NPZ/zip doubles the writer surface for no gain at our
  scale. FTDS is ~300 lines of pure Rust using only existing dependencies
  (`serde`, `serde_json`, `sha2`, `ndarray`) and is fully self-describing.
- **Why `f64` only in v1?** `SimulationOutput` carries `f64` everywhere.
  Introducing `f32`/`bf16` is a minor-version additive change (the dtype tag
  already reserves it).
- **Why per-shard checksums?** Sharded datasets are copied across nodes and
  storage tiers; a per-file SHA-256 lets the loader reject bit-rot without
  re-running the producing physics.

## References

- Issue #1778 (T5.3) — acceptance criteria.
- Depends on T5.2 (`src/ai/batch_runner.rs` `BatchResults`).
- `docs/synthetic_data_schema.md` — the companion Parquet schema for the
  Python-side `v2.1` pipeline.
- `src/ai/tensor_dataset.rs` — reference implementation.
