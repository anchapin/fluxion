# On-demand diagnostic test scripts

These files are **diagnostic-only** investigation tools — `#[ignore]`'d tests that
print intermediate physics quantities (energy attribution tables, solar-gain
decompositions, mass-trajectory traces, etc.) for the ongoing ASHRAE 140
high-mass investigations (issues #2452 / #2453 / #2454 / #917, routed to the
GaugeSolver rework #1465 / #1462). They hold no regression value today because
the physics they inspect is **known-broken** (see `docs/KNOWN_ISSUES.md`
§LIMIT-05 / §SOLAR-02); gating them as real assertions would fail CI on every PR
until that rework lands.

## Why they live here (not directly under `tests/`)

Cargo auto-discovers every `tests/*.rs` as a separate integration-test target
and compiles it on every `cargo test` / `cargo check --tests`. Files placed in
this subdirectory are **not** auto-discovered, so they are:

- **Not compiled** by `cargo test`, `cargo check --tests`, clippy, or the CI
  build — zero build-time / maintenance cost on the main test tree.
- **Not run** by any CI gate — no false "ignored" noise in test summaries.

This resolves issue #2708 (per the #2536 quarantine policy): the diagnostic
tooling is preserved for investigators, but it no longer inflates the gated
test tree with zero-signal `#[ignore]` entries.

## Running a diagnostic on demand

Because these scripts are not declared test targets, pick the option that best
fits your workflow:

**Option A — temporary target (recommended for a one-off run):** copy the file
back into `tests/`, run it, then remove it.

```sh
cp tests/diagnostics/diag_phim.rs tests/_tmp_diag.rs
cargo test --profile ci --test _tmp_diag -- --ignored --nocapture
rm tests/_tmp_diag.rs
```

**Option B — declare an explicit `[[test]]` target** in `Cargo.toml` while you
are actively iterating, then remove the entry when done:

```toml
[[test]]
name = "diag_phim"
path = "tests/diagnostics/diag_phim.rs"
```

```sh
cargo test --profile ci --test diag_phim -- --ignored --nocapture
```

> Whichever option you choose, **do not commit** a `[[test]]` entry or a
> `tests/_tmp_*.rs` file — that would re-add the diagnostic to the default
> build and defeat the purpose of this directory.

## Converting a diagnostic into a real gated test

Once the underlying physics (GaugeSolver #1465 / #1462) lands and a diagnostic's
printed values fall inside an ASHRAE 140 tolerance band, promote it: move the
file back under `tests/`, replace the `println!` dump with a real `assert!`
against the reference data in `tests/reference_data/`, and drop the `#[ignore]`.
See `docs/KNOWN_ISSUES.md` for the current status of each investigation.

## Inventory

| File | Investigation |
|------|---------------|
| `diag_917_energy.rs` | Case 600FF energy-balance / 917 diagnostics |
| `diag_917_solar.rs`  | Case 600FF solar-gain / 917 diagnostics |
| `diag_917_v2.rs`     | Case 917 v2 diagnostics |
| `diag_phim.rs`       | phi*_m / solar-noon peak-day trace |
| `diag_mass_traj.rs`  | Mass-node temperature trajectory |
| `diag_solfields.rs`  | Solar field inspection |
| `diag_solar_hr.rs`   | Hourly solar diagnostic |
| `diag_check.rs`      | Peak-temperature sanity dump |
| `case_940_setback_diagnostic.rs` | Case 940 setback attribution (#2452) |
| `case_920_orientation_attribution.rs` | Case 920 per-orientation solar decomposition (#2454) |

Quarantined per #2536; relocated per #2708.
