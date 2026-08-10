# Adding a New ASHRAE 140 Case

Walkthrough for registering a new Standard 140 (BESTEST) case end-to-end:
where the case spec lives, how to register it in the validator, where the
reference data goes, and which CI gates must stay green.
Audience: contributors who already read `AGENTS.md` §Validation Strategy and
`ARCHITECTURE.md` §"Cycle break (#1441)". Docs only — no code changes here.
Closes #2542.

*Last Updated: 2026-08-10*

## TL;DR — the five touch points

A new ASHRAE 140 case touches **five files** (plus one optional CSV). Nothing
else needs to change:

| # | File | What you add |
|---|------|--------------|
| 1 | `fluxion-core/src/ashrae_cases.rs` | New leaf type **only if** the case introduces a new enum/struct that `sim` or `validation` will share (e.g. a new `Orientation` variant, a new `BuildingType`). Most cases skip this. |
| 2 | `src/validation/ashrae_140_cases.rs` | (a) new variant on `ASHRAE140Case` enum, (b) new `CaseBuilder::case_XXXX()` factory, (c) match arm in `ASHRAE140Case::spec()`. |
| 3 | `src/validation/benchmark.rs` | `data.insert("XXX".to_string(), BenchmarkData { … })` entry with ASHRAE 140-2023 reference min/max ranges. |
| 4 | `src/validation/ashrae_140_validator.rs` | Add `ASHRAE140Case::CaseXXX` to the `cases` vec in `validate_with_diagnostics` (and `validate_analytical_engine` if it should run by default). |
| 5 | `tests/ashrae_140_validation.rs` | Add the case ID to the `case_ids` array in `test_all_cases_instantiation`. |
| 6 (opt) | `tests/reference_data/ashrae140/monthly/case_XXX_monthly_reference.csv` | Monthly EnergyPlus reference series, if you want bottom-up module checks alongside the annual benchmark. |

That's the whole surface. The rest of this guide walks each step, then shows
the verification commands and the cycle-guard contract you must not break.

## Step 0 — Pick the case number and decide on leaf types

ASHRAE 140 cases are identified by a number (`600`, `610`, `650FF`, `900`,
`195`, …). Look at the existing variants on `pub enum ASHRAE140Case`
(`src/validation/ashrae_140_cases.rs:61`) and follow the naming pattern:
`Case<number>` for standard cases, `Case<number>FF` for free-floating,
`Case<number><Modifier>` for diagnostic variants (e.g. `Case195HighMass`).

**Decide now: does this case introduce a *new shared domain type*?** A shared
domain type is any enum/struct that both `src/sim/**` and `src/validation/**`
need to import. Examples: a new compass direction on `Orientation`, a new
`BuildingType`, a new `GlassType`. If yes, the type lives in `fluxion-core`;
if no, skip Step 1.

## Step 1 — (Optional) Add a shared leaf type to `fluxion-core`

The cycle-breaking rule from issue #1441 (see `ARCHITECTURE.md` §"Cycle break
(#1441 — ASHRAE-140 leaf types → `fluxion-core`)") is:

> All shared ASHRAE 140 domain types — `Orientation`, `WindowArea`,
> `ConstructionType`, `ShadingType`, `ShadingDevice`, `GlassType`,
> `WindowSpec`, `InternalLoads`, `HvacSchedule`, `NightVentilation`,
> `BuildingType`, `GeometrySpec`, `ConductanceReferences` — live in
> `fluxion-core/src/ashrae_cases.rs`, **not** in `src/validation/`.

This file is a *leaf*: it may only depend on `serde` and `std`. To add a new
variant, edit `fluxion-core/src/ashrae_cases.rs` and add it to the relevant
`pub enum`. For a brand-new type, add a `pub struct`/`pub enum` with
`#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]` (mirror the
existing types).

> **The cycle guard.** `scripts/check_ashrae_cases_cycle.py` (CI gate
> `Ashrae Cases Cycle Check` #1441) enforces three invariants:
>
> 1. `fluxion-core/src/**/*.rs` must NOT contain any `crate::sim::*`,
>    `crate::physics::*`, `crate::ai::*`, `crate::validation::*`,
>    `crate::interop::*`, … import. **Leaf means leaf.**
> 2. `src/sim/**/*.rs` must NOT `use crate::validation::ashrae_140_cases::Orientation`
>    (or any other type that was hoisted in #1441). Import from
>    `fluxion_core::ashrae_cases::*` instead.
> 3. `fluxion_core::ashrae_cases` must still define all 13 moved leaf types.
>
> If your change breaks any of these, the gate fails and the PR is blocked.
> The fix is always "import from `fluxion_core::ashrae_cases`", never "move
> the type back into `validation`".

## Step 2 — Register the case in the validator's case module

Open `src/validation/ashrae_140_cases.rs`. Three edits, in order:

### 2a. Add a variant to the `ASHRAE140Case` enum

Add `CaseXXX,` to `pub enum ASHRAE140Case` (`src/validation/ashrae_140_cases.rs:61`).
Group it with its siblings (600-series together, 900-series together,
diagnostic 195–470 together, HVAC 800-series together).

### 2b. Write a `CaseBuilder::case_XXX()` factory

Add a `pub fn case_XXX_description() -> CaseSpec` to
`impl CaseBuilder` (around `src/validation/ashrae_140_cases.rs:1271`). Use the
builder API — `CaseBuilder::new()`, `.with_case_id(...)`, `.with_dimensions(...)`,
`.add_zone(...)`, `.with_window(...)`, `.with_hvac(...)`, etc. — and finish
with `.build().expect("case_XXX spec is well-formed")`. Copy a neighbouring
factory (e.g. `case_610_south_shading` at L1752) as a template.

The leaf types you construct with (`Orientation::South`,
`WindowSpec::double_clear_glass()`, `HvacSchedule::constant(...)`,
`NightVentilation::case_650()`, `BuildingType::LowMass`, …) come from
`fluxion_core::ashrae_cases` — they are re-exported at the top of
`ashrae_140_cases.rs`. **Do not** redefine them here.

### 2c. Wire the variant to the factory in `spec()`

Add one match arm to `ASHRAE140Case::spec()` (`src/validation/ashrae_140_cases.rs:796`):

```rust
ASHRAE140Case::CaseXXX => CaseBuilder::case_XXX_description(),
```

The match is exhaustive; the compiler will remind you if you forget this arm.

## Step 3 — Add the ASHRAE 140 reference ranges

Open `src/validation/benchmark.rs` and add an entry to
`get_all_benchmark_data()` (L113). The key is the case ID string; the value is
a `BenchmarkData` struct with `annual_heating_min/max`, `annual_cooling_min/max`,
`peak_heating_min/max`, `peak_cooling_min/max`, and the free-floating min/max
ranges (all in MWh for annual, kW for peak, °C for free-floating).

Reference values come from **ASHRAE Standard 140-2023 Annex B** (raw
multi-program ranges: EnergyPlus, ESP-r, TRNSYS, DOE-2). Cite the source in a
comment above the `data.insert(...)`, exactly like the Case 600 entry at L118
does:

```rust
// Case XXX - <short description>
// ASHRAE 140-2023 Annex B raw reference values (issue #NNNN)
data.insert(
    "XXX".to_string(),
    BenchmarkData {
        annual_heating_min: 4.36,
        // …
    },
);
```

> **Rule (AGENTS.md §Validation Strategy #2):** never tune reference values
> to make a failing case pass. Fix the underlying math. Reference values are
> an external standard, not a free parameter.

For a free-floating-only case, set the annual heating/cooling ranges to
`0.00..0.00` (see Case 650 at L217).

### Optional: add a monthly EnergyPlus reference series

If you also want bottom-up, per-module comparison (see AGENTS.md §Validation
Strategy #3), drop a CSV into `tests/reference_data/ashrae140/monthly/`
named `case_XXX_monthly_reference.csv`. Use the column layout of the existing
`case_600_monthly_reference.csv`. Generation scripts live in
`tests/reference_data/` (`generate_reference_data.py` needs EnergyPlus 25.2.0
on PATH; see `tests/reference_data/README.md`).

## Step 4 — Add the case to the validator's run set

Open `src/validation/ashrae_140_validator.rs` and add `ASHRAE140Case::CaseXXX`
to the `cases` vec inside `validate_with_diagnostics` (L446) and/or
`validate_analytical_engine` (L1145), grouped with its series. The first is
the full diagnostic run; the second is the analytical-engine run used by the
`ASHRAE 140 Strict Energy Gate` branch-protection check (issue #1333).

If the case belongs to a named diagnostic range (`"800-810"`, `"195-470"`,
`"non-residential"`, `"solid-conduction"`, `"solar-gain"`), also extend the
corresponding arm of `expand_diagnostic_range` (L355) so
`validator.add_diagnostic_case_range(...)` picks it up.

## Step 5 — Update the instantiation test

Open `tests/ashrae_140_validation.rs` and add `"XXX"` to the `case_ids` array
in `test_all_cases_instantiation` (L48), plus the corresponding
`"XXX" => ASHRAE140Case::CaseXXX,` arm in the `match` directly below it. This
test asserts `spec.case_id == id` and `spec.validate().is_ok()` for every
case, which catches the most common mistakes (mismatched ID, invalid
geometry, mismatched `num_zones`, heating setpoint above cooling setpoint).

## Step 6 — Verify the round-trip

From the repo root, in this order:

```bash
# 1. Cycle guard stays green (the gate that closes the #1441 cycle).
python3 scripts/check_ashrae_cases_cycle.py

# 2. Architecture drift: ARCHITECTURE.md still matches the code.
python3 scripts/check_architecture_drift.py

# 3. Code quality (CI's exact invocations).
cargo fmt -- --check
cargo clippy --lib -- -D warnings

# 4. Instantiation test (fast; runs the new case's CaseBuilder factory).
cargo test -p fluxion --test ashrae_140_validation test_all_cases_instantiation -- --nocapture

# 5. Full validation suite (slow; runs the analytical engine + benchmarks).
cargo test --test ashrae_140_validation -- --nocapture

# 6. Energy-conservation gate (must never print "violated energy conservation").
cargo test --test zone_balance_eplus_isolation -- --nocapture
```

A "round-trip" means: the new `CaseBuilder` factory produces a `CaseSpec` that
(1) passes `CaseSpec::validate()` (`src/validation/ashrae_140_cases.rs:1048`),
(2) the analytical engine can step through a full TMY year without violating
energy conservation, and (3) the resulting annual/peak loads fall inside the
`BenchmarkData` reference ranges you entered in Step 3.

## Common pitfalls

- **Don't move leaf types back into `validation`.** If `sim` needs the type,
  it lives in `fluxion-core/src/ashrae_cases.rs`. The cycle guard will catch
  you.
- **Don't add `use crate::validation::*` under `fluxion-core/src/`.** Same
  guard, same outcome.
- **Don't forget the `spec()` match arm.** The compiler will tell you, but
  only if you've added the enum variant first.
- **Don't tune reference values to make a system test pass.** Fix the math
  (AGENTS.md §Validation Strategy #2). Cases **600** and **900** have known
  structural failures documented in `docs/KNOWN_ISSUES.md`; the strict
  ±15 % annual-energy gate (#1333) is the only place those tolerances are
  enforced.
- **Free-floating cases have zero HVAC load.** Set
  `annual_heating_min/max` and `annual_cooling_min/max` to `0.00..0.00` and
  put the real signal in `min/max_free_float_min/max` (°C).
- **Release-gate thresholds.** `release_gates.yaml` sets the minimum ASHRAE
  140 pass rate at 60 % (40 % for patches) and the strict annual-energy gate
  at ±15 % for Cases 600/900. Adding a case that consistently fails will drag
  the pass rate down — flag it in the PR description.

## Cross-references

- `ARCHITECTURE.md` §"Cycle break (#1441 — ASHRAE-140 leaf types →
  `fluxion-core`)" — the canonical cycle contract.
- `AGENTS.md` §Validation Strategy — the four bottom-up rules and the
  release-gate thresholds.
- `AGENTS.md` §CI Gates — `Ashrae Cases Cycle Check` (#1441),
  `ASHRAE 140 Strict Energy Gate` (#1333), `Energy Conservation` (#1295).
- `docs/ASHRAE140_RESULTS.md` — current pass rates per case.
- `docs/KNOWN_ISSUES.md` — open physics limitations (CI fails if the
  `*Last Updated:*` line is >60 days old).
- ML-surrogate swap-point traits you may want to exercise with the new case:
  `HeatConductionSolver` (`src/physics/solver_trait.rs`),
  `VentilationSchedule` (`src/sim/ventilation.rs`),
  `ThermalModelTrait` (`src/sim/thermal_model.rs`).
