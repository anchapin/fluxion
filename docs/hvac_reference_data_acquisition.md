# HVAC Reference-Data Acquisition Plan (issue #3942)

Acquisition plan and milestone for the real AHRI / EnergyPlus HVAC equipment reference data that `tests/all_tests/hvac_bottom_up_validation.rs` and `src/sim/hvac/efficiency_curves.rs` still lack.
Audience: maintainers who will obtain, provenance-label and commit the reference data, and reviewers checking that placeholder values are not mistaken for reference values.
Covers: what data is needed (AHRI-rated chiller / boiler / heat-pump points, EnergyPlus equipment outputs), the plausible public sources, where the data will be committed, and what stays placeholder until then.
Related: `tests/all_tests/hvac_bottom_up_validation.rs` (placeholder fixtures), `src/sim/hvac/efficiency_curves.rs` (`default_ahri_coefficients()`), `tests/reference_data/equipment/` (existing analytical equipment CSVs and their provenance).
Status: **PLAN ONLY** — nothing has been downloaded, no reference values are asserted here, and no physics or test assertion changes with this document.
Action: close issue #3942 by completing the milestone in §4; until then the deferred items in the test header stay marked `TODO(#3942)`.

## 1. Why this issue exists

Issue #3942 is the TODO-linked follow-up to the bottom-up HVAC equipment
validation added for issue #1925.  The validation file exercises the chiller,
boiler, heat-pump, VAV and CAV physics formulas in isolation, but every
equipment fixture it uses is a hand-picked placeholder:

- `tests/all_tests/hvac_bottom_up_validation.rs` — the design constants
  (`CHILLER_CAPACITY`, `CHILLER_COP`, `BOILER_CAPACITY`, `BOILER_EFFICIENCY`,
  `HP_HEATING_COP`, `HP_COOLING_EER`, …) are representative numbers chosen
  by the author of the test, not published rated points.
- `src/sim/hvac/efficiency_curves.rs::default_ahri_coefficients()` — the
  polynomial coefficients it returns are documented in the source as
  "placeholder values".

Consequently the file proves that the implementation is *self-consistent*
(the polynomial returns 1.0 at rated PLR, capacity degrades in the expected
direction, power = load / COP), but it does **not** prove agreement with
rated equipment data.  The header of the test file lists exactly which
deferred items this issue owns; each one now carries a `TODO(#3942)` marker.

The only other reference material in the repo is
`tests/reference_data/equipment/` (issue #1933).  Those CSVs are explicitly
labelled `Status: ANALYTICAL` — they are correlation-derived, **not**
EnergyPlus outputs — and their `PROVENANCE.md` already describes the
replacement path this issue must execute.

## 2. Data needed

### 2.1 AHRI-rated equipment points

For each of the three equipment classes exercised by the validation file:

| Equipment | Rating standard (public) | Points needed |
|-----------|--------------------------|---------------|
| Chiller (water-cooled centrifugal) | AHRI 550 / 590 | rated capacity, rated COP at the standard rating condition, IPLV/NPLV part-load fractions, CAPFT capacity-correction coefficients |
| Boiler (hot-water, non-condensing) | AHRI 148 / 202 | rated input & output capacity, rated thermal efficiency, part-load (stack-loss) efficiency curve |
| Heat pump (air-source) | AHRI 210 / 240 (and 340 for ducted) | rated heating capacity/COP at the standard rating conditions, rated cooling capacity/EER, capacity derating vs outdoor dry-bulb, balance-point temperature |

The rating-condition temperatures and the curve *forms* are already
documented with citations in `tests/reference_data/equipment/PROVENANCE.md`;
issue #3942 needs the **values** for the specific rated units that the
validation file will compare against, together with the citation for each.

### 2.2 EnergyPlus equipment outputs

Direct E+ outputs for the same equipment classes and the same operating
envelope (PLR sweep × outdoor-air-temperature sweep):

- Chiller: capacity and COP/EIR versus outdoor entering-water temperature and
  part-load ratio.
- Boiler: efficiency and fuel/electric consumption versus part-load ratio.
- Heat pump: heating/cooling capacity, COP and power versus outdoor dry-bulb
  temperature, including the mode switchover region.

These become the CSV reference files the placeholder assertions are
converted into comparisons against.

## 3. Plausible public sources

No source below has been used yet; each is a candidate that must be
recorded in the provenance file when it is actually used.

1. **EnergyPlus example files shipped with the EnergyPlus distribution.**
   The official release archives published by NREL on
   `https://github.com/NREL/EnergyPlus/releases` bundle the
   `ExampleFiles/` directory (IDFs plus their baseline result files) and the
   `datasets/` directory, which contains equipment performance data used by
   the example models.  Running the bundled IDFs with
   `energyplus --readvars` reproduces the reference outputs locally.
2. **EnergyPlus source repository.** `https://github.com/NREL/EnergyPlus`
   carries the same `ExampleFiles/` tree on the `develop` branch, plus the
   Engineering Reference and the Technical Support Document that document
   the equipment-object defaults (e.g. `Curve:*` coefficient defaults and
   the reference `Boiler:HotWater` / `Chiller:Electric:EIR` objects).
3. **DOE / NREL reference building models.** The DOE commercial and
   residential reference building sets (small office, large office,
   medium office, large space cooling) publish IDFs with fully populated
   `HVACTemplate:*` / `Chiller:*` / `Boiler:*` / `HeatPump:*` objects and
   their simulation outputs, i.e. a complete E+ equipment configuration
   plus results at a real weather file.
4. **AHRI certification rating tables.** The published standard rating and
   certification tables (AHRI 550/590 for chillers, 148/202 for boilers,
   210/240 and 340 for heat pumps) provide the rated capacity, rated
   efficiency and the part-load weighting used to compute IPLV/NPLV.
   These are the values that the "calibrated AHRI coefficients" placeholder
   in `default_ahri_coefficients()` must eventually be derived from.

Sources 1–3 are EnergyPlus-side; source 4 is the AHRI-side.  The
milestone requires both: an E+ output **and** the rated point it is
compared against.

## 4. Milestone (definition of done for #3942)

The issue closes when all of the following are true:

1. **Data committed under a reference-data directory with provenance.**
   The acquired E+ equipment outputs land next to the existing analytical
   data in `tests/reference_data/equipment/` (proposed: an `eplus/`
   sub-directory, mirroring the "_eplus suffix" convention already
   described in that directory's `PROVENANCE.md`), each file carrying a
   header with `Status: ENERGYPLUS`, the EnergyPlus version, the source
   IDF, the simulation command and the citation of the AHRI standard used.
   `tests/reference_data/equipment/PROVENANCE.md` is updated with the
   acquisition date, the exact URLs / archive versions and the licensing
   notes for the redistributed data.
2. **Placeholder tests converted to reference comparisons.** The
   placeholder fixtures in `tests/all_tests/hvac_bottom_up_validation.rs`
   are replaced by (or supplemented with) assertions that read the
   committed reference data and compare the Fluxion capacity / COP /
   efficiency curves against it at the rated and part-load points.
3. **Placeholder coefficients retired.** The `TODO(#3942)` marker on
   `default_ahri_coefficients()` in `src/sim/hvac/efficiency_curves.rs` is
   removed only when the function returns coefficients fitted to the
   acquired AHRI-rated points, and the placeholder coefficients it
   currently returns are updated in the same change.
4. **Deferred header items closed.** The `TODO(#3942)` markers in the
   "What is NOT covered here" section of the validation file are removed as
   each deferred item becomes covered (E+ CSV reference data, chiller
   capacity vs OAT against E+, equipment cycling/transient response vs E+,
   calibrated AHRI coefficients).

Until step 3 and step 4 are done, the coefficients, the test fixtures and
the assertions remain placeholders and must be described as such.

## 5. What stays placeholder until then

- `src/sim/hvac/efficiency_curves.rs::default_ahri_coefficients()` — the
  current polynomial coefficients, temperature coefficients and design
  temperatures are unchanged placeholders.
- The design constants in `tests/all_tests/hvac_bottom_up_validation.rs`
  (`CHILLER_CAPACITY`, `CHILLER_COP`, `BOILER_CAPACITY`,
  `BOILER_EFFICIENCY`, `HP_HEATING_COP`, `HP_COOLING_EER`, …) remain
  representative values, not published rated points.
- `tests/reference_data/equipment/*.csv` remain `Status: ANALYTICAL`; they
  are correlation-derived and are *not* a substitute for E+ output.
- The validation file continues to assert only the internal algebra of the
  physics formulas.  No tolerance band is widened to accommodate missing
  reference data.

## 6. Verification (unchanged behaviour)

The validation suite must behave exactly as before this issue:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo test --test all_tests hvac_bottom_up
```

This issue adds documentation and `TODO(#3942)` markers only — no test was
added or removed, and no coefficient or assertion was modified.

## 7. Related material

- `tests/all_tests/hvac_bottom_up_validation.rs` — bottom-up HVAC equipment
  validation (issue #1925) and the `TODO(#3942)` markers.
- `src/sim/hvac/efficiency_curves.rs` — `default_ahri_coefficients()` and
  the `TODO(#3942)` marker on it.
- `tests/reference_data/equipment/README.md` and
  `tests/reference_data/equipment/PROVENANCE.md` — the existing analytical
  equipment data, its citation chain and the replacement path this plan
  executes.
- `src/sim/hvac/part_load_curves.rs` — the ASHRAE / EnergyPlus-sourced
  part-load curve coefficients already in the code.
- `docs/KNOWN_ISSUES.md` — the repository-wide register of validation gaps.
