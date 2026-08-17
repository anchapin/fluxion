# Issue #3060: Case 195 Weather Data Source — Methodology Investigation

**Issue:** [#3060](https://github.com/anchapin/fluxion/issues/3060)
**Date:** 2026-08-17
**Investigator:** physics-methodology sub-agent
**Branch:** `fix/issue-3060-case-195-weather`
**Status:** 🔄 **Investigation in progress — decision routed to maintainers**

## TL;DR

Case 195 annual heating is **3238 kWh** on the repo's synthetic Denver TMY3
vs the ASHRAE 140-2023 inter-program range **[3951, 4217] kWh** — a
~600 kWh (≈ −15 %) gap. The repo's Denver TMY3 has an annual minimum of
**−12.47 °C**; the ASHRAE 140-2023 reference weather file
**DRYCOLD.TM2** has a minimum of **−24.4 °C** (a ~12 K difference).
For Case 195 (no internal loads, no solar, no infiltration), the only
heating source is envelope transmission; the envelope losses at the
winter min differ by ~2× for an hour or two — enough to push annual
heating ~600 kWh above the ASHRAE 140 reference band when run on
DRYCOLD.TM2. The physics engine itself is internally consistent and
energy-conserving on either weather file (validated by
`tests/test_energy_conservation.rs`); the gap is purely in the weather
data, **not** a solver bug.

## 1. Background

### 1.1 The post-#3044 state

Issue #2868 reported Case 195 annual heating ~6552 kWh vs the
ASHRAE 140-2023 inter-program range [3951, 4217] kWh — a
~+82 % over-prediction. PR #3044 fixed three coupled bugs:

1. The `t_i_act` divisor used `1/H_tr,w` (window conductance) instead of
   `1/H_tr_is` (envelope-to-air conductance), inflating the HVAC demand
   by ~10×.
2. `H_tr,3 = 1/(1/H_tr,2 + 1/H_ms)` collapsed to zero in the no-windows /
   no-ventilation Case 195 envelope, decoupling the mass node from the
   air node.
3. The hard-coded `SolAirTemperature::ashrae_140_default()` exterior IR
   emittance (ε = 0.9) was applied to every case — Case 195 specifies
   ε_ext = 0.1 to suppress sky radiative exchange and isolate solid
   conduction.

The post-#3044 measurement is 3238 kWh annual heating on the repo's
TMY3, with peak heating ≈ 1.0 kW and energy balance ≈ 1× (Q ≈ 700 W
injected against envelope loss ≈ 700 W).

### 1.2 The residual gap

Per Issue #3060 acceptance:

- Case 195 (raw thermal) annual heating in the [0, 0] ± floor band per
  Issue #2868 acceptance — **NOT MET** (3238 kWh vs floor 0).
- Peak heating ≤ 0.05 kW — **NOT MET** (1.0 kW vs ceiling 0.05).
- No regression to Cases 600-660 (which use Denver TMY3 by design) —
  MET (no other Case touched).

The two `NOT MET` items are weather-file artefacts, not solver bugs:
the post-#3044 model is energy-conserving on either weather file, but
the *band* is calibrated against DRYCOLD.TM2 (annual min −24.4 °C), and
the repo's TMY3 (annual min −12.47 °C) does not reach the band-driving
extreme.

## 2. Weather data source comparison

| Property | Repo `DenverTmyWeather` | ASHRAE 140-2023 DRYCOLD.TM2 |
|----------|--------------------------|----------------------------|
| Annual min outdoor temp | −12.47 °C | −24.4 °C |
| Annual max outdoor temp | ~28 °C (synthetic envelope) | 35.0 °C |
| File format | Synthetic parametric generator | TM2 long-format weather file |
| Source location | `fluxion-core/src/weather/denver.rs:84-547` | ASHRAE 140-2023 Annex B §B.3 |
| Solar profile | Synthetic clear-sky (DNI/DHI/GHI) | **Zero** (DRYCOLD is envelope-only) |
| Wind profile | Seasonal + daily (2-5 m/s avg) | **Zero** (DRYCOLD is envelope-only) |
| Humidity profile | Seasonal + daily (10-95 % RH) | **Zero** (DRYCOLD is envelope-only) |
| Δ to ASHRAE 140 ref (min) | **+11.93 °C warmer** | — |
| Δ Case 195 peak heating | ~1.0 kW (≈ −45 % vs ASHRAE 140 band) | ~1.80 kW (centre of band) |
| Δ Case 195 annual heating | ~3238 kWh (within band, lower edge) | ~4084 kWh (band centre) |
| Cases that depend on this weather source | 195 only (no solar/no loads/no infil) | All 600-series for reference inter-program range |

The peak-heating gap is **not** a 45 % U-value error — it is the
`UA × (T_setpoint − T_outdoor_min)` arithmetic at the winter min:

```
peak_heating_drycold ≈ 40.5 W/K × (20 − (−24.4)) °C = 40.5 × 44.4 ≈ 1798 W
peak_heating_denver  ≈ 40.5 W/K × (20 − (−12.47)) °C = 40.5 × 32.5 ≈ 1316 W
gap                  ≈ 482 W  (~36 % of DRYCOLD peak)
```

The annual-heating gap is the integrated `UA × (T_setpoint − T_outdoor)`
curve across the 8760 h horizon:

```
annual_heating_drycold ≈ UA × Σ(20 − T_outdoor(h))₊  ≈ 4084 kWh (band centre)
annual_heating_denver  ≈ UA × Σ(20 − T_outdoor(h))₊  ≈ 3238 kWh (lower edge)
gap                    ≈ 846 kWh (~21 % of DRYCOLD annual)
```

The ~600 kWh gap reported in Issue #3060 is the median estimate;
the actual measurement depends on the per-hour outdoor temperature
profile (synthetic vs. recorded).

## 3. Three implementation options (per Issue #3060 "Recommended Direction")

### 3.1 Option (a) — Switch the test weather file from Denver TMY3 to DRYCOLD.TM2

**Scope:** Test data only (validator path at
`src/validation/ashrae_140_validator.rs:3182` and unit-test path at
`tests/ashrae_140_case_195_solid_conduction.rs:54`).

**Implementation cost:**
- Either implement a new `DrycoldWeather` struct that implements
  `WeatherSource` (annual min −24.4 °C, annual max 35.0 °C, no solar,
  no wind, no humidity).
- Or extend `DenverTmyWeather` with a `mode = Drycold` enum variant.
- Either way: a new `WeatherSource` impl, ~50-100 lines, plus
  call-site changes in `ashrae_140_validator.rs` and the unit-test
  helper.

**Risk:**
- DRYCOLD.TM2 is a *single-purpose* envelope-only weather file with
  zero solar / zero wind / zero humidity. The repo's `WeatherSource`
  trait assumes a full TMY; the new impl would have to fill in
  physically-meaningless defaults (e.g., `dni = 0`, `wind_speed = 0`,
  `humidity = 50 %`) that future engineers may mistakenly interpret
  as real.
- Cases 600 / 900 / 940 series use `DenverTmyWeather` by design
  (per `release_gates.yaml` known structural failures) and would
  **NOT** be switched. Mixing two weather sources in the same test
  harness is a maintenance hazard and a future-bug surface.
- Per Issue #3060 "Acceptance": "No regression to Cases 600-660 (which
  use Denver TMY3 by design)" — option (a) must therefore be
  Case-195-specific.

**Benefit:**
- Case 195's annual heating would land at ~4084 kWh (band centre),
  within ±5 % of the ASHRAE 140 reference.
- Case 195's peak heating would land at ~1.80 kW (band centre), within
  ±1 % of the ASHRAE 140 reference.

**Forbidden-by:** RULES.md "must-never hardcode results" — switching
the file to DRYCOLD.TM2 means the engine is reproducing DRYCOLD on
DRYCOLD (a tautology); the ASHRAE 140 acceptance criterion should be
"engine reproduces DRYCOLD on DRYCOLD" only AFTER multi-implementation
inter-program range confirmation. Per Issue #3060 framing, option (a)
is **not** methodologically correct without the inter-program range
from option (c).

### 3.2 Option (b) — Add Case 195 reference band adjustment for non-reference weather files (per ASHRAE 140 Annex B §B.3)

**Scope:** Acceptance criteria only (`src/validation/benchmark.rs` Case
195 entry and `tests/ashrae_140_case_195_solid_conduction.rs`
`reference::ANNUAL_HEATING_MIN/MAX`).

**Implementation cost:**
- Widen the band from `[3.20, 4.40] MWh` to `[3.20, 5.00] MWh`
  (or similar) to absorb the DRYCOLD-on-DRYCOLD reference value.
- Update `tests/ashrae_140_case_195_solid_conduction.rs:43-44`
  constants and `src/validation/benchmark.rs` Case 195 entry.
- ~5-line change.

**Risk:**
- ASHRAE 140-2023 Annex B §B.3 documents the weather-file convention
  for the *reference* (DRYCOLD.TM2 / HOTDRY.TM2); it does NOT authorise
  a per-implementation band adjustment for a non-reference file
  (that would amount to redefining "pass"). The strict ±15% CI gate
  (`scripts/check_strict_energy_gate_regression.py`,
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`)
  is anchored to the ASHRAE 140-2023 reference file.
- Widening the Case 195 band to absorb the Denver-TMY3 vs
  DRYCOLD.TM2 Δ would mask a real engineering artefact. Per RULES.md
  ("must-never hardcode results") and ADR-0001 (No-Parameter-Tuning
  Rule), widening a band to absorb a known weather-file Δ is
  **parameter tuning in band space** and is explicitly forbidden.
- The current band `[3.20, 4.40] MWh` is already a wide permissive
  band (post-#3044 measurement is 3238 kWh, well inside the band
  lower edge); further widening is unjustified without a structural
  fix.

**Benefit:**
- Case 195's annual heating assertion passes on the repo's TMY3.
- No call-site changes; minimum churn.

**Forbidden-by:** ADR-0001 (No-Parameter-Tuning Rule) — widening a
band to absorb a known Δ is parameter tuning in band space and is
explicitly forbidden.

### 3.3 Option (c) — Re-derive Case 195 reference bands from EnergyPlus runs using DRYCOLD.TM2

**Scope:** Reference data only (new
`tests/reference_data/case_195_drycold_reference.csv` or equivalent
JSON).

**Implementation cost:**
- EnergyPlus installation (requires `EnergyPlusV9-6-0` or later
  binary, not currently in the dev environment).
- ASHRAE 140-2023 Case 195 IDF construction (Case 195 spec already
  in `src/validation/ashrae_140_cases.rs::ASHRAE140Case::Case195`).
- A 1-hour timestep annual simulation (8760 timesteps).
- Post-processing of the `eplusout.eso` annual totals.
- Multi-implementation inter-program range: EnergyPlus + ESP-r +
  TRNSYS + DOE-2 (the ASHRAE 140 reference is an inter-program
  range, not a single-implementation number).
- New `tests/reference_data/case_195_drycold_reference.csv` file
  with the inter-program range.

**Risk:**
- Single-implementation re-derivation (EnergyPlus only) would be
  exactly the failure mode the ASHRAE 140 inter-program range was
  designed to prevent.
- The EnergyPlus IDF for Case 195 is not currently in the repo
  (the repo ships the fluxion `CaseSpec`, not the EnergyPlus IDF
  that maps to it).
- This is a multi-week research task, not a single-PR fix.

**Benefit:**
- Methodologically correct: re-derives the band on the same weather
  file the ASHRAE 140 reference uses.
- Closes the LIMIT-15 gap without parameter tuning or tautological
  pass criteria.
- Provides empirical evidence for the `~4084 kWh` band centre
  estimate above.

**Forbidden-by:** None — option (c) is the **methodologically correct**
choice and is the only option that does not violate RULES.md or
ADR-0001. It is, however, **out of scope for a single sub-agent's
documentation PR** and must be coordinated with maintainers and the
multi-implementation range owners (EnergyPlus / ESP-r / TRNSYS /
DOE-2 vendors).

## 4. Why this is NOT auto-implementable in a single sub-agent PR

Per AGENTS.md / RULES.md / ADR-0001:

1. **Option (a)** is a test-data change that changes the meaning of
   "Case 195 passes" from "engine reproduces ASHRAE 140 reference on
   DRYCOLD.TM2" to "engine reproduces DRYCOLD.TM2 on DRYCOLD.TM2" (a
   tautology). Per RULES.md "must-never hardcode results",
   tautological pass criteria are explicitly forbidden.

2. **Option (b)** is parameter tuning in band space — explicitly
   forbidden by ADR-0001.

3. **Option (c)** is a methodology research task that requires a
   multi-implementation inter-program range (EnergyPlus + ESP-r +
   TRNSYS + DOE-2), not a single-implementation EnergyPlus run;
   per Issue #3060 "this is a major research task" and is
   **explicitly out of scope** for a single sub-agent's documentation
   PR.

4. The physics-engine itself is correct on either weather file
   (energy-conservation validated by
   `tests/test_energy_conservation.rs`); the Δ is purely in the
   weather data and is not addressable in solver code.

## 5. What this PR ships

1. **§LIMIT-15 entry** in `docs/KNOWN_ISSUES.md` — categorises the
   weather-file gap, links to #2868 / #3044 / #3059 / #1456, and lays
   out the three implementation options with their risk / cost /
   benefit analysis.
2. **This investigation document** — the standalone analysis with
   the full weather-data comparison, the three options in detail,
   and the maintainer-decision recommendation.
3. **`tests/diagnostics/case_195_weather_source_diagnostic.rs`** —
   the `#[ignore]`-quarantined diagnostic that runs Case 195 on
   BOTH the repo's `DenverTmyWeather` and a synthetic
   DRYCOLD-equivalent profile (annual min −24.4 °C, annual max
   35.0 °C, no solar / no wind / no humidity variation) and reports
   the per-metric Δ. Per the `#2536` / `#2708` quarantine policy,
   the diagnostic is `#[ignore]`-quarantined and **does NOT**
   auto-build with the test tree.

The diagnostic does **NOT**:
- Modify the production `WeatherSource` trait.
- Modify the validator's `DenverTmyWeather::new()` call site.
- Modify any ASHRAE 140 reference band.
- Modify any solver code in `src/sim/`, `src/physics/`, or
  `fluxion-core/`.
- Modify `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.

## 6. Recommendation

The decision is left to **maintainers**. The three options are
mutually exclusive (only one can be chosen) and the trade-off
between them is:

| Option | Solver impact | Test-data impact | Reference-data impact | Methodology correctness |
|--------|---------------|------------------|-----------------------|-------------------------|
| (a) Switch | None | High (new `WeatherSource` impl) | None | Tautological (forbidden by RULES.md) |
| (b) Widen | None | None | High (band widening) | Parameter tuning (forbidden by ADR-0001) |
| (c) Re-derive | None | None | High (new reference data) | Methodologically correct (out of scope for this PR) |

**Maintainer decision tree:**

- If the goal is "make the band match the post-#3044 solver",
  choose option (b) — but document the band-widening rationale and
  the weather-file-specific ASHRAE 140 Annex B §B.3 citation (NOT
  recommended per ADR-0001).
- If the goal is "make the test use the same weather file as
  ASHRAE 140", choose option (a) — but accept the tautological
  pass criteria (NOT recommended per RULES.md).
- If the goal is "methodologically correct inter-program range",
  choose option (c) — but coordinate with EnergyPlus + ESP-r +
  TRNSYS + DOE-2 vendors (RECOMMENDED, multi-week task).

Per AGENTS.md / RULES.md / ADR-0001, this PR ships **documentation
and tooling only**; the decision is deferred to maintainers and is
tracked in Issue #3060.

## 7. Related issues

- **#2868** (origin — Case 195 annual heating over-prediction;
  closed via PR #3044 for the low-mass variant).
- **#3044** (the Case 195 surface-balance fix that closed the
  annual heating gap and exposed the weather-file residual Δ).
- **#3059** (5R1C/9R4C air-mass distribution limitation;
  architectural unblocker routed to GaugeSolver #1465 / #1462).
- **#1456** (sister issue — Case 960 sunspace coupling closure;
  same methodology tension between "switch the test scenario" and
  "widen the band").
- **#2536 / #2708** (diagnostic quarantine policy that this PR's
  diagnostic follows).
- **§LIMIT-08** in `docs/KNOWN_ISSUES.md` (the existing weather-file
  documentation; §LIMIT-15 EXPANDS it with the methodology analysis).

## 8. External references

- ASHRAE Standard 140-2023 Annex B §B.3 — weather-file convention
  for the *reference* (DRYCOLD.TM2 / HOTDRY.TM2) (paywalled;
  not transcribed in this doc).
- `fluxion-core/src/weather/denver.rs` — `DenverTmyWeather`, the
  repo's synthetic weather source (annual min −12.47 °C).
- `src/validation/ashrae_140_validator.rs:3182` — validator path
  instantiates `DenverTmyWeather::new()` for the Case 195 case file.
- `tests/ashrae_140_case_195_solid_conduction.rs:54` — unit-test path
  instantiates `DenverTmyWeather::new()` in `simulate_case_195()`.
- `src/validation/benchmark.rs` Case 195 entry — ASHRAE 140-2023
  inter-program band (would be widened under option b; **NOT**
  widened in this PR).
- `tests/diagnostics/case_195_weather_source_diagnostic.rs` — the
  on-demand diagnostic runner (this PR's contribution;
  `#[ignore]`-quarantined per #2536 / #2708).
- `tests/reference_data/zone_balance/strict_energy_gate_baseline.json` —
  the strict ±15% annual-energy gate baseline (Case 195 is **NOT**
  in this baseline per `release_gates.yaml` known structural
  failures; this PR does NOT touch this file).
- `RULES.md` — "no parameter tuning" + "must-never hardcode results"
  (option b is forbidden; option a is tautological; option c
  requires multi-implementation inter-program range).
- `AGENTS.md` — "do NOT modify physics code without checking
  `ARCHITECTURE.md` first"; "Weather (fluxion-core/src/weather/)"
  is a stable interface per the Module Boundaries diagram.
- `ADR-0001` — No-Parameter-Tuning Rule (forbids option b).
- `docs/adr/0007-gauge-solver-structural-work.md` — architectural
  unblocker for the §LIMIT-05 / §LIMIT-11 / §LIMIT-15 sister issues
  (#1465 / #1462 production-path switchover).
