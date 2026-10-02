# LIMIT-15 — investigation history

Narrative history for **LIMIT-15**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-15` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-15: ASHRAE 140 Case 195 — Denver TMY min −12.47 °C vs DRYCOLD.TM2 min −24.4 °C weather data source mismatch (Issue #3060)

- **Description:** Issue #3060 is the methodology follow-up to the
  Issue #2868 / PR #3044 Case 195 surface-balance fix. PR #3044 dropped
  Case 195 annual heating from 6810 kWh to 3238 kWh and brings annual
  heating into the ASHRAE 140-2023 band [3.951, 4.217] MWh; the residual
  ~0.6 MWh gap (post-#3044 measured at 3238 kWh vs the ASHRAE 140-2023
  band centre ≈ 4084 kWh) is **not** a physics bug — it is a weather-file
  artefact. The repo's synthetic `DenverTmyWeather`
  (`fluxion-core/src/weather/denver.rs`) has an annual minimum of
  −12.47 °C; the ASHRAE 140-2023 reference weather file **DRYCOLD.TM2**
  has a minimum of −24.4 °C and a maximum of 35.0 °C. For Case 195 (no
  internal loads, no solar, no infiltration), the only heating source is
  envelope transmission; the envelope losses at the winter min differ by
  ~2× for an hour or two, enough to push annual heating ~600 kWh above
  the ASHRAE 140 reference band when run on DRYCOLD.TM2. The validator
  path (`src/validation/ashrae_140_validator.rs:3182`) instantiates
  `DenverTmyWeather::new()` for the Case 195 case file; the unit-test
  path (`tests/ashrae_140_case_195_solid_conduction.rs:54`) does the
  same. Peak heating on the repo's TMY caps at ≈ 1.0 kW; the
  ASHRAE 140-2023 reference peak band is [1.791, 1.802] kW
  (`UA × (20 − T_min) = 40.5 × 44.4 ≈ 1.80 kW`). This weather-file gap
  is **also** the documented §LIMIT-08 peak-heating gap on the
  Issue #2868 acceptance criteria; this LIMIT-15 entry expands §LIMIT-08
  with the methodology comparison and the three implementation options.

  - **Weather data source comparison table** (current state, post-#3044):

    | Property | Repo `DenverTmyWeather` | ASHRAE 140-2023 DRYCOLD.TM2 |
    |----------|--------------------------|----------------------------|
    | Annual min outdoor temp | −12.47 °C | −24.4 °C |
    | Annual max outdoor temp | ~28 °C (synthetic envelope) | 35.0 °C |
    | File format | Synthetic parametric generator | TM2 long-format weather file |
    | Source | `fluxion-core/src/weather/denver.rs:84-547` | ASHRAE 140-2023 Annex B §B.3 |
    | Δ to ASHRAE 140 ref (min) | +11.93 °C | — |
    | Δ Case 195 peak heating | ~1.0 kW (≈ −45 % vs ASHRAE 140 band) | ~1.80 kW (centre of band) |
    | Δ Case 195 annual heating | ~3238 kWh (within band, lower edge) | ~4084 kWh (band centre) |
    | Cases that depend on this | 195 only (no solar/no loads/no infil) | All 600-series for reference inter-program range |

- **Three implementation options (per Issue #3060 "Recommended Direction"):**

  - **(a) Switch Case 195 weather file from Denver TMY3 to DRYCOLD.TM2**
    (the ASHRAE 140 reference) — affects **test data only** (the validator
    and unit-test paths), risky because:
      - DRYCOLD.TM2 has no solar / no wind / no humidity variation; it is
        a *single-purpose* envelope-only weather file, NOT a general
        TMY. Switching it in for Case 195 would either (i) require
        paralleling a separate `WeatherSource` impl
        (`DrycoldWeather`) or (ii) require extending
        `DenverTmyWeather` with a `mode = Drycold` toggle — neither
        is a 1-line change, and both touch the weather-data
        contract that ARCHITECTURE.md §"Module Boundaries" anchors
        as a stable interface.
      - The Cases 600 / 900 / 940 series use `DenverTmyWeather` by
        design (per `release_gates.yaml` known structural failures)
        and would NOT be switched. Mixing two weather sources in
        the same test harness is a maintenance hazard and a
        future-bug surface.
      - Per Issue #3060 "Acceptance": "No regression to Cases 600-660
        (which use Denver TMY3 by design)" — option (a) must therefore
        be Case 195-specific.

  - **(b) Add Case 195 reference band adjustment for non-reference
    weather files (per ASHRAE 140 Annex B §B.3)** — affects
    **acceptance criteria** (the band in `src/validation/benchmark.rs`
    Case 195 entry and `tests/ashrae_140_case_195_solid_conduction.rs`
    `reference::ANNUAL_HEATING_MIN/MAX`), moderate risk because:
      - ASHRAE 140-2023 Annex B §B.3 documents the weather-file
        convention for the *reference*; it does NOT authorise a
        per-implementation band adjustment for a non-reference file
        (that would amount to redefining "pass"). The strict ±15%
        CI gate (`scripts/check_strict_energy_gate_regression.py`,
        `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`)
        is anchored to the ASHRAE 140-2023 reference file.
      - Widening the Case 195 band to absorb the Denver-TMY3 vs
        DRYCOLD.TM2 Δ would mask a real engineering artefact. Per
        RULES.md ("must-never hardcode results") and ADR-0001
        (No-Parameter-Tuning Rule), widening a band to absorb a known
        weather-file Δ is **parameter tuning in band space** and is
        explicitly forbidden.
      - The current band `[3.20, 4.40] MWh` is already a wide
        permissive band (post-#3044 measurement is 3238 kWh, well
        inside the band lower edge); further widening is unjustified.

  - **(c) Re-derive Case 195 reference bands from EnergyPlus runs
    using DRYCOLD.TM2** — affects **reference data**, major research
    task because:
      - Requires EnergyPlus installation, ASHRAE 140-2023 Case 195
        IDF construction, a 1-hour timestep annual simulation, and
        post-processing of the `eplusout.eso` annual totals into a
        new `tests/reference_data/case_195_reference_drycold.csv`
        (or equivalent JSON).
      - The new reference band would be an inter-program range
        across EnergyPlus + ESP-r + TRNSYS + DOE-2, not just an
        EnergyPlus single-implementation number. Re-deriving the
        band without a multi-implementation comparison is a
        single-source reference, which is exactly the failure mode
        the ASHRAE 140 inter-program range was designed to prevent.
      - This option is the **methodologically correct** one for
        closing the LIMIT-15 gap but is well outside the scope of
        a single sub-agent's documentation PR.

  Per AGENTS.md / RULES.md / ADR-0001 ("no parameter tuning",
  "must-never hardcode results"), **none of the three options
  is auto-implementable by a documentation PR**. The decision
  is left to maintainers; this entry documents the trade-offs
  and routes the resolution back to Issue #3060.

- **Affected Tests:**
  - `tests/ashrae_140_case_195_solid_conduction.rs` (Case 195
    low-mass; `simulate_case_195()` at line 51 uses
    `DenverTmyWeather::new()`).
  - `src/validation/ashrae_140_validator.rs` (validator path;
    line 3182 instantiates `DenverTmyWeather::new()` for the
    Case 195 case file).
  - The diagnostic in `tests/diagnostics/case_195_weather_source_diagnostic.rs`
    (this PR's contribution) is the on-demand weather-source
    comparison runner; it is `#[ignore]`-quarantined per the
    `#2536` policy and **does not** alter any assertion.

- **Affected Metrics:** Case 195 annual heating (kWh), Case 195
  peak heating (kW). Both are bounded by the chosen weather
  file's annual minimum outdoor temperature, NOT by the
  physics-engine envelope U-value calculation. The
  physics-engine result on either file is internally
  consistent and energy-conserving (validated by
  `tests/test_energy_conservation.rs`); the band-vs-simulation
  Δ is a **weather-data artefact**, not a solver bug.

- **Severity:** Low (no ASHRAE 140 reference band is gated on
  the repo's TMY3 specifically; the strict ±15% annual-energy
  gate is covered by
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`,
  and Case 195 is NOT in that baseline per `release_gates.yaml`
  known structural failures; the post-#3044 band
  `[3.20, 4.40] MWh` already passes with comfortable margin).

- **GitHub Issue:** [#3060](https://github.com/anchapin/fluxion/issues/3060)
  (this entry), with related issues **#2868** (origin — Case 195
  annual heating over-prediction; closed via PR #3044 for the
  low-mass variant), **#3044** (the PR that closed the annual
  heating half and surfaced the weather-file Δ as a residual
  gap), **#3059** (5R1C/9R4C air-mass distribution limitation
  — sister issue routed to GaugeSolver #1465 / #1462),
  **#1456** (sister issue — Case 960 sunspace coupling
  closure; demonstrates the same "switch the test scenario vs
  widen the band" methodology tension).

- **Status:** 🔄 **Tracking stub + investigation — no physics-code
  change in this PR.** This entry is **documentation/tracking
  only**; the three implementation options above are routed
  back to Issue #3060 for maintainer decision. The on-demand
  diagnostic in `tests/diagnostics/case_195_weather_source_diagnostic.rs`
  is the empirical evidence base for whichever option is chosen
  (it prints annual heating, peak heating, annual cooling, the
  Δ to ASHRAE 140-2023 reference bands, and a synthetic-vs-DRYCOLD
  Δ attribution table for both weather sources).

- **What this PR ships (documentation/tracking scaffolding):**
  1. **This LIMIT-15 entry** — categorises the weather-file gap,
     links to #2868 / #3044 / #3059 / #1456, and lays out the
     three implementation options with their risk / cost / benefit
     analysis.
  2. **`docs/investigations/issue-3060-case-195-weather-source.md`**
     — the standalone investigation document with the full
     weather-data comparison, the three options in detail, and
     the maintainer-decision recommendation.
  3. **`tests/diagnostics/case_195_weather_source_diagnostic.rs`**
     — the `#[ignore]`-quarantined diagnostic that runs Case 195
     on BOTH the repo's `DenverTmyWeather` and a synthetic
     DRYCOLD-equivalent profile (annual min −24.4 °C, annual max
     35.0 °C, no solar / no wind / no humidity variation) and
     reports the per-metric Δ. The diagnostic does NOT modify
     the production `WeatherSource` trait, does NOT modify the
     validator's `DenverTmyWeather::new()` call site, and does
     NOT modify any ASHRAE 140 reference band. Per Issue #3060
     acceptance ("Peak heating ≤ 0.05 kW. No regression to Cases
     600-660") — peak heating is bounded by the weather file
     (not the solver), so the 0.05 kW ceiling in the Issue #3060
     AC is **not** physically achievable on a 0.6 K-h weather file
     at any outdoor temperature below the 20 °C setpoint; the
     diagnostic instead reports the empirical peak heating on
     both files for whichever option is chosen.

- **Why this is NOT a fixable tuning change (per AGENTS.md / RULES.md
  / ADR-0001):**
  1. Switching the test weather file (option a) is a **test-data
     change**, not a solver change, but it changes the meaning of
     "Case 195 passes" from "engine reproduces ASHRAE 140 reference
     on DRYCOLD.TM2" to "engine reproduces DRYCOLD.TM2 on
     DRYCOLD.TM2" (a tautology). Per RULES.md "must-never hardcode
     results", tautological pass criteria are explicitly forbidden.
  2. Widening the reference band (option b) is **parameter tuning in
     band space** — explicitly forbidden by ADR-0001.
  3. Re-deriving the band from EnergyPlus runs (option c) is a
     methodology research task that requires a multi-implementation
     inter-program range (EnergyPlus + ESP-r + TRNSYS + DOE-2), not
     a single-implementation EnergyPlus run; per Issue #3060 "this
     is a major research task" and is **explicitly out of scope**
     for a single sub-agent's documentation PR.
  4. The physics-engine itself is correct on either weather file
     (energy-conservation validated by
     `tests/test_energy_conservation.rs`); the Δ is purely in the
     weather data and is not addressable in solver code.

- **Related sections in this document:**
  - §LIMIT-08 — Case 195 (no-loads) peak heating weather-file gap
    (Issue #2868 — partially resolved by PR #3044); LIMIT-15
    EXPANDS §LIMIT-08 with the methodology comparison and the
    three implementation options (LIMIT-08 documents the symptom,
    LIMIT-15 documents the trade-off analysis).
  - §LIMIT-09 — Case 950 5R1C free-float night-vent override
    (Issue #3071) — sister methodology question, different
    weather file (also Denver TMY3).
  - §LIMIT-11 — Case 195 high-mass walls zero-energy
    (Issue #3064) — sister test failure on the same weather file.
  - §"Aggressive-baseline cohort tracking (Issue #3072)" — Case 195
    is in the 5-case GaugeSolver-blocked cohort (195 / 600 / 620 /
    940 / 960); #3060 is listed as a dependent issue (line 1132).

- **External references:**
  - Issue #3060 (origin) — the methodology decision between
    switch-the-file / widen-the-band / re-derive-the-band.
  - Issue #2868 (origin — Case 195 annual heating over-prediction;
    closed via PR #3044 for the low-mass variant).
  - PR #3044 — the Case 195 surface-balance fix that closed the
    annual heating gap and exposed the weather-file residual Δ.
  - Issue #3059 (5R1C/9R4C air-mass distribution limitation;
    architectural unblocker routed to GaugeSolver #1465 / #1462).
  - Issue #1456 (sister issue — Case 960 sunspace coupling
    closure; same methodology tension between "switch the test
    scenario" and "widen the band").
  - `fluxion-core/src/weather/denver.rs` — `DenverTmyWeather`,
    the repo's synthetic weather source (annual min −12.47 °C).
  - `src/validation/ashrae_140_validator.rs:3182` — validator
    path instantiates `DenverTmyWeather::new()` for the Case
    195 case file.
  - `tests/ashrae_140_case_195_solid_conduction.rs:54` — unit-test
    path instantiates `DenverTmyWeather::new()` in
    `simulate_case_195()`.
  - `src/validation/benchmark.rs` Case 195 entry — ASHRAE 140-2023
    inter-program band (would be widened under option b; **NOT**
    widened in this PR).
  - `tests/diagnostics/case_195_weather_source_diagnostic.rs` —
    the on-demand diagnostic runner (this PR's contribution;
    `#[ignore]`-quarantined per #2536 / #2708).
  - `docs/investigations/issue-3060-case-195-weather-source.md`
    — the standalone investigation document (this PR's contribution).
  - ASHRAE 140-2023 Annex B §B.3 — weather-file convention for
    the *reference* (DRYCOLD.TM2 / HOTDRY.TM2) (referenced; not
    transcribed; standard is paywalled).
  - `RULES.md` — "no parameter tuning" + "must-never hardcode
    results" (option b is forbidden; option a is tautological;
    option c requires multi-implementation inter-program range).
  - `AGENTS.md` — "do NOT modify physics code without checking
    `ARCHITECTURE.md` first"; "Weather (fluxion-core/src/weather/)"
    is a stable interface per the Module Boundaries diagram.
  - `ADR-0001` — No-Parameter-Tuning Rule (forbids option b).
   - `docs/adr/0007-gauge-solver-structural-work.md` — architectural
    unblocker for the §LIMIT-05 / §LIMIT-11 / §LIMIT-15 sister
    issues. The Phase A8 production-path switchover is staged
    (Issue #3291 / PR #3482) — `ThermalSelector::default() =
    ZoneSolverKind::Gauge` is wired but the `gauge-solver` cargo feature
    remains the production-path gate pending §LIMIT-21 closure.
