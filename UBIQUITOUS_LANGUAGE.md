# UBIQUITOUS LANGUAGE — fluxion

The shared domain language for this repo, in the Domain-Driven Design
sense: one name per concept, used the same way in briefs, issues,
code, tests, and docs — by humans and by agent workers alike.
Definitions are traced to the code and docs on `develop`
(`ARCHITECTURE.md`, `CODEBASE_MAP.md`, `AGENTS.md`, `RULES.md`,
`CONTRACT.md`, `docs/ASHRAE140_TERMINOLOGY.md`, `docs/KNOWN_ISSUES.md`).
Terms that could not be traced to a source go in **Open questions**,
not in the glossary — resolve them there rather than inventing a
definition in a brief. (None open at this revision; see Resolved
questions.)

**How to use it:** One name per concept, used the same way in briefs,
issues, code, tests, and docs. If you mean the thing, use its word here.
If a word is not here, check the *Don't say / say instead* table before
inventing one. When a definition changes, change it here and rename in
code in the same PR — a glossary that drifts from the code is worse
than none.

Companion files: `CONTRACT.md` (never-regress / hard stops),
`REPOS.md` at home (cross-repo context map), `SCORECARD.md`
(generated status snapshot).

---

## Bounded contexts — the map

| Context | Where it lives | Its language is about |
|---|---|---|
| Simulation core | `src/sim/` | Zones, timesteps, the energy balance |
| Physics / conduction | `src/physics/`, `fluxion-core/src/per_surface_conduction/` | How heat moves through constructions |
| Construction & assembly | `fluxion-core/src/construction/`, `fluxion-core/src/assembly/` | What surfaces are made of |
| Weather, solar & sky | `fluxion-core/src/weather/`, `src/sim/solar.rs`, `src/sim/sky_radiation.rs` | What the outside does to the building |
| HVAC & fluid | `src/sim/hvac*`, `fluxion-fluid/` | Equipment that adds/removes heat on purpose |
| Validation | `src/validation/`, `tests/all_tests/ashrae_140*` | ASHRAE 140 cases, reference bands, gates |
| Surrogate / AI | `src/ai/` | ONNX models that approximate physics, never replace its meaning |
| Interop & bindings | `src/interop/`, `src/python/`, `src/napi/`, `fluxion-wasm/` | OSM / gbXML / FMI / PyO3 / NAPI surfaces |

A term means what its context says it means. The same word in two
contexts (e.g. *model* in Simulation core vs. Surrogate) is two terms.

---

## Simulation core

- **Zone** — The unit of thermal control and energy balance: one air
  node (plus its mass nodes in multi-node models) with one temperature
  trajectory and one setpoint schedule. *Not* a room: a room is an
  architectural unit (see Matchline's **Space**); several rooms can be
  one zone, and one open space can span zones.
- **Timestep** — One simulation step (1 hour in ASHRAE 140 work). Loads
  are rates over a timestep; energies are loads integrated over it.
- **Thermal model** — The engine object that advances zones through
  timesteps (`ThermalModel`); the trait hierarchy rooted at
  `ThermalModelTrait` is what Physics / Mock / Surrogate / Hybrid /
  Unified implementations plug into. See also the *Model (usage
  rule)* entry below.
- **Energy balance** — The invariant that must hold for any zone:
  conduction + convection + radiation + solar + HVAC sum to zero.
  (CONTRACT.md, from RULES.md §Must-Always 1.) A balance failure is a
  bug, never a tuning opportunity.
- **Setpoint** — The target zone temperature for heating / cooling
  control, °C. Examples use 19.0–24.0 °C as a design range, not a
  definition.
- **Free-float / free-floating (FF)** — HVAC disabled; zone temperature
  floats with physics alone. Case names carry the suffix: **Case 600FF**.
  Free-float cases test thermal response without control interference.
  (`docs/ASHRAE140_TERMINOLOGY.md`)
- **Internal loads / internal gains** — Heat added inside a zone by
  people, lights, and equipment, per an `InternalLoads` spec. Say
  *internal loads* for the scheduled input, *gains* when talking about
  the heat actually entering the balance.
- **Ventilation** — Outdoor-air exchange with the zone, via
  **VentilationSchedule** (a named swap point). **Night ventilation** is
  the scheduled night-flush variant (`NightVentilation`), not a
  synonym for ventilation in general.
- **Infiltration** — Uncontrolled air leakage. Distinct from
  ventilation, which is intentional/scheduled.
- **Warm-up** — The pre-conditioning period of an annual simulation,
  per ASHRAE 140 §B2 (`src/sim/warmup.rs`, `WarmupConfig`, enabled by
  default): the model runs until it reaches periodic steady state
  *before* results are recorded, so the arbitrary initial condition
  (all zones at 20 °C) does not bias early timesteps — worst in
  free-float cases. Two modes: **fixed-duration** (default 14 days,
  `DEFAULT_WARMUP_DAYS`; weather wraps periodically, hour % 8760) and
  **convergence-based** (full-year iterations until max ΔT < 0.01 °C,
  max 4 iterations). Warm-up output is never recorded results.
- **Warm-up bridge** — The parallel-in-time use of warm-up (Issue
  #4699): a month chunk is simulated as [N preceding days on the
  *real preceding weather*, Dec→Jan wrap] + [the month], warm-up
  output discarded (overlap-and-discard). The 14-day standard is
  measured, not assumed: it holds for low-mass (Case 600, #4728) and
  high-mass (Case 900, #4750) against the periodic reference. The
  N-vs-2N self-check stays conservative — it declares convergence
  only at 30 days for high-mass — so a self-check-driven gate picks
  30 even though 14 measurably suffices.
- **Model (usage rule)** — Bare "the model" in a brief means the
  assembled `ThermalModel` for a case (what topology export shows).
  Every other use carries its qualifier: *thermal model* (anything
  implementing `ThermalModelTrait` — Physics / Mock / Surrogate /
  Hybrid / Unified), *surrogate model*, *sky model*
  (`PerezSkyModel`), *input model* (`FluxionModel` / CaseSpec).

## Physics / conduction

- **HeatConductionSolver** — The trait swap point for conduction
  (`src/physics/solver_trait.rs`). Implementations are interchangeable;
  briefs must name which one they mean.
- **5R1C** — The ISO 13790 lumped model: 5 resistances, 1 capacitance —
  one lumped thermal-mass node. Canonical full name on first use:
  *5R1C (FiveROneC in code, `ZoneSolverKind::FiveROneC`)*.
- **9R4C** — The high-mass multi-node sibling. HighMass specs
  auto-promote from 5R1C to 9R4C via `from_spec_with_selector`.
- **Gauge / GaugeSolver** — The equation-based prototype on the DAE
  path (`ZoneSolverKind::Gauge`, gated by `--features gauge-solver`).
  Per ADR-0017 it is a *prototype*, not the production solver.
- **DAE teacher** — The equation-based DAE model adopted in ADR-0017
  as the single production *target*. The unconditional default flip to
  it is gated on the #3986 teacher validation suite. Do not call Gauge
  "the teacher": Gauge is the prototype, the teacher is the target.
- **ThermalSelector** — The explicit, cfg-dependent choice of zone
  solver (`src/sim/thermal_selector.rs`). There is no silent
  fall-through: an explicit `Gauge` selector in a default build panics
  loudly naming the feature flag. (ADR-0017)
- **CTF** — Conduction Transfer Function solver. Currently the 5R1C
  cross-check, per ADR-0017 — not the production path.
- **FD** — Finite Difference solver / multi-node path used for
  conditioned high-mass cases.
- **CTA** — Continuous Tensor Abstraction (`physics/cta.rs`), the
  tensor formulation under the 5R1C implementation. Not a solver kind
  in briefs unless the brief is about CTA itself.
- **Thermal mass node** — A capacitance node representing stored heat
  in construction mass (`ThermalMassNode`, `MultiNodeThermalMass` in
  `fluxion-core`). The recurring structural limitation in KNOWN_ISSUES
  is the *single lumped* mass node (the `dt/τ ≈ 3.6` pathology) — name
  it that way, not as a generic "mass bug."
- **Conductances (`h_tr_*`)** — Named heat-transfer conductances in
  W/K: `h_tr_is` (interior surface ↔ air), `h_tr_ms` (mass ↔ surface),
  `h_tr_em` (exterior ↔ mass). Use the code name; do not paraphrase
  into "the surface coefficient."
- **Film coefficient** — Surface air-film coefficient, W/m²K. The
  canonical exterior value is **18.3** (`fluxion-core/src/construction.rs`).
  29.3 is a legacy value and must not reappear in production paths.

## Construction & assembly

- **Surface** — One bounded face of a zone (wall, roof, floor, window)
  with an orientation, a construction, and an area. Fluxion's topology
  export is a zone / surface / construction / HVAC graph.
- **Construction** — The layered build-up assigned to a surface
  (`Construction` in `fluxion-core`). One homogeneous layer inside
  it is a **construction layer** (`ConstructionLayer`).
- **Material layer / MaterialLayer** — One homogeneous layer inside a
  construction (part of `BuildingAssembly` / `AssemblyBuilder`).
- **Assembly / BuildingAssembly** — A named, reusable set of
  constructions for a building or case.
- **MassClass** — LowMass / HighMass classification of a construction
  or case; it drives the 5R1C → 9R4C auto-promotion. "CaseMass" in
  ASHRAE specs is the same distinction — say MassClass in prose.
- **SurfaceType** — The kind of a surface for construction lookup
  (wall / roof / floor / window class), distinct from `SurfaceKind`
  in per-surface conduction. If a brief says just "surface type,"
  it is ambiguous — specify which enum.
- **Orientation** — Compass orientation of a surface/case
  (`fluxion_core::ashrae_cases::Orientation`). Tilt and azimuth are
  its geometric companions in solar work; they are not synonyms.

## Weather, solar & sky

- **Weather** — The external forcing time series (TMY3 / EPW family in
  `fluxion-core/src/weather/`). Weather is input, never a result.
- **Solar distribution fractions** — Where window solar is routed:
  `solar_distribution_to_air` (fraction direct to the air node) and
  the to-mass fraction. These named fractions are load-bearing in the
  Case 600-series audits — use the code names in physics briefs.
- **Sky radiation** — Longwave exchange with the sky
  (`sim/sky_radiation.rs`, `STEFAN_BOLTZMANN` now in `fluxion-core`).
- **Shading** — `ShadingType` / `ShadingDevice` / overhangs and fins
  that reduce solar on a surface. A shading *device* is geometry; a
  shading *type* is the case-spec category.

## HVAC & fluid

- **HVAC template** — The 1.0-scope way a system is specified *from
  input* (fluxion #3997): a named system pattern instantiated onto
  zones, as opposed to hand-wired equipment in code.
- **Equipment** — A concrete device instance (`AnyEquipment`,
  `AnyZoneEquipment`). Template ≠ equipment: the template is the
  pattern, the equipment is what runs.
- **Acausal HVAC / fluxion-fluid** — The port-based, acausal HVAC
  crate is **`fluxion-fluid/`**. It is *not* `fluxion-core/src/fluid/`
  (which does not own this meaning). Briefs that say "the fluid
  module" without the crate name are ambiguous.
- **Design day** — A 24-hour extreme-weather day used for sizing.
  Two sources, do not conflate them: a **DDY design day**
  (`DesignDaySpec`, parsed from EnergyPlus `.ddy` files —
  `SizingPeriod:DesignDay`, Winter/SummerDesignDay, with max
  dry-bulb, daily range, and humidity condition) and a **selected
  design day** (`DesignDaySelector`, deriving the heating day =
  lowest average dry-bulb and cooling day = highest average
  dry-bulb from TMY hourly data, per ASHRAE Fundamentals Ch. 14).
  The CLI run period is "design-day" vs "annual."
- **Sizing** — Determining equipment capacity from design-day peak
  loads, × a safety factor (typically 1.1–1.2). Sizing is the
  *purpose* of a design day; there is no "sizing day" — that phrase
  is external shorthand and is not used in briefs.
- **Sensible vs. latent load** — Sensible = temperature change;
  latent = moisture. Judge latent-load/moisture work against humid
  climates (e.g. Baltimore, MD) — Denver is too dry to show its value.
  (Alex's standing climate lens, 2026-09-29.)

## Validation

- **ASHRAE 140 case / CaseSpec** — One standard test building +
  conditions (`CaseSpec`, `ASHRAE140Case`). Prose form is spaced —
  *Case 600*, *Case 900*, *Case 600FF*; code/ids are unspaced.
  600 = low-mass baseline family, 900 = high-mass family; FF = the
  free-float variant of the same case, not a different building.
- **CaseSpec leaf types** — The pure-data spec types a case is built
  from, in `fluxion_core::ashrae_cases`: `WindowSpec`, `GlassType`,
  `BuildingType`, `GeometrySpec`, `ConstructionType`, `ShadingType`,
  `Orientation`, `InternalLoads`, `NightVentilation`. They are leaf
  types (no upward dependencies) — in briefs, name the specific leaf
  type, not "the case's window/glass/geometry settings."
- **Reference band** — The ASHRAE-published min–max range a result
  must fall in. Bands belong to the standard, not to Fluxion.
- **Load vs. energy (validation sense)** — Fluxion reports *thermal
  loads* (heat added/removed, kW peak / MWh annual); some ASHRAE
  reference values are *HVAC system energy* (electricity, after COP /
  efficiency). They are not directly comparable without the efficiency
  step. (`docs/ASHRAE140_TERMINOLOGY.md`) Never shorten both to
  "energy" in a validation brief.
- **EUI** — In the toy examples only, EUI is a raw cumulative
  temperature-gap metric, *not* calibrated kWh/m²/yr. Do not cite
  example EUI as physical EUI.
- **Gate** — A CI check in the Gate Registry. Changing a gate is an
  edit to `.github/ci-gates.yaml` — never an edit to a workflow file
  to achieve the same effect. **Strict (energy) gate** = the ±15%
  annual-energy regression gate on Case 600/900.
- **Baseline / re-pin** — The recorded expected value a guard compares
  against. Output values moving is not by itself a regression: re-pin
  the baseline to current values and proceed (standing directive,
  2026-10-25). *Never* raise the strict-gate baseline to hide a
  600/900 regression — that specific re-pin is forbidden.
- **LIMIT-\*** — A numbered structural limitation in
  `docs/KNOWN_ISSUES.md` (LIMIT-05, LIMIT-21, …). A LIMIT is a known
  architectural gap with a routed fix, not a failing test and not
  permission to tune constants.
- **Quarantine** — A test marked `#[ignore = "awaiting #N"]` (or
  LIMIT-*) *with* a matching row in `tests/QUARANTINE.md`. Ignored
  without a registry row is not quarantined, it's lost.
- **Scorecard** — `SCORECARD.md`, the *generated* canonical status
  snapshot. Never hand-edit; regenerate it.
- **Blind validation** — The v1.3 posture: no post-simulation
  correction factors, no case-type hints — physics only, measured
  against true reference values. Do not call any remaining constant a
  "calibration factor" without checking
  `docs/CORRECTION_FACTORS_INVENTORY.md`.

## Surrogate / AI

- **Surrogate** — A trained ONNX model that approximates a physics
  component for speed (`src/ai/surrogate.rs`, `SurrogateManager`),
  swappable with the physics component via the trait swap points.
  Surrogates are **post-1.0** scope. A surrogate result is an
  approximation of the physics term it replaces — name the term.
- **BatchOracle** — The parallel population evaluator (NAPI
  `BatchOracle`). It parallelizes *populations only*; there is no
  nested inner-loop Rayon parallelism.
- **Mock mode** — The explicit fallback when no valid ONNX model is
  present. ONNX loading is fail-closed against `<model>.sha256`;
  mock is a stated mode, never a silent substitution.

## Interop & repo language (for workers)

- **Topology export / lint** — `fluxion topology export|lint`: the
  assembled zone/surface/construction/HVAC graph as deterministic,
  schema-versioned JSON, plus its E/W-rule lint. Diagram drift is
  gated — regenerate diagrams in the same PR that moves the graph.
- **fluxion-core (leaf)** — The dependency-light leaf crate. It must
  not import root `sim`, `physics`, `ai`, or `validation`. Types that
  moved there keep **re-export shims** at their old paths — a shim is
  a forwarding re-export, not a copy and not a second definition.
- **Swap point** — One of the four named trait boundaries where an
  implementation may be exchanged: `HeatConductionSolver`,
  `VentilationSchedule`, `ThermalSelector`, `ThermalModelTrait`.

---

## Don't say / say instead

| Don't say | Say | Why |
|---|---|---|
| room (for a thermal unit) | zone | Rooms are architectural; zones are thermal/control units |
| the solver (unspecified) | 5R1C / 9R4C / Gauge / CTF / FD | Five different things share that nickname |
| the teacher = Gauge | Gauge (prototype) / DAE teacher (target) | ADR-0017 keeps them distinct |
| energy (for a load) | thermal load, or HVAC system energy | ASHRAE bands mix the two; Fluxion reports loads |
| calibration / tuning (for a fix) | root-cause fix | Blind validation forbids tuning to the band |
| fluid module | `fluxion-fluid/` crate | Two different locations could claim it |
| test count (from memory) | the count in `tests/test_inventory.json` | It is the canonical citation; counts drift weekly |
| Case600 | Case 600 (prose) | Spaced in prose, unspaced only in code ids |
| sizing day | design day (for sizing) | "Sizing day" appears nowhere in the repo; design day is the term, sizing is its purpose |
| the model (unqualified, non-case sense) | thermal / surrogate / sky / input model | Bare "the model" is reserved for the assembled ThermalModel of a case |

## Resolved questions (2026-11-11, operator's verdict)

The first draft's open questions, resolved on Alex's word after
research against develop — the full definitions live in the sections
above; this is the decision record:

1. **Warm-up** — pinned as glossed (Simulation core), including the
   distinct **warm-up bridge** sub-term from Issue #4699.
2. **Design day / sizing day** — design day is a Fluxion term of
   art (DDY-specified and selector-derived senses); sizing is its
   purpose. "Sizing day" is not used.
3. **Model** — the usage rule in Simulation core applies: bare
   "the model" = the assembled `ThermalModel`; all other uses are
   qualified. No renames.

No open questions at this revision. New untraced terms go here as
questions, not into the glossary as invented definitions.

---

## Core terms (machine-checked)

The curated core of this glossary — the terms a brief may rely on
without re-defining them. Checked by plan-coder's
`bin/glossary_check.py` (run in the weekly harness-tune pass,
report-only): each row's **Symbol** must still exist at its
**Source**, and each **Term** must still have a glossary entry
above. This is a ratchet on a curated set, *not* a coverage target —
the repo has hundreds of public types that are implementation
detail, not domain language, and they do not belong here.

Adding a core term = adding a row here plus its entry above.
Renaming or deleting a core symbol = updating its row in the same
PR, per the rename rule in *How to use it*.

| Term | Symbol | Source |
|---|---|---|
| Zone | Zone | src/sim |
| Surface | Surface | src/sim |
| Construction | Construction | fluxion-core/src/construction.rs |
| Construction layer | ConstructionLayer | fluxion-core/src/construction.rs |
| Material layer | MaterialLayer | fluxion-core/src/assembly.rs |
| Assembly | BuildingAssembly | fluxion-core/src/assembly.rs |
| Mass class | MassClass | fluxion-core/src/construction.rs |
| Surface type | SurfaceType | fluxion-core/src/construction.rs |
| Film coefficient | EXTERIOR_FILM_COEFF | fluxion-core/src/construction.rs |
| Orientation | Orientation | fluxion-core/src/ashrae_cases.rs |
| CaseSpec leaf types | WindowSpec | fluxion-core/src/ashrae_cases.rs |
| Internal loads | InternalLoads | fluxion-core/src/ashrae_cases.rs |
| Night ventilation | NightVentilation | fluxion-core/src/ashrae_cases.rs |
| Thermal model | ThermalModel | src/sim |
| Thermal selector | ThermalSelector | src/sim/thermal_selector.rs |
| Heat conduction solver | HeatConductionSolver | src/physics/solver_trait.rs |
| Ventilation schedule | VentilationSchedule | src/sim/ventilation.rs |
| Warm-up | WarmupConfig | src/sim/warmup.rs |
| Warm-up bridge | — | docs/perf/parallel-in-time-4699.md |
| Design day | DesignDaySpec | fluxion-core/src/weather/ddy.rs |
| Selected design day | DesignDaySelector | fluxion-core/src/weather/design_day_selector.rs |
| Equipment | AnyEquipment | src/sim |
| ASHRAE 140 case | CaseSpec | src/validation |
| Surrogate | SurrogateManager | src/ai/surrogate.rs |
| BatchOracle | BatchOracle | src/napi |
