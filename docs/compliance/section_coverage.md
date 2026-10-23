# ASHRAE 140 Section Coverage Tracker

> **Summary 1/6:** Committed snapshot of the #4267 full-standard conformance section-coverage
> tracker. It records, per ASHRAE 140-2023 section, what fluxion implements and which issue owns
> the remaining work.
> **Summary 2/6:** **"Status" in this table is coverage, not a score.** Per #4277, Standard 140
> defines no acceptance criteria and issues no pass/fail certificate; the example-result ranges are
> diagnostic context, not thresholds.
> **Summary 3/6:** **Correction (2026-10-23, ADR-0026): §6 is the weather-interpretation section
> (Weather Driver Tests, WD100–WD600) and §12 is Building Thermal Envelope and Fabric Load Tests
> (Class II)** — the §7 envelope cases re-run in a second climate. The earlier mapping that treated
> §12 as a weather-interpretation section was wrong.
> **Summary 4/6:** §12 is **IN SCOPE**, sequenced immediately after §7 reaches passing status and
> **explicitly blocked on §7** — it shares an implementation with §7 and must not be picked up
> early.
> **Summary 5/6:** fluxion's committed validation suites implement **§6 and §7 (Class I) only**.
> Every published 140 number must carry its section and class qualifier; a bare percentage reads as a
> claim about the whole standard.
> **Summary 6/6:** Source of truth for the tracker issue is #4267; this file is the committed,
> reviewable copy. Update both together when coverage moves.

- **Maintainer:** fluxion maintainers
- **Last Updated:** 2026-10-23
- **Tracker issue:** [#4267](https://github.com/anchapin/fluxion/issues/4267)
- **Decision record:** [`docs/adr/0026-ashrae140-section12-class-ii-scope.md`](../adr/0026-ashrae140-section12-class-ii-scope.md)
- **Related:** #4277 (no acceptance criteria in Standard 140), #4276 (Annex A2 Standard Output Report)

---

## What "compliance" means here

Settled in #4277: **Standard 140 defines no acceptance criteria and issues no pass/fail
certificate.** Every statistics table in Informative Annex B carries the sentence "These statistics
do not have any substantial importance and are not to be interpreted as acceptance criteria"
verbatim, §4.4 ("Comparing Output to Other Results") is informative, and NREL's own reference
implementation (BESTEST-GSR) contains no scoring logic anywhere.

So conformance is: run the required cases, produce every required output, publish the results and
the modeler report publicly. The example-result ranges are **diagnostic context**, not thresholds.
Falling outside one is a signal to find the bug or document the difference; falling inside one is
not a certificate.

The only absolute targets in the entire standard are the analytical verification cases (§8 GC10a =
2433 W, the §11 quasi-analytical solutions) and a handful of physics identities. Those are tracked
in `data/reference/ashrae140/analytical_acceptance_criteria.json`.

**Consequence for this tracker: "status" below is coverage, not a score.**

---

## Section coverage

| Section | Test set | Class | Fluxion status | Issue |
|---------|----------|-------|----------------|-------|
| 6 | Weather Driver Tests (WD100–WD600) | I | **Implemented.** All six WD cases ran and every §6.2.1 required output is reported — per-case table `data/reference/section6/section6_conformance.csv` marks WD100–WD600 `conformant`; the release gate carries that artifact as a per-case checklist. **This is the weather-file interpretation suite.** | #4269 |
| 7 | Building Thermal Envelope and Fabric Load Tests | **I** | **Implemented.** All 18 cases ran; the per-case §6.2.1 table `data/reference/section7/section7_conformance.csv` marks each case `conformant` or `incomplete` — `incomplete` means a required output is still missing (the 6.2.1.8 day cells), not a score. | v1.3 milestone + ADR-0017 series |
| 7 (Add. a) | Hourly time-series outputs | I | **Partially implemented.** The emitted results CSV carries the GSR-contract 24-hour day profiles (Feb 1 / May 4 / Jul 14); the §7 table holds 6.2.1.8 at `partial` for the affected cases pending a clean copy of the Addendum a Table 6-1 day cells. No full hourly time series is published. | #4268 |
| 8 | Ground-Coupled Slab-on-Grade Analytical Verification | I | **Partially implemented.** The GC10a analytical case is implemented (closed-form oracle plus an FD measuring solver, `fluxion-core/src/ground_coupling.rs`) and its measurement is published in `data/reference/ashrae140/analytical_results.json` and checked by the release gate. The b-series (transient / periodic / convectively coupled) is unimplemented. | #4270 |
| 9a | Space-Cooling Equipment Performance Analytical Verification | I | **Not implemented as a test set.** CE100 performance map and the 14 CE case definitions are transcribed; a CE200 steady-state dispatch exists (`tests/validation/hvac_equipment_rated_performance/section9_ce.rs`). No CE case runs end-to-end, so no §9 outputs are produced. | #4271 |
| 9b | Space-Cooling Equipment Performance Comparative Tests | I | **Not implemented.** Comparative evidence ranges are transcribed (Task 22 ensemble bounds, `tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv`); no comparative CE-case run exists. | #4272 |
| 10 | Space-Heating Equipment Performance Tests | I | **Not implemented.** No §10 furnace case data in the tree; `bestest_he_reporting` stays out of scope until the equipment layer exists. | #4273 |
| 11 | Air-Side HVAC Equipment Performance Analytical Verification | I | **Analytical targets implemented, test set not.** QASv2 reimplemented (`fluxion-core/src/qas_airside.rs`); cases 1–6 × FC/SZ/CAV/VAV are published in `data/reference/ashrae140/section11_qas_targets.json` and `data/reference/ashrae140/analytical_results.json`, and the release gate checks the case-1 anchors. fluxion's own air-side model is not yet run against those targets. | #4274 |
| 12 | Building Thermal Envelope and Fabric Load Tests | **II** | **Not started.** No Class II weather file and no Class II case run in the tree. **In scope** (ADR-0026), blocked on §7. | Issue to file — title and body in ADR-0026 |

### Correction to an earlier version of this table

The previous revision listed §12 as "Class II test procedures" with no subject matter, and planning
notes had been treating it as a *weather data interpretation* section on the strength of
`Colorad.TMY` and `Lasvega.TMY` shipping with it. **That was wrong.** Per the 140-2023 section
source map:

- **§6** is Weather Driver Tests, sourced from old 4.3.1.6 / 5.6 / 6.6. This is where correct
  reading of a supplied weather file is the thing under test.
- **§12** is Building Thermal Envelope and Fabric Load Tests (Class II), sourced from old
  4.3.2 / 7.2 / 8.2. The real-location TMY files appear there because Class II cases are run in
  real climates, not because §12 tests the file reader.

§7 and §12 therefore test the same subject matter and differ by procedure class, not physics. The
general procedures split upstream at §5.1 (Class I, from old 5 / 5.1 / 6.1) and §5.2 (Class II,
from old 7 / 7.1 / 8.1).

This correction *reduces* estimated effort: §12 is a second-climate re-run of an engine §7 already
needs, so it costs close to nothing once §7 works, and it is the cheapest available check against
climate-specific tuning in the conduction core. See
[ADR-0026](../adr/0026-ashrae140-section12-class-ii-scope.md) and #4275.

---

## Three structural consequences

1. **#3984 (pressure-driven equation-based HVAC air loop) is a gating prerequisite, not a roadmap
   tail item.** Sections 9a, 9b, 10 and 11 are all equipment-performance test sets. The current
   ideal-load controller reads a setpoint and returns a load; it cannot express a CE100-series
   part-load case. Nothing in the equipment sections can start before #3984 lands.
2. **Section 8 needs a model that does not exist — except for GC10a.** The GC10a analytical
   verification now has its own closed-form oracle and FD measuring solver
   (`fluxion-core/src/ground_coupling.rs`, #4270), which is what the §8 row above records. The
   remaining §8 b-series (transient, periodic, convectively coupled) still needs 2D/3D slab
   conduction against a spec that the in-repo sources do not derive. #746 added a constant
   ground-temperature boundary for the Section 7 cases, which is not the same thing. The FD
   multi-node substrate from #3980 / #3983 is the natural host.
3. **Section 12 is not a parallel workstream.** It shares an implementation with §7 and must be
   explicitly blocked on it, or it gets picked up early and re-implements what §7 is already
   building.

---

## Critical path

```text
#3983 CTF per-surface conduction  (§7 core, and the host for §8)
        │
        ├──► §7 conforming ──► §12 (near-free second climate)
        │
        └──► #3980 FD substrate ──► §8 ground coupling (GC10a done; b-series open)

#3984 pressure-driven air loop  (gates everything below)
        │
        ├──► §9a ──► §9b
        ├──► §10
        └──► §11

§6 weather drivers  (independent; cheapest section in the standard — implemented)
```

§6 was worth doing early precisely because it is independent and cheap, and it has now been done:
its per-case conformance table is published and is a release-gate artifact (§6 row above), so the
reporting and publication pipeline (#4276, #4277, #4283) has been exercised end to end on a section
where the physics cannot be the thing that fails.

---

## Definition of done for this tracker

Every section above is either (a) implemented, with every required output produced and published, or
(b) carries a written, dated scope decision explaining why it is out of scope for the conformance
claim. A conformance claim that silently omits a section is the failure mode this tracker exists to
prevent.

Wherever fluxion states its 140 status, the **section and class scope must appear next to any
number**. A bare percentage with no qualifier reads as a claim about the whole standard.

---

## Refresh log

**2026-10-23 — statuses re-verified against current `develop` (#4267 refresh).** Every row was
checked against the committed evidence in the tree; no status is asserted from a plan or an issue
title. What changed and what did not:

| Row | Change | Evidence checked |
|------|--------|------------------|
| §6 | Was "Not implemented" — now implemented | `data/reference/section6/section6_conformance.csv` (WD100–WD600 `conformant`, every §6.2.1 requirement cell `report`), `data/reference/section6/section6_workflow_results.csv`, `tests/ashrae_140_section6_weather_driver.rs` (six-case finite-output run, Table 6-1 daily CSV, value pins), `scripts/release_gate_checker.py` conformance artifacts list (§6) |
| §7 | "Implemented, not conforming" → per-case coverage statement | `data/reference/section7/section7_conformance.csv` (per-case `conformant` / `incomplete`; `incomplete` = 6.2.1.8 `partial`), `data/reference/section7/section7_conformance.md`, `tests/ashrae_140_section7_conformance.rs`, release-gate checklist rendering per case |
| §7 Add. a | Issue `TBD` → **#4268**; status now "partially implemented" | `data/reference/section7/workflow_results.csv` (GSR-contract 24-hour day profiles `temp_0201`, `temp_0714`, `sens_htg_clg_0201`…), `data/reference/ashrae140/bestest_gsr_csv_contract.json` (`day_profile`: 24 hourly values), §7 table 6.2.1.8 `partial` cells |
| §8 | "No ground model in the suite" → GC10a implemented, b-series open | `fluxion-core/src/ground_coupling.rs` (closed-form oracle + FD measuring solver, module doc states the b-series remains unimplemented), `data/reference/ashrae140/analytical_results.json` → `section_8_ground_coupled`, release-gate `analytical_targets` row |
| §9a | Refined: not implemented as a test set; CE200 steady-state dispatch exists | `data/reference/ashrae140/section9_ce_cases.json` (`engine_coverage.cases_run: 0`), `tests/validation/hvac_equipment_rated_performance/section9_ce.rs`, `docs/KNOWN_ISSUES.md` LIMIT-42 |
| §9b | Refined: comparative evidence ranges transcribed, no comparative run | `data/reference/ashrae140/section9_ce_cases.json` → `acceptance.comparative` (evidence range, not a criterion), `tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv` |
| §10 | Unchanged — still not implemented | No §10 furnace case data in `data/reference/`; `data/reference/ashrae140/bestest_gsr_csv_contract.json` `_scope` keeps `bestest_he_reporting` out of scope until the equipment layer exists |
| §11 | "Not implemented" → analytical targets implemented, test set not | `fluxion-core/src/qas_airside.rs` (QASv2 port, cases 1–6 × FC/SZ/CAV/VAV), `data/reference/ashrae140/section11_qas_targets.json`, `data/reference/ashrae140/analytical_results.json` → `section_11_airside` (cases 1–6), release-gate `analytical_targets` row (case-1 anchors) |
| §12 | Unchanged — still not started, still blocked on §7 | `docs/adr/0026-ashrae140-section12-class-ii-scope.md`; no `Colorad` / `Lasvega` Class II weather file and no Class II case run in the tree |

The framing above is unchanged: **conformance is coverage** (the required cases ran and every
required output was produced and published), not a pass rate and not a score. Standard 140 defines
no acceptance criteria (#4277), so no row in this tracker carries a percentage, a threshold, or a
pass/fail verdict; per-case `conformant` / `incomplete` states are coverage states, and the
release-gate checklist renders them as a checklist for exactly that reason.

---

## Related

- #4267: the tracker issue this file snapshots.
- #4275: §12 scope decision (ADR-0026).
- #4277: no acceptance criteria exist; replace the pass-rate gate with conformance plus analytical
  targets.
- #4276: Annex A2 Standard Output Report and conformance table.
- #4281: release-over-release output diff as the CI regression gate.
- #4283: emit the Section 7 results CSV from the engine, to the BESTEST-GSR contract.
- ADR-0017 (equation-based DAE teacher) and its #3979–#3987 series own the Section 7 physics.
- #4239 is the research spike for the Section 9a methodology.
- #4268 owns the §7 Addendum a hourly time-series output gap.
