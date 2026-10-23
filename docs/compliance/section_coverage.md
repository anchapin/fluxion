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
> **Summary 5/6:** fluxion's committed validation suite implements **§7 (Class I) only**. Every
> published 140 number must carry its section and class qualifier; a bare percentage reads as a
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
| 6 | Weather Driver Tests (WD100–WD600) | I | Not implemented. **This is the weather-file interpretation suite.** | #4269 |
| 7 | Building Thermal Envelope and Fabric Load Tests | **I** | Implemented, not conforming | v1.3 milestone + ADR-0017 series |
| 7 (Add. a) | Hourly time-series outputs | I | Not implemented (validator reports annual + peak only) | TBD |
| 8 | Ground-Coupled Slab-on-Grade Analytical Verification | I | No ground model in the suite. Reference data transcribed. | #4270 |
| 9a | Space-Cooling Equipment Performance Analytical Verification | I | Not implemented. CE100 performance map transcribed. | #4271 |
| 9b | Space-Cooling Equipment Performance Comparative Tests | I | Not implemented | #4272 |
| 10 | Space-Heating Equipment Performance Tests | I | Not implemented | #4273 |
| 11 | Air-Side HVAC Equipment Performance Analytical Verification | I | Not implemented. QAS oracle reimplemented, targets generated. | #4274 |
| 12 | Building Thermal Envelope and Fabric Load Tests | **II** | Not started. **In scope** (ADR-0026), blocked on §7. | Issue to file — title and body in ADR-0026 |

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
2. **Section 8 needs a model that does not exist.** Ground-coupled slab-on-grade analytical
   verification requires 2D/3D slab conduction against analytical solutions. #746 added a constant
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
        └──► #3980 FD substrate ──► §8 ground coupling

#3984 pressure-driven air loop  (gates everything below)
        │
        ├──► §9a ──► §9b
        ├──► §10
        └──► §11

§6 weather drivers  (independent; cheapest section in the standard)
```

§6 is worth doing early precisely because it is independent and cheap. It is a plausible *first*
section to reach full conformance, which exercises the whole reporting and publication pipeline
(#4276, #4283) on a section where the physics cannot be the thing that fails.

---

## Definition of done for this tracker

Every section above is either (a) implemented, with every required output produced and published, or
(b) carries a written, dated scope decision explaining why it is out of scope for the conformance
claim. A conformance claim that silently omits a section is the failure mode this tracker exists to
prevent.

Wherever fluxion states its 140 status, the **section and class scope must appear next to any
number**. A bare percentage with no qualifier reads as a claim about the whole standard.

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
