# ADR-0026: ASHRAE 140 §12 (Class II) is an Envelope Re-Run — In Scope, Sequenced After §7

> **Summary 1/6:** Decision: ASHRAE 140-2023 **§12 is Building Thermal Envelope and Fabric Load Tests (Class II)** — the §7 envelope case set re-run in a second and third climate against a Class II output template. It is **not** a weather-interpretation section.
> **Summary 2/6:** **Weather interpretation is §6** (Weather Driver Tests, WD100–WD600), sourced from old 4.3.1.6 / 5.6 / 6.6. The real-location TMY files (`Colorad.TMY`, `Lasvega.TMY`) that ship with §12 are there because Class II cases are run in real climates, not because §12 tests the weather-file reader.
> **Summary 3/6:** **Scope decision: §12 is IN SCOPE** for the fluxion ASHRAE 140 conformance claim, sequenced **immediately after §7 reaches passing status**, before the §9/§10/§11 equipment work completes. Rationale: it is the cheapest available increase in the breadth of a future conformance claim and the check most likely to expose a climate-specific tuning defect in the conduction core.
> **Summary 4/6:** **Sequencing constraint: §12 is explicitly BLOCKED on §7.** It shares an implementation with §7 (no new conduction, solar, infiltration or mass model — point the existing engine at a different TMY file, run the case set, populate a different output template) and must not be picked up before §7 works, or it will re-implement what §7 is already building.
> **Summary 5/6:** Scope-label rule adopted repo-wide: **every number fluxion publishes as an ASHRAE 140 figure must carry its section and class** (e.g. "ASHRAE 140 §7 (Class I)"). A bare percentage reads as a claim about the whole standard. Applied to `README.md`, `SCORECARD.md` and `docs/ASHRAE140_RESULTS.md` by this ADR.
> **Summary 6/6:** This ADR records the decision only. **No GitHub issue was filed by this change** (supervisor rule): the exact title and body of the blocked §12 implementation issue is preserved verbatim in the "Follow-up issue to file (blocked on §7)" section below for the maintainer to file.

- **Status:** 🟢 **Accepted** (Issue #4275, 2026-10-23)
- **Date:** 2026-10-23
- **Deciders:** Alex Chapin (issue author), fluxion maintainers
- **Supersedes:** None
- **Depends on:** #4267 (section coverage tracker), #4277 (no acceptance criteria in Standard 140), ADR-0017 (Section 7 physics ownership)
- **Related:** #4275 (this decision's source issue), #4269 (§6 Weather Driver Tests), `docs/compliance/section_coverage.md` (committed coverage tracker), `data/reference/ashrae140/analytical_acceptance_criteria.json` (140-2023 section source map)

## Context

Issue #4275 establishes what ASHRAE 140-2023 §12 actually is, and corrects a mis-mapping that
had travelled through fluxion's planning notes.

§12 was previously described only as "the Class II test procedures set", without saying what it
tests. That vagueness let an incorrect assumption take hold: §12 was being treated as a **weather
data interpretation** section, on the strength of `Colorad.TMY` and `Lasvega.TMY` appearing in its
supplemental files. That is wrong, and the correction changes the cost estimate substantially — a
weather-interpretation suite would need its own parsing, diagnostics and output plumbing, whereas
§12 needs none of it.

### The 140-2023 section source map

From the 140-2023 section source map (old 140-2020 clause numbers on the right; the same mapping
is transcribed in `data/reference/ashrae140/analytical_acceptance_criteria.json` →
`_standard_structure.section_numbering_2020_to_2023`):

| 2023 section | Title | Sourced from |
|---|---|---|
| 5.1 | Class I Test Procedures | 5 |
| 5.1.1 / 5.1.2 | Modeling Approach / Output Requirements | 5.1 / 6.1 |
| 5.2 | Class II Test Procedures | 7 |
| 5.2.1 / 5.2.2 | Modeling Approach / Output Requirements | 7.1 / 8.1 |
| 6 | Weather Driver Tests | 4.3.1.6, 5.6, 6.6 |
| 7 | Building Thermal Envelope and Fabric Load Tests (**Class I**) | 4.3.1, 5.2, 6.2.1 |
| 8 | Ground-Coupled Slab-on-Grade | 4.3.1.1.4, 5.2.4, 6.2.2 |
| 9 | Space-Cooling Equipment | 4.3.1.2 / 4.3.1.3, 5.3, 6.3 |
| 10 | Space-Heating Equipment | 4.3.1.4, 5.4, 6.4 |
| 11 | Air-Side HVAC Equipment | 4.3.1.5, 5.5, 6.5 |
| 12 | Building Thermal Envelope and Fabric Load Tests (**Class II**) | 4.3.2, 7.2, 8.2 |

Two consequences follow directly:

1. **§7 and §12 test the same subject matter.** Both are envelope and fabric load tests. They
   differ by *procedure class*, not by physics: §12 draws its inputs from old clause 7.2 and its
   outputs from old 8.2, both of which sit under the Class II umbrella at 5.2.
2. **Weather data interpretation is §6, not §12.** §6 (Weather Driver Tests, the WD100–WD600
   cases) is where correct reading of a supplied weather file is the thing under test. The
   real-location TMY files that ship with §12 are there because Class II cases are run in real
   locations.

### Why this makes §12 cheap rather than expensive

Once §7 genuinely passes, §12 costs roughly: point the existing engine at a different TMY file, run
the case set, populate a different output template. **No new conduction, solar, infiltration or
mass model.** The marginal engineering is close to zero and the marginal *credibility* is high,
because running the same physics through unrelated climates is exactly the check that catches
climate-specific tuning.

## Decision

**§12 is IN SCOPE** for the fluxion ASHRAE 140 conformance claim.

**Sequencing:** §12 is sequenced **immediately after §7 reaches passing status**, and before the
§9/§10/§11 equipment work is complete.

**Blocking:** §12 is **explicitly blocked on §7**. It is not a parallel workstream and must not be
staffed as one; it cannot be attempted before §7 works, and it becomes nearly free the moment it
does.

### Rationale

- It is the cheapest available increase in the breadth of a future conformance claim.
- It is the section most likely to expose an envelope bug that a single-climate test suite hides.
- Deferring it until after the equipment layer would mean discovering climate-sensitivity defects
  in the conduction core long after that core has been declared done.

### The argument against, and why it loses

The one argument against is that §12 widens the surface area before any section is finished. That
argument loses here, because §12 shares an implementation with §7 and therefore does not widen the
surface area at all — **it widens the evidence**.

### Consequences

1. **§12 must not be started before §7 passes.** The implementation issue below is blocked on §7.
2. **The §6/§12 mis-mapping must not reappear.** §6 is the weather-interpretation section; §12 is
   the envelope Class II section. The corrected coverage table lives in
   [`docs/compliance/section_coverage.md`](../compliance/section_coverage.md) (committed snapshot of
   the #4267 tracker) and is mirrored in the decision record below.
3. **Every published 140 number carries its section and class** (see the scope-label rule below).
4. **A 140 status claim covers §7 only until further notice.** fluxion's committed validation suite
   implements §7 (Class I) only; §6, §8, §9, §10, §11 and §12 are separate sections with their own
   issues under #4267.

## Scope-label rule (mandatory for published 140 status)

> **Wherever fluxion states its ASHRAE 140 status, the section and class scope must appear next to
> any number. A bare percentage with no section qualifier reads as a claim about the whole
> standard.**

Canonical qualifier strings used across the status documents:

| Surface | Qualifier to use |
|---|---|
| README badge / status blockquote / metric table | `ASHRAE 140 §7 (Class I)` |
| `SCORECARD.md` headline and pass-rate section | `ASHRAE 140 §7 (Class I)` |
| `docs/ASHRAE140_RESULTS.md` summary and per-case tables | `ASHRAE 140 §7 (Class I)` |

`§7 (Class I)` = Building Thermal Envelope and Fabric Load Tests — the 600/900 series, 195,
800/810, 960/970 and the free-float variants.

## Corrected section coverage table

Snapshot of the #4267 tracker, **with the §6/§12 correction applied**. The committed copy is
[`docs/compliance/section_coverage.md`](../compliance/section_coverage.md); the tracker issue itself
remains #4267.

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
| 12 | Building Thermal Envelope and Fabric Load Tests | **II** | Not started. **In scope** (this ADR), blocked on §7. | issue to file — see below |

### Correction to the earlier version of this table

The previous revision listed §12 as "Class II test procedures" with no subject matter, and planning
notes had been treating it as a *weather data interpretation* section on the strength of
`Colorad.TMY` and `Lasvega.TMY` shipping with it. **That was wrong.** Per the 140-2023 section
source map:

- **§6** is Weather Driver Tests, sourced from old 4.3.1.6 / 5.6 / 6.6. This is where correct
  reading of a supplied weather file is the thing under test.
- **§12** is Building Thermal Envelope and Fabric Load Tests (Class II), sourced from old
  4.3.2 / 7.2 / 8.2. The real-location TMY files appear there because Class II cases are run in
  real climates, not because §12 tests the file reader.

The general procedures split upstream at §5.1 (Class I, from old 5 / 5.1 / 6.1) and §5.2 (Class II,
from old 7 / 7.1 / 8.1).

## Follow-up issue to file (blocked on §7)

> **Maintainer action required.** Per the supervisor rule for this change, **no GitHub issue was
> filed**. File the issue below verbatim, with `blocked-by: §7` recorded (GitHub's native blocked-by
> relation, or an explicit "Blocked by: §7" line in the body, whichever the tracker supports).

**Title:**

```
[140-§12] Implement the Class II envelope-and-fabric re-run (second climate), blocked on §7
```

**Body:**

```markdown
## Summary

Section 12 of ASHRAE 140-2023 is **Building Thermal Envelope and Fabric Load Tests (Class II)**.
It is the Section 7 envelope-and-fabric case set re-run in a second and third climate against the
Class II output template — it is **not** a weather-interpretation section (§6 is: Weather Driver
Tests, WD100–WD600).

Per the 140-2023 section source map, §12 is sourced from old clause 4.3.2 (test descriptions),
7.2 (inputs) and 8.2 (outputs), and sits under the Class II general procedures at §5.2 (from old
7 / 7.1 / 8.1). The real-location TMY files that ship with §12 (`Colorad.TMY`, `Lasvega.TMY`) are
there because Class II cases are run in real climates — not because §12 tests the weather-file
reader.

Source: [ASHRAE Standard 140 Resource Files](https://data.ashrae.org/standard140/accompany.html).

## Why this is cheap

§12 shares an implementation with §7. There is **no new conduction, solar, infiltration or mass
model**: once §7 works, the work is

1. point the existing engine at the Class II weather file(s),
2. run the §7 case set against it,
3. populate the Class II output template (§8.2 outputs) rather than the Class I one (§6.2.1).

That makes §12 the cheapest available increase in the breadth of a future conformance claim, and
the check most likely to expose a climate-specific tuning defect in the conduction core.

## Blocked by §7

**This issue is blocked on Section 7 reaching passing status** (the v1.3 milestone / ADR-0017
series). It must not be picked up before then: it re-implements nothing precisely because §7 is
already building the shared implementation, and picking it up early means building that
implementation twice. The §12 coverage row in the #4267 tracker carries the same block.

## Proposed work

1. Obtain the §12 (Class II) case definitions, weather files and reference results from the
   140-2023 supplemental download; record their provenance hashes alongside the existing §6/§7
   provenance chain.
2. Run the §7 case set (600/900 series, 195, 800/810, 960/970, free-float variants) against the
   Class II weather file(s) with no physics change — the point is to re-run the *same* physics in a
   different climate.
3. Emit the §8.2 Class II outputs (Annex A2 Standard Output Report form, per #4276) and publish
   them with the §7 results.
4. Compare against the Class II reference results with per-metric citations. Per #4277 these are
   diagnostic context, **not** acceptance criteria — Standard 140 defines no pass/fail threshold.

## Acceptance criteria

- §12 cases runnable and deterministic, from the committed case definitions.
- Class II outputs produced in the Annex A2 Standard Output Report format and published.
- Results compared against the Class II reference results, with per-metric citations.
- A written statement of what the Class II re-run does and does not cover (e.g. whether the
  Class II case set is a subset of §7, and which §7 cases have no Class II counterpart).
- The §12 row in the #4267 coverage tracker is updated from "Not started" to the achieved state.

## Context

- Parent: #4267 (section coverage tracker).
- Blocked by: §7 (Building Thermal Envelope and Fabric Load Tests, Class I) — the v1.3 milestone
  and the ADR-0017 series.
- Scope decision: ADR-0026 (`docs/adr/0026-ashrae140-section12-class-ii-scope.md`) — §12 is IN
  SCOPE, sequenced immediately after §7.
- Related: #4277 (no acceptance criteria exist in Standard 140), #4276 (Annex A2 Standard Output
  Report), #4269 (§6 Weather Driver Tests — the actual weather-interpretation section), #4283
  (§7 BESTEST-GSV CSV emitter).
```

## Files touched by this decision

| File | Change |
|---|---|
| `docs/adr/0026-ashrae140-section12-class-ii-scope.md` | This decision record (new). |
| `docs/compliance/section_coverage.md` | Committed coverage tracker with the corrected §6/§12 mapping (new). |
| `README.md` | Section/class qualifier next to every published 140 number (badge, status blockquote, metric table). |
| `SCORECARD.md` | Same, via `scripts/generate_scorecard.py` (SCORECARD.md is generated — do not hand-edit). |
| `scripts/generate_scorecard.py` | Emits the §7 (Class I) qualifier in the headline and pass-rate sections. |
| `docs/ASHRAE140_RESULTS.md` | Section/class qualifier on the summary tables and the context header. |

## Verification

```bash
grep -rn "Class II" docs/ README.md SCORECARD.md
grep -rn "§7" README.md SCORECARD.md docs/ASHRAE140_RESULTS.md | head
python3 scripts/generate_scorecard.py --check      # SCORECARD.md is not stale
python3 scripts/check_scorecard_data_sources_consistent.py
python3 scripts/check_docs_summaries.py
python3 scripts/check_doc_inventory_fresh.py
python3 scripts/check_doc_link_integrity.py
```
