# LIMIT-23b — Case 970: reference-band provenance and blind-path envelope ledger

Investigation record for **loop round 20** (2026-10-12, America/New_York),
following up §LIMIT-23 / Issue #3552 on develop tip `7e321a22`. Companion to
`limit-23-case-970-5-zone-multi-zone.md` (verbatim history, unchanged).

## Reproduction on the tip

`cargo test --release --features ort --test all_tests
test_case_970_annual_energy_ashrae140_tolerance -- --include-ignored
--nocapture` on `7e321a22`:

- Annual heating **H = 4.616 MWh** (band [10.54, 14.26]) — matches the
  strict-energy-gate baseline exactly.
- Annual cooling **C = 2.341 MWh** (band [7.391, 9.999]) — matches.

Known-fail by design; the gate compares against the recorded gaps
(47.77 % / 58.08 % of band midpoint).

## Finding 1 — the "published" band has no published source

The 970 reference band (heating 10.54–14.26 MWh, cooling 7.39–10.00 MWh)
is cited everywhere as the "ASHRAE 140-2017 §B6.7 / 140-2023 Annex B8-3
inter-program envelope". Both in-repo primary sources contradict that:

1. **ASHRAE 140-2017 (with Addendum a), `docs/140_2017_a_20200901.pdf`.**
   Full-text search finds no Case 970 anywhere (only unrelated hits:
   a molecular-weight figure, an ach conversion, and the Annex B8 figure
   lists, which cover window cases 660–695 and 980–995, the sunspace 960,
   and the 195–950 families). There is no §B6.7 five-zone cross-coupling
   case in the published standard this repository carries.
2. **ASHRAE 140-2023 Annex B8 §B8.1, authoritative in-repo transcription
   `data/reference/ashrae140/annex_b8_section7_statistics.json`** (the
   round-18/#4168 source, verified against Std140_TF_Results.pdf).
   Its scope line reads "Cases 195-995 and 600FF-980FF" and its
   Table B8-1 (annual heating, six programs) case list is 600…695, 900…950,
   **960, 980, 985 — no 970**.

So the band was introduced at Issue #1446 as if published, but no ASHRAE
140 edition or annex available to this repository contains a Case 970, and
the real Annex B8-3 is the *peak heating* statistics table for the basic
cases, not an annual-heating band for a five-zone case. The band should be
treated as an **internal target of unverified origin**, not an
inter-program envelope.

Physical consistency supports this: Case 970 shares the Case 900 envelope
(8 m × 6 m × 2.7 m high-mass, 12 m² south glazing, same setpoints), whose
published annual heating reference is ≈ 1.6 MWh. An 8× heating increase
(12.4 MWh midpoint) from adding interior partitions to the same building
is not physically credible; a 48 m² Denver building at 258 kWh/m²·yr
heating is far outside the 900-series family behaviour.

**Consequence for §LIMIT-23:** "4/4 metrics out of band" has been measured
against an unfit target since #1446. The engine's under-band gaps
(47.77 % / 58.08 %) quantify distance from an internal target, not
non-conformance with ASHRAE 140. No band value is changed by this round
(RULES.md: never move a band to make a gap disappear — in either
direction); the baseline `_doc` and the provenance text are corrected
instead.

## Finding 2 — the blind-path envelope ledger over-counts exterior surfaces

`ThermalModel::from_spec_with_selector`
(`src/sim/thermal_model_core/mod.rs`) builds, for **every** zone, six
surfaces from that zone's own geometry with exterior U-values and outdoor
(sol-air) exposure. For Case 970:

- Zone 0 (4 × 6 × 2.7): S/N 10.8 m² each, E/W 16.2 m² each.
- Zones 1–4 (4 × 1.5 × 2.7): S/N 10.8 m² each, E/W 4.05 m² each.
- Engine exterior-wall total: 2·54 + 2·28.35 = **164.7 m²**.
- Real building envelope (perimeter 28 m × 2.7 m): **75.6 m²**.

Every interior partition interface appears **twice as outdoor-exposed
exterior wall** (once per adjacent zone) *in addition* to the separate
common-wall conductances added from `spec.common_walls`. The partition
faces also receive full exterior film + sol-air treatment instead of
interior conditions. Roof (48 m²) and floor (48 m²) sum correctly.

Per-facade detail (from the layout in `series_970.rs`: west core 4 × 6,
east strip of four 4 × 1.5 zones stacked along the 6 m depth):

| Facade | Real | Engine | Note |
|---|---:|---:|---|
| South | 21.6 | 54.0 | zones 1–3 S walls are interior |
| North | 21.6 | 54.0 | zones 2–4 N walls are interior |
| East  | 16.2 | 28.35 | zone 4 E alone is exterior (4.05) |
| West  | 16.2 | 28.35 | only core W (16.2) is exterior |

The engine is simultaneously over-exposed on partition faces and
under-areas the true east facade; the net error is +89.1 m² of exterior
wall (2.18×). This is consistent in direction and rough magnitude with the
engine's 970/900 heating ratio (4.616 / 1.566 ≈ 2.95) against a physical
expectation only moderately above 1.

The same construction applies to Case 960 (2-zone sunspace), whose
current band positions were reached *with* this behaviour — any fix moves
a case that is partially in band today.

## Why this is not fixed in this round

A principled repair needs to know which zone faces lie on the building
perimeter. `CaseSpec` carries per-zone geometry and a common-wall list but
**no zone coordinates or orientation of the shared faces**, so perimeter
exposure is not derivable from the spec alone. That is a schema change
(`CaseSpec` zone layout) plus re-derivation of the 960/970 surface
builders and validators, moving cases that are partially in band —
outside loop merge authority. Recorded here and in the §LIMIT-23 row for
an Alex-level scope decision.

## Is 970 still "GaugeSolver-blocked"?

No. The gauge unification was retired in loop round 14 (bypass stays), and
the strict-gate 970 test
(`tests/all_tests/zone_balance_eplus_isolation.rs::test_case_970_annual_energy_ashrae140_tolerance`)
drives the **legacy blind path**
(`ThermalModel::from_spec_with_selector` + `run_blind_annual_energy`)
directly — no GaugeSolver in the loop. The QUARANTINE.md rows whose
un-ignore condition reads "GaugeSolver (#1465/#1462) ships" carry a stale
premise: the residual lives in the legacy multi-zone spec path (Finding 2)
against an unfit band (Finding 1). The framing in §LIMIT-23's title
("GaugeSolver-blocked") is superseded by this document.

## Round-20 numbers moved

None. Documentation/provenance only; engine, band values, baseline gaps
and statuses unchanged. Reproduction confirmed the recorded strict-gate
values bit-for-basis (H 4.616, C 2.341).
