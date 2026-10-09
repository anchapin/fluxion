# §LIMIT-39 — Case 630 annual cooling over band after the E+ Perez coefficient-table fix

Opened: 2026-10-09. Owner issue: #4332 (solar-delivery chain).

## Symptom

`ashrae_140_case_600_series::case_630::test_annual_cooling` (quarantined
`#[ignore]` 2026-10-09): Case 630 annual cooling 4.03 MWh vs the published
band [2.13, 3.70] MWh (+9% over the upper bound). The test was in band on the
previous (Perez 1990 journal) coefficient set and moved out the UPPER side
when the coefficient table was corrected to the E+ 1999 set — the opposite
direction from every cooling gap in the §LIMIT-05/§SOLAR-02 family, which
were UNDER band and narrowed.

## Mechanism

Case 630 is the east/west-shading variant of the 600 series. The corrected
coefficient set raises sun-facing vertical-surface irradiance most strongly in
the circumsolar-heavy clear bins; on the 630 geometry (large east/west glazing
fraction relative to its shading surfaces) the extra beam-plus-circumsolar
delivery lifts annual cooling past the band's upper bound. The same fix moved
630 annual heating TOWARD its band (6.95 → 6.65 MWh vs [5.05, 6.47], still
over) and 630 peak heating into consideration range.

## Open suspect

The residual east/west irradiance asymmetry (limit-35 doc §13): the engine
amplifies the EPW's own morning DNI weighting (engine E/W annual beam ratio
1.69 vs the reference engine's 1.16) through hour-start solar-position
sampling; an intra-hour integration convention would redistribute the
east/west (and thereby 630's) annual cooling. Not resolvable from in-repo
reference data alone.

## Closes when

A physics decision brings Case 630 annual cooling inside [2.13, 3.70] without
moving any other metric out of its published band. No band widening; the
quarantine is recorded in tests/QUARANTINE.md.
