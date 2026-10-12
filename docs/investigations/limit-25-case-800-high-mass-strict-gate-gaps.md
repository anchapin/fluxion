# LIMIT-25 — Case 800 (high-mass base, conditioned) strict-gate known-fails

Provenance history for the `LIMIT-25` row in [`docs/KNOWN_ISSUES.md`](../KNOWN_ISSUES.md). Added 2026-10-12 (physics loop round 19, issue #4168 follow-up) so the strict-gate baseline's `_doc_issue3572` note (issue #3572) no longer points at a missing structural entry.

## What the row tracks

The strict ±15% annual-energy gate (issue #2506, extended by #3572 to the eight-case cohort) has carried Case 800's heating and cooling as `known_fail` since the cohort was extended, but Cases 800/810 were the only two cohort members without a formal `§LIMIT-NN` structural entry — the rationale lived only in the baseline JSON's `_doc_issue3572` field.

## Current numbers

Measured on develop `20ab09e1` (deterministic re-measure recorded in the round-18 re-record, PR #4387); bands are the ASHRAE 140-2023 Annex B8 §B8.1 inter-program envelope transcribed by #4387:

| Metric | Engine value | Published envelope | Gap (% of midpoint) | Status |
|---|---|---|---|---|
| Case 800 heating | 7.039 MWh | [4.906, 5.403] | 21.55 (over) | known_fail |
| Case 800 cooling | 3.591 MWh | [0.282, 0.473] | 836.29 (over) | known_fail |

## Structural reading

- Heating: Case 800 is the high-mass conditioned base; its heating metric shares the 9R4C mass-path structural gap of §LIMIT-35 (Case 900). Case 810's heating equals Case 900's by construction (both `case_900_baseline()` specs with the HVAC equipment swapped, §LIMIT-36); Case 800 carries the same mass-path family on its own spec.
- Cooling: the 836% gap is the standing cooling-chain under-prediction family (§LIMIT-16, §LIMIT-28), not an 800-specific defect.

## Closes when

GaugeSolver multi-node mass path (heating) and the cooling-chain fix (cooling). Per RULES.md the baseline gaps are recorded, never tuned.
