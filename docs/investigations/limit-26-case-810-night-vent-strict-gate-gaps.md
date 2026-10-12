# LIMIT-26 — Case 810 (high-mass + night vent) strict-gate known-fails

Provenance history for the `LIMIT-26` row in [`docs/KNOWN_ISSUES.md`](../KNOWN_ISSUES.md). Added 2026-10-12 (physics loop round 19, issue #4168 follow-up) so the strict-gate baseline's `_doc_issue3572` note (issue #3572) no longer points at a missing structural entry.

## What the row tracks

Case 810's strict-gate heating gap was already structurally tracked in §LIMIT-36 (the honest below-band movement 3.823 → 2.455 → 1.680 MWh caused by the #4336 9R4C mass coupling), but the case as a whole had no formal `§LIMIT-NN` entry. #4387 (round 18) made this row newly meaningful: the Case 800/810 bands in the strict-gate baseline are now the transcribed ASHRAE 140-2023 Annex B8 §B8.1 inter-program envelope (Std140_TF_Results.pdf Tables B8-1..B8-4), not synthetic-fitted ranges.

## Current numbers

Measured on develop `20ab09e1` (deterministic re-measure recorded in the round-18 re-record); bands per #4387:

| Metric | Engine value | Published envelope | Gap (% of midpoint) | Status |
|---|---|---|---|---|
| Case 810 heating | 1.680 MWh | [2.038, 2.610] | 12.69 (under) | known_fail |
| Case 810 cooling | 1.684 MWh | [1.295, 1.606] | 1.10 (over) | known_fail |

The 810 cooling gap FELL 46.73 → 1.10 in #4387 purely because the synthetic band it was measured against was itself unfit; the engine value 1.684 MWh re-measured byte-identical. No gap was raised or lowered to hide anything.

## Structural reading

- Heating: the §LIMIT-36 movement — the #4336 mass coupling, shared with Case 900 by construction.
- Cooling: night-vent / cooling-chain family (§LIMIT-28 and the §LIMIT-16 chain), now evaluated against the real published envelope.

## Closes when

The same structural closures as §LIMIT-36 (heating) and the night-vent/cooling chain (cooling). Per RULES.md the baseline gaps are recorded, never tuned.
