# REF-01 — investigation history

Narrative history for **REF-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `REF-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### REF-01: Blind-validation monthly reference data — recast as v1.3 documented-shape reference (issues #2677 → #2748)

- **Description:** The monthly heating/cooling reference CSVs at
  `tests/reference_data/ashrae140/monthly/case_{600,900}_monthly_reference.csv`
  — consumed by the Phase D ±10% monthly criterion in
  `tests/ashrae_140_blind_validation.rs::test_monthly_energy_validation_baseline`
  — are a **documented-shape reference**, not direct EnergyPlus monthly
  outputs. They are a degree-day-derived *shape* (computed from the repo's own
  Denver TMY3 hourly weather, ASHRAE Fundamentals degree-day method, balance
  point 18.3 °C) applied to the authoritative *annual* midpoint
  (NREL/TP-472-6231 Table 3-2 / ASHRAE 140-2023 Annex B). The annual totals are
  authoritative; the monthly *distribution* is a physically-reasonable
  approximation (winter heating peaks Dec/Jan/Feb, summer cooling peaks
  Jun/Jul/Aug, shoulder-season near-zero) rather than a fabrication.
- **Why no authoritative monthly data exists in-repo (#2748 investigation,
  2026-08-13):**
  1. ASHRAE 140-2023 Annex B publishes only annual + peak figures (no monthly
     breakdown).
  2. The IEA SHC Task 12 / BESTEST report (NREL/TP-472-6333) and the EnergyPlus
     BESTEST validation reports carry monthly figures as plots only, not
     citeable tabulated values.
  3. EnergyPlus 25.2.0 runs on the in-repo Case 600/900 IDFs but reproduces
     cooling ~50× below (Case 600) and ~5× below (Case 900) the ASHRAE band;
     Case 900 heating is 8.6× above and inverted in direction from Case 600 —
     the IDFs need insulation/glazing/concrete-mass fixes before E+ output can
     serve as the monthly reference. Using those numbers would itself be a
     different shape of fabrication. The IDF physics fix is tracked under
     §SOLAR-02 UPDATE (Issue #2239) and §LIMIT-05.
- **Affected Cases:** 600, 900 (monthly metric only — annual/peak metrics use
  the authoritative annual bands and are unaffected).
- **Affected Metrics:** Phase D ±10% monthly heating/cooling energy.
- **Severity:** High (the v1.3 DoD phrase *"true ASHRAE reference values"* is
  not literally satisfied because ASHRAE 140-2023 does not publish a monthly
  breakdown; the documented-shape reference is the strongest physically-
  defensible substitute available without new E+ physics work or a new
  published monthly source).
- **GitHub Issues:** #2677 (origin: placeholder is fabricated, not
  authoritative), #2748 (resolution: recast as documented-shape reference +
  un-ignore the test + add E+ regeneration tooling).
- **Status:** � **Resolved (#2748)** — the v1.3 DoD-blocker framing is
  retired: CI no longer reports false-confidence pass/fail against fabricated
  data, every monthly PASS/FAIL is against the documented-shape reference
  derived from the authoritative annual midpoint. Specifically:
  1. Both CSVs at `tests/reference_data/ashrae140/monthly/case_{600,900}_monthly_reference.csv`
     carry a new "STATUS: v1.3 Reference (documented derivation)" header that
     documents the method (ASHRAE Fundamentals Ch. 19 degree-day
     redistribution) and the deferred-work path (E+ regeneration once
     Issue #2239 closes).
  2. The monthly `README.md` §STATUS block was rewritten to reflect the new
     interpretation, the §Caveats block was updated to drop the
     "no-signal" framing, and a §Regeneration path replaces the §TODO.
  3. The dependent CI gate `test_monthly_energy_validation_baseline` was
     un-`#[ignore]`'d and now runs against the documented-shape reference
     in CI. The gate remains **reporting-only** (no assert) because the
     engine cooling under-prediction means the pass rate will be low until
     Issue #2239 closes — the correct signal. Once #2239 closes, the gate
     can be hardened to assert a Phase D pass-rate target.
  4. A new `scripts/generate_monthly_aggregate.py` (with 41 unit tests in
     `scripts/ci/test_generate_monthly_aggregate.py`) is the in-place
     replacement for these CSVs: it consumes
     `tests/reference_data/zone_balance/case_<id>_energy_hourly.csv` (produced
     by `generate_case_600_900_energy.py` from the in-repo IDFs) and emits
     the same-schema `case_<id>_monthly_reference.csv` here. Its
     `--validate` subcommand re-runs the reduction and asserts Σ(monthly) is
     inside the annual band (catches E+ regenerator regressions before they
     reach the monthly CSVs).
- **Phase Addressed:** v1.3 (Phase C: BENCH-01 — true ASHRAE reference data
  replaces calibrated ranges).
- **Resolution Notes:** The DoD language *"true ASHRAE reference values"* is
  not literally satisfied for the monthly dimension because no published
  source carries it (issue #2748's investigation explored ASHRAE 140-2023
  Annex B, the NREL/TP-472-6333 BESTEST report, and EnergyPlus BESTEST
  validation reports — none cite tabulated monthly values for Cases 600/900).
  The documented-shape reference is the strongest physically-defensible
  substitute: it sums to the authoritative annual midpoint **exactly**, has a
  documented physical method (ASHRAE Fundamentals Ch. 19), and does not
  require new E+ physics work or a new published source. The v1.3 DoD
  blocker is resolved by making the situation honest: CI no longer reports
  pass/fail against fabricated data, the reference derivation is documented
  in the CSV header + README + this entry, and the regeneration path is in
  place for when Issue #2239 closes. When E+ reproduces the ASHRAE band, run
  `scripts/generate_monthly_aggregate.py --case 600` (and `--case 900`) to
  overwrite these CSVs with direct-E+ monthly totals in the same schema.
