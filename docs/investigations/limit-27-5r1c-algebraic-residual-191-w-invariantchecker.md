# LIMIT-27 — investigation history

Narrative history for **LIMIT-27**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-27` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-27: 5R1C algebraic residual ~191 W — InvariantChecker evaluates gains at T_old while heat flows use T_new (Issue #3647)

- **Description:** The `InvariantChecker`'s algebraic formulation for
  5R1C has a known systematic residual of ~191 W due to evaluating
  gains at `T_old` while heat flows use `T_new`. This is a
  **test-infrastructure tolerance**, not a physics gate:
  `tests/test_energy_conservation.rs` sets
  `ENERGY_BALANCE_RESIDUAL_THRESHOLD: f64 = 200.0` (headroom above the
  ~191 W residual) for the `InvariantChecker` algebraic-consistency
  assertions across Cases 600 / 900 / 960 / 600FF. The actual energy
  conservation of the simulation is validated by the zone balance
  tests (`tests/zone_balance_eplus_isolation.rs`), which PASS with
  zero violations.
- **Affected Cases:** 600, 900, 960, 600FF (the
  `test_case_*_energy_conservation_residual` family in
  `tests/test_energy_conservation.rs`)
- **Affected Metrics:** None — this is an algebraic-identity
  tolerance on the `InvariantChecker`, not an ASHRAE 140 reported
  metric.
- **Severity:** Low
- **GitHub Issue:** [#3647](https://github.com/anchapin/fluxion/issues/3647)
  (doc-vs-code drift fix); original tolerance introduced by
  [#2225](https://github.com/anchapin/fluxion/issues/2225) / PR
  [#2230](https://github.com/anchapin/fluxion/pull/2230) (commit
  `12425d8`, 2026-07-31)
- **Status:** 🟡 **Docs-drift resolved (#3647); 5R1C algebraic residual remains a documented test-infra tolerance — structural fix routed to GaugeSolver #1465/#1462.**
- **History / naming note:** The tolerance was introduced under the
  citation "documented as FREE-04 in docs/KNOWN_ISSUES.md" (commit
  `12425d8` / PR #2230), but no `FREE-04` entry ever existed — the
  FREE-\* family in this document covers free-floating temperature
  validation findings (FREE-01/02/03), not solver-checker algebraic
  residuals. Issue #3647 repointed the test citation to this entry.
  The `LIMIT-25` / `LIMIT-26` numbers are reserved for the Cases
  800 / 810 structural entries requested by the
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
  `_doc_issue3572` note (Issue #3572), hence `LIMIT-27`.
- **Sibling framing:** Same `InvariantChecker` post-step
  algebraic-invariant family as §LIMIT-19 / Issue #3103
  (`test_one_watt_artificial_gain_increases_imbalance`) and
  §MULTI-03 / Issue #3066 (the 88.7 W hand-balanced stub residual on
  the 9R4C BE-implicit identity). Deleting or widening the 200 W
  threshold would de-facto relax the strict-energy regression guard
  (Issues #2506 / #3572) — treat threshold changes as gate changes,
  not cleanup.
