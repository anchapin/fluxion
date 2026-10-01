# SOLAR-02 — investigation history

Narrative history for **SOLAR-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `SOLAR-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 2. No wording was changed, softened or deleted.

---

#### SOLAR-02: Annual Cooling Energy Under-Prediction (High-Mass)

- **Description:** Annual cooling energy for high-mass cases (900 series) is under-predicted by 30-80%. While the 5R1C model has known limitations for high-mass buildings, the magnitude of error exceeds acceptable tolerance. This likely relates to solar gain timing and thermal mass coupling - high-mass buildings distribute cooling load over time, but total seasonal cooling should still match reference.
- **Affected Cases:** 900, 910, 920, 930, 940, 950
- **Affected Metrics:** Annual Cooling (MWh)
- **Severity:** High
- **GitHub Issue:** #275
- **Status:** 🔄 Open (partially mitigated)
- **Phase Addressed:** Phase 3
- **Resolution Notes:** Mode-specific coupling corrections (heating vs cooling) improved peak loads but annual cooling still low. Model limitation acknowledged but magnitude too large - requires further solar gain integration fixes.

#### SOLAR-02 UPDATE (Issue #2239, 2026-07-31): Case 900 residual annual-energy deviation — confirmed structural

- **Status:** ✅ Confirmed as known 5R1C architectural limitation (routed to GaugeSolver #1465)
- **Context:** After the combined fixes from #2227 (`derived_h_tr_3` ISO 13790 §6.3
  combined conductance replacing `h_tr_ms` in the HVAC coupling path) and #2229
  (`h_ms_coeff` 9.1 → 13.4 W/(m²·K) for HighMass), Case 900 still falls outside
  the ASHRAE 140 reference ranges:

  | Metric  | Fluxion  | ASHRAE 140 Ref | Deviation from midpoint |
  |---------|----------|----------------|-------------------------|
  | Heating | 2.362 MWh | [1.17, 2.04] MWh | +47 % above midpoint (+15.8 % over upper bound) |
  | Cooling | 1.330 MWh | [2.13, 3.67] MWh | −54 % below midpoint (−37.6 % under lower bound) |

- **Pattern:** Heating **too high** AND cooling **too low** is the textbook
  signature of a single lumped thermal-mass node integrated on a 1-hour timestep
  (documented in §LIMIT-05 UPDATE). The mass node cannot simultaneously:
  (a) release stored solar heat fast enough during shoulder/cooling seasons
  (driving annual heating up), and (b) absorb enough daytime solar to charge the
  thermal mass for night-time cooling release (driving annual cooling down).
  No single `h_ms_coeff` or `derived_h_tr_3` adjustment can move both metrics
  into band simultaneously — see the #1522 air-node investigation which proved
  this trade-off is structurally infeasible at `dt/τ ≈ 3.6`.

- **Investigated & ruled out (per issue #2239):**
  1. **f_furniture adjustment** — would shift heating and cooling in the *same*
     direction (both up or both down); cannot close the bidirectional gap.
  2. **derived_h_tr_3 formula revisiting** — already correct per ISO 13790 §6.3
     (verified in `docs/research/iso13790_equation_mapping.md` Eq C.8). The
     `h_tr_ms=1608 W/K → derived_h_tr_3=43.2 W/K` change (#2227) was the major
     advance (H: 5.835 → 2.343 MWh); further tuning of the series formula does
     not help.
  3. **South wall bypass (#715)** — the #715 fix is applied (see diagnostic
     output `R_ext_to_mass=0.888`); reverting it would worsen, not close, the gap.

- **Why this is NOT a fixable bug (per AGENTS.md / RULES.md):**
  - The deviation magnitudes (+47 % / −54 % from midpoints) fall **squarely
    within** the documented LIMIT-01 range ("heating 30–200 % above reference")
    and SOLAR-02 range ("cooling under-predicted by 30–80 %").
  - Closing the gap by adjusting `h_ms_coeff`, `f_furniture`, or
    `derived_h_tr_3` constants would be **parameter tuning to pass system tests**
    — explicitly forbidden by AGENTS.md ("fix the underlying math").
  - The correct fix is the **GaugeSolver** (#1465 / #1462), which treats solar
    as geometric curvature rather than per-timestep energy injection, or
    **sub-hour air-node sub-stepping** — both out of scope for #2239.

- **Resolution:** Documented as a known limitation. No physics-code change.
  Diagnostic infrastructure test `test_case_900_blind_energy_infrastructure`
  passes (it reports values, not a reference-bound gate). Tracked by #1465.
