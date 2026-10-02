# BASE-03 — investigation history

Narrative history for **BASE-03**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `BASE-03` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### BASE-03: Thermal Mass Capacitance Incorrect

- **Description:** Thermal mass capacitance (Cm) values were either missing or incorrectly derived from construction materials. ASHRAE 140 cases specify precise thermal mass properties that must be matched exactly. Incorrect Cm causes wrong time constant and thermal lag.
- **Affected Cases:** High-mass cases (900, 910, 920, 930, 940, 950, 900FF, 950FF) and any case with significant thermal mass.
- **Affected Metrics:** Temperature swing, thermal lag, free-floating temperatures, seasonal energy
- **Severity:** High
- **Status:** ✅ Fixed (Phase 2)
- **Phase Addressed:** Phase 2
- **Resolution Notes:** Construction specifications now correctly compute thermal mass capacitance from material layers (volumetric heat capacity × volume). Case 900 thermal mass properly configured.
- **v1.3 No-Tuning Resolution (Issue #2706, 2026-08-11):** The empirical thermal-mass "correction factor" in `src/validation/thermal_mass.rs` — `clamp(1/sqrt(C/2.4e6 J/K), 0.2, 1.0)` with a hardcoded 2.4e6 J/K reference capacitance — was REMOVED. It had no first-principles derivation (lumped-capacitance damping follows `τ=RC` / `1/sqrt(1+(ωτ)²)`, not `1/sqrt(C)`; semi-infinite effusivity `sqrt(k·ρ·cₚ)` also depends on conductivity `k`), and it was never part of the ASHRAE 140 validation pipeline (`docs/CORRECTION_FACTORS_INVENTORY.md` §4.1: "not in pipeline"; no callers outside the file). Its removal therefore changes **no** ASHRAE 140 validation result — it only eliminates a v1.3 DoD "zero correction factors" violation. The genuine structural checks (capacitance ratio ≥ 3.0, 6R2C envelope/internal mass fractions) remain, since they are measured model properties, not post-hoc tuning.
