# FLUID-01 — investigation history

Narrative history for **FLUID-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `FLUID-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### FLUID-01: Analytical Jacobian Saturation/Clamping Errors

- **Description:** Analytical Jacobians for Chiller, Boiler, CoolingCoil, Pump, and VavBox do not properly handle non-smooth behavior at clamping/saturation points. When model inputs are clamped to bounds (e.g., COP clamped to minimum 0.1 in `Chiller::evaluate`, efficiency clamped in `Boiler::evaluate`), the analytical derivative formulas continue to compute values for the unsaturated case. This causes the analytical Jacobian to differ significantly from the finite-difference Jacobian, which correctly captures the saturated behavior (zero derivative at clamp points).
- **Affected Tests:** `fluxion-fluid` — `test_chiller_jacobian_accuracy`, `test_boiler_jacobian_accuracy`, `test_cooling_coil_jacobian_accuracy`, `test_pump_jacobian_accuracy`, `test_vav_box_jacobian_accuracy`
- **Affected Metrics:** N/A (unit tests only)
- **Severity:** Medium
- **GitHub Issue:** #2330
- **Status:** 🔄 **Known Limitation** — Analytical Jacobians compute derivatives assuming smooth functions, but `max()`/`clamp()` in `evaluate` create non-smooth points. Fixing would require subgradient or automatic differentiation support.
