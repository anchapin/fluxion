# FLUID-02 — investigation history

Narrative history for **FLUID-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `FLUID-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### FLUID-02: VAV Box Gradient Descent Test Input Size Mismatch

- **Description:** The `test_vav_box_gradient_descent_convergence` test passes `damper = vec![0.5]` (1 element) to `optimize_with_gradient_descent`, but `VavBox::evaluate` expects 3 inputs `[damper, static_pressure, t_inlet]`. The optimizer calls `jacobian_input` which accesses `input[STATIC_PRESSURE_IDX]` (index 1), causing an "index out of bounds" panic. This is a test design bug: the test intended to optimize only the damper but the API requires all inputs.
- **Affected Tests:** `fluxion-fluid` — `test_vav_box_gradient_descent_convergence`
- **Affected Metrics:** N/A (unit test only)
- **Severity:** Low
- **GitHub Issue:** #2330
- **Status:** 🔄 **Known Limitation** — Test would require redesign to either pass all 3 inputs or support partial-input optimization.
