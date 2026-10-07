# CI-04 — investigation history

Narrative history for **CI-04**. The current state of this limitation is the
`CI-04` row in `docs/KNOWN_ISSUES.md`; this file is the provenance behind it.

---

#### CI-04: Node/NAPI Bindings (ubuntu-24.04 + windows-latest) and Surrogate Drift Tolerance Gate (Issue #1784) red on develop — blocked on the LIMIT-33 physics fix (Issue #4314)

- **Discovered:** 2026-10-06 triage of `develop` @ `e93ba790`.
- **Symptom:** all three checks fail deterministically on every run since
  2026-09-29 (PR #4258 merge):
  - Node/NAPI (both OSes): `npm/test.js` — `ASHRAE 600 annual heating 7181.51 kWh
    outside ±15% published band [4314, 5836]` (identical value on both runners —
    deterministic physics, not toolchain).
  - Surrogate Drift Tolerance (#1784): fallback mode, `Failed: 1, Panics: 1,
    Peak drift 50.0000%` vs `case_900_analytical_fallback_baseline.json`.
- **Root cause:** shared — the #4241/#4258 ideal-HVAC storage residual (see
  §LIMIT-33 / Issue #4314). The npm test pins the ASHRAE 140 published band and
  the drift gate compares against a pre-#4241 recorded baseline.
- **Closes when:** the LIMIT-33 fix lands and the recorded-value baselines are
  re-generated from the corrected engine; the npm heating assertion stays pinned
  to the published band and the drift baseline is regenerated via the documented
  `cargo test --release --features ort --test surrogate_drift_fallback_regression
  -- --ignored --nocapture fallback_annual_hvac_diagnostic` command. No gate
  threshold is widened and no gate is skipped.
- **Owns:** Issue #4314 (blocks: LIMIT-33).
