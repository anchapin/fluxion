# PeakHeatingLimit-01 — investigation history

Narrative history for **PeakHeatingLimit-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `PeakHeatingLimit-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### PeakHeatingLimit-01: Case 960 Peak Heating < 2 kW (5R1C architectural)

- **Description:** Fluxion's 5R1C/9R4C Norton-equivalent `h_coeff` (≈ 76 W/K for
  Case 960 back-zone) under-predicts peak heating at the coldest hour because
  the single lumped-mass node buffers the air-side free-floating temperature.
  EnergyPlus reports ~3.9 kW peak heating at hour 8000 (T_out = -9°C) while
  Fluxion's 5R1C gives ~0.9 kW at the coldest step (T_out = -12°C, t_free ≈ 8°C).
- **Affected Cases:** 960
- **Affected Metrics:** Peak Heating (kW)
- **Severity:** Medium (accepted limitation)
- **GitHub Issue:** #1456 follow-up
- **Status:** ⚠️ Won't Fix in scope (architectural — requires 9R4C multi-surface
  time-constant integration with finer timestep)
- **Resolution Notes:** `test_peak_load_validation` allows a documented 5R1C
  under-prediction tolerance (< 85% error from the 5 kW reference midpoint).
  See `tests/ashrae_140_case_960_sunspace.rs:633-643`.
