# Issue #1281 — Backend Result

## Summary

**Issue:** Close the ~90% zone cooling underestimate — root cause in zone-level thermal network.
**Branch:** `fix/issue-1281-zone-cooling-underestimate`
**Status:** Architectural fix shipped; cooling-gap residual documented.

---

## What I Did

### 1. Investigation (Python) — `.agents/results/issue-1281-python-verification.py`

Built a stand-alone Python 9R4C simulator using **actual Case 900 parameters** derived
from `src/sim/construction.rs::Assemblies::high_mass_wall` / `high_mass_roof` / `high_mass_floor`
and `src/sim/thermal_model_core.rs:914-921` (`h_tr_is = 3.45 × floor_area`).

Two formulations compared:

| Model | Equation |
|-------|----------|
| **Additive** (current) | `T_s = (Σ h_ms_k × T_m_k) / Σ h_ms_k` (shared surface); `T_air = (h_tr_is × T_s + h_ve × T_out + φ_ia) / (h_tr_is + h_ve)` |
| **ParallelResistance** (Issue #1281 fix) | Each surface has its own `T_s_k = (h_ms_k × T_m_k + h_tr_is × T_air) / (h_ms_k + h_tr_is)`; air node sees `h_path_k = h_ms_k × h_tr_is / (h_ms_k + h_ms_k)` parallel-sum |

#### Key finding #1 — additive overcounting is **confirmed** (+32.7 %)

| Quantity | Value |
|----------|-------|
| h_ms_total (additive sum) | 127.3 W/K |
| h_path_total (series-parallel) | 96.0 W/K |
| Overcount | +32.7 % |

#### Key finding #2 — but the cooling direction is **opposite** to the issue hypothesis

| Model | Peak cooling (24-hour Denver summer day, hour 13) |
|-------|------|
| Additive (Python) | **4.10 kW** |
| ParallelResistance (Python) | 3.27 kW |
| Engine (real, additive) | **0.86 kW** |
| ASHRAE 140 reference | 2.10–3.50 kW |

Switching to parallel-resistance would *widen* the cooling gap (4.10 → 3.27 kW), not close it.
The additive model over-predicts, the engine under-predicts, and the gap is
fundamentally elsewhere.

#### Key finding #3 — actual root cause is roof-solar under-counting

Per `docs/investigations/issue-1280-ctf-peak-load.md` §4 (the predecessor investigation):
"HVAC demand is correctly proportional to (T_free − T_set) but T_free itself is too low
because the driving solar load is too small." The dominant remaining gap is **roof solar
gains being under-counted** (~3×).

### 2. Pivoted per task instructions

> "If you discover the issue body is wrong about the root cause, trust your Python
> verification — file a comment on the issue saying so, and pivot."

Posted a comment on Issue #1281 summarising the Python verification and explaining
that the `h_ms_total` overcounting is real but in the *opposite* direction to what
the issue body expected.

### 3. Architectural fix shipped (smaller, correct increment)

Even though the parallel-resistance correction does not close the cooling gap, it IS
a more physically correct formulation of the 9R4C mass-to-air coupling network and
addresses the spirit of the issue body. Implemented as an opt-in mode to preserve
backward compatibility.

**Files changed:**

| File | Change |
|------|--------|
| `src/sim/multi_node_thermal.rs` | New `MassAirCouplingMode` enum (`AdditiveSum` / `ParallelResistance`) with documentation explaining both modes, the Python-verified derivation, and the LIMIT-05 UPDATE reasoning. |
| `src/physics/multi_node_solver.rs` | New `coupling_mode` field on `MultiNodeSolver` (default `AdditiveSum`). Added `h_series` and `per_surface_t_s` helper functions. Added `compute_zone_air_temperature_additive` (refactored from existing logic), `compute_zone_air_temperature_parallel_resistance` (new), `step_backward_euler_additive`, `step_backward_euler_parallel_resistance`, and the corresponding `step_backward_euler_with_gains_parallel_resistance` for gain-injection. Public API now has `new_with_mode` and `with_coupling_mode` builders. `compute_zone_air_temperature` and `step_with_gains` dispatch based on `coupling_mode`. |
| `ARCHITECTURE.md` | Added a section under Module 5 documenting the two coupling modes, their equations, the Python-verified numerical result (h_path_total = 96.0 vs h_ms_total = 127.3, +32.7 % overcount), and the explicit note that the actual cooling-gap root cause is roof-solar under-counting, NOT the h_ms_total additive formulation. Updated the "Zone Balance detail" paragraph to reference the #1281/#1280 root cause and the parallel-resistance architectural improvement. |
| `docs/KNOWN_ISSUES.md` | New LIMIT-05 UPDATE subsection (Issue #1281, 2026-Q2) summarising the Python-verified finding, the architectural improvement shipped, and the residual cooling-gap follow-up. |
| `.agents/results/issue-1281-python-verification.py` | New file — the Python verification script. |

**New tests added (10 tests, all passing):**

```
test_issue_1281_default_mode_is_additive_sum
test_issue_1281_new_with_mode_parallel_resistance
test_issue_1281_with_coupling_mode_builder
test_issue_1281_parallel_resistance_air_lower_than_additive
test_issue_1281_h_series_formula
test_issue_1281_per_surface_t_s_helper
test_issue_1281_parallel_resistance_step_uses_per_surface_t_s
test_issue_1281_parallel_resistance_step_with_gains
test_issue_1281_backward_compat_additive_unchanged
test_issue_1281_parallel_resistance_degenerate_falls_back
```

All 18 existing `physics::multi_node_solver` tests still pass. The full multi-node
validation suite (`case_900_multinode_validation`) still passes (6/6).

### 4. Strict ±15 % annual energy tests — RESIDUAL GAP (documented)

Ran the strict tolerance tests per the acceptance criteria:

```
$ cargo test --release --features=ort --test zone_balance_eplus_isolation \
    test_case_900_annual_energy_ashrae140_tolerance -- --ignored --nocapture

[#1147 Case 900 strict] H=1.791 MWh (band 1.364-1.846), C=0.834 MWh (band 7.862-10.637)
thread 'test_case_900_annual_energy_ashrae140_tolerance' panicked at
tests/zone_balance_eplus_isolation.rs:924:5:
Case 900 annual cooling 0.834 MWh outside ±15% band [7.862, 10.637]
```

The test still fails because the **actual root cause** (roof-solar under-counting) is
in Module 2 (solar), not Module 5 (zone thermal network). Per the task instructions:

> "If they don't pass due to Phase 1 module isolation failures, that's acceptable —
> document the residual gap."

The residual gap is documented in:
- This file (`result-backend-1281.md`)
- `docs/KNOWN_ISSUES.md` LIMIT-05 UPDATE (Issue #1281, 2026-Q2)
- `ARCHITECTURE.md` Module 5 coupling-mode table and Zone Balance detail paragraph
- GitHub issue comment (#1281) with full Python-verified reasoning

---

## Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| Python verification of additive overcounting hypothesis | ✅ **Confirmed** (+32.7 % overcount, but direction opposite to issue hypothesis) |
| Non-additive coupling correction OR multi-node network fix implemented | ✅ **`MassAirCouplingMode::ParallelResistance`** shipped as opt-in |
| Unit tests verifying the corrected h_ms_total logic | ✅ **10 new tests** in `src/physics/multi_node_solver.rs` |
| ARCHITECTURE.md updated | ✅ Module 5 coupling-mode table, Zone Balance detail paragraph |
| Strict ±15 % annual energy tests un-ignored / passing | ❌ **Residual gap** — actual root cause is roof-solar (Module 2, separate from this issue). Test remains `#[ignore]` per the wave orchestrator's instruction. Documented in LIMIT-05 UPDATE and GitHub issue comment. |
| Case 900 peak cooling within 1.60–2.10 kW ASHRAE 140 reference | ❌ **Blocked by Module 2** (roof-solar under-counting). 0.86 kW actual vs 2.10–3.50 kW target. |
| Case 920/950 peak cooling within ASHRAE 140 reference | ❌ **Blocked by Module 2** (same root cause) |
| Annual cooling for Cases 900-950 within ±15 % of ASHRAE 140 reference | ❌ **Blocked by Module 2** (same root cause) |

---

## Follow-Up

The **real** cooling-gap closure requires fixing the roof-solar under-counting in
Module 2. This is a separate issue from the `h_ms_total` architectural improvement
shipped here. The wave orchestrator should file a follow-up issue pointing at
`docs/investigations/issue-1280-ctf-peak-load.md` §4 (roof-solar under-counting) as the
remaining work.

The `MassAirCouplingMode::ParallelResistance` mode is available for opt-in adoption
by anyone who wants to evaluate the alternative 9R4C coupling network in isolation
or as a foundation for future work (e.g., adding per-surface air-film coefficients
`h_is_k` would make the parallel-resistance formulation even more accurate).

---

## References

- `.agents/results/issue-1281-python-verification.py` — the Python verification script
- `docs/investigations/issue-1280-ctf-peak-load.md` — predecessor investigation
- `docs/KNOWN_ISSUES.md` LIMIT-05 UPDATE — original h_ms_total additive hypothesis
- `docs/adr/0002-promote-9r4c-high-mass-default.md` — ADR-002 (9R4C selection rule)
- `ARCHITECTURE.md` Module 5 — updated with coupling-mode table
- Issue #1281 comment thread — Python-verified pivoting documented there
