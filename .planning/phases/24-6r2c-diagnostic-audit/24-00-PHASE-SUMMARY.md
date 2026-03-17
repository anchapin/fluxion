# Phase 24: 6R2C/8R3C Diagnostic Audit — Summary

**Status:** PLANNING COMPLETE
**Plans:** 10 (24-01 through 24-10)
**Waves:** 5 (Specification Audit, Component Testing, Integration Tracing, Reference Comparison, Root Cause Report)
**Estimated Effort:** Comprehensive forensic investigation

---

## Phase Goal

Diagnose why 6R2C/8R3C shows no accuracy improvement over 5R1C for high-mass buildings. Answer the critical question:

> Is there a **fixable bug** in the 6R2C implementation, or is the RC network structure itself **fundamentally incapable** of capturing high-mass thermal dynamics?

---

## Requirements Coverage

| Requirement | Plans | Status |
|-------------|-------|--------|
| **DIAG-01:** 6R2C implementation audit | 24-01, 24-02 | Covered |
| **DIAG-02:** 6R2C node placement verification | 24-04 | Covered |
| **DIAG-03:** Time constant analysis | 24-05 | Covered |
| **DIAG-04:** Heat flow path tracing | 24-06 | Covered |
| **DIAG-05:** Reference program comparison | 24-07, 24-08, 24-09 | Covered |
| **Root cause report** | 24-10 | Covered |

---

## Plan Breakdown

### Wave 1: Specification Audit (Plans 24-01, 24-02)

**Goal:** Establish ground truth and verify code conformance.

| Plan | Title | Type | Dependencies |
|------|-------|------|--------------|
| 24-01 | ISO 13790 6R2C Equation Verification | auto | None |
| 24-02 | 6R2C Code-to-Spec Conformance Audit | auto | 24-01 |

**Deliverables:**
- `docs/ISO_13790_6R2C_SPECIFICATION.md` — Complete 6R2C equations from ISO 13790
- `docs/6R2C_CODE_AUDIT_DISCREPANCIES.md` — Any spec-vs-code discrepancies

---

### Wave 2: Component Testing (Plans 24-03, 24-04, 24-05)

**Goal:** Verify every component with comprehensive unit tests.

| Plan | Title | Type | Dependencies |
|------|-------|------|--------------|
| 24-03 | Conductance Calculation Unit Tests | auto | 24-01, 24-02 |
| 24-04 | Node Placement and Boundary Condition Tests | auto | 24-01, 24-02 |
| 24-05 | Time Constant and Timestep Sensitivity Analysis | auto | 24-01, 24-02 |

**Deliverables:**
- `tests/6r2c_conductance_tests.rs` — 30-40 conductance unit tests
- `tests/6r2c_node_placement_tests.rs` — 30-40 node placement tests
- `tests/6r2c_time_constant_analysis.rs` — 15-20 time constant tests
- `docs/6R2C_TIMESTEP_RECOMMENDATIONS.md` — Timestep guidance

---

### Wave 3: Integration Tracing (Plans 24-06, 24-07)

**Goal:** Trace heat flow and compare 5R1C vs 6R2C side-by-side.

| Plan | Title | Type | Dependencies |
|------|-------|------|--------------|
| 24-06 | Heat Flow Path Tracing Instrumentation | auto | 24-03, 24-04, 24-05 |
| 24-07 | 5R1C vs 6R2C Side-by-Side Comparison | auto | 24-06 |

**Deliverables:**
- Heat flow tracing instrumentation (engine code)
- Trace data CSV files (~5-10 MB)
- `docs/6R2C_HEAT_FLOW_ANALYSIS.md` — Heat flow analysis
- `tests/5r1c_vs_6r2c_comparison.rs` — 20-30 comparison tests
- `docs/5R1C_VS_6R2C_COMPARISON.md` — Comprehensive comparison

---

### Wave 4: Reference Comparison (Plans 24-08, 24-09)

**Goal:** Compare against EnergyPlus (reference implementation).

| Plan | Title | Type | Dependencies |
|------|-------|------|--------------|
| 24-08 | EnergyPlus Installation and Test Case Setup | auto | None |
| 24-09 | EnergyPlus Internal State Comparison | auto | 24-08 |

**Deliverables:**
- EnergyPlus installed on system
- `Case600.idf`, `Case900.idf` — EnergyPlus input files
- `tools/compare_fluxion_energyplus.py` — Comparison script
- `docs/ENERGYPLUS_COMPARISON_GUIDE.md` — Setup guide
- `docs/FLUXION_VS_ENERGYPLUS_COMPARISON.md` — Comparison report

---

### Wave 5: Root Cause Report (Plan 24-10)

**Goal:** Synthesize findings into root cause report with recommendation.

| Plan | Title | Type | Dependencies |
|------|-------|------|--------------|
| 24-10 | Comprehensive Root Cause Analysis and Recommendation | auto | 24-01 through 24-09 |

**Deliverables:**
- `docs/CASE_900_ROOT_CAUSE.md` — Comprehensive root cause report (~30-50 pages)
- Root cause fault tree diagram
- Go/No-Go recommendation with justification

---

## Success Criteria

Phase 24 is complete when:

1. ✅ **ISO 13790 6R2C specification** documented and verified
2. ✅ **Code audit** complete (any discrepancies documented)
3. ✅ **Conductance calculations** verified with 30-40 unit tests
4. ✅ **Node placement** verified with 30-40 unit tests
5. ✅ **Time constants** analyzed and timestep recommendations documented
6. ✅ **Heat flow paths** traced and analyzed
7. ✅ **5R1C vs 6R2C comparison** complete (20-30 tests)
8. ✅ **EnergyPlus comparison** complete (internal states compared)
9. ✅ **Root cause report** with go/no-go recommendation

---

## Expected Findings (Hypotheses)

Based on prior research, we expect to find:

### Hypothesis 1: Timestep Too Coarse (Confidence: Medium)
- **Finding:** 1-hour timestep is too coarse for τ_m = 4.82 hours
- **Evidence:** Rule of thumb says Δt < τ/10 = 29 minutes
- **Fix:** Sub-stepping (6×10 minutes per hour)
- **Expected impact:** Reduce error 20-30%

### Hypothesis 2: RC Structural Limitation (Confidence: High)
- **Finding:** RC network cannot capture multi-layer thermal lag
- **Evidence:** EnergyPlus uses CTF, which captures layer-by-layer conduction
- **Fix:** Adopt CTF or finite difference method
- **Expected impact:** Reduce error 80-90%

### Hypothesis 3: Implementation Bug (Confidence: Low)
- **Finding:** Wrong conductance formula or node placement
- **Evidence:** Code audit or unit tests reveal discrepancy
- **Fix:** Correct the bug
- **Expected impact:** Unknown (depends on bug severity)

---

## Decision Gate

After Phase 24, we will know:

```
If implementation bug found:
  └─→ Fix bug and proceed to Phase 26 (comparative evaluation)

If timestep issue only:
  └─→ Implement sub-stepping and proceed to Phase 26

If structural limitation (RC vs CTF):
  └─→ Proceed to Phase 25 (alternative physics: CTF, finite difference, or ML)

If inconclusive:
  └─→ Extend Phase 24 with additional diagnostics
```

---

## Technical Approach

### Test Strategy

- **Unit tests:** Isolate individual calculations (conductance, node placement, time constants)
- **Integration tests:** Compare 5R1C vs 6R2C with identical inputs
- **Reference tests:** Compare Fluxion vs EnergyPlus with identical inputs
- **Property tests:** Verify invariants (energy conservation, non-negative conductances)

### Comparison Methodology

- **Apples-to-apples:** Match inputs exactly (geometry, construction, weather, schedules)
- **Internal states:** Compare temperatures and heat flows, not just annual energy
- **Time-series analysis:** Identify when and where models diverge
- **Statistical metrics:** MAE, RMSE, correlation, time lag

### Documentation

- **Specifications:** ISO 13790 equations documented from first principles
- **Test results:** All test outputs saved for future reference
- **Comparison data:** CSV files with trace data preserved
- **Analysis reports:** Comprehensive reports with plots and conclusions

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| EnergyPlus installation fails | Low | Medium | Use Docker container or manual download |
| Cannot match EnergyPlus inputs exactly | Medium | Low | Document differences, adjust interpretation |
| Inconclusive findings | Low | High | Extend diagnostics, consult literature |
| Multiple root causes (not single) | Medium | Medium | Quantify impact of each, prioritize fixes |
| Fix requires major rewrite | High | High | Document for v1.0, consider ML surrogate for v0.6 |

---

## Performance Considerations

- **Test execution:** ~200-300 new tests, expect 5-10 minutes for full suite
- **Trace data:** ~5-10 MB per case, manageable
- **EnergyPlus runtime:** ~1-2 minutes per case (fast for ideal HVAC)
- **Comparison scripts:** Python with pandas, < 1 minute per comparison

---

## Long-Term Value

Even if Phase 24 concludes that RC networks are fundamentally limited, the investment provides:

1. **Comprehensive test suite** — Tests remain valuable for regression testing
2. **Specification document** — ISO 13790 equations documented for future reference
3. **Comparison framework** — Can compare future implementations against EnergyPlus
4. **Root cause understanding** — Know exactly why RC networks fail, informs alternative approach
5. **Confidence in decision** — Data-driven go/no-go, not guesswork

---

## Next Steps

1. **Execute Phase 24** — Run all 10 plans in sequence (waves 1-5)
2. **Review findings** — Present root cause report and recommendation
3. **Make decision** — Go (fix bugs), Conditional Go (sub-stepping), or No-Go (pivot to alternative)
4. **Proceed to Phase 25 or 26** — Based on decision

---

*Phase 24 planning complete: March 17, 2026*
*Ready for execution pending user approval*
