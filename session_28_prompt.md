# Physics-Based Refactoring - Session 28 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 27 Recap
- **Approach**: Investigated root causes of empirical corrections and fixed predictive controller for setback schedules
- **Result**: Predictive controller now uses dynamic setpoints from schedule (properly enables setback behavior)
- **Key Finding**: The 5R1C model has fundamental limitations - improvements require better thermal physics
- **Pass Rate**: ~14% without empirical corrections

---

## Session 28 Task: Improve Thermal Model Physics (Multi-Node CTF)

### Objective
Implement multi-node Conduction Transfer Function (CTF) thermal modeling to improve the underlying physics and reduce reliance on empirical correction factors.

### Background
After 27 sessions, the 5R1C single-node thermal model still requires significant empirical corrections to match ASHRAE 140 reference values. The root cause is that single-node models cannot adequately capture:
1. **Thermal gradient in walls**: Different parts of wall have different temperatures
2. **Thermal mass buffering**: Heat storage/release over time varies through wall depth
3. **Dynamic heat transfer**: Response to solar gains differs from steady-state assumptions

Multi-node CTF addresses these by modeling the wall as multiple thermal nodes in series, each with its own capacitance and conductance.

### Priority 1: Enable Multi-Node CTF Solver for 900-Series Cases

**Current Status**: Multi-node CTF infrastructure exists in `src/physics/multi_node_ctf.rs`

**Investigation Steps**:
1. Check if `enable_multi_node_ctf()` is being called for 900-series cases
2. Verify the CTF solver is being used in `step_physics_6r2c()` vs `step_physics_5r1c()`
3. Check if CTF coefficients are being calculated correctly for wall constructions

**Expected Outcome**: Multi-node CTF should provide better thermal mass modeling, reducing need for empirical corrections

### Priority 2: Verify CTF Coefficient Calculation

**Check**:
- Are CTF coefficients being computed correctly for each construction type?
- Is the time step (3600s = 1 hour) correctly used?
- Are the coefficients being applied in the thermal solver?

**Files to Check**:
- `src/physics/ctf_coefficients.rs` - CTF calculation
- `src/physics/multi_node_ctf.rs` - Multi-node implementation
- `src/sim/engine.rs` - Integration with thermal model

### Priority 3: Analyze Heat Transfer Paths

**For 900-series cases (high-mass)**:
- Current: h_tr_em (exterior-to-mass coupling) is the key path
- Issue: Single-node cannot capture thermal gradient through wall thickness
- Solution: Multi-node models the wall as series of nodes

**Check thermal network**:
- Exterior surface node → Node 1 → Node 2 → ... → Node N → Interior surface node
- Each node has: C (capacitance), R (resistance to next node)
- Heat flow: q = (T_exterior - T_interior) / ΣR

### Priority 4: Test Multi-Node vs 5R1C Comparison

**Create comparison test**:
1. Run same case with both solvers
2. Compare zone temperatures and energy consumption
3. Identify where multi-node provides better results
4. Determine if multi-node reduces need for empirical corrections

### Expected Outcomes
1. **Improved thermal modeling**: Multi-node captures wall thermal gradient
2. **Reduced empirical corrections**: Physics-based model matches reference better
3. **Better dynamic response**: Solar gains buffered correctly through wall mass

### Files to Investigate
- `src/physics/ctf_coefficients.rs` - CTF calculation
- `src/physics/multi_node_ctf.rs` - Multi-node implementation  
- `src/sim/engine.rs` - Thermal model integration
- `src/sim/construction.rs` - Wall construction definitions

### Success Criteria
- [ ] Multi-node CTF enabled for 900-series cases
- [ ] CTF coefficients calculated correctly
- [ ] Comparison test shows improvement over 5R1C
- [ ] At least one empirical correction can be reduced/removed
- [ ] No regressions in existing passing cases

### Important Notes
- Use RUST_MIN_STACK=16777216 for release builds
- Run full validation after any changes
- Focus on physics improvements, not empirical tweaks
- Document any new issues found