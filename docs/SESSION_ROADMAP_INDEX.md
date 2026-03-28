# ASHRAE 140 Complete Validation - Session Roadmap

**Created**: 2026-03-27
**Status**: 📋 PLANNED
**Approach**: Enable advanced thermal models (CTF, FD) with sub-hourly timesteps
**Target**: ≥90% overall pass rate

---

## Session Overview

### Completed Sessions

| Session | Topic | Status | Key Outcome |
|---------|-------|--------|-------------|
| 47 | Peak Load Investigation | ✅ Complete | Identified 5R1C limitations, recommended advanced models |

### Planned Sessions

| Session | Topic | Duration | Priority | Dependencies |
|---------|-------|----------|----------|--------------|
| **48** | CTF Solver Audit | 1 week | 🔴 Critical | None (first) |
| **49** | CTF for 900-Series | 1 week | 🔴 Critical | Session 48 |
| **50** | FD Solver Audit | 1 week | 🔴 Critical | Session 49 |
| **51** | FD for 600-Series | 1 week | 🔴 Critical | Session 50 |
| **52** | Sub-Hourly Timesteps | 1 week | 🟡 High | Sessions 49,51 |
| **53** | Multi-Method Solver | 1 week | 🟡 High | Sessions 49,51,52 |
| **54** | Free-Floating Validation | 1 week | 🟢 Medium | Session 53 |
| **55** | Special Cases | 1 week | 🟢 Medium | Session 54 |
| **56** | Final Validation | 2 weeks | 🔴 Critical | All previous |

**Total Timeline**: 8-12 weeks (aggressive) to 12-18 weeks (conservative)

---

## Detailed Session Flow

```
Session 47: Peak Load Investigation (COMPLETE)
    ↓
    ├─→ Identified 5R1C limitations
    └─→ Recommended CTF/FD approach
         ↓
Session 48: CTF Solver Audit (1 week)
    ↓
    ├─→ Audit CTF implementation
    ├─→ Enable for Case 900
    └─→ Proof of concept
         ↓
Session 49: CTF for All 900-Series (1 week)
    ↓
    ├─→ Enable CTF for Cases 900-950
    ├─→ Validate all high-mass cases
    └─→ Target: ≥67% pass rate
         ↓
Session 50: FD Solver Audit (1 week)
    ↓
    ├─→ Audit FD implementation
    ├─→ Enable for Case 600
    └─→ Proof of concept
         ↓
Session 51: FD for All 600-Series (1 week)
    ↓
    ├─→ Enable FD for Cases 600-650
    ├─→ Validate all low-mass cases
    └─→ Target: ≥50% pass rate
         ↓
Session 52: Sub-Hourly Timesteps (1 week)
    ↓
    ├─→ Implement 15-minute timesteps
    ├─→ Fix peak load discrepancies
    └─→ Target: Peak loads within 15%
         ↓
Session 53: Multi-Method Solver Manager (1 week)
    ↓
    ├─→ Implement auto-selection
    ├─→ 5R1C/CTF/FD based on building
    └─→ Target: ≥85% overall pass rate
         ↓
Session 54: Free-Floating Validation (1 week)
    ↓
    ├─→ Fix temperature extremes
    ├─→ Solar gain redistribution
    └─→ Target: Temps within 5°C
         ↓
Session 55: Special Cases (1 week)
    ↓
    ├─→ Fix Case 960 (sunspace)
    ├─→ Fix Case 195 (steady-state)
    └─→ Target: ≥50% special cases
         ↓
Session 56: Final Validation (2 weeks)
    ↓
    ├─→ Comprehensive validation
    ├─→ Performance optimization
    ├─→ Complete documentation
    └─→ Target: ≥90% overall pass rate
```

---

## Session Files Created

### Session Prompt Files
- `session_48_prompt.md` - CTF Solver Audit and Initial Enablement
- `session_49_prompt.md` - CTF Integration for All 900-Series Cases
- `session_50_prompt.md` - FD Solver Audit and Initial Enablement
- `session_51_prompt.md` - FD Integration for All 600-Series Cases
- `session_52_prompt.md` - Sub-Hourly Timesteps for Peak Loads
- `session_53_prompt.md` - Multi-Method Solver Manager
- `session_54_prompt.md` - Free-Floating Temperature Validation
- `session_55_prompt.md` - Special Cases Validation
- `session_56_prompt.md` - Final Validation and Documentation

### Documentation Files
- `docs/ASHRAE140_ROADMAP.md` - Comprehensive 47-page roadmap
- `docs/ASHRAE140_QUICKSTART.md` - Quick start task list
- `docs/SESSION_ROADMAP_INDEX.md` - This file

### Summary Files (To Be Created)
- `SESSION_48_SUMMARY.md` through `SESSION_56_SUMMARY.md`

---

## Progress Tracking

### Current Status
- **Overall Pass Rate**: 1.6% (1/64 metrics)
- **5R1C Model**: Used for all cases
- **Advanced Models**: Implemented but not enabled

### Target Status
- **Overall Pass Rate**: ≥90% (58/64 metrics)
- **600-Series**: ≥67% pass rate (16/24)
- **900-Series**: ≥90% pass rate (22/24)
- **Free-Floating**: ≥50% pass rate (4/8)
- **Special Cases**: ≥50% pass rate (4/8)

### Progression by Session

| After Session | 600-Series | 900-Series | Peak Loads | Overall |
|---------------|-------------|-------------|------------|---------|
| **Current** | 0% | ~50% | 8-33% below | 1.6% |
| **Session 48** | 0% | ~60% | ~25% below | ~5% |
| **Session 49** | 0% | **≥67%** | ~20% below | ~15% |
| **Session 50** | ~30% | **≥67%** | ~20% below | ~25% |
| **Session 51** | **≥50%** | **≥67%** | ~20% below | ~35% |
| **Session 52** | **≥50%** | **≥67%** | **≤15%** | ~45% |
| **Session 53** | **≥50%** | **≥67%** | **≤15%** | **≥85%** |
| **Session 54** | **≥50%** | **≥67%** | **≤15%** | **≥85%** |
| **Session 55** | **≥50%** | **≥67%** | **≤15%** | **≥85%** |
| **Session 56** | **≥67%** | **≥90%** | **≤15%** | **≥90%** |

---

## Key Milestones

### Milestone 1: CTF Working (After Session 49)
- ✅ CTF enabled for all 900-series
- ✅ 900-series pass rate ≥67%
- ✅ Proof that advanced models work

### Milestone 2: FD Working (After Session 51)
- ✅ FD enabled for all 600-series
- ✅ 600-series pass rate ≥50%
- ✅ Low-mass buildings improved

### Milestone 3: Sub-Hourly Working (After Session 52)
- ✅ 15-minute timesteps implemented
- ✅ Peak loads within 15% of reference
- ✅ Annual energies unchanged

### Milestone 4: Auto-Selection Working (After Session 53)
- ✅ SolverManager auto-selects optimal solver
- ✅ Overall pass rate ≥85%
- ✅ No manual solver selection needed

### Milestone 5: Complete Validation (After Session 56)
- ✅ Overall pass rate ≥90%
- ✅ All categories meeting targets
- ✅ Documentation complete
- ✅ Production ready

---

## Risk Assessment

### High-Risk Sessions
- **Session 48**: CTF may have implementation bugs
- **Session 50**: FD may be unstable or slow
- **Session 52**: Sub-hourly may break energy conservation

### Medium-Risk Sessions
- **Session 53**: Solver switching may cause issues
- **Session 54**: Free-floating may not be fixable
- **Session 56**: May not achieve 90% target

### Mitigation Strategies
1. **Incremental Testing**: Test each component before integration
2. **Fallback Options**: Keep 5R1C as fallback
3. **Parallel Development**: Can work on issues in parallel
4. **Acceptable Limitations**: Some issues may be acceptable

---

## Quick Start

### To Begin Session 48:
```bash
# 1. Read session prompt
cat session_48_prompt.md

# 2. Audit CTF implementation
code src/physics/ctf_solver.rs
code src/physics/ctf_coefficients.rs

# 3. Check current usage
grep -r "ctf_enabled" src/

# 4. Build and test
cargo build --release
cargo test --release

# 5. Create summary template
touch SESSION_48_SUMMARY.md
```

### To Track Progress:
```bash
# Check pass rate after each session
cargo run --release --bin fluxion validate --all | grep "PASS\|FAIL"

# Compare with baseline
diff baseline_results.txt current_results.txt

# Update roadmap
# Edit docs/ASHRAE140_ROADMAP.md with progress
```

---

## Success Criteria

### Session-Level Success
Each session should:
- [ ] Achieve stated pass rate target
- [ ] Run without errors
- [ ] Maintain energy conservation
- [ ] Document findings
- [ ] Create summary

### Overall Success
- [ ] ≥90% overall pass rate
- [ ] All primary metrics passing
- [ ] Performance acceptable (<5s/case)
- [ ] Documentation complete
- [ ] Production ready

---

## Next Steps

1. **Review Session Roadmap**: Read all session prompts
2. **Plan Resources**: Allocate time for 8-12 weeks
3. **Start Session 48**: Begin CTF audit
4. **Track Progress**: Update this index after each session
5. **Celebrate Milestones**: Acknowledge progress

---

## References

- **`docs/ASHRAE140_ROADMAP.md`**: Comprehensive technical roadmap
- **`docs/ASHRAE140_QUICKSTART.md`**: Quick start task list
- **`SESSION_47_SUMMARY.md`**: Peak load investigation
- **`physics_based_refactor.md`**: Complete history
- **`ASHRAE 140 Standard`**: Validation requirements

---

**Last Updated**: 2026-03-27
**Status**: 📋 Ready to begin Session 48
**First Session**: Session 48 (CTF Solver Audit)
**Target Completion**: Session 56 (Final Validation) - 8-12 weeks from start
