# Fluxion Parallel Development - Quick Start

**Overview**: Work on 18 ASHRAE 140 validation issues across 4 parallel tracks to achieve 70%+ pass rate.

---

## 🚀 One-Minute Setup

```bash
# 1. Setup all worktrees
./scripts/setup_parallel_worktrees.sh

# 2. Launch 4 parallel agents (see PARALLEL_AGENTS_GUIDE.md)
#    Track 1: Solar & HVAC (#303, #299, #281, #278, #276)
#    Track 2: Physics Core (#273, #294, #280, #272)
#    Track 3: Zone Physics (#302, #295, #279, #274)
#    Track 4: Ventilation & Tools (#301, #297, #304, #275, #277)

# 3. Monitor progress
#    Use TaskOutput to check agent status

# 4. Create PRs when done
#    cd ../feature-issue-XXX
#    git push && gh pr create
#    cd ../fluxion && git worktree remove ../feature-issue-XXX
```

---

## 📊 Issue Priority Breakdown

| Priority | Count | Issues |
|----------|-------|--------|
| **CRITICAL** (≥150 pts) | 2 | #302, #273 |
| **HIGH** (120-149 pts) | 6 | #303, #301, #299, #297, #295, #294 |
| **MEDIUM** (80-119 pts) | 9 | #304, #281-#272 (investigations + enhancements) |
| **LOW** (<80 pts) | 1 | #277 (roadmap) |

---

## ⚠️ CRITICAL PATH (Do First)

These 2 issues block all validation progress:

1. **#302**: Refine Inter-Zone Longwave Radiation (Case 960)
   - Fixes multi-zone radiation coupling
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`

2. **#273**: Case 960 multi-zone investigation (20x higher cooling)
   - Diagnoses 20x discrepancy in cooling energy
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`

---

## 📁 Files Modified by Component

| Component | Files |
|-----------|-------|
| **Solar** | `src/sim/solar.rs`, `src/sim/shading.rs`, `src/sim/sky_radiation.rs` |
| **Physics Core** | `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs` |
| **Ventilation** | `src/sim/ventilation.rs`, `src/sim/occupancy.rs` |
| **HVAC** | `src/sim/hvac.rs`, `src/sim/demand_response.rs` |
| **Validation** | `src/validation/`, `src/validation/ashrae_140/` |
| **Tools** | `tools/`, `src/validation/diagnostic.rs` |

---

## 🎯 Success Criteria

### Phase 1 (Week 1-2)
- [ ] Case 960 cooling energy within 2x of reference (down from 20x)
- [ ] Inter-zone radiation coupling validated

### Phase 2 (Week 2-4)
- [ ] ASHRAE 140 pass rate ≥ 50% (18/36 metrics)
- [ ] Mean Absolute Error < 50%

### Phase 3 (Week 4-6)
- [ ] ASHRAE 140 pass rate ≥ 70% (25/36 metrics)
- [ ] Mean Absolute Error < 25%
- [ ] Hourly delta analysis tool operational

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `docs/PARALLEL_EXECUTION_PLAN.md` | Full execution plan with all details |
| `docs/PARALLEL_AGENTS_GUIDE.md` | Agent launch commands and workflow |
| `docs/PARALLEL_QUICKSTART.md` | This quick reference |
| `scripts/setup_parallel_worktrees.sh` | Automated worktree setup |

---

## 🔧 Development Commands

```bash
# Format code
cargo fmt

# Run linter
cargo clippy

# Run all tests
cargo test

# Run ASHRAE 140 validation
fluxion validate --all

# Create PR
git push origin feature/issue-XXX
gh pr create --title "Fix #XXX: Title" --body "Closes #XXX"

# Remove worktree
git worktree remove ../feature-issue-XXX
```

---

## 📈 Current Status

- **Repository**: anchapin/fluxion
- **Open Issues**: 18
- **Max Parallel Tracks**: 4
- **Target Pass Rate**: 70% (currently 27.8%)

---

## 🆘 Troubleshooting

**"Module not found" error**:
```bash
maturin develop
```

**Test failures**:
```bash
# Run single-threaded for debugging
cargo test -- --test-threads=1

# Run with output
cargo test -- --nocapture
```

**Git worktree conflicts**:
```bash
# List worktrees
git worktree list

# Remove worktree
git worktree remove ../feature-issue-XXX

# Clean up stale worktrees
git worktree prune
```

---

## 📞 Help

- **Full Plan**: See `docs/PARALLEL_EXECUTION_PLAN.md`
- **Agent Guide**: See `docs/PARALLEL_AGENTS_GUIDE.md`
- **Architecture**: See `docs/ARCHITECTURE.md`
- **Developer Guidelines**: See `CLAUDE.md`
- **Roadmap**: See GitHub Issue #277

---

**Generated**: 2026-02-19
**Total Priority Points**: 1760 pts
**Estimated Timeline**: 4-6 weeks with 4 parallel tracks
