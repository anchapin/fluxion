# Phase 8 Verification Checklist

## Entry Criteria (Phase 7 Complete)

- [x] Phase 7 plan 07-10 (MREF-03 remote reference tests) - SUMMARY exists, code changes in src/validation/commands.rs
- [x] Phase 7 plan 07-11 (Sensitivity BatchOracle) - SUMMARY exists, code changes in src/lib.rs, src/analysis/sensitivity.rs
- [ ] Code compiles without errors (currently some test compilation errors need fixing)
- [x] All Phase 7 changes committed or documented

## In-Process Verification (During Phase 8)

### Task 2: Diagnostic Test
- [ ] `tests/debug_960_summer.rs` created and compiles
- [ ] Test runs successfully and produces hourly logs
- [ ] Logs captured to file for analysis
- [ ] Key metrics extracted: zone temps, solar gains, inter-zone Q

### Task 3: Solar Gains Investigation
- [ ] Zone 1 (sunspace) solar gains > 0 during summer hours
- [ ] Window area for Zone 1 confirmed (6 m² south)
- [ ] Weather data DNI/DHI valid during simulation
- [ ] No shading blocking all solar
- [ ] `calculate_zone_solar_gain` returns correct total

### Task 4: Inter-Zone Heat Transfer Verification
- [ ] Inter-zone conductance (h_iz) value recorded
- [ ] Summer hourly logs show q_iz_total sign and magnitude
- [ ] Expected: T_sunspace > T_back => q_iz_total > 0 (heat to back)
- [ ] Actual: Verify against logs

### Task 5: HVAC Efficiency Check
- [ ] Confirm whether model reports thermal or electrical energy
- [ ] Compare with reference source (EnergyPlus electrical)
- [ ] Determine if COP correction needed
- [ ] If needed, implement COP division for cooling (and heating)

### Task 6: Reference Data Collection
- [ ] Web search completed for Case 960 reference values
- [ ] Expected cooling range confirmed (1.0-3.5 MWh)
- [ ] Typical sunspace temperature behavior documented
- [ ] Findings recorded in 08-RESEARCH.md

## Exit Criteria (Phase Complete)

- [ ] Case 960 annual cooling within 1.0-3.5 MWh (benchmark range)
- [ ] Annual heating still within 5.0-15.0 MWh (no regression)
- [ ] Peak cooling and heating within acceptable ranges
- [ ] Full ASHRAE 140 suite runs (`cargo test validate_all_cases` or `fluxion validate --all`)
- [ ] No regression in Cases 600-950 (pass rates unchanged)
- [ ] Root cause documented in `docs/CASE_960_ROOT_CAUSE.md`
- [ ] `KNOWN_ISSUES.md` updated (MULTI-01 resolved or updated)
- [ ] All changes committed with conventional commit message
- [ ] Pre-commit hooks pass (cargo fmt, clippy, audit)
- [ ] 08-SUMMARY.md created with completion details

## Test Commands

```bash
# Build and check
cargo check --all-targets

# Run Case 960 specific test
cargo test test_ashrae_140_case_960 --release -- --nocapture

# Run diagnostic
cargo test test_960_summer_debug --release -- --nocapture

# Full validation
cargo test validate_all_cases --release
# or
fluxion validate --all
```

## Quality Gates

1. **No Compilation Errors**: `cargo check --all-targets` clean
2. **Test Coverage**: Existing tests pass; new diagnostic test passes
3. **Performance**: No >5% slowdown in other cases
4. **Code Quality**: `cargo clippy` warnings addressed; `cargo fmt` applied
5. **Documentation**: Root cause and fix clearly explained

## Regression Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fix breaks other cases (600-950) | High | Run full suite after fix; isolate change to multi-zone code |
| COP correction over-corrects heating | Medium | Apply COP only to cooling; verify heating range still passes |
| Inter-zone conductance change affects only 960 | Low | Only Case 960 uses common walls; limited risk |
| Solar gain fix changes single-zone behavior | Low | Solar calculation zone-specific; should be safe |

## Sign-off

- [x] Investigator: Claude Code (Phase 8 planning complete)
- [ ] Lead Reviewer: TBD (after implementation)
- [ ] Merge Approval: TBD (after validation)
