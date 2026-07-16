# Doc Inventory

Self-healing inventory of all documentation in the Fluxion repository. Each doc carries a 7-line summary (lines 2-8) for rapid context-setting in AI sessions.

| Doc | Purpose | Status |
|-----|---------|--------|
| [ARCHITECTURE.md](../../ARCHITECTURE.md) | Physics module boundaries, I/O contracts, Mermaid diagram | ✅ Has summary |
| [RULES.md](../../RULES.md) | Coding rules, hard constraints, must-always rules | ✅ Has summary |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | Contribution guide, PR workflow, hotfix process | ✅ Has summary |
| [CODEBASE_MAP.md](../../CODEBASE_MAP.md) | Code navigation, module dependency graph, Rust/Python/JS overview | ✅ Has summary |
| [FIX.md](../../FIX.md) | Known bugs placeholder, ASHRAE 140 CI gate fixes | ✅ Has summary |
| [docs/KNOWN_ISSUES.md](../../docs/KNOWN_ISSUES.md) | Known systematic issues, ASHRAE 140 validation issues | ✅ Has summary |
| [documentation/performance_guide.md](../../documentation/performance_guide.md) | Performance validation user guide, CLI usage | ✅ Has summary |
| [documentation/performance.md](../../documentation/performance.md) | Performance benchmarks, optimization, validation targets | ✅ Has summary |
| [validation_report.md](../../validation_report.md) | ASHRAE 140 validation results, pass/fail rates | ✅ Has summary |
| [docs/worksheets/README.md](../../docs/worksheets/README.md) | Index of worksheets by issue/tag | 🆕 New |

## 7-Line Summary Convention

Every doc MUST have a 7-line summary at lines 2-8:

```
# Doc Title

<!-- Exactly 6 lines of summary context — one line per concept -->
<!-- Line 1: What this doc is about -->
<!-- Line 2: Who should read it -->
<!-- Line 3: Key concepts covered -->
<!-- Line 4: How it relates to other docs -->
<!-- Line 5: Current status / freshness -->
<!-- Line 6: Any action required -->

## Rest of document...
```

**Agent Instruction**: After ANY change to a module, update the 7-line summary of the relevant doc.

## Case 600 Series #[ignore] Test Sweep (Post-#1465)

Issue #1621: Tracking issue for un-ignore sweep of Case 600/900 series `#[ignore]` tests
left quarantined after the Case 600 strict CI conversion (#1465 closed).

### Ignored Tests and Acceptance Criteria

| Test | Reason for #[ignore] | Acceptance Criterion for Un-ignore |
|------|---------------------|----------------------------------|
| `tests/zone_balance_eplus_isolation.rs:test_case_600_annual_energy_ashrae140_tolerance` | #1333 strict gate wired; post-#1323 cooling-gap regression (H=3.17 MWh vs band [4.31–5.84], C=2.67 MWh vs band [4.28–5.78]). Root cause: steady-state 5R1C limitation. | Un-ignore when A#4 reports PASS after #1465 closes — GaugeSolver Phase 3 closure |
| `tests/zone_balance_eplus_isolation.rs:test_case_900_annual_energy_ashrae140_tolerance` | #1333 strict gate wired; post-#1323 cooling-gap regression (C=1.20 MWh vs band [2.47–3.34], peak 1.03 kW vs 2.10 kW lower bound). Root cause: CTF transient wall modeling. | Un-ignore when A#4 reports PASS after #1465 closes — GaugeSolver Phase 3 closure |
| `tests/case_900_multinode_validation.rs:test_case_900_peak_cooling_spec_band_closure` | Blocked by CTF transient wall modeling — post-#1356 peak cooling measured 1.06 kW vs [2.10–3.50] kW band. | Un-ignore when CTF transient wall modeling follow-up lands (owned by B#5 per #1328 scope) |
| `tests/known_issues_regression.rs:test_issue1457_remaining_600_series_metrics` | #1457: 14 Case 600-series metrics await GaugeSolver #1465 (discrete-node solar injection). | **Un-ignore immediately after #1465 closes** — GaugeSolver Phase 3 brings 14 metrics into band |
| `tests/ashrae_140_blind_validation.rs:test_blind_mode_case_800_annual_energy_within_band` | Pending case_800_energy_reference.csv (EnergyPlus regeneration tracked in #1331/#1168). | Un-ignore when case_800_energy_reference.csv lands from #1331 |
| `tests/ashrae_140_blind_validation.rs:test_blind_mode_case_810_annual_energy_within_band` | Pending case_810_energy_reference.csv (EnergyPlus regeneration tracked in #1331/#1168). | Un-ignore when case_810_energy_reference.csv lands from #1331 |
| `tests/ashrae_140_blind_validation.rs:test_blind_mode_case_920_annual_energy_within_band` | Case 920 reference CSV (PR #1331) now in place; engine still under-predicts annual heating (1.71 MWh vs band [3.26–4.30]). Same root cause as #1213/#1323 (high-mass peak cooling + roof-solar under-counting). | Un-ignore when #1323 closes |
| `tests/ashrae_140_blind_validation.rs:test_blind_mode_case_950_annual_energy_within_band` | Case 950 reference CSV (PR #1331) now in place; strict band check stays ignored pending #1323 close (same root cause as Case 600/900/920). | Un-ignore when #1323 closes |
| `tests/ashrae_140_blind_validation.rs:test_blind_mode_case_960_annual_energy_within_band` | Pending case_960_energy_reference.csv (EnergyPlus regeneration tracked in #1331/#1168). | Un-ignore when case_960_energy_reference.csv lands from #1331 |
| `tests/ashrae_140_case_900.rs:test_900_series_regression` | Test pollution — Case 920 shows 7.49 MWh (2.2x overprediction) when run in regression suite; individual case tests pass correctly. | Un-ignore when test pollution (shared state / execution-order) issue is diagnosed and fixed |

### Immediate Post-#1465 Un-ignore Candidates (≥3 required)

Per acceptance criterion 3, at least 3 tests are un-ignorable immediately after #1465 closes:

1. **`tests/known_issues_regression.rs:test_issue1457_remaining_600_series_metrics`** — GaugeSolver #1465 is the direct fix for all 14 quarantined metrics
2. **`tests/zone_balance_eplus_isolation.rs:test_case_600_annual_energy_ashrae140_tolerance`** — GaugeSolver Phase 3 addresses the steady-state 5R1C limitation that blocks Case 600
3. **`tests/zone_balance_eplus_isolation.rs:test_case_900_annual_energy_ashrae140_tolerance`** — GaugeSolver Phase 3 addresses the CTF transient wall modeling gap for Case 900

## Maintenance

This inventory is self-healing: run `scripts/doc_inventory_check.sh` to verify all docs have 7-line summaries and the table is accurate.
