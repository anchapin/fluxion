#!/usr/bin/env python3
"""
Test-inventory drift gate — Issue #3442.

Compares the live ``tests/test_inventory.json`` (or a freshly
regenerated one) against the frozen baseline at
``tests/reference_data/test_inventory_baseline.json`` and fails when
counts drift past documented thresholds.

Why this exists
---------------
``AGENTS.md`` test-count citations drifted on every merged PR from
2026-09-04 through 2026-09-08 (Issue #3442). Neither AGENTS.md nor
the ``docs/ci/nextest-rollout.md`` runbook had any regeneration step,
so the numbers went stale within a day of any test-adding PR. This
gate is the structural countermeasure: every PR that mutates the
test inventory gets rejected with the explicit drift message until
the baseline is updated in the same PR.

Two layers of protection
------------------------
1. **Drift threshold** — a ``DIFF_TOLERANCE_PCT`` (default 5%) or
   ``DIFF_TOLERANCE_ABS`` (default 25) on the headline counts.
   Catches accidental deletion of large test files or expansion of
   the test suite beyond expected growth.

2. **BASELINE_* ratchet** — the baseline constants
   (``BASELINE_LIB_TESTS``, ``BASELINE_WORKSPACE_TESTS``, etc.) are
   the *highest* values the gate has ever accepted. Shrinking the
   suite (test deletion, blocking-issue resolution) is the only
   allowed direction; growth without an explicit baseline bump is
   rejected. This mirrors the ``BASELINE_KNOWN_ORPHANS`` /
   ``BASELINE_WIRED_BUT_DEAD`` patterns from
   ``scripts/check_orphan_modules.py`` (Issue #3459 / #3458).

Usage
-----
  python3 scripts/check_test_inventory_drift.py                       # regenerate + check
  python3 scripts/check_test_inventory_drift.py --live-inventory path # use a pre-existing inventory
  python3 scripts/check_test_inventory_drift.py --baseline path        # custom baseline path
  python3 scripts/check_test_inventory_drift.py --update-baseline      # update baseline to live counts
  python3 scripts/check_test_inventory_drift.py --json                 # machine-readable output

Exit codes
----------
  0 — drift within thresholds AND no ratchet violation
  1 — drift exceeds thresholds OR baseline ratchet violation OR
      inventory generation failed
  2 — script error (e.g. baseline file missing)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY = REPO_ROOT / "tests" / "test_inventory.json"
DEFAULT_BASELINE = REPO_ROOT / "tests" / "reference_data" / "test_inventory_baseline.json"
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "generate_test_inventory.py"

# ---------------------------------------------------------------------------
# Drift tolerance — Issue #3442 acceptance: "5% drift OR documented exception".
# These thresholds are *per-metric*; the gate compares the live count to the
# baseline count and only fails if either the relative or absolute delta
# exceeds the threshold. The default 5%/25 mirrors the order-of-magnitude
# drift between develop and main over Issue #3442's lifetime.
# ---------------------------------------------------------------------------
DIFF_TOLERANCE_PCT = float(os.environ.get("TEST_INVENTORY_DRIFT_PCT", "5.0"))
DIFF_TOLERANCE_ABS = int(os.environ.get("TEST_INVENTORY_DRIFT_ABS", "25"))

# ---------------------------------------------------------------------------
# BASELINE_* ratchet (Issue #3442).
#
# This is the *highest* value the gate has ever accepted for each metric.
# The gate FAILS (exit 1) the moment the live count exceeds the baseline
# unless the constant is raised with a justifying comment. Lowering the
# baseline is the only authorised change; companion cleanup PRs that
# *resolve* a test (e.g. by removing a redundant coverage binary) are
# expected to LOWER the matching baseline.
#
# The drift gate runs in two modes:
#
#   * ``--no-verify`` (default in CI for speed) regenerates the
#     inventory via the AST-regex pass. Counts are deterministic for a
#     given *commit* — since Issue #4069 the scan enumerates strictly
#     git-tracked ``*.rs`` files, so working-tree dirt (untracked
#     scratch files) cannot shift the numbers. The BASELINE_* seed
#     below is the AST scan count, not the cargo runtime count.
#
#   * ``--verify`` (recommended for accuracy) cross-checks the AST
#     counts against ``cargo test --workspace --exclude fluxion-tauri
#     -- --list`` and prefers the cargo-derived numbers. The verified
#     counts at HEAD ``12856a9`` are documented as the
#     ``verified_at_HEAD_*`` constants further below — operators can
#     sanity-check the drift gate against cargo's actual output by
#     comparing the AST and verified deltas.
#
# History:
#   - 2026-09-08 (Issue #3442): seed values from the AST scan at HEAD
#     ``12856a9`` (post-#3449 / #3450 / #3285 / #3451 / #3443). The
#     ratchet constants are deliberately the AST values (4311 / 7 / 8680
#     / 108 / 301) so the ``--no-verify`` CI fast-path is consistent
#     with the committed ``tests/test_inventory.json``. The verified
#     counts at HEAD are 3894 / 4 / 7923 / 125 / 298 (lower because
#     cargo's ``--list`` honours ``#[cfg(test)]`` boundaries and
#     feature gates the AST scan does not).
#   - 2026-09-08 (Issue #3546): bumped ``BASELINE_TEST_BINARIES`` to 303
#     and ``BASELINE_WORKSPACE_IGNORED`` to 123 to accommodate the new
#     ``tests/cli_run_with_perf_600_900.rs`` integration binary and
#     the 6 new lib tests in
#     ``src/validation/ashrae140/cases/build_case_routing_tests.rs``
#     wired into the ``build_case`` router (Case 600 / 900 / 960 / 970
#     regression of #3555 partial split).
#   - 2026-09-09 (Issue #3599): bumped ``BASELINE_LIB_IGNORED`` from 7
#     to 8 to accommodate 3 newly ``#[ignore]``-quarantined 9R4C
#     legacy solver scratch-pool tests under Phase A8 / §LIMIT-21
#     (Issue #3291). The 3 tests live in
#     ``src/sim/thermal_model_physics/physics_impl/`` and panic on
#     ``cargo test --features wiring-tracing`` — see QUARANTINE.md
#     "Phase A8 / Issue #3599" subsection. ``BASELINE_WORKSPACE_IGNORED``
#     stays at 123 (workspace ignored rises from 108 → 111, well below
#     the ratchet).
#   - 2026-09-11 (Issue #3650): bumped ``BASELINE_LIB_TESTS`` from 4311
#     to 4314 and ``BASELINE_WORKSPACE_TESTS`` from 8680 to 8683 for the
#     three new CWE-209 regression tests in ``src/api/server/tests.rs``
#     (``probe_weather_ok_detail_omits_operator_supplied_path``,
#     ``probe_weather_err_detail_omits_operator_supplied_path``, and
#     ``readyz_weather_semantics_and_body_keep_path_private``). No new
#     binaries and no ignore-count changes.
#   - 2026-09-11 (PR improve/quarantine-burndown): lowered
#     ``BASELINE_WORKSPACE_IGNORED`` from 123 to 118 — the quarantine
#     burndown un-ignored
#     ``test_case_970_validator_accepts_canonical_midpoints``
#     (``tests/ashrae_140_case_970_validation.rs``) after verifying it
#     passes live; its assertions validate the validator, not the
#     engine band. Verified workspace ignored count at HEAD is 118.
#     (Merged resolution: #3650's test-count bumps land first; the
#     burndown's ignore-ratchet reduction stacks on top of them.)
#   - 2026-09-11 (PR #3704, stacked after the burndown): raised
#     ``BASELINE_WORKSPACE_IGNORED`` from 118 to 119 — the wasm
#     FFI smoke-test quarantine (``fluxion-wasm/tests/
#     wasm_integration_tests.rs``, Issue #3703) added one workspace
#     ``#[ignore]`` on top of the burndown's reduction. Cargo-verified
#     workspace ignored count at HEAD is 119.
# ---------------------------------------------------------------------------
#   - 2026-09-11 (Issue #3629, stacked after #3650/#3693-burndown): bumped
#     ``BASELINE_LIB_IGNORED`` from 8 to 9 for the newly quarantined
#     ``test_thermal_mass_temperature_damping`` placeholder in
#     ``src/validation/thermal_mass.rs`` (a ``src/`` unit test outside
#     the auditor's ``tests/**`` scan).
#   - 2026-09-12 (Issue #3711): raised ``BASELINE_WORKSPACE_IGNORED``
#     from 121 to 133 — the cargo-verified count (``cargo test
#     --workspace --exclude fluxion-tauri -- --list --ignored``), which
#     the gate's canonical ``--verify`` mode prefers, has been 133 since
#     Wave 8 (commit ``6bb98e1``, Issue #3590, 2026-09-09), when the
#     ratchet was correctly set to 133. The subsequent 133 -> 123
#     (#3595) -> 119 (#3693 burndown) -> 121 (#3689) recalibrations were
#     matched against AST-mode captures (their committed inventories
#     carry no ``verify`` block), and AST mode under-counts cargo by
#     omitting the 38 `` ```ignore `` doc-comment tests and counting
#     cfg/feature-gated ``#[ignore]``-shaped attributes a
#     default-features build does not compile or ignore (e.g. the root
#     lib's gauge-solver ``cfg_attr`` LIMIT-22 cohort: AST 9 vs cargo 6;
#     net AST over-count 24: 119 + 38 - 24 = 133). Per-crate attribution
#     of the verified 133 (lib + integration + ignored doctests):
#     ``fluxion`` (root) 113 = 6 + 76 + 31; ``fluxion-fluid`` 16 =
#     11 + 0 + 5; ``fluxion-core`` 2 = 0 + 0 + 2; ``fluxion-twin`` 2 =
#     1 + 0 + 1; all other crates 0. No net quarantine growth since
#     Wave 8: #3689's thermal_mass placeholder (+1) was offset by
#     #3707/#3714 un-ignoring the occupancy statistical and wasm
#     ``wasm_run_full_annual_*`` smoke tests. Issue #3711 is the
#     required documentation for this bump.
#   - 2026-09-13 (Issue #3754): bumped ``BASELINE_LIB_TESTS`` from 4314
#     to 4319 and ``BASELINE_WORKSPACE_TESTS`` from 8683 to 8688 for the
#     five new closed-channel tests in ``src/sim/orchestrator.rs``
#     (``batched_worker_batch_captured_when_receiver_dropped_mid_run``,
#     ``batched_worker_batch_delivered_when_receiver_alive``,
#     ``batched_finalize_fails_loudly_preserving_received_results``,
#     ``batched_finalize_ok_when_nothing_dropped``, and
#     ``batched_happy_path_returns_ok_with_full_population``). No new
#     binaries and no ignore-count changes.
#   - 2026-09-14 (Issue #3749): bumped ``BASELINE_LIB_TESTS`` from 4319
#     to 4325 and ``BASELINE_WORKSPACE_TESTS`` from 8688 to 8694 for the
#     six new binding-level effective-solver truth tests (two in
#     ``src/python/bindings.rs`` and one in
#     ``src/python/model_bindings/model.rs`` for the PyO3 accessor, three
#     in ``src/napi/state_extractor.rs`` for the ``StateMatrices``
#     ``effective_solver`` field). Also un-blocked the latent compile
#     breakage of the pre-existing ``python-bindings`` feature-gated test
#     modules (stale one-arg ``from_case_spec`` calls and moved
#     ``ThermalModel`` mass fields) so these tests actually run again.
#     No new binaries and no ignore-count changes.
#   - 2026-09-15 (Issue #3741): bumped ``BASELINE_LIB_TESTS`` from 4325
#     to 4326 and ``BASELINE_WORKSPACE_TESTS`` from 8694 to 8695 for the
#     new fail-closed egress allow-list release-decision table test in
#     ``src/api/email_notification.rs``
#     (``endpoint_allowlist_release_decision_table``). No new binaries
#     and no ignore-count changes.
#   - 2026-09-18 (Issue #3728): bumped ``BASELINE_LIB_TESTS`` from 4326
#     to 4346 and ``BASELINE_WORKSPACE_TESTS`` from 8695 to 8702 for the
#     20 new inline unit tests in ``src/api/security/path_validation.rs``
#     (the exporter write-path confinement validator: extension pin,
#     parent existence, symlink refusal, containment, traversal, dotdot
#     collapsing inside the allow-list, happy path, env-driven entry
#     point, opt-out failure modes) and the 7 new integration tests in
#     ``tests/all_tests/exporter_path_confinement.rs`` (the headline
#     `/etc/passwd` rejection, traversal outside, per-exporter
#     extension pin, symlinked parent, unrestricted bypass,
#     extension-pin-still-fires-under-bypass, and the default-dir
#     sanity). No new binaries and no ignore-count changes.
# 2026-09-24 (PR #3968, Issues #3963–#3966): bumped to the chain's live AST
# counts (lib 4346 -> 4240 is the corrected AST basis after the topology
# module move; workspace 8702 -> 8720 = +18 tests from the topology export /
# lint suites; ignored 9 -> 8 and 147 -> 131 absorb the AST-vs-cargo
# calibration delta; binaries 309 -> 49 reflects the post-#3764 consolidated
# runner basis). Cargo-verified basis is tracked in
# tests/reference_data/test_inventory_baseline.json::ratchet.
# (BASELINE_LIB_IGNORED stays 9: the harness's mode-aware selection
# fixture pins a lib_ignored of 9, and the live AST count 8 < 9 keeps
# the ratchet green either way.)
# 2026-09-25 (Issue #3978 / ADR-0017, PR #4017): bumped
# ``BASELINE_LIB_TESTS`` from 4240 to 4241 (+1: the Phase-A8 single
# default-selector unit test became the cfg-dependent ADR-0017 pair) and
# ``BASELINE_WORKSPACE_TESTS`` from 8720 to 8725 (+5: three new
# ``tests/gauge_dispatcher_cases.rs`` integration tests — default-build
# explicit-legacy default, loud Gauge panic, HighMass promotion truth —
# plus the two AST-counted cfg variants of the selector unit test; the
# net cargo-verified delta is workspace_tests 7927 -> 8018 per
# ``tests/test_inventory.json``).
# CUMULATIVE bumps from PR-A (#3986-A) and PR-B (#3986-B).
# - 2026-09-27 (Issue #3986-A, PR #4119): 4268 -> 4273 — PR-A adds 5 lib tests
#   (4 inline tests in tests.rs guard module, plus 1 extra in the existing
#   validator test module net); the AST workspace count grew by 7 (the
#   consolidated `ashrae_140_validator_selector_parity` runner module —
#   7 behavior tests); these ratchet bumps track both metrics.
# - 2026-09-27 (Issue #3986-B, PR #4120): 4273 -> 4274 — PR-B adds 1 inline
#   test in `pcm_test_box.rs::tests` (the `default_layer_thickness_matches_documented_value`
#   guard test) plus the AST-scan delta for the new `tests/all_tests/teacher_validation_pcm_box.rs`
#   module's integration tests. Stacked on PR-A's 4273 baseline.
# 2026-11-09 (Issue #4705, PR #4710 rebase wave 4): BASELINE_LIB_TESTS 4411 -> 4416
# 2026-11-09 (Issue #3982, PR #4651 rebase wave 5): BASELINE_LIB_TESTS 4416 -> 4424
# 2026-11-09 (Issue #4706, PR #4714 rebase wave 6): BASELINE_LIB_TESTS 4424 -> 4431
# 2026-11-09 (Issue #3983, PR #4652 rebase wave 7): BASELINE_LIB_TESTS 4431 -> 4441
BASELINE_LIB_TESTS = 4448  # 2026-11-12 (Issue #4775, PR #4785 merge-update): 4446 -> 4448 — merged-tree recount (branch acceptance tests in src/sim/boundary.rs on current develop). Prior: 2026-11-12 (Issue #4270, PR #4647 merge-update): 4441 -> 4446 — merged-tree recount after develop #4767 landed; regen-verified. Prior: 2026-11-09 (Issue #4693): 4401 -> 4411 — +10 error-controlled adaptive-timestep unit tests in src/sim/adaptive_timestep.rs (net of AST count). Prior: 2026-11-09 (Issue #4692): 4394 -> 4401 — +7 warm-start predictor unit tests in src/physics/bdf_engine.rs. Prior: 2026-11-09 (Issue #4707): 4390 -> 4394 — +4 pressure_air_loop unit tests (fan efficiency/shaft-power + fan-rise analytic pin, specified-flow design-flow reproduction, inverse zone-node round-trip, default-behaviour preservation). Prior: 2026-11-08 (Issue #3991, PR #4611 rebase wave 12): 4384 -> 4390 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #3984, PR #4597 rebase wave 11): 4371 -> 4384 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4272, PR #4660 rebase wave 10): 4361 -> 4371 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4631, PR #4638 rebase wave 9): 4351 -> 4361 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4633, PR #4643 rebase wave 8): 4349 -> 4351 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #3985, PR #4628 rebase wave 7): 4348 -> 4349 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #3997, PR #4589 rebase wave 6): 4345 -> 4348 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4634, PR #4641 rebase wave 5): 4331 -> 4345 — branch tests recounted on current develop. Prior: 2026-10-28 (Issue #3985 validation slice): 4330 -> 4331 — +1 hybrid ventilation invalid-ACH guard test in src/sim/thermal_model/tests.rs (hybrid_ventilation_invalid_ach_is_ignored_not_poisoning_h_ve). Prior: 2026-10-26 (Issue #4632): 4327 -> 4330 — +3 net server tests in src/api/server/tests.rs (named-HVAC-template guard). Prior: 2026-10-26 (Issue #4635): 4320 -> 4327 — +7 validate-case range-expansion unit tests in src/cli/mod.rs (parse/expand/unsupported/FF-exclusion/registry-sync/malformed). Prior: 2026-10-26: 4317 -> 4320 — +6 unit tests in src/io/idf/convert.rs (Issue #3991 slice: basic IDF HVAC import — ZoneHVAC:IdealLoadsAirSystem detection via EquipmentConnections -> EquipmentList chain, direct zone-named IdealLoads object, no-HVAC zone, second zone unmarked, schema-level check, real reference-model check). Prior: 2026-10-25 (Issue #4104): 4307 -> 4317 — +10 unit tests in src/api/report.rs (standard results report: sections, annual/monthly consistency, per-zone unmet order, byte-determinism, sorted-keys/fixed-float format, checksum sensitivity, missing-series fallback, non-finite handling, format_fixed pins). Prior: 2026-10-25 (loop round 98, Issue #4598): 4306 -> 4307 — +1 fixed-split evaporative contract (fanger_iso7730_fixed_evaporative_split_cold_side_and_sweating, src/sim/thermal_model/comfort.rs; ISO 7730:2005 fixed HL1 + sweating split replaces the wettedness regulator; the fan-26 °C contract re-pins in place — no count change). Prior: 2026-10-24 (Issue #3985): 4304 -> 4306 — +2 hybrid outdoor-temperature-series tests in src/sim/thermal_model/tests.rs (supplied series drives the run loop cold-vs-warm; short series panics loudly). Prior: 2026-10-24 (loop round 95, coverage remedy): 4299 -> 4304 — +5 Fanger/adaptive comfort unit tests (src/sim/thermal_model/comfort.rs: neutral-point PMV=0/PPD=5 pin, still-air monotone band, moving-air convection correlation, dry-air evaporative-term engagement, extreme-condition clamp arms; comfort.rs was 0% covered — 0/12 branch targets). Prior: 2026-10-24 (loop round 95): 4298 -> 4299 — +1 gauge scheduled-ACH wiring pin (test_r95_scheduled_ach_scratch_reaches_gauge_inputs, step_dispatcher.rs; gauge-gated; the +1 hybrid EUI/zone-temp pin split is cfg-constant, not a new test). Prior: 2026-10-24 (Issue #3967 slice 2): 4289 -> 4298 - +9 topology CLI model-format fail-loud unit tests in src/cli/topology.rs (export/lint: .osm rejected, .xml rejected, uppercase .OSM rejected, valid CaseSpec JSON still loads, .json garbage still yields the JSON parse error). Prior: 2026-10-24 (Issue #3967 slice 1): 4280 -> 4289 - +9 topology tests (HVAC system bridging in src/validation/topology_bridge.rs + E008_HVAC_DISCONNECTED rule tests in src/sim/topology_lint.rs). Prior: 2026-10-24 (Issue #3989): 4277 -> 4280 - +3 per-surface schema tests in src/api/schema.rs (per-surface round-trip, pre-#3989 back-compat deserialisation, case600_per_surface.flux example load). Prior: 2026-10-23 (Issue #3985): 4275 -> 4277 - +2 hybrid ventilation wiring tests in src/sim/thermal_model/tests.rs (scheduled ACH plumbed into h_ve; flag-off bit-identity pin). Prior: 2026-10-23 (PR #4568, Issue #4555): 4269 -> 4275 - +6 thermostat sensor-node tests in src/sim/sensor_node.rs. Prior: 2026-10-23 (PR #4565, Issue #4268): 4262 -> 4269 - +7 Addendum a hourly time-series validator tests in addendum_a.rs. Prior: 2026-10-18 (loop round 63, #4296 part 2): 4254 -> 4262 — +8 DX performance-map unit tests (src/sim/hvac/dx_performance_map.rs: grid validation, AHRI point reconciliation, node-exact reproduction, adjacent-column linearity, SHR variation, EnergyPlus cycling formulation, edge clamping, out-of-grid finiteness). # 2026-10-16 (loop round 38): 4306 -> 4254 — -52 unit tests removed with the archived src/quantum/ tree (qubo_mapping, dwave_client, qubo_scaling). # 2026-10-15 (loop round 35, #4343): 4305 -> 4306 — +1 default-model construction-equality unit test (default_model_matches_case_600_via_default_selector, batch_oracle.rs). # 2026-10-15 (loop round 35, GATE-04 / issue #4175): 4309 -> 4305 — -4 duplicate-routing-enum tests removed with the dead `thermal_model::ThermalModelType` enum + its From<&CaseSpec> impl (test_thermal_model_type_from_case_spec_{low_mass,high_mass,case_960}, test_thermal_model_type_default). # 2026-10-14 (loop round 31): 4308 -> 4309 — +1 gauge multi-zone §C.3 metering mechanism test (gauge_multi_zone_meters_c3_closed_form_not_ledger, step_dispatcher.rs; gauge-gated). # 2026-10-13 (loop round 24, issue #3061): 4306 -> 4307 — +1 per-zone constructions validation/Case-960-split unit test (test_zone_constructions_validation_and_case960_spec_split, ashrae_140_cases.rs). # 2026-10-13 (loop round 23, issue #3061): 4303 -> 4306 — +3 implicit inter-zone coupling unit tests (solve_linear_system: known 2x2, conservation+equilibration, heterogeneous global conservation; step_9r4c.rs). # 2026-10-12 (loop round 19, issue #4384 follow-up): 4298 -> 4303 — +5 single-source derive_solar_step_date unit tests (solar_position.rs: legacy-math full-year, days-in-month walks, day-of-year round-trip, boundaries, timestep_to_date agreement). # 2026-10-12 (loop round 18, issue #4168): 4296 -> 4298 — +2 Cases 800/810 informed/blind byte-equality + transcribed-envelope unit tests (benchmark.rs). # 2026-10-12 (loop round 15, issue #4169): 4295 -> 4296 — +1 Case 195 informed/blind byte-equality unit test (benchmark.rs). # 2026-10-12 (loop round 15, issue #4167): 4294 -> 4295 — +1 report/diagnostic setpoint-parity unit test (issue_4167_report_and_diagnostic_paths_parity, ashrae_140_validator/tests.rs). # 2026-10-11 (loop round 12, PR #4371): 4293 -> 4294 — +1 gauge exterior-film liveness unit test (gauge_exterior_film_is_wind_correlation_not_zone_conductance, step_dispatcher.rs). # 2026-10-10 (loop round 11 PR-b): 4292 -> 4293 — +1 FD volumetric-source conservation unit test (r11_pr_b_volume_source_conserves_energy, fd_solver.rs). Prior (loop round 11 PR-a): 4291 -> 4292 — +1 quasi-steady window-face unit test (r11_pr_a_window_face_is_quasi_steady_pass_through, gauge_zone_solver.rs). Prior (loop round 10): 4290 -> 4291 — +1 floor ground-boundary liveness unit test (floor_ground_boundary_is_live, gauge_zone_solver.rs, PR #4363). Prior: 4290 (loop round 9 PR-b): 4289 -> 4290 — +1 gauge infiltration ACH liveness unit test (test_infiltration_ach_parameter_is_live, gauge_zone_solver.rs). Prior baseline: 4289 (2026-09-29, #4172). (Issue #4172, PR #4299 supersede): 4300 -> 4289 — hoist SimulationDiagnostics to fluxion-core (12 new tests in fluxion-core/src/diagnostics.rs, net -11 after -23 tests moved out of fluxion::validation::diagnostics that became redundant with the new core diagnostics module). Stacked on the #4292 supersede's 4300 baseline.
                        # 2026-09-29 (Issue #4241): 4296 -> 4297 — analytic residual-formulation test (test_residual_formulation_analytic); stacked on #4193's 4296 baseline.
                        # 2026-09-29 (Issue #4193): 4295 -> 4296 — multi-zone validator reference-data invariant test (test_multi_zone_validator_uses_real_reference_data_not_placeholders); +1 on the post-#4242 4295 baseline during #4247 rebase.
                        # 2026-09-27: 4274 -> 4277 — post-rebase AST delta for PR-A + PR-B's combined inline tests (the 7 selector-parity tests are counted as workspace-integration rather than lib, so the lib bump comes from the 4 inline tests in `phase_change_material.rs::tests` + `pcm_test_box.rs::tests` plus the AST scan delta for PR-A's `tests.rs` guards).
                        # (+2 hvac setpoint CLI tests in src/cli/hvac_commands.rs:
                        # ZoneControl delegation + CLI-to-system propagation);
                        # stacked on the merged 4263 fluxion-#4086 bump.
                        # Issue #4069: the AST scan now enumerates strictly
                        # git-tracked .rs files, so a dirty developer tree
                        # scans identically to a fresh CI checkout — the old
                        # "dirty tree scans up to 4 lower" caveat is retired.
BASELINE_LIB_IGNORED = 9
# CUMULATIVE bumps from PR-A (#3986-A) and PR-B (#3986-B).
# - 2026-09-27 (Issue #3986-A, PR #4119): 8804 -> 8821 — PR-A adds 7 tests via
#   the new `ashrae_140_validator_selector_parity` consolidated runner module
#   (selector_round_trip, new_delegates_to_new_with_selector_default,
#   case_600_default_vs_new_with_default_parity, case_900_explicit_legacy_selector_runs,
#   multi_zone_selector_round_trip, multi_zone_new_delegates_to_new_with_selector_default,
#   multi_zone_explicit_legacy_selector_accepted) plus 10 lib tests across the
#   existing validator+multi-zone tests.rs modules.
# - 2026-09-27 (Issue #3986-B, PR #4120): 8821 -> 8829 — PR-B adds 8 tests via
#   the new `teacher_validation_pcm_box` consolidated-runner module
#   (rt27_constructs_with_nominal_properties, enthalpy_linear_below_solidus,
#   enthalpy_linear_above_liquidus, apparent_cp_sensible_outside_melting_band,
#   apparent_cp_latent_band_height, test_box_constructs_with_default_pcm,
#   test_box_solid_fraction_returns_none_without_reference_data,
#   test_box_apparent_cp_at_wall_delegates_to_material). Stacked on PR-A's
#   8821 baseline.

# 2026-11-09 (Issue #4691, PR #4721 rebase wave 1): BASELINE_WORKSPACE_TESTS 9136 -> 9137
# 2026-11-09 (Issue #4705, PR #4710 rebase wave 4): BASELINE_WORKSPACE_TESTS 9147 -> 9152
# 2026-11-09 (Issue #3982, PR #4651 rebase wave 5): BASELINE_WORKSPACE_TESTS 9152 -> 9160
# 2026-11-09 (Issue #4706, PR #4714 rebase wave 6): BASELINE_WORKSPACE_TESTS 9160 -> 9167
# 2026-11-09 (Issue #3983, PR #4652 rebase wave 7): BASELINE_WORKSPACE_TESTS 9167 -> 9177
# 2026-11-09 (Train PR #4672 union ratchet, post-weather-unblock merge wave): BASELINE_WORKSPACE_TESTS 9177 -> 9185
# 2026-11-09 (Train PR #4673 union ratchet, post-#4672 wave): BASELINE_WORKSPACE_TESTS 9185 -> 9191
BASELINE_WORKSPACE_TESTS = 9313  # 2026-11-12 (Issue #4774, PR #4787 merge-update): 9309 -> 9313 — merged-tree recount (branch +4 periodic verification tests in tests/ashrae_140/ground_periodic.rs on current develop). Prior: 2026-11-12 (loop round 135): 9305 -> 9309 - +4 Case-900FF spec-path engine-truth contracts (tests/step_9r4c_free_float_contract.rs extended in place - same standalone binary, test_binaries unchanged: the r133 remainder's last item, the production from-spec construction route pinned bit-exactly; teeth x3 exact). Prior: 2026-11-12 (Issue #4775, PR #4785 merge-update): 9303 -> 9305 — merged-tree recount on current develop. Prior: 2026-11-12 (loop round 134): 9294 -> 9303 — +9 AdditiveSum coupling-mode contracts (tests/step_9r4c_free_float_contract.rs extended in place — same standalone binary, test_binaries unchanged: the r133 remainder item; the mode-dispatched additive arms pinned bit-exactly at the flipped r131/r132 fixtures — the shared-T_s mass Euler, the film-sum committed-air balance + twin denominators 141.6/153.5/610.6, the implicit-iz stash arm, the per-surface channel's first explicit pin with the write-back clobber chain, the two-step composition, and the #4495 guard's additive shape with the finite-floor finding; Python-replicated and engine-verified bit-for-bit before authoring, teeth x3 exact). Prior: 2026-11-12 (Issue #4774): 9292 -> 9294 — +2 §8 transient verification tests in tests/ashrae_140/ground_transient.rs (erf closed-form step + settle-onto-steady), counted in the existing ashrae_140 binary. Prior: 2026-11-12 (loop round 133): 9291 -> 9292 — +2 mn committed-air correction contracts (tests/step_9r4c_free_float_contract.rs extended in place — same standalone binary, test_binaries unchanged: the two r132 findings fixed, the committed-air implicit solve's diag seam + the solver-less-zone boundary compaction); the committed inventory and the reference baseline realigned to the fresh AST scan in the same entry (workspace 9292 / integration 3687 / lib_root 4446 / source files 410) — resolves the issue #4773 stale-chain symptom on the committed surfaces (the prior chain predates the #4647 merge's test additions; the #4647 recount baseline read the merged tree one test below this scan). Prior: 2026-11-12 (Issue #4270, PR #4647 merge-update): 9282 -> 9291 — merged-tree recount after develop #4767 landed; regen-verified. Prior: 2026-11-12 (loop round 132): 9274 -> 9282 — +8 multi-zone implicit-iz air-solve contracts (tests/step_9r4c_free_float_contract.rs extended in place — same standalone binary, test_binaries unchanged: the r131 remainder item; the solve_implicit_iz (num_zones > 1) 5R1C-estimate and mn committed-air implicit systems pinned bit-exactly at a two-zone asymmetric fixture, Python-replicated and verified against the engine before authoring (57/60 observables exact; the 3 boundary commits are the #4495 guard rejecting the 0/0 NaN the replication transcribes); teeth ×3 with exact failing sets). Prior: 2026-11-12 (loop round 131): 9266 -> 9274 — +8 9R4C free-float committed-temperature chain contracts (tests/step_9r4c_free_float_contract.rs, new standalone test binary — the blitz's remaining mass-node component: the step_physics_9r4c orchestration chain at the ParallelResistance production mode, Python-replicated bit-exactly). Prior: 2026-11-11 (loop round 130): 9258 -> 9266 — +8 longwave pool-exchange contracts (tests/longwave_pool_exchange_contract.rs, new standalone test binary, plan §7 item (iv) — the InteriorSurfaceNetwork view-factor/pairwise-Stefan–Boltzmann closed forms, reciprocity, step + clamp routes; the asymmetric-area floor↔ceiling conservation break pinned as-is with its exact closed form). Prior: 2026-11-11 (loop round 129): 9251 -> 9258 — +7 window solar split contracts (tests/window_solar_split_contract.rs, new standalone test binary, plan §7 item (ii) remainder — the per-orientation transmittance/absorption split upstream of the class fractions). Prior: 2026-11-10 (loop round 126): 9240 -> 9251 — +11 per-surface sol-air composition contracts (tests/sol_air_composition_contract.rs, new standalone test binary, plan §7 item (i) remainder: the Issue #4166 exterior-boundary closed forms pinned at their public seam; teeth ×3 with predicted failing sets). Prior: 2026-11-10 (loop round 125): 9237 -> 9240 — +3 surface-pool class-drive contract pins (tests/surface_pool_class_drive_contract.rs, new standalone test binary, plan §7 rank-1 remainder: the Table 5-138 fraction constants + the r94 dual-component partition pinned bit-exactly at the class-drive seam; pin teeth proven both directions). Prior: 2026-11-10 (loop round 124 landing of PR #4686, onto the #4684 wave): 9230 -> 9237 — +7 sky-diffuse view-factor re-record/contract pins (loop r120 SOLAR-03).  # 2026-11-10 (loop round 124 landing of PR #4684, onto the #4683 wave): 9225 -> 9230 — +5 shading-overhang contract/re-record pins (loop r119).  # 2026-11-10 (loop round 124 landing of PR #4683, onto the #4682 wave): 9220 -> 9225 — +5 HVAC-masked probe fixture pins (loop r118).  # 2026-11-10 (loop round 124 landing of PR #4682, onto the #4681 wave): 9217 -> 9220 — +3 R90 lag-channel closure contract pins (loop r117).  # 2026-11-10 (loop round 124 landing of PR #4681, onto the #4709 recount wave): 9213 -> 9217 — +4 archaeology-route contract pins (loop r116).  # 2026-11-10 (PR #4709 final wave, post weather-train develop): 9209 -> 9213 — +4 export_schema integration tests (Issue #4702 glazing + IdealLoads IDF export). Prior: 2026-11-10 (loop round 124 landing of PR #4678, onto the #4676 wave): 9208 -> 9209 — +1 sustained-probe self-response continuation contract (the air record's ninth np slot, loop r115 plan section 5 increment F).  # 2026-11-10 (PR #4676 landing, weather train): 9203 -> 9208 — +5 inter-zone cross-response contract pins (tests/multizone_cross_response_contract.rs, new standalone test binary, loop r113 plan §5 increment E). Prior: 2026-11-10 (PR #4675 landing, weather train): 9196 -> 9203 — +7 multi-zone vent routing contract pins (tests/multizone_vent_routing_contract.rs, new standalone test binary, loop r112 plan §7 slice). Prior: 2026-11-10 (loop round 124 landing of PR #4674, onto the #4673 wave): 9191 -> 9196 — +5 solar-split archaeology contract pins (tests/sol2st_split_archaeology.rs, new standalone test binary, loop r111 plan §7 slice; the solar-partition queue closed there).  # 2026-11-09 (Issue #4693): 9137 -> 9147 — same +10 adaptive-timestep unit tests. Prior: 2026-11-09 (Issue #4692): 9129 -> 9136 — same +7 warm-start predictor unit tests. Prior: 2026-11-09 (Issue #4707): 9125 -> 9129 — same +4 pressure_air_loop unit tests. Prior: 2026-11-09 (Issue #4666, PR #4704 rebase wave 13): 9117 -> 9125 — branch tests recounted on current develop after #4701 merged. Prior: 2026-11-08 (Issue #4696, PR #4701 rebase wave 13): 9114 -> 9117 — branch tests recounted on current develop after #4611 merged. Prior: 2026-11-08 (Issue #3991, PR #4611 rebase wave 12): 9105 -> 9114 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #3984, PR #4597 rebase wave 11): 9092 -> 9105 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4272, PR #4660 rebase wave 10): 9082 -> 9092 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4631, PR #4638 rebase wave 9): 9072 -> 9082 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4633, PR #4643 rebase wave 8): 9070 -> 9072 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #3985, PR #4628 rebase wave 7): 9069 -> 9070 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #3997, PR #4589 rebase wave 6): 9066 -> 9069 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4634, PR #4641 rebase wave 5): 9052 -> 9066 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4271, PR #4659 rebase wave 2): 9047 -> 9052 — branch tests recounted on current develop. Prior: 2026-11-08 (Issue #4636, PR #4637 rebase): 9046 -> 9047 — +1 WD200 smoke integration test in tests/ashrae_140_section6_weather_driver.rs (lib count unchanged). Prior: 2026-10-31 (PR #4658, Issue #4274): 9037 -> 9046 — +9 §11 engine-run integration tests (tests/validation/hvac_equipment_rated_performance/section11_engine_run.rs). Prior: 2026-10-29 (loop round 108, §LIMIT-22 increment D): 9036 -> 9037 — +1 legacy gauge-proxy identity air-record consumption contract (tests/all_tests/invariant_checker_test.rs::test_legacy_identity_air_route_consumes_air_record: surgical legacy-entry state — zone-0 R77 mass record AND per-class boundary/t_s states cleared, air record live — probes +1 kW through the legacy lumped identity's only load route h_tr_3·t_i with the committed air blend consuming the self-contained seven-entry air record via the shared composed_air_route_shift helper; closed form âh_tr_3·r_air_N·Δ at the 1e-9 bar, affine pin, exactly-0-W fall-through discriminator; counts unchanged elsewhere). Prior: 2026-10-28 (loop round 107, §LIMIT-22 increment C): 9035 -> 9036 — +1 per-class t_i route air-record consumption contract (tests/all_tests/invariant_checker_test.rs::test_perclass_ti_route_consumes_air_record: surgical round-104 entry state — zone-0 R77 mass record cleared, air record live — probes +1 kW through the per-class t_s-first mirror's surface identity with the t_i route consuming the self-contained seven-entry air record [D,M,W,den,h3od,a,bl]; closed form (rad_frac + h_is_zone·r_air_N)·Δ at the 1e-9 bar, affine pin, committed-air fall-through discriminator; counts unchanged elsewhere). Prior: 2026-10-28 (Issue #3985 validation slice): 9034 -> 9035 — same +1 hybrid ventilation invalid-ACH guard test in src/sim/thermal_model/tests.rs. Prior: 2026-10-28 (loop round 105, §LIMIT-19/#1344 increment A): 9033 -> 9034 — +1 air-node record composition contract (tests/all_tests/invariant_checker_test.rs::test_r77_air_record_composes_engine_counterfactual, free-float two-model counterfactual re-running the engine at a +1 kW zone-0 load delta and compositing MassState::r77_air_substep_coeffs [D,M,W,den,h3od] to the committed air+mass response exactly at FLX_R77_COUPLED_N ∈ {2,12,96}; the record is write-only, zero consumers this PR, zero-delta replay untouched; counts unchanged elsewhere). Prior: 2026-10-27 (loop round 104, §LIMIT-22 per-class rebuild): 9032 -> 9033 — +1 per-class t_s-first mirror gain-probe leverage contract (tests/all_tests/invariant_checker_test.rs::test_perclass_ts_mirror_gain_probe_leverages_5r1c_state, explicit-legacy-selector 5R1C state with the R77 record surgically cleared so the checker falls through to the round-73 per-class arm; engine records phi_st_zone_load_coef = rad_frac on the boundary-record branch, checker rebuilds phi_st_c at the probe load; counts unchanged elsewhere). Prior: 2026-10-26 (loop round 103, §LIMIT-22 rebuild): 9031 -> 9032 — +1 R77 composite gain-probe leverage contract (tests/all_tests/invariant_checker_test.rs::test_r77_composite_gain_probe_leverages_5r1c_state, explicit-legacy-selector 5R1C state; the same commit lifts the gauge-build cfg_attr quarantine on test_different_zones_respond_differently_to_targeted_gain — statically-counted ignored total unchanged). Prior: 2026-10-26 (Issue #4632): 9028 -> 9031 — same +3 server tests. Prior: 2026-10-26 (Issue #4635): 9021 -> 9028 — same +7 src/cli/mod.rs unit tests. Prior: 2026-10-26: 9019 -> 9021 — same +6 src/io/idf/convert.rs unit tests (Issue #3991 slice). Prior: 2026-10-25 (Issue #4104): 9009 -> 9019 — same +10 src/api/report.rs unit tests. Prior: 2026-10-25 (Issue #4296, PR #4601, shepherd CI fix): 9002 -> 9009 — +7 §9 CE engine-run tests in tests/validation/hvac_equipment_rated_performance/section9_engine_run.rs (new test source file 387 -> 388, integration tests 3542 -> 3549; lib counts unchanged). Prior: 2026-10-25 (loop round 98, Issue #4598): 9000 -> 9002 — +1 fixed-split evaporative contract (fanger_iso7730_fixed_evaporative_split_cold_side_and_sweating, src/sim/thermal_model/comfort.rs) + 1 in fluxion-behavior/src/comfort.rs (evaporative_split_is_fixed_iso7730_not_load_regulated); seven behavior contracts (25 °C, 18 °C, 30 °C met-1.2, 20 °C, met-1.5 respiration pair, 23 °C) and the sim fan-26 °C contract re-pin in place — no count change. Prior: 2026-10-25 (loop round 97, comfort ISO 7730 fixes): 8999 -> 9000 — +1 respiration contract (respiration_uses_exhaled_air_saturation_pressure_iso7730, fluxion-behavior/src/comfort.rs; E_re = 1.7e-5·M·(5867 − p_a), 5867 Pa = exhaled-air saturation pressure; the thermal_model + behavior comfort contracts re-pin in place — no count change). Prior: 2026-10-24 (Issue #3985): 8997 -> 8999 — same +2 hybrid outdoor-temperature-series tests. Prior: 2026-10-24 (loop round 96, comfort units fix): 8996 -> 8997 — +1 pinned humidity-monotonicity evaporative-capacity contract in fluxion-behavior/src/comfort.rs (ISO 7730:2005 Annex D Pa-form units fix; the five thermal_model comfort contracts re-pin in place — no count change). Prior: 2026-10-24 (loop round 95, coverage remedy): 8991 -> 8996 — same +5 comfort unit tests. Prior: 2026-10-24 (loop round 95): 8990 -> 8991 — same +1 gauge scheduled-ACH wiring pin. Prior: 2026-10-24: 8986 -> 8990 — +4 integration-cli topology tests for the #3967 slice-2 `--model` unsupported-format rejection (export .osm, export .xml, lint .osm, and a .json file with garbage still failing as a JSON parse error). Prior:  2026-10-24 (loop round 94, merge union): 8983 -> 8986 - same +3 surface-pool component-partition contracts (tests/mass_advance_contract.rs slice 6: internal-gain radiative keeps the §C.3 conductance partition under the T5138 route, solar-component route untouched by the decomposition, lumped-zone partition untouched; the engine fix decomposes the surface pool at the class drive + lag site under FLX_R94_T5138_GAINS_C3) landing on top of #4587 slice 2's topology row. 2026-10-24 (loop round 94): 8974 -> 8977 - same +3 surface-pool component-partition contracts (tests/mass_advance_contract.rs slice 6: internal-gain radiative keeps the §C.3 conductance partition under the T5138 route, solar-component route untouched by the decomposition, lumped-zone partition untouched; the engine fix decomposes the surface pool at the class drive + lag site under FLX_R94_T5138_GAINS_C3). Prior: 2026-10-24 (Issue #3967 slice 1): 8963 -> 8974 - same +9 topology lib tests + 2 integration-cli topology lint tests (E008 miswired/correctly-wired end-to-end). Prior: 2026-10-24 (Issue #3989): 8960 -> 8963 - same +3 per-surface schema tests in src/api/schema.rs. Prior: 2026-10-24 (loop round 93, #4582): 8959 -> 8960 - +1 T5138 air-share retire contract (tests/mass_advance_contract.rs contract 19: t5138_airshare_retired_default_and_falsy_restore; the lag-site input retires under the class route, falsy-token restore bit-identical). Prior: 2026-10-23 (Issue #3985): 8957 -> 8959 - same +2 hybrid ventilation wiring tests. Prior: 2026-10-23 (loop round 92): 8953 -> 8957 — +4 solar-partition + retained-lag-filter contract tests (tests/mass_advance_contract.rs slice 5: partition-site window-loss-sink scaler + retired-knob-default pins, T5138 default-token gate-leak bit-identity, T5138-vs-C.3 route-choice mass-direction, retained lag-filter route transient-direction + 960 h DC parity + exact persisted lag-state parity). # Prior: 2026-10-23 (PR #4568, Issue #4555): 8947 -> 8953 - same +6 sensor-node tests. Prior: 2026-10-23 (PR #4565, Issue #4268): 8940 -> 8947 - same +7 Addendum a hourly tests. Prior: 2026-10-22 (loop round 89b): 8936 -> 8940 — +4 mass-node time-coverage contracts (tests/mass_advance_contract.rs slice 4: default-polarity bit-identity unset-vs-"2", legacy-vs-default non-inertness, convergence re-anchored to the default N=12→N=96 limit, A/B/C decomposition pin coarse-vs-inner-loop ≈ 0.4–1.4% of the legacy movement; FLX_R89_MASS_DT_ALIGN flip). Stacked on the round-88 baseline below. # 2026-10-22 (loop round 88): 8933 -> 8936 — +3 mass-advance surface-pool share contracts (tests/mass_advance_contract.rs slice 3: flag-off bit-identity unset-vs-falsy on air+mass nodes, flag-on mass-direction + air-node non-identity on the per-class path under free-float solar driving, lumped-site pairing for tensor-less zones; FLX_R88_PHIST_MASS_SHARE measurement arm). Stacked on the round-87 baseline below. # 2026-10-22 (loop round 87, PR #4539): 8932 -> 8933 — +1 machine-fair pop-100 floor-match guard (tests/all_tests/throughput_benchmark.rs: test_batch_oracle_pop100_floor_matches_yaml — release_gates.yaml::benchmark.batch_oracle_pop100.min_configs_per_sec vs the reader fallback constant, #4534; mirrors the three #4527 hybrid floor-match guards). Stacked on the round-86 baseline below. # 2026-10-22 (loop round 86): 8929 -> 8932 — +3 night-vent coupling contract tests (tests/mass_advance_contract.rs: vent-off inertness at both live coupling sites, vent-on monotone approach toward outdoor, per-class-vs-lumped agreement under live vent driving — blitz slice 2 per FREE_FLOAT_BRAINSTORM_R85 rank 3; the same commit serializes the suite through ENV_LOCK since the engine reads FLX_R84_PERCLASS per step and non-mutating stepping tests observed contract 4/7's in-process flip from other threads). Stacked on the round-85 baseline below. # 2026-10-22 (loop round 85, PR #4535): 8925 -> 8929 — +4 mass-node advance contract tests (tests/mass_advance_contract.rs: equilibrium invariance, relaxation hull, conductance monotonicity, per-class-vs-lumped agreement under uniform night driving — first slice of the free-float physics TDD blitz, Alex-directed 2026-10-21; new standalone test binary, +1 to BASELINE_TEST_BINARIES). # 2026-10-20 (loop round 77): 8924 -> 8925 — +1 coupled single-N scratch probe (tests/all_tests/r77_convergence_probe.rs: ignored in-process N-sweep probe for the FLX_R77_COUPLED_N arm — 650FF/600FF/900FF Tmin/Tmax full-precision convergence sweep N=1..384, non-inert assert). # 2026-10-20 (loop round 76): 8923 -> 8924 — +1 sub-hour mass-state scratch probe (tests/all_tests/r76_mass_probe.rs: ignored in-process N-sweep probe for the FLX_R76_MASS_SUBSTEPS arm — 650FF Tmin/Tmax full-precision payoff vs the hourly-mass baseline, N=1..96, non-inert assert). # 2026-10-20 (loop round 75, #4509): 8922 -> 8923 — +1 vent-transition event-resolution scratch probe (tests/all_tests/r75_vent_probe.rs: ignored in-process flag-toggle probe, transition-hour full-precision deltas for the FLX_R75_VENT_RAMP arm; asserts the arm is non-inert). # 2026-10-19 (loop round 70 rerun, #4500/#4501): 8919 -> 8922 — +3 fuzz-crash regression tests (tests/all_tests/issue_4500_fuzz_signed_energy.rs: signed-net-energy contract pin, accumulator non-negativity pin, overflow-demand rejection + state/accumulator-untouched pin). # 2026-10-19 (loop round 70, #4495): 8916 -> 8919 — +3 input-envelope rejection tests (tests/all_tests/issue_4495_zone_temp_nonfinite.rs: air-node inf rejection contract, mass-node inf rejection + state-untouched pin, legacy step_physics 0.0-kWh degradation pin). # 2026-10-18 (loop round 64, #4296 part 3): 8913 -> 8916 — +3 §9 CE dispatch tests (tests/validation/hvac_equipment_rated_performance/section9_ce.rs: CE200-vs-map-rating-point provenance, 13-case entering-wet-bulb refusal guard, CE200 steady-state dispatch with load-balance and physics invariants). # 2026-10-18 (loop round 63, #4296 part 2): 8905 -> 8913 — same +8 dx_performance_map unit tests. # 2026-10-18 (loop round 62): 8901 -> 8905 — +4 §6.2.1 conformance-output tests (tests/ashrae_140_section7_gsr_emitter.rs: companion-CSV shape/presence, companion-CSV byte-identity, round-62 value pins with dated provenance, +1 ignored emit artifact test; tests/ashrae_140_section7_conformance.rs gains no new test fns — its pins/eval were re-recorded in place). # 2026-10-18 (loop round 58, #4274): 8896 -> 8901 — +5 §11 QASv2 tests (fluxion-core/src/qas_airside.rs: case-1 workbook-anchor reproduction, case-2 wet-bulb dew-point derivation, six-case determinism, warm-case heating-sign sanity, oracle cross-check of OA state). # 2026-10-18 (loop round 57, #4270): 8893 -> 8896 — +3 §8 GC10a tests (fluxion-core/src/ground_coupling.rs: closed-form vs published 2433 W pin, closed-form determinism, coarse-FD sanity band). # 2026-10-18 (loop round 56, #4295): 8886 -> 8893 — +7 TMY2 reader tests (fluxion-core/src/weather/tm2.rs: header parse from the 140-2001 Addendum a Figure A1-1 sample, tenths-unit scalings, direct/diffuse solar split, illuminance source-flag handling, missing-sentinel hard-error policy, synthetic 8,760-record year contract, WeatherSource hour-bounds). # 2026-10-17 (loop round 55, #4276): 8881 -> 8886 — +5 §7 conformance-table tests (tests/ashrae_140_section7_conformance.rs: committed-CSV parse/pin, per-case §6.2.1 status-matrix pin, conformance-CSV byte-identity staleness, conformance-MD byte-identity staleness, +1 ignored emit artifact test). — +6 §7 results-CSV emitter tests (tests/ashrae_140_section7_gsr_emitter.rs: pinned-case-set, finite-outputs, contract-shape, committed-CSV byte-identity staleness, headline value pins, +1 ignored emit artifact test). # 2026-10-17 (loop round 51, §6 missing rows): 8872 -> 8875 — +3 §6 conformance-upgrade tests (tests/ashrae_140_section6_weather_driver.rs: Table 6-1 daily-CSV coverage test, daily-CSV artifact byte-equality staleness test, §6.2.1.2/§6.2.1.8 solar-sky + Table-6-1 day value-pin test — the round-51 harness extension emits the formerly missing §6.2.1.2.1–.3 and §6.2.1.8 outputs). # 2026-10-17 (loop round 50, #4157 slice 3): 8870 -> 8872 — +2 teacher-validation reachability tests (tests/all_tests/teacher_validation_core_only_construction.rs: constructs and steps a PhysicsThermalModel from a CaseSpec struct literal built entirely from fluxion_core::ashrae_cases / fluxion_core::construction types, proving the #3986 DAE-teacher zone path is reachable without the validation layer; acceptance item 5 of #4157). # 2026-10-17 (loop round 49): 8869 -> 8870 — +1 §6 value-pin test (tests/ashrae_140_section6_weather_driver.rs: section6_output_values_pinned_to_recorded_baseline — pins each WD case's annual loads, peaks, and peak occurrences to the committed section6_workflow_results.csv values at ±0.5%, closing the #4456 review finding that the conformance CSV carries statuses, not values). # 2026-10-17 (loop round 48, #4283/#4276): 8866 -> 8869 — +3 §6 conformance tests (tests/ashrae_140_section6_weather_driver.rs: conformance-table evaluation from the committed workflow CSV, conformance-CSV artifact byte-equality, +1 ignored conformance-artifact emit test). # 2026-10-16 (loop round 47, #4269): 8858 -> 8866 — +8 §6 case-harness tests (tests/ashrae_140_section6_weather_driver.rs: six-case finite-output run, EPW-header site metadata, WD100/WD600 full-engine albedo discrimination, WD300 southern-hemisphere seasonality, WD400 arctic extremes, WD500 cooling dominance, workflow-CSV shape, +1 ignored CSV emit test). # 2026-10-16 (loop round 46, #4269): 8854 -> 8858 — +4 §6 weather-driver smoke tests (tests/ashrae_140_section6_weather_driver.rs: WD100/WD600 albedo discrimination, WD500 half-hour UTC offset, WD300 southern-hemisphere declination, WD400 polar finiteness). # 2026-10-16 (loop round 38): 8929 -> 8854 — -75 with the archived src/quantum/ tree (52 unit tests + 3 integration modules: quantum_python_verification, quantum_topology_stress, qubo_error_handling). # 2026-10-15 (loop round 35, #4343): 8928 -> 8929 — same +1 default-model unit test. # 2026-10-15 (loop round 35, GATE-04 / issue #4175): 8932 -> 8928 — same -4 duplicate-routing-enum test removal. # 2026-10-14 (loop round 31): 8931 -> 8932 — same +1 gauge multi-zone §C.3 metering unit test. # 2026-10-14 (loop round 27): 8930 -> 8931 — +1 gauge-gated integration test test_gauge_dispatch_consumes_setback_schedule (ashrae_140_setback_ventilation.rs; gauge-path schedule-consumption mechanism pin, §LIMIT-41 follow-up). # 2026-10-13 (loop round 26 fixup): 8929 -> 8930 — +1 DailySchedule::is_constant unit test (schedule.rs; guards the #2826 vector-write contract). # 2026-10-13 (loop round 26): 8928 -> 8929 — +1 integration test test_case_940_setback_schedule_reaches_demand (setback schedule consumption mechanism pin, ashrae_140_setback_ventilation.rs). # 2026-10-13 (loop round 25, issue #3061): 8927 -> 8928 — +1 integration test sim_per_zone_u_lw.rs (per-zone U-values threaded to the 5R1C LW path; lives in the integration suite because the #1441/#2495 cycle guard rejects new sim->validation edges under src/sim). # 2026-10-13 (loop round 24, issue #3061): 8926 -> 8927 — same +1 per-zone constructions unit test. # 2026-10-13 (loop round 23, issue #3061): 8923 -> 8926 — same +3 implicit inter-zone coupling unit tests. # 2026-10-13 (loop round 22, issue #3061): 8920 -> 8923 — +3 Case 960 sunspace round-22 tests (glazing geometry pin, common-wall-only coupling pin, free-floating-not-pinned). # 2026-10-12 (loop round 19, issue #4384 follow-up): 8915 -> 8920 — same +5 solar step-date unit tests. # 2026-10-12 (loop round 18, issue #4168): 8913 -> 8915 — same +2 Cases 800/810 band-provenance unit tests. # 2026-10-12 (loop round 15, issue #4169): 8912 -> 8913 — same +1 Case 195 table-parity unit test. # 2026-10-12 (loop round 15, issue #4167): 8911 -> 8912 — same +1 setpoint-parity unit test. # 2026-10-11 (loop round 12, PR #4371): 8910 -> 8911 — same +1 gauge exterior-film unit test. # 2026-10-10 (loop round 11 PR-b): 8909 -> 8910 — same +1 FD volumetric-source conservation unit test. Prior (loop round 11 PR-a): 8908 -> 8909 — same +1 quasi-steady window-face unit test. Prior (loop round 10): 8907 -> 8908 — same +1 floor ground-boundary unit test (PR #4363). Prior (loop round 9 PR-b): 8906 -> 8907 — +1 gauge infiltration unit test. Prior: 8906 (2026-10-09, #4170): 8893 -> 8906 — 13 new tests in tests/all_tests/zone_balance_eplus_isolation.rs (nine strict annual-energy gate tests for the previously-ungated cases 610/620/630/640/650/910/930/940/195 + four free-float strict-gate tests 600FF/650FF/900FF/950FF). Stacked on the 2026-10-07 #4188 baseline of 8893.
                                  # 2026-10-07 (Issue #4204): 8890 -> 8891 — one new steady-state test in tests/dhat_batched_surrogate_zero_growth.rs (predict_loads_into_with_scratch_zero_steady_state_growth). Stacked on the 2026-09-29 #4299-supersede baseline of 8890.

                                  # 2026-09-29 (Issue #4192): 8873 -> 8881 — collapse triplicated EPW decoder; 8 new tests in fluxion-core/src/weather/epw.rs (shared-fixture agreement, truncated/missing-field skips, 8760-count fixture guard, sentinel coercion). Stacked on #4155's 8873 baseline.
                                  # Previous: 2026-09-28: 8836 -> 8839 — three new `fabric_case_*_measurement` tests in `tests/all_tests/ashrae_140_fabric_multiselector.rs` (PR-A+2 fabric harness, Refs #3986-A+2 / #4117). CI's authoritative cargo --list count rises 8836 -> 8839 (local 8131). Ratchet must equal or exceed CI live count per Issue #3442 protocol.
                                  # Previous: 2026-09-27: 8829 -> 8833 — post-rebase AST delta for PR-A + PR-B's combined consolidated-runner modules (7 selector-parity + 8 PCM box tests).
                                 # unmet-hours lib tests in src/api/schema.rs
                                 # (tolerance default, all-hours divergence,
                                 # multi-zone summation); no binary/ignore
                                 # changes. Stacked on the 8796 #4102 bump.
                                 # Previous: 2026-09-27: 8785 -> 8796 — Issue
                                 # #4102 adds 11 monthly_end_use integration
                                 # tests (bin edges, leap-year, sub-hourly,
                                 # multi-year, reconciliation, peaks, REST
                                 # shape, Case 600 cross-check); no
                                 # lib/binary/ignore changes. Stacked on the
                                 # 8785 #4101 bump.

# 2026-09-12 (Issue #3711): 121 -> 133 — see the history entry above for
# the per-crate attribution and the AST-vs-cargo calibration analysis.
# 2026-09-18 (Issue #3729): 133 -> 138 — absorbs the cargo-verified
# workspace_ignored growth between Wave 8 (commit ``6bb98e1``, Issue #3590)
# and develop HEAD. The five new ignored tests are quarantines that
# pre-date #3729 (the new commit adds zero ignored tests of its own);
# bumping the ratchet here keeps the nightly gauge soak's
# ``scripts-tests.yml`` invocation green until a dedicated cleanup PR
# can attribute them per-crate.
BASELINE_WORKSPACE_IGNORED = 152  # 2026-11-12 (Issue #4774, PR #4787 merge-update): 153 -> 152 — merged-tree recount. Prior: 2026-11-12 (Issue #4777): 154 -> 153 — test_issue532_case195_energy_regression is un-ignored now that #532 is closed (2026-04-18); it measures 4.49 MWh annual heating and 1.53 MWh annual cooling for Case 195, so the near-zero-energy assertion passes. Prior: 2026-11-12 (Issue #4759): 155 -> 154 — the Case 195 high-mass-walls variant (tests/all_tests/ashrae_140_solid_conduction_variants.rs::test_case_195_high_mass_walls) is un-ignored now that its blockers closed (#3059/#3064/#1465/#1462); it measures -18.13 kWh annual energy against a -18.22 kWh baseline, so the zero-energy assertion passes. Prior: 2026-11-12 (Issue #4755): 156 -> 155 — the IDF Case 600 strict ±15 % band gate (tests/all_tests/idf_ashrae_140_acceptance.rs::idf_case_600_annual_heating_within_15_percent_strict) is un-ignored now that #1577 (develop CI broken) is closed; it measures 5.379 MWh inside the 4.314–5.836 MWh reference band. Prior: 2026-10-23 (Issue #4210): 167 -> 156 — 11 FLUID-01 fluxion-fluid Jacobian tests un-ignored after the analytical Jacobians were fixed (from_row_slice orientation, raw-input saturation gates, Chiller COP-floor factor, Boiler column order). Prior: 167 (2026-10-10, loop round 9 PR-c): 161 -> 167 — 6 gauge-routed 600-series metrics re-quarantined under the round-5 #[ignore=reason] convention pending PR-d re-pin. Prior: 161 (2026-10-09, #4170).
                                  # round 5 quarantines the two Case 610 premise
                                  # tests (case_610::test_annual_heating / cooling)
                                  # that left the published Annex B ranges under
                                  # the spec-correct shading geometry; dated
                                  # reasons + QUARANTINE.md rows in the same PR.
                                  # Previous: 2026-10-09 (Issue #4170): 153 -> 160 — all
                                  # thirteen new strict-gate tests in
                                  # tests/all_tests/zone_balance_eplus_isolation.rs
                                  # are #[ignore]d (nine annual-energy tests for the
                                  # previously-ungated cases 610/620/630/640/650/
                                  # 910/930/940/195 + four free-float strict-gate
                                  # tests 600FF/650FF/900FF/950FF). Previous:
                                  # Issue #4058 bump
                                  # from 147 (+6) — the six
                                  # FD-vs-EnergyPlus step-response tests quarantined
                                  # ``awaiting #4058`` (their former flux channel was
                                  # a circular identity exposed by the #3981
                                  # conservative Robin extraction; redesign tracked
                                  # in Issue #4058, rows in tests/QUARANTINE.md).
                                  # Previous: Issue #3869 bump from 138 (+9) — eight new
                                  # doctests marked ``ignore`` in the workspace-doctests
                                  # fix (Issue #3869 unblocking develop merges): the
                                  # ``MaterialLayer`` / ``CTFMaterial`` / ``WallSpec``
                                  # doctests that referenced pre-#2462 crate-split
                                  # types were marked ``ignore`` rather than rewritten
                                  # to maintain historical-doc-link integrity, plus
                                  # the new ``rust,ignore`` blocks in
                                  # ``src/sim/thermal_model_solvers.rs`` (4 blocks)
                                  # that flag the deprecated ``enable_*`` methods as
                                  # illustrative-only. Live cargo-verified count is
                                  # 153 on this branch after the #3981 module lands.
# 2026-09-12 (Issue #3685): bumped from 308 to 309 for the new
# ``tests/cold_start_guard_test.rs`` binary — the always-compiled
# (feature-independent) unit tests for the Multi-Zone Cold Start
# Gate's warm-sample epsilon guard (``tests/cold_start_guard/mod.rs``).
# AST-scan test_binaries at HEAD is 309 (308 + this binary); no other
# ratchet moves (lib/workspace/ignored AST counts are unchanged or
# below their constants).
# 2026-09-24 (PR #3968): 309 -> 49 — corrected to the live AST basis after
# the #3764 consolidated-runner restructure (273 standalone root binaries
# folded into ``all_tests``); the 309 figure predated the restructure's
# AST-scan accounting.
# 2026-09-26 (Issue #4005): 49 -> 50 — new ``grid_adapter_integration``
# test binary (Case 600 engine-driven grid-adapter tests,
# ``required-features = ["grid"]``); no other ratchet moves.
# 2026-09-27 (Issue #3986-A): 50 -> 51 — new ``ashrae_140_validator_selector_parity``
# test binary (ThermalSelector wiring through ASHRAE 140 validator; PR-A of #3986
# teacher validation suite); no other ratchet moves.
# CUMULATIVE bumps from PR-A (#3986-A) and PR-B (#3986-B).
# - 2026-09-27 (Issue #3986-A, PR #4119): 50 -> 51 — new
#   `ashrae_140_validator_selector_parity` test binary (ThermalSelector
#   wiring through ASHRAE 140 validator; PR-A of #3986 teacher validation
#   suite).
# - 2026-09-27 (Issue #3986-B, PR #4120): 51 -> 51 — no AST-scan delta;
#   the new `teacher_validation_pcm_box` module is hosted inside the
#   existing consolidated `all_tests` runner and does not add a new
#   binary. The AST scanner's module-detection heuristic is consistent
#   with cargo's `--list` reporting.
# 2026-11-09 (Train PR #4672 union ratchet): BASELINE_TEST_BINARIES 58 -> 59 (new tests/surface_pool_contract.rs binary)
# 2026-11-09 (Train PR #4673 union ratchet, post-#4672 wave): BASELINE_TEST_BINARIES 59 -> 60 (surface_pool_contract + solar_lag_carry_contract binaries both landed)
BASELINE_TEST_BINARIES = 72  # 2026-11-12 (loop round 131): 71 -> 72 — new tests/step_9r4c_free_float_contract.rs target (9R4C free-float committed-temperature chain contract suite). Prior: 2026-11-11 (loop round 130): 70 -> 71 — new tests/longwave_pool_exchange_contract.rs target (longwave pool-exchange contract suite, plan §7 item (iv)). Prior: 2026-11-11 (loop round 129): 69 -> 70 — new tests/window_solar_split_contract.rs target (window solar split contract suite, plan §7 item (ii) remainder). Prior: 2026-11-10 (loop round 126): 68 -> 69 — new tests/sol_air_composition_contract.rs target (per-surface sol-air composition contract suite, plan §7 item (i) remainder). Prior: 2026-11-10 (loop round 125): 67 -> 68 — new tests/surface_pool_class_drive_contract.rs target (surface-pool class-drive contract suite, plan §7 rank-1 remainder). Prior: 2026-11-10 (loop round 124 landing of PR #4684): 66 -> 67 — new tests/shading_overhang_contract.rs target (loop r119).  # 2026-11-10 (loop round 124 landing of PR #4683): 65 -> 66 — new tests/hvac_mask_contract.rs target (loop r118).  # 2026-11-10 (loop round 124 landing of PR #4682): 64 -> 65 — new tests/solar_lag_carry_contract.rs target (loop r117).  # 2026-11-10 (loop round 124 landing of PR #4681): 63 -> 64 — new tests/archaeology_route_contract.rs target (loop r116).  # 2026-11-10 (PR #4676 landing, weather train): 62 -> 63 — new tests/multizone_cross_response_contract.rs target (inter-zone cross-response suite, loop r113). Prior: 2026-11-10 (PR #4675 landing, weather train): 61 -> 62 — new tests/multizone_vent_routing_contract.rs target (multi-zone vent routing suite, loop r112). Prior: 2026-11-10 (loop round 124 landing of PR #4674): 60 -> 61 — new tests/sol2st_split_archaeology.rs target (solar-split archaeology suite, loop r111).  # 2026-11-08 (Issue #4696, PR #4701 rebase): 57 -> 58 — branch adds tests/batch_oracle_output_materialisation.rs. Prior: 2026-10-22 (loop round 85, PR #4535): 56 -> 57 — new tests/mass_advance_contract.rs target (mass-node advance contract suite, blitz slice 1). # 2026-10-17 (loop round 55, #4276): 55 -> 56 — new tests/ashrae_140_section7_conformance.rs target (§7 conformance table, whitelisted past the tests/ashrae_140_*.rs ignore). # 2026-10-17 (loop round 53, #4283): 54 -> 55 — new tests/ashrae_140_section7_gsr_emitter.rs target (§7 results-CSV emitter harness). # 2026-10-16 (loop round 46, #4269): 53 -> 54 — new tests/ashrae_140_section6_weather_driver.rs target (§6 WD smoke tests, whitelisted past the tests/ashrae_140_*.rs ignore). # 2026-10-07 (Issue #4188): 52 -> 53 — new batch-oracle-memory-budget [[test]] target.
                             # 2026-09-28: 51 -> 52 — schedule conformance tests (#4196) add fluxion-core/tests/schedule_conformance.rs test binary.

# Sanity-check constants — the verified cargo counts at HEAD
# ``12856a9``. Operators checking the drift gate's accuracy can
# compare ``--no-verify`` vs ``--verify`` against these. The drift
# gate itself does NOT use them (they exist as documentation only).
VERIFIED_AT_HEAD_LIB_TESTS = 3894
VERIFIED_AT_HEAD_LIB_IGNORED = 4
VERIFIED_AT_HEAD_WORKSPACE_TESTS = 7923
VERIFIED_AT_HEAD_WORKSPACE_IGNORED = 125
VERIFIED_AT_HEAD_TEST_BINARIES = 298


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _regenerate_inventory(cargo_target_dir: str | None, verify: bool) -> dict:
    """Run the generator and parse the JSON it emits.

    When ``verify`` is True (the default), the generator cross-checks
    its AST counts against ``cargo test -- --list`` and prefers the
    cargo-derived numbers where available. This is the canonical
    inventory snapshot — pure-AST mode is only used when ``verify``
    is explicitly disabled (legacy migration scenarios) or when cargo
    is not available on the path.

    The generator is pointed at a scratch temp file (Issue #4134), so
    this gate never rewrites the committed
    ``tests/test_inventory.json`` — a read-only gate must not dirty
    the working tree or silently downgrade the cargo-verified snapshot
    to a pure-AST one.
    """
    with tempfile.TemporaryDirectory(prefix="fluxion-test-inventory-") as tmpdir:
        tmp_inventory = Path(tmpdir) / "test_inventory.json"
        cmd = [
            "python3",
            str(GENERATOR_SCRIPT),
            "--output",
            str(tmp_inventory),
        ]
        if verify:
            cmd += ["--verify"]
        if cargo_target_dir:
            cmd += ["--cargo-target-dir", cargo_target_dir]
        # 60s without ``--verify`` is generous; ``--verify`` triggers a
        # full ``cargo test -- --list`` which can take 30+ min on cold
        # rebuilds, so we use a larger timeout for that path.
        timeout = 1800.0 if verify else 60.0
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # returncode handled explicitly below (ruff PLW1510)
        )
        if proc.returncode != 0:
            print(
                f"ERROR: generator failed (exit={proc.returncode}):\n{proc.stderr}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not tmp_inventory.exists():
            print(
                f"ERROR: generator did not produce {tmp_inventory}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return json.loads(tmp_inventory.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    """Repo-relative display form, falling back to the absolute path.

    ``--baseline`` / ``--live-inventory`` may point outside the repo
    (e.g. a scratch copy under /tmp); ``relative_to`` would crash on
    those, so degrade gracefully instead of dying in a print statement.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_baseline(path: Path) -> dict:
    if not path.exists():
        print(
            f"ERROR: baseline not found at {_display_path(path)}. "
            f"Generate one with --update-baseline.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_baseline(path: Path, baseline: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _drift_message(metric: str, baseline: int, live: int) -> str:
    diff = live - baseline
    rel = (diff / baseline * 100.0) if baseline else 0.0
    direction = "↑" if diff > 0 else "↓"
    return (
        f"  {metric}: baseline={baseline}, live={live} "
        f"({direction} {abs(diff)} tests, {rel:+.2f}%)"
    )


def _check_drift_threshold(
    metric: str, baseline: int, live: int, violations: list[str]
) -> tuple[bool, bool]:
    """Compare a live count against the baseline for ``metric``.

    Returns ``(shrink_failed, grow_failed)``. ``shrink_failed`` is
    True when ``live < baseline`` (test deletion beyond threshold);
    ``grow_failed`` is True when ``live > baseline`` (growth beyond
    threshold). The ratchet layer in ``main()`` will translate these
    into a richer exit code.

    The threshold is OR-semantics: the gate fails if EITHER the
    absolute change exceeds ``DIFF_TOLERANCE_ABS`` OR the relative
    change exceeds ``DIFF_TOLERANCE_PCT``. This catches both the
    "delete 10 tests in a tiny suite" (rel-large, abs-small) and
    "delete 1000 tests in a giant suite" (abs-large, rel-small)
    failure modes that a single-sided threshold misses.
    """
    diff = live - baseline
    abs_diff = abs(diff)
    rel_diff = (abs_diff / baseline * 100.0) if baseline else 0.0
    exceeded = abs_diff > DIFF_TOLERANCE_ABS or rel_diff > DIFF_TOLERANCE_PCT
    if not exceeded:
        return (False, False)
    if diff > 0:
        violations.append(_drift_message(metric, baseline, live))
        return (False, True)
    violations.append(_drift_message(metric, baseline, live))
    return (True, False)


def _check_ratchet(
    metric: str,
    baseline_const_name: str,
    baseline_const_value: int,
    live: int,
    failures: list[str],
) -> None:
    """Issue #3442 downward-only ratchet — reject growth above baseline.

    Companion cleanup PRs that *resolve* a test (e.g. by deleting a
    redundant coverage binary) are expected to LOWER
    ``baseline_const_value`` BY ONE. Adding tests without raising the
    baseline is the failure mode we are protecting against — the
    drift threshold is informational; the ratchet is binding.
    """
    if live > baseline_const_value:
        failures.append(
            f"  {metric} = {live} > {baseline_const_name} = "
            f"{baseline_const_value} (growth above documented baseline)"
        )


def _check_agents_md_citations(
    agents_md_path: Path,
    live_totals: dict,
) -> list[str]:
    """Issue #4180 — verify AGENTS.md citations agree with live inventory.

    Extracts the four headline figures cited in ``AGENTS.md`` from the
    ``tests/test_inventory.json::totals`` block and compares them against
    the live inventory. Mismatch means the documentation drifted from
    reality — a PR that changes test counts without refreshing AGENTS.md
    must be rejected.

    Extracted figures:
      - lib_tests_root (from the table's ``cargo test --lib`` row)
      - workspace_tests (from the table's ``cargo test --workspace`` row)
      - workspace_ignored (from the table's ``cargo test --workspace`` row)
      - test_binaries (from the table's ``Cargo test binaries`` row)
      - lib_tests_root (from the workspace-scope rule paragraph)
      - sibling_crate_tests (from the workspace-scope rule paragraph)

    Returns a list of failure messages (empty == all citations match).
    """
    failures: list[str] = []

    try:
        text = agents_md_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # A missing AGENTS.md is itself a docs-hygiene failure: the check is
        # fail-closed, never silently green on absent input.
        return [f"  AGENTS.md: file not found at {agents_md_path}"]

    live_lib = live_totals.get("lib_tests_root", 0)
    live_workspace = live_totals.get("workspace_tests", 0)
    live_sibling = live_workspace - live_lib

    # Table-row citations — process each row exactly once.
    # Markdown table format (pipe-split with leading/trailing empty cells):
    #   | Source | Suite | Tests | Ignored | Notes |
    #   idx 0      1       2       3         4       5
    #
    # row_checks: (description_substring, expected_tests_field, expected_ignored_field)
    # expected_tests_field / expected_ignored_field are keys into live_totals
    # (None = no check for that column)
    row_checks = [
        (
            "`cargo test --lib`",
            "lib_tests_root",
            "lib_ignored_root",
        ),
        (
            "`cargo test --workspace --exclude fluxion-tauri`",
            "workspace_tests",
            "workspace_ignored",
        ),
        (
            "Cargo test binaries",
            "test_binaries",
            None,  # no Ignored column for this row
        ),
    ]

    lines = text.split('\n')
    for desc, tests_field, ignored_field in row_checks:
        cited_tests_raw = None
        cited_ignored_raw = None
        for line in lines:
            if desc in line and line.strip().startswith('|'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5:
                    cited_tests_raw = parts[3]
                    cited_ignored_raw = parts[4]
                    break
        if cited_tests_raw is None:
            failures.append(
                f"  AGENTS.md: could not find table row for {tests_field!r}"
            )
            continue
        expected_tests = live_totals.get(tests_field, 0)
        try:
            cited_tests_val = int(cited_tests_raw.replace(",", ""))
        except ValueError:
            failures.append(
                f"  AGENTS.md {tests_field}: unparseable Tests value "
                f"{cited_tests_raw!r}"
            )
        else:
            if cited_tests_val != expected_tests:
                failures.append(
                    f"  AGENTS.md {tests_field}: cited={cited_tests_val}, "
                    f"inventory={expected_tests} "
                    f"(diff {cited_tests_val - expected_tests:+,d})"
                )
        if ignored_field is not None:
            expected_ignored = live_totals.get(ignored_field, 0)
            try:
                cited_ignored_val = (
                    None
                    if cited_ignored_raw == "n/a"
                    else int(cited_ignored_raw.replace(",", ""))
                )
            except ValueError:
                failures.append(
                    f"  AGENTS.md {tests_field} (ignored): "
                    f"unparseable Ignored value {cited_ignored_raw!r}"
                )
            else:
                if cited_ignored_val != expected_ignored:
                    failures.append(
                        f"  AGENTS.md {tests_field} (ignored): "
                        f"cited={cited_ignored_val}, inventory={expected_ignored}"
                    )

    # Workspace-scope paragraph: "N,NNN lib tests" and "M,MMM sibling-crate tests"
    scope_pattern = re.compile(
        r"bare\s+`cargo test`\s+therefore runs the root crate ONLY "
        r"\(([^\)]+)\)\s+and silently SKIPS the remaining "
        r"(\d[\d,]*)\s+sibling-crate tests",
        re.IGNORECASE,
    )
    m_scope = scope_pattern.search(text)
    if m_scope:
        # group 1 is "4,276 lib tests + the root `[[test]]` entries" — extract leading number
        lib_text = m_scope.group(1)
        lib_match = re.match(r"(\d+)", lib_text.replace(",", ""))
        if lib_match is None:
            failures.append(
                "  AGENTS.md: could not parse lib-test count in workspace-scope paragraph"
            )
            return failures
        cited_root = int(lib_match.group(1))
        cited_sibling = int(m_scope.group(2).replace(",", ""))
        if cited_root != live_lib:
            failures.append(
                f"  AGENTS.md lib_tests_root (scope paragraph): "
                f"cited={cited_root}, inventory={live_lib} "
                f"(diff {cited_root - live_lib:+,d})"
            )
        if cited_sibling != live_sibling:
            failures.append(
                f"  AGENTS.md sibling_crate_tests (scope paragraph): "
                f"cited={cited_sibling}, inventory={live_sibling} "
                f"(diff {cited_sibling - live_sibling:+,d})"
            )
    else:
        failures.append(
            "  AGENTS.md: could not parse workspace-scope paragraph "
            "lib/sibling-crate test counts"
        )

    return failures


# ---------------------------------------------------------------------------
# Prose citation sites — Issue #4754.
#
# The AGENTS.md gate-inventory table rows are covered by the table-row
# checks above; the *prose* citations (the inline comment on the
# workspace command, the "bare `cargo test` skips N sibling-crate
# tests" line, and the README quick-start comment) were NOT scanned by
# any gate, so they silently drifted away from the canonical inventory
# (Issue #4754: "8,865 / 143 ignored / 4,158 / ~4,400" vs. the
# inventory's 9,266 / 156 / 4,825).
#
# The patterns below are deliberately tolerant: a site that is not
# present is not a failure here (the synthetic-tree unit tests build
# minimal AGENTS.md fixtures). Presence is enforced separately by
# ``scripts/ci/test_test_inventory_prose_citations.py``, which asserts
# that every citation site exists *and* carries the right number.
# ---------------------------------------------------------------------------
_PROSE_WORKSPACE_COUNT_RE = re.compile(
    r"full workspace \(([\d,]+) cargo-listed tests / ([\d,]+) ignored\)"
)
_PROSE_AGENTS_SIBLING_RE = re.compile(
    r"bare `cargo test` skip(?:s)? the ([\d,]+) sibling-crate tests"
)
_PROSE_README_SIBLING_RE = re.compile(
    r"skips (~?[\d,]+) sibling-crate tests"
)


def _check_prose_citations(repo_root: Path, live_totals: dict) -> list[str]:
    """Verify the prose test-count citations in AGENTS.md / README.md.

    Issue #4754 — the inline prose figures (as opposed to the
    AGENTS.md gate-inventory table rows) were never checked by the
    drift gate, so they drifted to stale values. Each citation site is
    compared against the live inventory:

    * AGENTS.md — the workspace command comment
      ``(N cargo-listed tests / M ignored)`` must cite
      ``workspace_tests`` / ``workspace_ignored``.
    * AGENTS.md — the bare ``cargo test`` line must cite
      ``workspace_tests - lib_tests_root`` sibling-crate tests.
    * README.md — the quick-start comment must cite the same
      sibling-crate count (same definition: workspace tests minus the
      root crate's lib tests).

    A citation site that is absent from the file is skipped (not a
    failure) so that synthetic-tree fixtures stay valid; presence is
    asserted by the dedicated prose-citation test.

    Returns a list of failure messages (empty == all citations match).
    """
    failures: list[str] = []

    live_workspace = live_totals.get("workspace_tests", 0)
    live_ignored = live_totals.get("workspace_ignored", 0)
    live_lib = live_totals.get("lib_tests_root", 0)
    live_sibling = live_workspace - live_lib

    def _read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    agents_text = _read(repo_root / "AGENTS.md")
    if agents_text is not None:
        for match in _PROSE_WORKSPACE_COUNT_RE.finditer(agents_text):
            cited_tests = int(match.group(1).replace(",", ""))
            cited_ignored = int(match.group(2).replace(",", ""))
            if cited_tests != live_workspace:
                failures.append(
                    f"  AGENTS.md workspace_tests (prose): cited={cited_tests}, "
                    f"inventory={live_workspace} "
                    f"(diff {cited_tests - live_workspace:+,d})"
                )
            if cited_ignored != live_ignored:
                failures.append(
                    f"  AGENTS.md workspace_ignored (prose): "
                    f"cited={cited_ignored}, inventory={live_ignored}"
                )
        for match in _PROSE_AGENTS_SIBLING_RE.finditer(agents_text):
            cited_sibling = int(match.group(1).replace(",", ""))
            if cited_sibling != live_sibling:
                failures.append(
                    f"  AGENTS.md sibling_crate_tests (prose): "
                    f"cited={cited_sibling}, inventory={live_sibling} "
                    f"(diff {cited_sibling - live_sibling:+,d})"
                )

    readme_text = _read(repo_root / "README.md")
    if readme_text is not None:
        for match in _PROSE_README_SIBLING_RE.finditer(readme_text):
            cited_sibling = int(match.group(1).replace("~", "").replace(",", ""))
            if cited_sibling != live_sibling:
                failures.append(
                    f"  README.md sibling_crate_tests (prose): "
                    f"cited={cited_sibling}, inventory={live_sibling} "
                    f"(diff {cited_sibling - live_sibling:+,d})"
                )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test-inventory drift gate (Issue #3442)."
    )
    parser.add_argument(
        "--live-inventory",
        type=Path,
        default=None,
        help="Use this inventory JSON file instead of regenerating (default: regenerate to a scratch temp file; the committed inventory is never modified).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Baseline JSON path (default: {DEFAULT_BASELINE.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline to match the live inventory (use ONLY in cleanup PRs that intentionally shrink the suite).",
    )
    parser.add_argument(
        "--cargo-target-dir",
        default=None,
        help="CARGO_TARGET_DIR for the regenerator invocation.",
    )
    parser.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=True,
        help="(default: True) Regenerate via the AST-regex + cargo cross-check path. Slower (~5 min cold, sub-second warm) but counts match what cargo test -- --list reports.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Regenerate via AST-only (skip cargo cross-check). Fast but counts may over-shoot by ~10% because the regex doesn't track cfg(test) boundaries.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output for CI consumption.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="(default: True) Fail on ratchet OR drift-threshold violations.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict mode; only ratchet violations fail (useful for the first migration run).",
    )
    args = parser.parse_args()

    # Stage 1: live inventory.
    if args.live_inventory:
        live_path: Path = args.live_inventory
        if not live_path.is_absolute():
            live_path = REPO_ROOT / live_path
        if not live_path.exists():
            print(f"ERROR: live inventory missing at {live_path}", file=sys.stderr)
            return 2
        live = json.loads(live_path.read_text(encoding="utf-8"))
    else:
        live = _regenerate_inventory(args.cargo_target_dir, verify=args.verify)

    totals = live["totals"]
    live_lib = totals.get("lib_tests_root", 0)
    live_lib_ignored = totals.get("lib_ignored_root", 0)
    live_workspace = totals.get("workspace_tests", 0)
    live_workspace_ignored = totals.get("workspace_ignored", 0)
    live_binaries = totals.get("test_binaries", 0)

    # Stage 0 (Issue #4180): AGENTS.md citation check.
    # Fail immediately if the docs cite stale numbers. The path is resolved
    # from REPO_ROOT at call time (not a module-level constant) so tests can
    # redirect REPO_ROOT at a synthetic tree.
    agents_md_path = REPO_ROOT / "AGENTS.md"
    agents_md_failures = _check_agents_md_citations(agents_md_path, totals)
    if agents_md_failures:
        for line in agents_md_failures:
            print(line, file=sys.stderr)
        print(
            "AGENTS.md citations do not match tests/test_inventory.json. "
            "Refresh the cited figures in AGENTS.md before submitting.",
            file=sys.stderr,
        )
        return 1

    # Stage 0b (Issue #4754): prose citation check (AGENTS.md inline
    # comments + README quick-start comment). The gate-inventory table
    # rows are covered by Stage 0; the *prose* figures were not scanned
    # by any gate and drifted (Issue #4754).
    prose_failures = _check_prose_citations(REPO_ROOT, totals)
    if prose_failures:
        for line in prose_failures:
            print(line, file=sys.stderr)
        print(
            "Prose test-count citations do not match tests/test_inventory.json. "
            "Refresh the cited figures in AGENTS.md / README.md before submitting.",
            file=sys.stderr,
        )
        return 1

    # Stage 2: baseline. For --update-baseline, skip the comparison
    # entirely and rewrite the file with the live snapshot. Both the
    # cargo-verified ``metrics`` and the AST-mode ``metrics_ast``
    # snapshot are recorded so the gate can compare like-for-like in
    # either mode (Issue #3711).
    if args.update_baseline:
        live_is_ast = not bool((live.get("verify") or {}).get("matched"))
        if live_is_ast:
            metrics_ast = {
                "lib_tests": live_lib,
                "lib_ignored": live_lib_ignored,
                "workspace_tests": live_workspace,
                "workspace_ignored": live_workspace_ignored,
                "test_binaries": live_binaries,
            }
        else:
            # Verified overlay active: the AST counts survive in the
            # per-crate table and the workspace_lib/integration split.
            ast_totals = live["totals"]
            ast_root = live.get("by_crate", {}).get("fluxion", {})
            metrics_ast = {
                "lib_tests": ast_root.get("lib_tests", 0),
                "lib_ignored": ast_root.get("lib_ignored", 0),
                "workspace_tests": ast_totals.get("workspace_lib_tests", 0)
                + ast_totals.get("workspace_integration_tests", 0),
                "workspace_ignored": ast_totals.get("workspace_lib_ignored", 0)
                + ast_totals.get("workspace_integration_ignored", 0),
                "test_binaries": ast_totals.get("test_binaries", 0),
            }
        baseline_snapshot = {
            "schema_version": live.get("schema_version", 1),
            "captured_at": _now_iso(),
            "captured_from": str(DEFAULT_INVENTORY.relative_to(REPO_ROOT)),
            "metrics": {
                "lib_tests": live_lib,
                "lib_ignored": live_lib_ignored,
                "workspace_tests": live_workspace,
                "workspace_ignored": live_workspace_ignored,
                "test_binaries": live_binaries,
            },
            "metrics_ast": metrics_ast,
            "ratchet": {
                "BASELINE_LIB_TESTS": BASELINE_LIB_TESTS,
                "BASELINE_LIB_IGNORED": BASELINE_LIB_IGNORED,
                "BASELINE_WORKSPACE_TESTS": BASELINE_WORKSPACE_TESTS,
                "BASELINE_WORKSPACE_IGNORED": BASELINE_WORKSPACE_IGNORED,
                "BASELINE_TEST_BINARIES": BASELINE_TEST_BINARIES,
            },
            "ratchet_drift_tolerance_pct": DIFF_TOLERANCE_PCT,
            "ratchet_drift_tolerance_abs": DIFF_TOLERANCE_ABS,
            "totals": totals,
            "by_crate": live.get("by_crate", {}),
        }
        baseline_path: Path = args.baseline
        if not baseline_path.is_absolute():
            baseline_path = REPO_ROOT / baseline_path
        _save_baseline(baseline_path, baseline_snapshot)
        print(
            f"Baseline updated: {_display_path(baseline_path)} "
            f"(lib_tests={live_lib}, workspace_tests={live_workspace}, "
            f"test_binaries={live_binaries})"
        )
        return 0

    # Stage 3: load baseline + compare.
    baseline_path = args.baseline
    if not baseline_path.is_absolute():
        baseline_path = REPO_ROOT / baseline_path
    baseline = _load_baseline(baseline_path)
    # Mode-aware comparison (Issue #3711): the two gate modes count
    # differently by construction (AST-regex scan vs cargo ``--list``
    # cross-check), so a single baseline metrics dict cannot serve both
    # — AST and verified counts differ by ~9% on the headline metrics,
    # which is far beyond the 5% drift tolerance. When the baseline
    # carries a ``metrics_ast`` snapshot, compare like-for-like: AST
    # live counts (``--no-verify``, the CI fast path, or a ``--verify``
    # run whose cargo cross-check failed) are compared against
    # ``metrics_ast``, and cargo-verified live counts (``--verify`` with
    # ``verify.matched``) against ``metrics``. Baselines without
    # ``metrics_ast`` keep the previous single-dict behavior. Newer
    # baselines (PR #4265 supersede, Issue #4156) carry the canonical
    # numbers in ``totals`` and drop the duplicated ``metrics`` /
    # ``metrics_ast`` blocks; read from ``totals`` when those blocks are
    # absent.
    live_is_ast = (not args.verify) or not bool(
        (live.get("verify") or {}).get("matched")
    )
    if "metrics" in baseline or "metrics_ast" in baseline:
        baseline_metrics = baseline.get("metrics", {})
        if live_is_ast and "metrics_ast" in baseline:
            baseline_metrics = baseline["metrics_ast"]
    else:
        baseline_totals = baseline.get("totals", {})
        baseline_metrics = {
            "lib_tests": baseline_totals.get("lib_tests_root", 0),
            "lib_ignored": baseline_totals.get("lib_ignored_root", 0),
            "workspace_tests": baseline_totals.get("workspace_tests", 0),
            "workspace_ignored": baseline_totals.get("workspace_ignored", 0),
            "test_binaries": baseline_totals.get("test_binaries", 0),
        }
    base_lib = baseline_metrics.get("lib_tests", 0)
    base_lib_ignored = baseline_metrics.get("lib_ignored", 0)
    base_workspace = baseline_metrics.get("workspace_tests", 0)
    base_workspace_ignored = baseline_metrics.get("workspace_ignored", 0)
    base_binaries = baseline_metrics.get("test_binaries", 0)

    drift_violations: list[str] = []
    shrink_failed: list[str] = []
    grow_failed: list[str] = []
    for metric, base_val, live_val in [
        ("lib_tests", base_lib, live_lib),
        ("lib_ignored", base_lib_ignored, live_lib_ignored),
        ("workspace_tests", base_workspace, live_workspace),
        ("workspace_ignored", base_workspace_ignored, live_workspace_ignored),
        ("test_binaries", base_binaries, live_binaries),
    ]:
        shrink, grow = _check_drift_threshold(
            metric, base_val, live_val, drift_violations
        )
        if shrink:
            shrink_failed.append(metric)
        if grow:
            grow_failed.append(metric)

    # Ratchet failures: any metric that grew above the documented
    # baseline constant is rejected.
    ratchet_failures: list[str] = []
    _check_ratchet(
        "lib_tests", "BASELINE_LIB_TESTS", BASELINE_LIB_TESTS, live_lib, ratchet_failures
    )
    _check_ratchet(
        "lib_ignored",
        "BASELINE_LIB_IGNORED",
        BASELINE_LIB_IGNORED,
        live_lib_ignored,
        ratchet_failures,
    )
    _check_ratchet(
        "workspace_tests",
        "BASELINE_WORKSPACE_TESTS",
        BASELINE_WORKSPACE_TESTS,
        live_workspace,
        ratchet_failures,
    )
    _check_ratchet(
        "workspace_ignored",
        "BASELINE_WORKSPACE_IGNORED",
        BASELINE_WORKSPACE_IGNORED,
        live_workspace_ignored,
        ratchet_failures,
    )
    _check_ratchet(
        "test_binaries",
        "BASELINE_TEST_BINARIES",
        BASELINE_TEST_BINARIES,
        live_binaries,
        ratchet_failures,
    )

    if args.json:
        out = {
            "executed_at": _now_iso(),
            "drift_tolerance_pct": DIFF_TOLERANCE_PCT,
            "drift_tolerance_abs": DIFF_TOLERANCE_ABS,
            "live": totals,
            "baseline_metrics": baseline_metrics,
            "ratchet_baselines": {
                "BASELINE_LIB_TESTS": BASELINE_LIB_TESTS,
                "BASELINE_LIB_IGNORED": BASELINE_LIB_IGNORED,
                "BASELINE_WORKSPACE_TESTS": BASELINE_WORKSPACE_TESTS,
                "BASELINE_WORKSPACE_IGNORED": BASELINE_WORKSPACE_IGNORED,
                "BASELINE_TEST_BINARIES": BASELINE_TEST_BINARIES,
            },
            "drift_violations": drift_violations,
            "shrink_failed": shrink_failed,
            "grow_failed": grow_failed,
            "ratchet_failures": ratchet_failures,
            "would_fail": bool(
                (drift_violations and args.strict) or ratchet_failures
            ),
        }
        json.dump(out, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(
            f"Test-inventory drift gate (Issue #3442)\n"
            f"  baseline: {_display_path(baseline_path)}\n"
            f"  live:     regenerated scratch snapshot "
            f"({DEFAULT_INVENTORY.relative_to(REPO_ROOT)} untouched)\n"
            f"  tolerance: ±{DIFF_TOLERANCE_PCT:.1f}% or ±{DIFF_TOLERANCE_ABS} tests "
            f"(whichever is larger)"
        )
        print(
            f"\nCounts (live → baseline):\n"
            f"  lib_tests:           {live_lib} → {base_lib}\n"
            f"  lib_ignored:         {live_lib_ignored} → {base_lib_ignored}\n"
            f"  workspace_tests:     {live_workspace} → {base_workspace}\n"
            f"  workspace_ignored:   {live_workspace_ignored} → {base_workspace_ignored}\n"
            f"  test_binaries:       {live_binaries} → {base_binaries}"
        )
        if drift_violations:
            print("\nDrift-threshold violations:")
            for line in drift_violations:
                print(line)
        else:
            print("\nNo drift-threshold violations.")
        if ratchet_failures:
            print("\nRatchet violations (BASELINE_* exceeded):")
            for line in ratchet_failures:
                print(line)
        else:
            print("\nNo ratchet violations.")

    failure = bool((drift_violations and args.strict) or ratchet_failures)
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())

#   - 2026-09-11 (PR improve/quarantine-placeholder-test, final stack):
#     119 -> 120 — the Issue #3629 thermal_mass placeholder quarantine
#     adds one more workspace `#[ignore]` on top of the wasm quarantine.

#   - 2026-09-12 (PR improve/quarantine-placeholder-test, Issue #3705):
#     120 -> 121 — the flaky occupancy statistical test (OS-seeded 10k-step
#     Markov assertion) quarantined per the Issue #3629 protocol.
