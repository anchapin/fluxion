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
BASELINE_LIB_TESTS = 4290  # 2026-10-10 (loop round 9 PR-b): 4289 -> 4290 — +1 gauge infiltration ACH liveness unit test (test_infiltration_ach_parameter_is_live, gauge_zone_solver.rs). Prior baseline: 4289 (2026-09-29, #4172). (Issue #4172, PR #4299 supersede): 4300 -> 4289 — hoist SimulationDiagnostics to fluxion-core (12 new tests in fluxion-core/src/diagnostics.rs, net -11 after -23 tests moved out of fluxion::validation::diagnostics that became redundant with the new core diagnostics module). Stacked on the #4292 supersede's 4300 baseline.
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

BASELINE_WORKSPACE_TESTS = 8907  # 2026-10-10 (loop round 9 PR-b): 8906 -> 8907 — same +1 gauge infiltration unit test. Prior: 8906 (2026-10-09, #4170): 8893 -> 8906 — 13 new tests in tests/all_tests/zone_balance_eplus_isolation.rs (nine strict annual-energy gate tests for the previously-ungated cases 610/620/630/640/650/910/930/940/195 + four free-float strict-gate tests 600FF/650FF/900FF/950FF). Stacked on the 2026-10-07 #4188 baseline of 8893.
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
BASELINE_WORKSPACE_IGNORED = 161  # 2026-10-09 (Issue #4170): 160 -> 161 — loop
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
BASELINE_TEST_BINARIES = 53  # 2026-10-07 (Issue #4188): 52 -> 53 — new batch-oracle-memory-budget [[test]] target.
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
    import re

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
