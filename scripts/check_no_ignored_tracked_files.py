#!/usr/bin/env python3
"""
Tracked-but-Ignored Check for Fluxion.

Fails CI when a tracked file is also matched by a ``.gitignore`` pattern.
Catches regressions of the #3174 / #3356 bug class — PR #3342 had to
reconstruct 5 silently-dropped modules because an unanchored ``lib/``
pattern in the root .gitignore matched ``fluxion-tauri/frontend/src/lib/``.

A tracked file matching a ``.gitignore`` pattern is a silent-data-loss
hazard: any future ``git add`` of similar files will be dropped, and any
rebase or filter-branch operation may silently untrack the file. The
:class:`~subprocess.CompletedProcess` exit code is ``1`` if any tracked
file matches an ignore pattern, ``0`` otherwise.

The detection uses ``git check-ignore --no-index -v`` for every tracked
file. Without ``--no-index`` git skips paths already in the index, so
the ``-v`` verbose output would never report a match and the gate would
be a silent no-op for exactly the cases it is supposed to catch.

Pre-existing tracked-but-ignored files (committed despite matching a
``.gitignore`` rule) are listed in :data:`KNOWN_TRACKED_BUT_IGNORED`
and tolerated. New tracked-but-ignored files — the regression we care
about — fail the gate. The baseline lives in this file because the set
is bounded and audited; promoting it to a separate file would risk a
silent drift on a future cleanup PR.

Usage::

    python3 scripts/check_no_ignored_tracked_files.py

Exit codes:
    0 — all tracked files pass the check
    1 — one or more tracked files match .gitignore patterns (regression)
    2 — script error (e.g. ``git`` not available, IO error)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Known tracked-but-ignored files (pre-existing violations of #3356).
#
# Each entry is a path relative to the repo root that is currently
# tracked in git AND matches a .gitignore pattern. These were committed
# deliberately (with ``git add -f`` or before the relevant gitignore rule
# was tightened) and must remain tracked. They are tolerated by this gate
# so the existing 131 entries do not block CI.
#
# Future regressions — tracked files added on or after the #3356 PR that
# match a .gitignore pattern — are NOT in this list and WILL fail the
# gate. To add a new intentionally-tracked-but-ignored file, document
# the rationale in the commit message and add the path here.
# ---------------------------------------------------------------------------
KNOWN_TRACKED_BUT_IGNORED: frozenset[str] = frozenset(
    {
        # Issue #3356 audit (PR #3356): every entry below was matched by
        # at least one .gitignore pattern at the time the baseline was
        # captured. They are intentionally tracked and the .gitignore
        # patterns that match them were not deleted (see commit message
        # for the rationale per category).
        # ----------------------------------------------------------------
        # Root `.editorconfig` matched by `.gitignore:62:.editorconfig`.
        # The file ships at repo root because editor tooling expects it
        # there; the gitignore rule prevents other directories from
        # committing per-dir `.editorconfig` overrides.
        ".editorconfig",
        # ----------------------------------------------------------------
        # `.issues/*` markdown files (issue intake forms, see AGENTS.md)
        # matched by `.gitignore:283:.issues/`. These were force-added
        # for historical issue tracking; they remain tracked.
        ".issues/issue_01_furniture_factor_C_me.md",
        ".issues/issue_02_h_tr_me_furniture_factor.md",
        ".issues/issue_03_building_type_enum.md",
        ".issues/issue_04_validate_thermal_coupling_ashrae140.md",
        ".issues/issue_05_internal_mass_time_constant.md",
        ".issues/issue_B_h_tr_me_value.md",
        ".issues/research_internal_mass_capacitance.md",
        # ----------------------------------------------------------------
        # `.planning/phases/**/*` matched by `.gitignore:217:.planning/phases/`.
        # These are per-milestone phase plan/summary records (issues
        # #31..#903, etc.). The gitignore rule blocks transient scratch
        # state under `.planning/`; permanent records are tracked.
        ".planning/ASHRAE_140_BLIND_VALIDATION_PLAN.md",
        ".planning/phases/31-full-validation-release/31-04-PLAN.md",
        ".planning/phases/32-ctf-thermal-mass-fix/PHASE-SUMMARY.md",
        ".planning/phases/33-peak-load-diagnostics/33-01-PLAN.md",
        ".planning/phases/33-peak-load-diagnostics/33-01-SUMMARY.md",
        ".planning/phases/33-peak-load-diagnostics/33-02-PLAN.md",
        ".planning/phases/33-peak-load-diagnostics/33-03-PLAN.md",
        ".planning/phases/33-peak-load-diagnostics/33-03-SUMMARY.md",
        ".planning/phases/34-peak-load-physics-fix/34-02-SUMMARY.md",
        ".planning/phases/34-peak-load-physics-fix/34-03-SUMMARY.md",
        ".planning/phases/36-01-validation/36-01-PLAN.md",
        ".planning/phases/36-v0.8.0-release/36-01-SUMMARY.md",
        ".planning/phases/36-v0.8.0-release/36-03-SUMMARY.md",
        ".planning/phases/36-v0.8.0-release/36-04-SUMMARY.md",
        ".planning/phases/40-case-expansion-foundation/40-01-PLAN.md",
        ".planning/phases/40-case-expansion-foundation/40-02-PLAN.md",
        ".planning/phases/40-case-expansion-foundation/40-03-PLAN.md",
        ".planning/phases/40-case-expansion-foundation/40-04-PLAN.md",
        ".planning/phases/40-case-expansion-foundation/40-04-SUMMARY.md",
        ".planning/phases/40-case-expansion-foundation/40-05-PLAN.md",
        ".planning/phases/40-case-expansion-foundation/40-06-SUMMARY.md",
        ".planning/phases/40-case-expansion-foundation/40-CONTEXT.md",
        ".planning/phases/40-case-expansion-foundation/40-PLAN-PROMPT.md",
        ".planning/phases/40-case-expansion-foundation/40-RESEARCH-PROMPT.md",
        ".planning/phases/40-case-expansion-foundation/40-RESEARCH.md",
        ".planning/phases/41-high-mass-physics-performance/41-02-SUMMARY.md",
        ".planning/phases/41-high-mass-physics-performance/41-03-SUMMARY.md",
        ".planning/phases/44-high-mass-physics-validation/44-01-SUMMARY.md",
        ".planning/phases/44-high-mass-physics-validation/44-03-SUMMARY.md",
        ".planning/phases/45-advanced-cross-validation-automation/45-01-SUMMARY.md",
        ".planning/phases/45-advanced-cross-validation-automation/45-02-SUMMARY.md",
        ".planning/phases/46-expanded-validation-coverage/46-02-SUMMARY.md",
        ".planning/phases/46-expanded-validation-coverage/46-03-SUMMARY.md",
        ".planning/phases/47-performance-validation-optimization/47-01-PLAN.md",
        ".planning/phases/47-performance-validation-optimization/47-02-PLAN.md",
        ".planning/phases/47-performance-validation-optimization/47-02-SUMMARY.md",
        ".planning/phases/47-performance-validation-optimization/47-03-PLAN.md",
        ".planning/phases/47-performance-validation-optimization/47-03-SUMMARY.md",
        ".planning/phases/47-performance-validation-optimization/47-04-PLAN.md",
        ".planning/phases/47-performance-validation-optimization/47-04-SUMMARY.md",
        ".planning/phases/47-performance-validation-optimization/47-05-PLAN.md",
        ".planning/phases/47-performance-validation-optimization/47-05-SUMMARY.md",
        ".planning/phases/47-performance-validation-optimization/47-06-PLAN.md",
        ".planning/phases/47-performance-validation-optimization/47-06-SUMMARY.md",
        ".planning/phases/47-performance-validation-optimization/47-07-PLAN.md",
        ".planning/phases/47-performance-validation-optimization/47-08-PLAN.md",
        ".planning/phases/47-performance-validation-optimization/47-08-SUMMARY.md",
        ".planning/phases/47-performance-validation-optimization/47-COMPLETION-REPORT.md",
        ".planning/phases/47-performance-validation-optimization/47-CONTEXT.md",
        ".planning/phases/47-performance-validation-optimization/47-RESEARCH.md",
        ".planning/phases/A-baseline-stripping/A-01-PLAN.md",
        ".planning/phases/A-baseline-stripping/A-02-PLAN.md",
        ".planning/phases/B-physics-fixes/B-01-PLAN.md",
        ".planning/phases/B-physics-fixes/B-02-PLAN.md",
        ".planning/phases/B-physics-fixes/B-03-PLAN.md",
        ".planning/phases/C-benchmark-correction/C-01-PLAN.md",
        ".planning/phases/D-blind-validation-pass/D-01-PLAN.md",
        ".planning/phases/E-sustained-validation/E-01-PLAN.md",
        ".planning/phases/M1-multi-zone-foundation/M1-02-SUMMARY.md",
        ".planning/phases/M1-multi-zone-foundation/M1-03-SUMMARY.md",
        ".planning/phases/M1-multi-zone-foundation/M1-VERIFICATION.md",
        ".planning/phases/M2-zone-hvac-controls/M2-01-PLAN.md",
        ".planning/phases/M2-zone-hvac-controls/M2-02-PLAN.md",
        ".planning/phases/M2-zone-hvac-controls/M2-03-PLAN.md",
        ".planning/phases/M2-zone-hvac-controls/M2-04-PLAN.md",
        ".planning/phases/M2-zone-hvac-controls/M2-05-PLAN.md",
        ".planning/phases/M2-zone-hvac-controls/M2-05-SUMMARY.md",
        ".planning/phases/M2-zone-hvac-controls/M2-06-PLAN.md",
        ".planning/phases/M2-zone-hvac-controls/M2-06-SUMMARY.md",
        ".planning/phases/M2-zone-hvac-controls/M2-07-PLAN.md",
        ".planning/phases/M2-zone-hvac-controls/M2-07-SUMMARY.md",
        ".planning/phases/M2-zone-hvac-controls/M2-08-PLAN.md",
        ".planning/phases/M2-zone-hvac-controls/M2-08-SUMMARY.md",
        ".planning/phases/M2-zone-hvac-controls/M2-CONTEXT.md",
        ".planning/phases/M3-ashrae-140-validation/M3-01-SUMMARY.md",
        ".planning/phases/M3-ashrae-140-validation/M3-02-SUMMARY.md",
        ".planning/phases/issue-903-600-test-failures/issue-903-01-SUMMARY.md",
        # ----------------------------------------------------------------
        # `assets/*.onnx`, `examples/dummy_surrogate.onnx` matched by
        # `.gitignore:292:*.onnx`. These ship as fixture inputs for the
        # surrogate examples; the blanket `*.onnx` rule guards against
        # new model artifacts leaking in without going through the
        # surrogate registry.
        "assets/dummy_surrogate.onnx",
        "assets/loads_predictor.onnx",
        "examples/dummy_surrogate.onnx",
        # ----------------------------------------------------------------
        # `benches/.../current_tdqs.json` matched by an exact-path rule
        # `.gitignore:232:benches/...`. The baseline file is a tracked
        # point of comparison for the TDQS regression gate; the rule
        # guards against local edits replacing it accidentally.
        "benches/orchestration_decisions/baselines/current_tdqs.json",
        # ----------------------------------------------------------------
        # `docs/.../*_PLAN.md`, `docs/.../*_ANALYSIS.md` matched by
        # `.gitignore:189:**/*_ANALYSIS.md` and `:192:**/*_PLAN.md`.
        # These are intentionally-tracked archival investigation
        # reports; the blanket `**/*_ANALYSIS.md` / `**/*_PLAN.md`
        # rules block new analysis/plan files at any depth.
        "docs/archive/planning/PARALLEL_ISSUES_PLAN.md",
        "docs/literature/ENERGYPLUS_CONDUCTION_ANALYSIS.md",
        # ----------------------------------------------------------------
        # `models/...` matched by `.gitignore:303:models/*` and the
        # `.onnx`/`.pt`/`.pkl` blanket rules. The model artifacts listed
        # here ship as fixture inputs to the surrogate regression
        # tests; the blanket rule guards against new models leaking in.
        "models/.gitignore",
        "models/rl_policy/policy.json",
        "models/rl_policy/policy.onnx",
        "models/rl_policy/policy.onnx.data",
        "models/surrogate_conduction_metrics.json",
        "models/surrogate_conduction_validation.json",
        "models/surrogate_solar_gain_metrics.json",
        "models/surrogate_solar_gain_validation.json",
        "models/surrogate_ventilation_metrics.json",
        "models/surrogate_ventilation_validation.json",
        "models/surrogate_zone_thermal_metrics.json",
        "models/surrogate_zone_thermal_validation.json",
        # ----------------------------------------------------------------
        # `tests/ashrae_140_*.rs` (minus the ones explicitly re-included
        # by `.gitignore:148-156`) matched by `.gitignore:147:tests/ashrae_140_*.rs`.
        # These are ASHRAE 140 test binaries not yet wired into the
        # formal validation gate. The blanket rule blocks new ones; the
        # tracked ones stay.
        "tests/ashrae_140_blind_validation.rs",
        "tests/ashrae_140_case_195_470.rs",
        "tests/ashrae_140_case_195_solid_conduction.rs",
        "tests/ashrae_140_case_600_series.rs",
        "tests/ashrae_140_case_900.rs",
        "tests/ashrae_140_case_960_sunspace.rs",
        "tests/ashrae_140_case_non_residential.rs",
        "tests/ashrae_140_cases_800_810.rs",
        "tests/ashrae_140_diagnostic_integration_test.rs",
        "tests/ashrae_140_diagnostic_test.rs",
        "tests/ashrae_140_free_floating.rs",
        "tests/ashrae_140_integration.rs",
        "tests/ashrae_140_setback_ventilation.rs",
        "tests/ashrae_140_solar_gain_variants.rs",
        "tests/ashrae_140_solid_conduction_variants.rs",
        "tests/ashrae_140_weather_comparison.rs",
        # ----------------------------------------------------------------
        # `tools/evolution/...` matched by `tools/evolution/.gitignore`
        # (rules `*.yaml`, `checkpoints/`, `*.json`). Per-campaign
        # reproducible artefacts committed for issues #3337 / #3338 / #3339.
        "tools/evolution/configs/ctf.yaml",
        "tools/evolution/configs/solar_simd.yaml",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/best_program.rs",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/best_program_info.json",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/metadata.json",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/programs/12e71212-aaa5-4a13-8e57-6f37bf91a147.json",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/programs/449fa7a2-7462-4616-a3d3-c571184f8e9b.json",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/programs/839bba43-2d4c-4473-8cd0-f6340178b108.json",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/programs/bdc9d250-2292-4183-aaea-e94220214651.json",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/programs/d2a2077e-d2de-4251-99b3-7123308cd7bd.json",
        "tools/evolution/results/ctf/bounded_run/checkpoints/checkpoint_5/programs/f9adae24-bdee-4b80-abce-b5ad85f2546a.json",
        # ----------------------------------------------------------------
        # `tools/test_synthetic_fallback_disabled.py` matched by
        # `.gitignore:211:tools/test_*.py`. This is the only intentionally
        # committed test-*.py file under tools/; the blanket rule blocks
        # new ones.
        "tools/test_synthetic_fallback_disabled.py",
    }
)


def find_tracked_but_ignored(tracked: list[str]) -> list[tuple[str, str]]:
    """Return ``[(ignore_rule, file_path), ...]`` for each tracked file that
    matches a ``.gitignore`` pattern.

    Uses ``git check-ignore --no-index <path>`` so tracked files are tested
    against the full standard ignore hierarchy. Without ``--no-index`` the
    call is a no-op for indexed paths.

    For the rule text we make a separate ``git check-ignore --no-index -v``
    call only when the bare call reports the file as ignored. The ``-v``
    flag is intentionally avoided for the go/no-go decision because it
    reports *every* matching rule including ``!`` re-includes, so the exit
    code alone is not a reliable "file is excluded" signal in the presence
    of re-include patterns.
    """
    hits: list[tuple[str, str]] = []
    for f in tracked:
        r = subprocess.run(
            ["git", "check-ignore", "--no-index", f],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if r.returncode != 0:
            continue
        v = subprocess.run(
            ["git", "check-ignore", "--no-index", "-v", f],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        # Prefer the verbose output for context, but if the file is
        # actually excluded the verbose call should agree. Defensive
        # fallback to the bare path keeps the gate correct even if the
        # git version somehow diverges.
        rule = v.stdout.strip() if v.returncode == 0 else f"(unknown rule for {f})"
        hits.append((rule, f))
    return hits


def main() -> int:
    print("=== Fluxion Tracked-but-Ignored Check (Issue #3356) ===")
    print(f"Repo: {REPO_ROOT}")
    print()

    # Validate that we are inside the repo the script was committed to.
    # ``git check-ignore`` would still work from a subdirectory of the
    # repo, but ``KNOWN_TRACKED_BUT_IGNORED`` is keyed on repo-relative
    # paths; failing fast prevents a misleading PASS if the script is
    # invoked from an unrelated worktree.
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: cannot resolve repo toplevel from {REPO_ROOT}", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        return 2
    if os.path.realpath(toplevel) != str(REPO_ROOT.resolve()):
        print(
            f"ERROR: script lives at {REPO_ROOT} but current git toplevel is "
            f"{toplevel}; refusing to run with stale baseline.",
            file=sys.stderr,
        )
        return 2

    # Tracked files (NUL-separated; the trailing ``\0`` produces one
    # empty string after split, which we drop).
    tracked_proc = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    tracked = [f for f in tracked_proc.stdout.split("\0") if f]

    hits = find_tracked_but_ignored(tracked)

    if not hits:
        print(f"OK: {len(tracked)} tracked files checked, none ignored")
        return 0

    # Partition into accepted (baseline) and new (regression).
    accepted = [(rule, f) for rule, f in hits if f in KNOWN_TRACKED_BUT_IGNORED]
    regressions = [(rule, f) for rule, f in hits if f not in KNOWN_TRACKED_BUT_IGNORED]

    if regressions:
        print(
            f"ERROR: {len(regressions)} NEW tracked file(s) match .gitignore "
            f"patterns (regression of the #3174 / #3356 bug class):",
        )
        for rule, f in regressions[:50]:
            print(f"  - {f}")
            print(f"      {rule}")
        if len(regressions) > 50:
            print(f"  ... and {len(regressions) - 50} more")
        print()
        print(
            "Fix by either anchoring the matching .gitignore pattern with a "
            "leading '/', tightening it to a more specific path, or adding a "
            "'!' re-include for the file. Do NOT 'git rm' tracked files in "
            "this PR — that is a separate cleanup.",
        )
        return 1

    print(
        f"OK: {len(tracked)} tracked files checked; "
        f"{len(accepted)} pre-existing tracked-but-ignored file(s) tolerated "
        f"(see KNOWN_TRACKED_BUT_IGNORED); no regressions.",
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — top-level CLI error boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
