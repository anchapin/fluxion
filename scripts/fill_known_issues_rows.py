#!/usr/bin/env python3
"""Fill the empty KNOWN_ISSUES rows from their investigation files (Issue #4291).

Every fill below was drafted by reading `docs/investigations/<token>-*.md` plus the
current strict-gate baseline (`tests/reference_data/zone_balance/strict_energy_gate_baseline.json`)
and the live code/baselines where the investigation's claim needed a premise check.
Nothing is hand-edited in docs/KNOWN_ISSUES.md: this script is the only writer, it
asserts the row currently has the empty `—` markers before touching it, and it is
idempotent (re-running after a fill is a no-op).

Usage: python3 scripts/fill_known_issues_rows.py [--check]
  --check: exit 1 if any row would still change (no write).
"""

from __future__ import annotations

import sys
from pathlib import Path

DOC = Path("docs/KNOWN_ISSUES.md")

# token -> (value cell, issue cell, closes-when cell, status cell)
# None means "leave this cell as-is" (e.g. keep an existing issue link).
FILLS: dict[str, tuple[str, str | None, str | None, str]] = {
    "BASE-01": (
        "resolved — Phase 1: window U-value applied only to `h_tr_w`; `apply_parameters()` "
        "separated it from the envelope conductance so it no longer reaches `h_tr_em`",
        None, None, "resolved",
    ),
    "BASE-02": (
        "resolved — Phase 1: HVAC mode and load determined from Ti_free per ISO 13790 "
        "(Ti_free = (num_tm + num_phi_st + num_rest)/den), not the conditioned Ti",
        None, None, "resolved",
    ),
    "BASE-03": (
        "resolved — Phase 2: Cm computed from material layers (volumetric heat capacity × volume); "
        "the empirical 1/sqrt(C/2.4e6 J/K) correction was removed under #2706 (2026-08-11) with zero "
        "ASHRAE 140 result change (never in the validation pipeline); structural capacitance checks retained",
        None, None, "resolved",
    ),
    "BASE-04": (
        "resolved — Phase 1: Denver TMY integrated; residual documented minor under #2429: the embedded "
        "DenverTmyWeather is a synthetic approximation vs the Golden-NREL.724666 TMY3 EPW behind the "
        "reference data (~45 km / 174 m elevation apart); Case 950 peak cooling 0.859 kW inside the 0.70–0.90 kW band",
        None, None, "resolved",
    ),
    "BASE-05": (
        "resolved — Phase 7A: h_tr_em now uses h_tr_op from actual construction U-values (was a fixed k/d "
        "formula 7.5× too high); annual heating over-prediction 2.5–3.5× → 1.2–1.6×; peak heating in range",
        None, None, "resolved",
    ),
    "SOLAR-01": (
        "Case 600 peak cooling 3.09 kW vs authoritative 4.8–6.2 kW band (benchmark.rs:124-127, ~36 % under — "
        "§LIMIT-05 carrier); Case 900 peak cooling 1.95 kW in the post-#1408 1.60–2.10 kW band. The 2026-03-30 "
        "'40–80 % under-predicted' framing is obsolete; residual high-mass gaps tracked under §LIMIT-05 / #1280",
        None, "superseded — live gaps tracked under §LIMIT-05", "tracking only",
    ),
    "SOLAR-03": (
        "open — 610/630/910/930 shading cases show smaller cooling reduction than the 30–60 % reference "
        "response; shading device configuration appears correct but the reduction does not propagate through "
        "the thermal network (mass-vs-glass solar distribution suspected); no measurement or fix recorded "
        "since the Phase 3 note",
        None, "shading sensitivity demonstrated on 610/630/910/930", "open",
    ),
    "FREE-02": (
        "Phase-2 '900FF passes / 950FF fails' framing superseded by the strict gate: 900FF min −6.71 °C vs "
        "[0.6, 2.2] °C and 950FF min −21.97 °C vs the Annex B8-corrected [−13.4, −12.5] °C band — both "
        "known_fail, both now COLDER than the band (sign opposite the original over-prediction framing); "
        "950FF tracked under §LIMIT-17 (strict_energy_gate_baseline.json, round-26 fan fix state)",
        None, "GaugeSolver air-trajectory rework (#1465/#1462) — same carrier as FREE-01/§LIMIT-17", "open",
    ),
    "FREE-03": (
        "resolved — #2339 sub-hour air-node sub-stepping (commit 645116d): all FF swings pass acceptance "
        "(600FF 86.3 °C ≥ 80, 650FF 91.1 ≥ 82, 900FF 62.8 ≥ 50, 950FF 66.8 ≥ 58)",
        None, None, "resolved",
    ),
    "TEMP-01": (
        "resolved — validated: temperature swing (amplitude) is the primary metric and passes; lag-timing "
        "differences within 1 h accepted for annual energy; not an energy blocker",
        None, None, "resolved",
    ),
    "MULTI-01": (
        "resolved — 100 kW anomaly chain closed (validator kWh/kW unit fix; #1396 accounting; #1399 mass-node "
        "sign): post-#1407 Case 960 peak heating ≈ 1.4 kW vs [2.0, 8.0] kW — the below-band residual is the "
        "architectural under-prediction tracked under PeakHeatingLimit-01, not this bug",
        None, None, "resolved",
    ),
    "MULTI-02": (
        "resolved — Phase 8: COP conversion (cooling COP 3.0, heating efficiency 0.9) added to validate_case_960 "
        "/ validate_analytical_engine; core engine thermal loads unchanged; Case 960 passes after correction",
        "#273", None, "resolved",
    ),
    "LIMIT-01": (
        "the original 'Won't Fix (by design)' framing is superseded: the 900-series annual-energy bidirectional "
        "over-prediction is live-tracked with measurements under §LIMIT-05 UPDATE (#2453); the accepted-5R1C-"
        "limitation rationale is retained only as historical framing",
        None, "— (live tracking under §LIMIT-05)", "tracking only",
    ),
    "LIMIT-02": (
        "superseded: 600FF max free-float is live-tracked under FREE-01 (57.25 °C vs [62.4, 68.4] °C, "
        "GaugeSolver-routed); the 'won't fix, diagnostic-only' framing is stale under the fidelity-first direction",
        None, "— (live tracking under FREE-01)", "tracking only",
    ),
    "LIMIT-03": (
        "resolved — Phase 7A: hardcoded 100 kW replaced by floor-area-based capacity (heating 500 W/m², cooling "
        "600 W/m² × floor area; Case 960: 100 → 32 kW); open TODO (investigation file): design-day load-based "
        "capacity with 1.1–1.2× safety margin",
        None, "—", "resolved",
    ),
    "LIMIT-04": (
        "Case 960 peak heating now 2.58 kW vs [2.0, 8.0] kW — in band (§LIMIT-14, round-21+): the Phase-7B 32 kW "
        "overprediction and its free-floating-sunspace root cause were closed by the #1456 6R2C-override removal "
        "and the round-21..24 envelope/spec/coupling fixes; residual sunspace physics tracked under §LIMIT-14",
        None, "— (residual under §LIMIT-14)", "resolved",
    ),
    "LIMIT-06": (
        "the legacy empirical h_corr/c_corr factors were tuned to pre-#1270 reference ranges now obsolete; "
        "superseded by the #1457/#1460 physics fix; #1421 re-validation against the post-#1270 authoritative "
        "CSVs (benchmark.rs:124-127: 600 H 4.36–5.79 MWh, C 3.92–6.14 MWh) still pending",
        None, "#1421 re-calibration vs the post-#1270 CSVs", "open",
    ),
    "LIMIT-21": (
        "process entry, no measured value: ADR-0017 (2026-09-25 UPDATE) — default flip to the equation-based DAE "
        "teacher gated on the #3986 teacher validation suite; β-soak (#3286) nightly authority at 0/30; GaugeSolver "
        "designated DAE-path prototype pending #3982 thermal-mass demonstration; the 2026-09-20 'no tunable fix' "
        "conclusion stands",
        None, "#3986 teacher validation suite passes → default flip", "open",
    ),
    "REF-01": (
        "resolved (#2748): the monthly references are a documented-shape degree-day redistribution (ASHRAE "
        "Fundamentals Ch. 19, balance point 18.3 °C) of the authoritative annual midpoint — no published monthly "
        "source exists (Annex B annual/peak only; E+ BESTEST plots only; in-repo IDFs not yet physics-fixed, "
        "#2239); the monthly CI gate is un-ignored, reporting-only; `scripts/generate_monthly_aggregate.py "
        "--validate` regeneration path ready",
        None, None, "resolved",
    ),
    "REPORT-01": (
        "open — `classify_systematic_issues()` in `src/validation/reporter.rs` is still heuristic (case-ID / "
        "metric-type based, verified on the current tip); the 05-04 data-driven analyzer plan citation is stale — "
        "it never landed",
        None, "data-driven classification in reporter.rs", "open",
    ),
    "REPORT-02": (
        "open — no automatic quality-metric (pass rate, MAE, max deviation) phase-over-phase tracking has landed; "
        "the 05-04 analyzer.rs/QualityMetrics plan citation is stale",
        None, "QualityMetrics tracked automatically per phase", "open",
    ),
    "REPORT-03": (
        "partially addressed — KNOWN_ISSUES.md rows now carry Issue columns, but ASHRAE140_RESULTS.md rows remain "
        "unlinked to GitHub issues; the 05-04 catalog plan citation is stale",
        None, "every ASHRAE140_RESULTS.md issue row linked to its GitHub issue", "open",
    ),
    "REPORT-04": (
        "open — the validation report still has no 'what's fixed in this phase' section; the 05-04 plan citation "
        "is stale",
        None, "phase-comparison section in the report", "open",
    ),
    "CI-02": (
        "environmental, open — debug builds of fluxion-rest SIGSEGV in rust-lld on disk/memory-constrained local "
        "systems (observed during #2280); not reproducible on GitHub-hosted runners; workaround: `cargo build/test "
        "--release` locally (CI uses release builds)",
        None, "reproducible environment identified or linker-level fix", "open",
    ),
    "CI-03": (
        "intentional RC pin, open — `ort` = 2.0.0-rc.13 verified in Cargo.toml and fluxion-behavior/Cargo.toml; no "
        "stable 2.0 on crates.io as of the investigation; opt-in everywhere (feature-gated in both crates, out of "
        "default). Closes when ort 2.0 stable ships: bump the pin, update comments, drop this entry",
        None, "ort 2.0 stable on crates.io → bump pin, drop entry", "open",
    ),
    "FLUID-01": (
        "known limitation, open — analytical Jacobians for Chiller/Boiler/CoolingCoil/Pump/VavBox assume smooth "
        "functions; clamp()/max() saturation points make the analytical derivative diverge from finite-difference "
        "(zero derivative at clamps); fix requires subgradient or automatic differentiation (#2330); affected "
        "jacobian-accuracy tests still present in fluxion-fluid",
        None, "subgradient or AD support in fluxion-fluid autodiff", "open",
    ),
    "FLUID-02": (
        "test-design bug, open — `test_vav_box_gradient_descent_convergence` passes a 1-element damper input to an "
        "API requiring 3 inputs ([damper, static_pressure, t_inlet]) → index-out-of-bounds in `jacobian_input` "
        "(test still present in fluxion-fluid, #2330); needs an all-3-input redesign or partial-input optimization "
        "support",
        None, "test redesigned to pass all 3 inputs or partial-input API added", "open",
    ),
    "FFD-02": (
        "structural, open — the FFD stub returns a constant 293.15 K (no zone air energy balance) so peak_cooling "
        "is 0.00 kW vs the 4.5 kW NIST HVAC BESTEST reference (100 % error); test `#[ignore]`-quarantined per "
        "#2612; closing requires a genuine coupled BES↔FFD solver (air-node ODE via the FfdCfdAdapter), not "
        "constants — tuning to pass is forbidden",
        None, "real coupled BES↔FFD zone energy balance lands, `#[ignore]` removed, 10 % tolerance asserted",
        "open",
    ),
    # Not in #4291's original 28, but the last row still carrying an empty value
    # cell; filled so the doc-wide 'no current measurement' caveat can be dropped.
    "PeakHeatingLimit-01": (
        "architectural 5R1C/9R4C under-prediction at the coldest hour (single lumped-mass node buffers the "
        "air-side free-float; E+ ~3.9 kW vs Fluxion ~0.9 kW in the original measurement). Current tip state: "
        "docs/ASHRAE140_MULTI_ZONE_RESULTS.md records ~1.4 kW (below the [2.0, 8.0] kW band) for the 5R1C/9R4C "
        "path, while the §LIMIT-14 round-21 update recorded 2.58 kW in band after the envelope/coupling fixes; "
        "the strict gate carries no peak metric, so no fresher re-measure exists. Accepted limitation: "
        "`test_peak_load_validation` allows a documented < 85 % under-prediction tolerance; fix needs 9R4C "
        "multi-surface time-constant integration at a finer timestep",
        None, "9R4C multi-surface time-constant integration (finer timestep)", "open",
    ),
}


def split_row(line: str) -> list[str]:
    stripped = line.strip()
    assert stripped.startswith("|") and stripped.endswith("|"), f"not a table row: {line!r}"
    return [c.strip() for c in stripped[1:-1].split(" | ")]


def build_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    text = DOC.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    filled: list[str] = []
    already: list[str] = []
    missing: list[str] = []

    for token, (value, issue, closes, status) in FILLS.items():
        marker = f"| **{token}** |"
        idxs = [i for i, ln in enumerate(lines) if ln.startswith(marker)]
        if len(idxs) != 1:
            missing.append(f"{token} (found {len(idxs)} rows)")
            continue
        i = idxs[0]
        cells = split_row(lines[i].rstrip("\n"))
        if len(cells) != 7:
            missing.append(f"{token} (row has {len(cells)} cells, expected 7)")
            continue
        if cells[2] != "—" or cells[4] != "—" or cells[5] != "open":
            already.append(token)
            continue
        cells[2] = value
        if issue is not None:
            cells[3] = issue
        if closes is not None:
            cells[4] = closes
        cells[5] = status
        lines[i] = build_row(cells) + ("\n" if lines[i].endswith("\n") else "")
        filled.append(token)

    if missing:
        for m in missing:
            print(f"FAIL: {m}", file=sys.stderr)
        return 1

    if not filled:
        print("OK: nothing to fill — all target rows already carry values")
    else:
        if check_only:
            print(f"CHECK: {len(filled)} rows would change: {', '.join(filled)}")
            return 1
        DOC.write_text("".join(lines), encoding="utf-8")
        print(f"OK: filled {len(filled)} rows: {', '.join(filled)}")
        if already:
            print(f"note: already filled (untouched): {', '.join(already)}")

    # Done-criterion of Issue #4291: once no row carries an empty "—" value cell,
    # drop the now-false "Entries with no current measurement in the row" caveat.
    remaining_empty = sum(
        1 for ln in lines
        if ln.startswith("| **") and len(split_row(ln.rstrip("\n"))) == 7
        and split_row(ln.rstrip("\n"))[2] == "—"
    )
    caveat_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("## Entries with no current measurement in the row"):
            caveat_start = i
            break
    if caveat_start is not None:
        if remaining_empty:
            print("note: caveat section kept — empty rows remain")
        else:
            end = None
            for j in range(caveat_start + 1, len(lines)):
                if lines[j].startswith("## "):
                    end = j
                    break
            assert end is not None, "caveat section has no following heading"
            del lines[caveat_start:end]
            DOC.write_text("".join(lines), encoding="utf-8")
            print("OK: removed the 'no current measurement' caveat section (no empty rows remain)")
    elif remaining_empty:
        print("note: empty rows remain but no caveat section present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
