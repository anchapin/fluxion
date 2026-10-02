# SOLAR-01 — investigation history

Narrative history for **SOLAR-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `SOLAR-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### SOLAR-01: Peak Cooling Load Under-Prediction

- **Description:** Peak cooling load gaps have evolved across the post-#1323 wave.
  The original 2026-03-30 framing ("peak cooling 40-80 % under-predicted across
  nearly all cases") was correct for the Phase 7B 5R1C/6R2C baseline but is now
  obsolete. Per `docs/ASHRAE140_RESULTS.md` (2026-06-24 snapshot) and the
  Case 900 peak-cooling verification PR #1362 / #1328: low-mass Cases 600/610
  peak cooling now sit within ±15 % of the post-#1270 reference envelope, and
  high-mass Case 900 peak cooling (1.95 kW) sits inside the post-#1408 reconciled
  reference band 1.60-2.10 kW. The remaining 9xx-series peak cooling under-prediction
  is the same root cause that LIMIT-05 tracks (roof-solar under-counting —
  `docs/investigations/issue-1280-ctf-peak-load.md` §4). The 40-80 % figure
  should not be cited by new contributors; refer to LIMIT-05 + the per-case engine
  numbers in `docs/ASHRAE140_RESULTS.md` instead.
- **Affected Cases (legacy):** 600, 610, 620, 630, 640, 650, 900, 910, 920, 940, 950, 960
- **Affected Cases (post-#1323):** 910, 920, 930, 940, 950, 960 peak cooling (the
  high-mass shading / setback / night-ventilation set); Cases 600/650 and 900
  peak cooling now within the post-#1270 reference envelope (±15 %).
- **Affected Metrics:** Peak Cooling (kW)
- **Severity:** High (downgraded from Critical — Case 900 peak cooling closes
  per #1362/#1328; remaining high-mass gaps tracked under LIMIT-05)
- **GitHub Issue:** #274 (legacy), supersedes #1280 follow-up chain
- **Status:** 🟡 **Partially Resolved** (low-mass + Case 900 peak cooling PASS
  per post-#1362 verification; high-mass shading/setback peak cooling tracked
  under LIMIT-05 root-cause investigation)
- **Phase Addressed:** Phase 7A (legacy partial), #1323 + #1367 + #1368 +
  #1362 + #1392 + #1394 (post-#1323 refresh)
- **Resolution Notes:**
  - **#1323 (`fix(#1323): restore ASHRAE 140/#1140 corrected constants in
    roof-solar`):** restored the #1140 film coefficient and solar absorptance
    constants into the roof-solar path. ARCHITECTURE.md §Current Module Status
    marks anything pre-#1323 as obsolete.
  - **#1367 (`feat(#1334): re-train Surrogate v3.1 against post-#1323 physics
    outputs`):** re-trained the v3.1 surrogate against the corrected roof-solar
    path; all surrogate-driven Case 600/650/900 numbers now reflect the
    post-#1323 baseline.
  - **#1368 (`feat(#1333): wire strict ±15 % annual-energy CI gate`):** made
    the ±15 % band the release-blocking gate, exposing any drift as a CI failure
    rather than a doc-only discrepancy.
  - **#1362 (`test(#1328): verify Case 900 peak cooling closes to ASHRAE 140
    band`):** closed Case 900 peak cooling (1.95 kW vs ref 1.60-2.10 kW).
  - **#1392 (`fix(surface-flux-provider): surface_heat_flux must be query-only,
    not mutating`):** removed a hidden mutation that was double-counting
    per-surface solar into the 5R1C air node, which suppressed apparent peak
    cooling on the 600 series.
  - **#1394 (`perf(solar): hoist calculate_solar_position out of 5R1C
    orientation lookup`):** the solar-position hoist eliminated a subtle
    wall-clock vs wall-clock-of-day inconsistency that masqueraded as solar
    under-counting in some Case 920/930 profiles.
  - **Low-mass status (post-#1362, post-#1392):** Case 600 peak cooling is
    within reference (per the **authoritative reference** in
    `benchmark.rs:124-127`: peak_cooling 4.8-6.2 kW, ±15 % accept band
    4.675-6.325 kW; the Case 600 reference CSV
    `tests/reference_data/zone_balance/case_600_energy_reference.csv` is
    unified to this value per #1421). Engine output reported in
    `docs/ASHRAE140_RESULTS.md` Case 600 row is 3.09 kW — below the
    band by ~36 % — which is tracked under #1421's Case 600 ref-range drift
    (now resolved) and the LIMIT-05 discrete-node solar-injection pathology.
    The empirical `c_corr` corrections listed in LIMIT-06 below are calibrated
    to the **pre-#1270** Case 600 reference (8.00-10.50 MWh cooling) and do
    not apply to the post-#1270 band; LIMIT-06 itself is marked open in the
    issue tracker pending re-calibration.
  - **High-mass status (post-#1362, Case 900 only):** Case 900 peak cooling
    PASSES the post-#1408 reconciled 1.60-2.10 kW band. Cases 920, 930, 940,
    950, 960 peak cooling still under-predict and are tracked under LIMIT-05
    + the #1280 roof-solar investigation.

**Phase 7A Findings (kept for historical traceability):**
1. Original behavior: `solar_distribution_to_air = 0.0` meant all radiative loads went to surface, but also meant solar had limited direct-to-air contribution
2. Tested approach: Decoupled internal radiative (now always 100% to surface) and adjusted solar_distribution_to_air for peak cooling
3. Test results with mass-specific values:
   - Low-mass (600 series): 0.7 solar-to-air → Peak C ≈ 5.8 kW (still under: 8.0-10.5 kW reference)
   - High-mass (900 series): 0.3 solar-to-air → Peak C ≈ 4.6 kW (still under: 2.1-3.7 kW reference)
4. Test approach 2: Added solar directly to phi_ia (air node) with solar_distribution_to_air parameter
5. Issue persists: Peak cooling still underpredicted for both mass classes
6. Root cause hypothesis: The problem may be deeper than just solar distribution parameters. Possible factors:
   - Solar gain calculation itself may be incorrect
   - Thermal mass dynamics (Cm values, time constants) may be wrong
   - Convective/radiative split (currently fixed at 40%/60%) may need adjustment
   - Window U-value application may need review

**Next Steps Required (post-#1323):**
1. Re-validate Case 600/650 peak cooling against the **authoritative
   reference** in `benchmark.rs:124-127` (peak_cooling 4.8-6.2 kW); the
   per-Case 600 number reported in `ASHRAE140_RESULTS.md` (3.09 kW) sits
   ~36 % below the post-#1270 band — a solver-level gap tracked under
   LIMIT-05 (discrete-node solar-injection pathology).
2. Continue LIMIT-05 / #1280 roof-solar follow-up to close Cases 920/930/940/
   950 peak cooling. The `MassAirCouplingMode::ParallelResistance` shipped in
   #1281 is the architecturally-correct 9R4C coupling but is **not** the cooling
   fix (per ARCHITECTURE.md:406 — Python verification shows parallel-resistance
   actually *lowers* peak cooling).
3. Detailed comparison with EnergyPlus hourly data to identify specific discrepancies
4. Review of solar gain calculation algorithm (beam vs diffuse distribution)
5. Validation of thermal mass capacitance calculations
6. Potential need for more sophisticated solar model (e.g., multi-zone, view factors)
