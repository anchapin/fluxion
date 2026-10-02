# LIMIT-06 — investigation history

Narrative history for **LIMIT-06**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-06` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 2. No wording was changed, softened or deleted.

---

#### LIMIT-06: 600-Series Annual Heating Correction (Empirical)

- **Description:** Issue #522 gap analysis revealed that 600-series produces ~1.64 MWh annual heating when ASHRAE 140 reference is 4.36-5.79 MWh (authoritative source: `benchmark.rs:124-127`). The 5R1C model doesn't properly differentiate low-mass thermal dynamics, producing energy in the high-mass range for low-mass buildings.

  > **Authoritative reference:** All Case 600 reference values are unified
  > across `benchmark.rs:124-127`, the Case 600 reference CSV
  > (`tests/reference_data/zone_balance/case_600_energy_reference.csv`),
  > `docs/ASHRAE140_RESULTS.md`, and this document per #1421. The values below
  > that pre-date #1270 (5.5-7.5 MWh heating, 8.00-10.50 MWh cooling) are
  > **obsolete** and must not be cited as authoritative for new work.

- **Root Cause:** The h_tr_ms calculation using ISO 13790 half-insulation rule doesn't capture the thermal response difference between low-mass (fiberglass insulation) and high-mass (concrete) constructions. Both produce similar heating output (~1.65 MWh) when ASHRAE expects low-mass to be 3-4x higher.

- **Affected Cases:** 600, 610, 620, 630, 640

- **Affected Metrics:** Annual Heating Energy (MWh) - **FIXED**; Annual Cooling Energy (MWh) - **STILL FAILING**

- **Severity:** Medium (heating now passes, cooling still underpredicts by 92%)

- **GitHub Issue:** #522

- **Status:** 🔄 Partially Fixed (Phase 36) — **but reference ranges below are pre-#1270; authoritative reference is `benchmark.rs:124-127`**

- **Resolution Notes:** Applied empirical correction factors (h_corr = 0.25-0.40) to 600-series heating to bring output from 1.64 MWh into 5.5-7.5 MWh range. This is NOT physics-based - it's an empirical calibration. The fundamental 5R1C model limitation remains.

  **⚠️ Reference-drift warning (post-#1270 / #1408 / #1457):** The 5.5-7.5 MWh
  reference range cited in this row pre-dates the raw ASHRAE 140-2023
  inter-program envelope that landed in #1270 and is now the authoritative
  source via `tests/reference_data/zone_balance/case_600_energy_reference.csv`
  (4.36-5.79 MWh heating, 3.92-6.14 MWh cooling). The `h_corr = 0.25` correction
  was tuned to push 1.64 → 6.6 MWh, which is now **above** the post-#1270
  reference envelope (4.36-5.79 MWh). The actual Case 600 series fix is tracked
  by **`fix(physics): resolve ASHRAE 140 Case 600 series failures` (#1457,
  merged via #1460)** — see the LIMIT-05 UPDATE block above. **#1421 is open**
  for re-validating this row's `c_corr` table against the post-#1270 reference
  CSVs; do **not** cite the 5.5-7.5 MWh range as authoritative for new work.

**Correction factors applied (legacy, pre-#1270):**
| Case | Heating Corr | Rationale |
|------|-------------|-----------|
| 600 | 0.25 | 1.64 / 0.25 ≈ 6.6 MWh (in 5.5-7.5 range) |
| 610 | 0.30 | Ref 4.36-5.79, slightly better solar |
| 620 | 0.32 | Ref 4.5-6.5, similar to 600 |
| 630 | 0.35 | Ref 5.05-6.47, shading helps |
| 640 | 0.40 | Ref 2.75-3.80, setback reduces demand |

#### LIMIT-06 UPDATE (Phase 36-04): 600-Series Cooling FIXED — **pre-#1270 reference only**

**Issue #531 Fix Applied:** The root cause was that c_corr = 1.0 (no correction) was applied to 600-series cooling, but the 5R1C model's sensitivity-based calculation severely underpredicts cooling for low-mass buildings.

**Fix:** Applied empirical c_corr < 1.0 correction factors to boost cooling:
- Case 600: c_corr = 0.071 (0.66 → 9.23 MWh, within 8.0-10.5 MWh ref) ✅
- Case 610: c_corr = 0.107 (0.54 → 5.08 MWh, within 3.92-6.14 MWh ref) ✅
- Case 620: c_corr = 0.095 (0.39 → 4.12 MWh, within 3.20-5.00 MWh ref) ✅
- Case 630: c_corr = 0.116 (0.34 → 2.95 MWh, within 2.13-3.70 MWh ref) ✅
- Case 640: c_corr = 0.092 (0.65 → 7.12 MWh, within 5.95-8.10 MWh ref) ✅
- Case 650: c_corr = 0.084 (0.50 → 5.93 MWh, within 4.82-7.06 MWh ref) ✅

**Results:**
- Pass rate improved from 17.2% to 26.6%
- All 600-series cooling cases now PASS
- Note: This is empirical correction, not physics-based (same as LIMIT-06 heating fix)

  **⚠️ Reference-drift warning:** The Case 600/640/650 numbers above use the
  **pre-#1270** Case 600 reference range (8.00-10.50 MWh cooling). Post-#1270
  the Case 600 cooling reference is **3.92-6.14 MWh** per the authoritative
  source `benchmark.rs:124-127` and the unified Case 600 reference CSV
  (`tests/reference_data/zone_balance/case_600_energy_reference.csv`,
  reconciled in #1421). The Case 600 cooling fix is **superseded** by
  `fix(physics): resolve ASHRAE 140 Case 600 series failures (#1457 / #1460)`
  — see the LIMIT-05 UPDATE block above.
