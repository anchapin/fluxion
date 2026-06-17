# ASHRAE 140 Reference Range Data

This directory contains the inter-program comparison reference ranges used for ASHRAE 140 Section 7 pass/fail validation.

## Files

| File | Contents | Source |
|-----|---------|-------|
| `section7_loads.json` | Annual heating/cooling (MWh) and peak loads (kW) for all Section 7 cases | NREL/TP-472-6231 Tables 3-2 to 3-4; target: ASHRAE 140-2023 Tables 7-2 to 7-8 |
| `section7_freeflot.json` | Annual max/min zone temperatures (°C) for free-floating cases | NREL/TP-472-6231 Table 3-5; target: ASHRAE 140-2023 Table 8-2 |

## Pass/Fail Methodology

Per **ASHRAE 140-2023 Section 1.5**, a simulation result passes if it falls within the published inter-program range `[min, max]`. No additional tolerance is applied — the range IS the acceptance criterion.

```
PASS if: ref_min ≤ result ≤ ref_max
FAIL if: result < ref_min  OR  result > ref_max
```

See `src/validation/ashrae_140_validator.rs` and issue [#723](https://github.com/anchapin/fluxion/issues/723) for the comparator fix.

## Provisional Values

Any entry with `"provisional": true` **must be verified** against ASHRAE140-2023 Tables 7-2 to 8-2 before the fluxion compliance report can be submitted.

Current provisional entries:
- **Cases 910, 920** (annual loads and peak loads): Derived from 1995 BESTEST delta tables; ranges are estimates
- **Part I Diagnostic Cases (195-395)<: Annual loads present, peak loads missing
- **Cases 640, 940
** (annual cooling): Not available in the 1995 report tables reviewed

## Sourcing

### Current Source (Provisional)
Judkoff, R. & Neymark, J. (1995). *Building Energy Simulation Test (BESTEST) and Diagnostic Method*. NREL/TP-472-6231.
Free download: https://www.nrel.gov/docs/legosti/old/6231.pdf

### Target Source (Required for Compliance Submission)
**ASHRAE Standard 140-2023** — purchase from https://www.ashrae.org
Tables 7-2 through 7-8 (loads), Table 8-2 (free-float temperatures)

### Alternative Free Source for 2017+ Reference Program Outputs
Run EnergyPlus 23.x against the official ASHRAE 140 test suite.

## Integrity Verification

To prevent accidental modification, a SHA-256 hash check is recommended in CI:

```rust
#[test]
fn reference_data_files_not_modified() {
    let expected_sha256_loads = "TODO: compute after finalized";
    // verify file hashes match expected at test startup
}
```

## Update Policy

1. When ASHRAE 140 is revised, create a new versioned subdirectory
2. All `provisional: true` entries must be resolved before the next compliance milestone
