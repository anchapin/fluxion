//! ASHRAE 140 regression test suite
//!
//! Tests run the full ASHRAE 140 validation (18 cases) to detect regressions.
//! This is the comprehensive regression test that runs nightly.

// TODO: Implement comprehensive regression test
// Test should:
// - Run all 18 ASHRAE 140 cases
// - Check for regressions from baseline (Case 195, 600, 620, 900, 960)
// - Generate markdown report for CI
// - Fail on regressions in critical cases (195, 600, 620)
// - Log warnings for non-critical cases (900-series still being calibrated)
