// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT
//
// [AI-assisted] Physics-loop diagnostic (NOT shipped): production-validator
// annual H/C for a set of ASHRAE 140 cases, kWh, for old→new comparisons.

use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::ashrae_140_validator::ASHRAE140Validator;

fn main() {
    let cases = [
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case620,
        ASHRAE140Case::Case630,
        ASHRAE140Case::Case900,
    ];
    for case in cases {
        let mut validator = ASHRAE140Validator::new();
        let (report, _diag) = validator.validate_single_case_with_diagnostics(case.clone());
        let id = case.number();
        for r in &report.results {
            println!(
                "case {} {:?}: {:.2} kWh (band {:.0}–{:.0}) {:?}",
                id,
                r.metric,
                r.fluxion_value * 1000.0,
                r.ref_min * 1000.0,
                r.ref_max * 1000.0,
                r.status
            );
        }
    }
}
