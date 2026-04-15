// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! ASHRAE 140 Validation Runner
//!
//! This binary runs the full ASHRAE 140 validation suite and outputs
//! the results to docs/ASHRAE140_RESULTS_v0.8.0.md and console.
//!
//! # v0.8.0 Changes
//! - Validates peak load fixes from Phase 34
//! - Validates free-floating temperature improvements from Phase 35
//! - Output file: docs/ASHRAE140_RESULTS_v0.8.0.md

use fluxion::validation::ashrae_140_validator::ASHRAE140Validator;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Running Full ASHRAE 140 Validation Suite ===");
    println!("Milestone: v0.8 Peak Load & Free-Float Validation");
    println!("Phase: 36 v0.8.0 Release");
    println!("------------------------------------------------");

    // Initialize validator
    let validator = ASHRAE140Validator::new();

    // Run full validation suite (analytical engine)
    println!("Executing 18 ASHRAE 140 cases with CTF/FD enabled...");
    let report = validator.validate_analytical_engine();

    // Print summary to console
    println!("\nValidation Results Summary:");
    report.print_summary();

    // Save to docs/ASHRAE140_RESULTS_v0.8.0.md
    let output_path = "docs/ASHRAE140_RESULTS_v0.8.0.md";
    println!("\nSaving report to {}...", output_path);
    report.save_to_file(Path::new(output_path))?;

    println!("Full validation complete.");

    Ok(())
}
