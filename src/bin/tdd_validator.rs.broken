//! TDD Physics Validation Binary
//!
//! This binary runs the Test-Driven Development framework to validate
//! physics accuracy against analytical references and EnergyPlus data.
//!
//! # Usage
//!
//! ```bash
//! # Run all physics domain tests
//! cargo run --bin tdd_validator
//!
//! # Run tests for a specific domain
//! cargo run --bin tdd_validator -- --domain heat-conduction
//!
//! # Generate report to specific file
//! cargo run --bin tdd_validator -- --output reports/tdd_report.md
//!
//! # Use reference data file
//! cargo run --bin tdd_validator -- --reference data/energyplus_references.json
//! ```

use clap::{Parser, ValueEnum};
use fluxion::testing::tdd_framework::{PhysicsDomain, TDDFramework};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "tdd_validator")]
#[command(about = "TDD Physics Validation for Fluxion")]
#[command(long_about = None)]
struct Args {
    /// Physics domain to test (default: all domains)
    #[arg(short, long, value_enum)]
    domain: Option<DomainArg>,

    /// Output file for the markdown report
    #[arg(short, long, default_value = "reports/tdd_physics_report.md")]
    output: PathBuf,

    /// Reference data file (JSON format)
    #[arg(short, long)]
    reference: Option<PathBuf>,

    /// Tolerance for all tests (overrides domain-specific defaults)
    #[arg(short, long)]
    tolerance: Option<f64>,

    /// Fail fast on first error (for debugging)
    #[arg(short, long, default_value = "false")]
    fail_fast: bool,

    /// Verbose output
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

#[derive(ValueEnum, Clone, Debug)]
enum DomainArg {
    HeatConduction,
    SolarRadiation,
    ThermalMass,
    HVACLoads,
    AirExchange,
    InterZoneTransfer,
    GroundCoupling,
    InternalGains,
    WindowHeatTransfer,
    LongwaveRadiation,
}

impl DomainArg {
    fn to_physics_domain(&self) -> PhysicsDomain {
        match self {
            DomainArg::HeatConduction => PhysicsDomain::HeatConduction,
            DomainArg::SolarRadiation => PhysicsDomain::SolarRadiation,
            DomainArg::ThermalMass => PhysicsDomain::ThermalMass,
            DomainArg::HVACLoads => PhysicsDomain::HVACLoads,
            DomainArg::AirExchange => PhysicsDomain::AirExchange,
            DomainArg::InterZoneTransfer => PhysicsDomain::InterZoneTransfer,
            DomainArg::GroundCoupling => PhysicsDomain::GroundCoupling,
            DomainArg::InternalGains => PhysicsDomain::InternalGains,
            DomainArg::WindowHeatTransfer => PhysicsDomain::WindowHeatTransfer,
            DomainArg::LongwaveRadiation => PhysicsDomain::LongwaveRadiation,
        }
    }
}

fn main() {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Fluxion TDD Physics Validation Framework            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create framework
    let mut framework = TDDFramework::new().with_fail_fast(args.fail_fast);

    // Load reference data if provided
    if let Some(ref_path) = &args.reference {
        println!("📚 Loading reference data from: {}", ref_path.display());
        framework = framework.with_reference_data(ref_path);
    }

    // Set global tolerance if provided
    if let Some(tolerance) = args.tolerance {
        println!("🎯 Setting global tolerance: {:.1}%", tolerance * 100.0);
        for domain in PhysicsDomain::all() {
            framework.set_tolerance(domain, tolerance);
        }
    }

    // Run tests
    let suites = if let Some(domain_arg) = &args.domain {
        let domain = domain_arg.to_physics_domain();
        println!("🔬 Running tests for: {:?}", domain);
        println!("   Description: {}", domain.description());
        vec![framework.run_tests(domain)]
    } else {
        println!("🔬 Running all physics domain tests...");
        framework.run_all_tests()
    };

    // Print summary
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("                         TEST SUMMARY                          ");
    println!("═══════════════════════════════════════════════════════════════");

    let mut total_tests = 0;
    let mut total_passed = 0;
    let mut total_failed = 0;
    let mut total_skipped = 0;

    for suite in &suites {
        let summary = suite.summary();
        total_tests += summary.total;
        total_passed += summary.passed;
        total_failed += summary.failed;
        total_skipped += summary.skipped;

        let status_icon = if summary.failed == 0 { "✅" } else { "❌" };
        println!(
            "{} {:<25} {:>3}/{:>3} passed ({:>5.1}%) - Max error: {:.2}%",
            status_icon,
            format!("{:?}", summary.domain),
            summary.passed,
            summary.total,
            summary.pass_rate * 100.0,
            summary.max_error * 100.0
        );
    }

    println!("───────────────────────────────────────────────────────────────");
    let overall_pass_rate = if total_tests > 0 {
        total_passed as f64 / total_tests as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "TOTAL: {}/{} passed ({:.1}%) | Failed: {} | Skipped: {}",
        total_passed, total_tests, overall_pass_rate, total_failed, total_skipped
    );
    println!("═══════════════════════════════════════════════════════════════");

    // Generate report
    println!();
    println!("📝 Generating report: {}", args.output.display());

    // Create output directory if it doesn't exist
    if let Some(parent) = args.output.parent() {
        if !parent.exists() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Warning: Could not create output directory: {}", e);
            }
        }
    }

    framework.generate_report(&suites, args.output.to_str().unwrap());

    // Print detailed results if verbose
    if args.verbose {
        println!();
        println!("═══════════════════════════════════════════════════════════════");
        println!("                      DETAILED RESULTS                         ");
        println!("═══════════════════════════════════════════════════════════════");

        for suite in &suites {
            println!();
            println!("{:?}", suite.domain);
            println!("{}", "─".repeat(60));

            for test in &suite.test_cases {
                let icon = match test.status {
                    fluxion::testing::tdd_framework::TestStatus::Pass => "✓",
                    fluxion::testing::tdd_framework::TestStatus::Fail => "✗",
                    fluxion::testing::tdd_framework::TestStatus::Skipped => "⊘",
                    fluxion::testing::tdd_framework::TestStatus::Error => "!",
                };
                println!(
                    "  {} {} ({}): {:.4} vs {:.4} ({:.2}%)",
                    icon,
                    test.name,
                    test.id,
                    test.computed_value,
                    test.reference_value,
                    test.relative_error * 100.0
                );
            }
        }
    }

    // Exit with appropriate code
    if total_failed > 0 {
        println!();
        println!(
            "⚠️  {} test(s) failed. Review the report for details.",
            total_failed
        );
        std::process::exit(1);
    } else {
        println!();
        println!("🎉 All tests passed!");
        std::process::exit(0);
    }
}
