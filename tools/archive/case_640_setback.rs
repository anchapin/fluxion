//! Case 640 Setback Diagnostic
//! Run: cargo run --release --bin case_640_setback

use fluxion::validation::report::MetricType;
use fluxion::validation::ASHRAE140Validator;

fn main() {
    println!("\n=== Case 640 Setback Diagnostic ===");
    println!("Comparing Case 600 (baseline) vs Case 640 (setback)");
    println!();

    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    // Extract Case 600 and 640 results
    let case_600 = report
        .results
        .iter()
        .find(|r| r.case_id == "600" && matches!(r.metric, MetricType::AnnualHeating));
    let case_640 = report
        .results
        .iter()
        .find(|r| r.case_id == "640" && matches!(r.metric, MetricType::AnnualHeating));

    if let (Some(c600), Some(c640)) = (case_600, case_640) {
        println!("Results:");
        println!(
            "  Case 600 (baseline): {:.2} MWh (Ref: {:.2}-{:.2})",
            c600.fluxion_value, c600.ref_min, c600.ref_max
        );
        println!(
            "  Case 640 (setback):  {:.2} MWh (Ref: {:.2}-{:.2})",
            c640.fluxion_value, c640.ref_min, c640.ref_max
        );
        println!();

        let savings = c600.fluxion_value - c640.fluxion_value;
        let savings_pct = (savings / c600.fluxion_value) * 100.0;

        let ref_savings_min = c600.ref_max - c640.ref_max; // Best case savings
        let ref_savings_pct_min = (ref_savings_min / c600.ref_max) * 100.0;

        let ref_savings_max = c600.ref_min - c640.ref_min; // Worst case savings
        let ref_savings_pct_max = (ref_savings_max / c600.ref_min) * 100.0;

        println!("Heating Savings:");
        println!("  Model: {:.2} MWh ({:.1}%)", savings, savings_pct);
        println!(
            "  Reference: {:.2}-{:.2} MWh ({:.1}%-{:.1}%)",
            ref_savings_min, ref_savings_max, ref_savings_pct_min, ref_savings_pct_max
        );
        println!();

        if savings_pct < ref_savings_pct_min {
            println!(
                "ISSUE: Model savings ({:.1}%) < Reference min ({:.1}%)",
                savings_pct, ref_savings_pct_min
            );
            println!("   Setback is not saving enough energy!");
        } else {
            println!("OK: Savings look reasonable");
        }
        println!();

        // Expected behavior analysis
        println!("Expected Setback Behavior:");
        println!("  - Setback: 23:00-7:00 at 10C (vs 20C normal)");
        println!("  - Duration: 8 hours/day = 33% of day");
        println!("  - Theoretical max savings: ~30-35% (ignoring thermal mass)");
        println!("  - Reference expects: ~40-50% savings (thermal mass 'heat bank' effect)");
        println!();

        println!("Root Cause Hypothesis:");
        println!("  1. 5R1C model has single thermal mass node");
        println!("  2. Cannot capture 'heat bank' effect (mass stores solar, releases at night)");
        println!("  3. Morning recovery heating is too aggressive");
        println!("  4. Zone cools to 14-15C at night, requires significant morning heating");
        println!();

        println!("Potential Fixes:");
        println!("  1. Add h_tr_ms_multiplier for Case 640 (like Case 940)");
        println!("     - Try 0.07-0.15 to slow heat release");
        println!("  2. Reduce morning recovery heating aggressiveness");
        println!("  3. Adjust coupling factors for setback hours");
        println!("  4. Accept as 5R1C model limitation");
    } else {
        println!("ERROR: Could not find Case 600/640 results");
    }
}
