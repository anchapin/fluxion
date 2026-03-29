//! 6R2C Heat Flow Balance Analysis
//!
//! Analyzes heat flow through 6R2C network to find imbalance.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};

fn main() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== 6R2C Heat Flow Balance Analysis ===");
    println!();

    // Key conductances
    let h_tr_em = model.h_tr_em[0]; // Envelope to exterior
    let h_tr_me = model.h_tr_me[0]; // Envelope to internal
    let h_tr_ms = model.h_tr_ms[0]; // Mass to surface
    let h_tr_is = model.h_tr_is[0]; // Surface to air
    let h_tr_w = model.h_tr_w[0]; // Windows
    let h_ve = model.h_ve[0]; // Ventilation

    println!("--- Conductance Ratios ---");
    println!("h_tr_em (env->ext): {:.2} W/K", h_tr_em);
    println!("h_tr_me (env->int): {:.2} W/K", h_tr_me);
    println!("h_tr_ms (mass->surf): {:.2} W/K", h_tr_ms);
    println!("h_tr_is (surf->air): {:.2} W/K", h_tr_is);
    println!();

    // Analyze heat loss paths
    let env_to_ext_ratio = h_tr_em / (h_tr_ms + h_tr_is);
    println!(
        "Envelope to Exterior / (Mass to Air) ratio: {:.2}",
        env_to_ext_ratio
    );

    let int_to_env_ratio = h_tr_me / (h_tr_ms + h_tr_is);
    println!(
        "Internal to Envelope / (Mass to Air) ratio: {:.2}",
        int_to_env_ratio
    );
    println!();

    // The problem: If envelope loses heat too fast relative to internal gains,
    // heating demand skyrockets.
    //
    // Heat flow envelope->exterior: Q_out = h_tr_em × (T_env - T_out)
    // Heat flow envelope->internal: Q_transfer = h_tr_me × (T_env - T_int)
    // Heat flow envelope->surface->air: Q_to_air = (h_tr_ms + h_tr_is) × (T_env - T_air)
    //
    // For steady state with internal gains:
    // - Heat from envelope must balance heat loss + internal gains
    // - Q_to_air ≈ Q_out + Q_int_gain
    //
    // If h_tr_em >> (h_tr_ms + h_tr_is):
    // - Envelope loses heat faster than it can supply to interior
    // - Heating demand increases dramatically

    println!("--- Heat Flow Analysis (Steady State) ---");
    println!("At steady state (heating season):");
    println!("  Envelope loses heat to exterior via h_tr_em");
    println!("  Envelope supplies heat to interior via h_tr_ms + h_tr_is");
    println!("  Internal gains add heat to interior");
    println!();
    println!("If h_tr_em is too high:");
    println!("  → Envelope temperature drops below interior");
    println!("  → Massive heat loss to exterior");
    println!("  → Heating demand increases to compensate");
    println!();

    // Calculate critical ratio
    let critical_ratio = h_tr_em / (h_tr_ms + h_tr_is);
    println!(
        "Critical ratio (h_tr_em / (h_tr_ms + h_tr_is)): {:.2}",
        critical_ratio
    );

    if critical_ratio > 0.5 {
        println!("⚠️  WARNING: Envelope loses heat > 50% as fast as it supplies!");
    }
    if critical_ratio > 1.0 {
        println!("⚠️  CRITICAL: Envelope loses heat faster than it supplies!");
    }
    if critical_ratio > 2.0 {
        println!("❌  SEVERE: Envelope heat loss is 2x+ the supply rate!");
    }
    println!();

    // Check if the issue is h_tr_ms being too high
    // If h_tr_ms is high, the envelope mass transfers heat quickly to surface
    // This causes the envelope to lose heat too fast to exterior
    println!("--- h_tr_ms Analysis ---");
    println!("h_tr_ms = {:.2} W/K", h_tr_ms);
    println!("  This controls heat transfer from envelope mass to surface/air");
    println!("  Higher h_tr_ms → Envelope heats/cools surface faster");
    println!("  → Faster envelope response → More heat loss via h_tr_em");
    println!();

    // Time constant comparison
    let c_env = model.envelope_thermal_capacitance[0];
    let tau_to_air = c_env / (h_tr_ms + h_tr_is); // Time for envelope to reach air temp
    let tau_to_ext = c_env / h_tr_em; // Time for envelope to reach outdoor temp

    println!("--- Time Constants ---");
    println!("τ_env->air: {:.2} hours", tau_to_air / 3600.0);
    println!("τ_env->ext: {:.2} hours", tau_to_ext / 3600.0);
    println!(
        "Ratio (τ_env->ext / τ_env->air): {:.2}",
        tau_to_ext / tau_to_air
    );
    println!();

    if tau_to_ext < tau_to_air * 0.1 {
        println!("⚠️  WARNING: Envelope responds 10x faster to exterior than interior!");
    }
}
