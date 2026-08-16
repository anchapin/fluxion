//! Hand-verification of H_total for ASHRAE 140 Case 600 against first-principles calculation.
//!
//! This test creates a Case 600 model, extracts all 5R1C conductance values,
//! and compares them against a hand-calculated reference derived from the
//! ASHRAE 140 construction specifications.
//!
//! ## Case 600 Specifications (ASHRAE 140-2023)
//!
//! - External dimensions: 8m × 6m × 2.7m
//! - South-facing window: 12 m² (double clear glass)
//! - Wall: plasterboard(12mm) + fiberglass(66mm) + wood_siding(9mm)
//! - Roof: plasterboard(10mm) + fiberglass(111.8mm) + roof_deck(19mm)
//! - Floor: timber(25mm) + fiberglass(197mm) (insulated slab-on-grade)
//! - Infiltration: 0.5 ACH
//! - HVAC setpoints: heating 20°C, cooling 27°C
//!
//! ## Hand-Calculated Reference Values
//!
//! ### Geometry
//! - Floor area: 48.0 m²
//! - Volume: 129.6 m³
//! - Total wall area: 75.6 m²
//! - Opaque wall area: 63.6 m²
//! - Window area: 12.0 m²
//! - Roof area: 48.0 m²
//!
//! ### U-values (with ASHRAE 140 v2023 film coefficients)
//! - Wall:  U = 0.512 W/m²K (R = 1.954 m²K/W)
//! - Roof:  U = 0.320 W/m²K (R = 3.127 m²K/W)
//! - Floor: U = 0.184 W/m²K (R = 5.444 m²K/W, includes R_ground=0.17)
//! - Window: U = 2.10 W/m²K
//!
//! ### Simple UA Approach
//! - H_wall   = 0.512 × 63.6 = 32.56 W/K
//! - H_roof   = 0.320 × 48.0 = 15.35 W/K
//! - H_floor  = 0.184 × 48.0 =  8.82 W/K
//! - H_window = 2.10  × 12.0 = 25.20 W/K
//! - H_ve     = ρ·cp·(ACH/3600)·V = 1.2×1005×(0.5/3600)×129.6 = 21.71 W/K
//! - H_total (simple) = 103.63 W/K
//!
//! ### ISO 13790 5R1C Conductances (what the model computes)
//! - h_tr_is  = 63.6×7.69 + 48.0×10.0 + 48.0×5.88 = 1251.3 W/K
//! - h_tr_ms  = 9.1 × A_m = 9.1 × (2.5 × 48.0) = 1092.0 W/K
//!   (low-mass: κ_wall=12,861 < 165,000 → A_m = 2.5 × floor_area per ISO 13790 Table C.2)
//! - h_op_walls_roof = U_wall(None,None)×63.6 + U_roof(None,None)×48.0 = 47.96 W/K
//!   (U(None,None) uses default h_int=8.29, h_ext=29.3)
//! - h_tr_em  = 1/(1/h_op - 1/h_ms) = 1/(1/47.96 - 1/1092.0) = 50.17 W/K
//! - h_opaque (5R1C series) = 1/(1/1251.3 + 1/1092.0 + 1/50.17) = 46.19 W/K
//! - h_tr_w   = 2.10 × 12.0 = 25.20 W/K
//! - h_ve     = 21.71 W/K
//! - **H_total (5R1C) = 46.19 + 25.20 + 21.71 = 93.10 W/K**
//!
//! ### Thermal Capacitance
//! - κ_wall  = 784×840×0.012 + 12×840×0.066 + 530×900×0.009 = 12,861 J/m²K
//! - κ_roof  = 784×840×0.010 + 12×840×0.1118 + 500×1300×0.019 = 20,063 J/m²K
//! - κ_floor = 600×1600×0.025 + 12×840×0.197 = 25,986 J/m²K
//! - Cm = κ_wall×63.6 + κ_roof×48.0 + κ_floor×48.0 = 3,028,278 J/K ≈ 3.03 MJ/K

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Tolerance for floating-point comparison (10% for conductances, 5% for geometry-derived)
const CONDUCTANCE_TOL_PCT: f64 = 0.10;

#[test]
fn test_case_600_htotal_hand_verification() {
    // ── Build the model (no warmup needed — conductances are static) ──
    let spec = ASHRAE140Case::Case600.spec();
    let model: ThermalModel<VectorField> = ThermalModel::from_spec(&spec);

    // ── Extract all 5R1C conductances ──
    let h_em = model.conduction.h_tr_em.as_ref()[0];
    let h_ms = model.conduction.h_tr_ms.as_ref()[0];
    let h_is = model.conduction.h_tr_is.as_ref()[0];
    let h_w = model.conduction.h_tr_w.as_ref()[0];
    let h_ve = model.conduction.h_ve.as_ref()[0];
    let h_floor = model.conduction.h_tr_floor.as_ref()[0];
    let cm = model.mass.thermal_capacitance.as_ref()[0];

    // ── Hand-calculated reference values ──

    // Geometry (from CaseBuilder::case_600_baseline → with_dimensions(8.0, 6.0, 2.7))
    let expected_floor_area = 48.0; // 8 × 6
    let expected_volume = 129.6; // 8 × 6 × 2.7
    let expected_wall_area = 75.6; // 2 × (8+6) × 2.7
    let expected_window_area = 12.0; // with_south_window(12.0)
    let expected_opaque_wall = expected_wall_area - expected_window_area; // 63.6
    let expected_roof_area = expected_floor_area; // 48.0

    // Film coefficients (v2023)
    let h_int_wall = 7.69; // W/m²K
    let h_int_ceil = 10.0;
    let h_int_floor = 5.88;
    let h_int_default = 8.29;
    let h_ext = 29.3;

    // Material R-values
    let r_pb12 = 0.012 / 0.16; // 0.075
    let r_fg66 = 0.066 / 0.04; // 1.65
    let r_ws9 = 0.009 / 0.14; // 0.06429
    let r_pb10 = 0.010 / 0.16; // 0.0625
    let r_fg112 = 0.1118 / 0.04; // 2.795
    let r_rd19 = 0.019 / 0.14; // 0.13571
    let r_tim25 = 0.025 / 0.14; // 0.17857
    let r_fg197 = 0.197 / 0.04; // 4.925

    // ── Verify U-values ──
    let u_wall_ref = 1.0 / (1.0 / h_int_wall + r_pb12 + r_fg66 + r_ws9 + 1.0 / h_ext);
    let u_roof_ref = 1.0 / (1.0 / h_int_ceil + r_pb10 + r_fg112 + r_rd19 + 1.0 / h_ext);
    let u_floor_ref = 1.0 / (1.0 / h_int_floor + r_tim25 + r_fg197 + 0.17); // R_ground=0.17
    let u_window_ref = 2.10; // WindowSpec::double_clear_glass()

    // U(None, None) for walls/roof — used in h_op_walls_roof calculation
    let u_wall_none = 1.0 / (1.0 / h_int_default + r_pb12 + r_fg66 + r_ws9 + 1.0 / h_ext);
    let u_roof_none = 1.0 / (1.0 / h_int_default + r_pb10 + r_fg112 + r_rd19 + 1.0 / h_ext);

    // ── h_tr_is: interior film × surface areas ──
    let expected_h_is = expected_opaque_wall * h_int_wall
        + expected_roof_area * h_int_ceil
        + expected_floor_area * h_int_floor;
    // Expected: 63.6×7.69 + 48×10 + 48×5.88 = 489.1 + 480 + 282.2 = 1251.3 W/K

    // ── h_tr_ms: ISO 13790 §7.2.2.2 ──
    let kappa_wall = 784.0 * 840.0 * 0.012 + 12.0 * 840.0 * 0.066 + 530.0 * 900.0 * 0.009;
    let kappa_roof = 784.0 * 840.0 * 0.010 + 12.0 * 840.0 * 0.1118 + 500.0 * 1300.0 * 0.019;
    let kappa_floor = 600.0 * 1600.0 * 0.025 + 12.0 * 840.0 * 0.197;
    let a_m = 2.5 * expected_floor_area; // VeryLight: κ_wall=12,861 < 165,000
    let expected_h_ms = 9.1 * a_m;
    // Expected: 9.1 × 120 = 1092.0 W/K

    // ── h_tr_em: ISO 13790 Eq. 64 ──
    let h_op_walls_roof = u_wall_none * expected_opaque_wall + u_roof_none * expected_roof_area;
    let expected_h_em = 1.0 / (1.0 / h_op_walls_roof - 1.0 / expected_h_ms);
    // Expected: 1/(1/47.96 - 1/1092.0) ≈ 50.17 W/K

    // ── h_tr_w: window conductance ──
    let expected_h_w = u_window_ref * expected_window_area;
    // Expected: 2.10 × 12 = 25.20 W/K

    // ── h_ve: infiltration conductance ──
    let expected_h_ve = 1.2 * 1005.0 * (0.5 * expected_volume / 3600.0);
    // Expected: 21.71 W/K

    // ── h_tr_floor ──
    let expected_h_floor = u_floor_ref * expected_floor_area;
    // Expected: 0.1837 × 48 ≈ 8.82 W/K

    // ── Cm: thermal capacitance ──
    let expected_cm = kappa_wall * expected_opaque_wall
        + kappa_roof * expected_roof_area
        + kappa_floor * expected_floor_area;
    // Expected: ~3,028,278 J/K ≈ 3.03 MJ/K

    // ── H_total for HVAC demand ──
    let h_opaque_5r1c = 1.0 / (1.0 / h_is + 1.0 / h_ms + 1.0 / h_em);
    let h_total_model = h_opaque_5r1c + h_w + h_ve;

    let expected_h_opaque_5r1c =
        1.0 / (1.0 / expected_h_is + 1.0 / expected_h_ms + 1.0 / expected_h_em);
    let expected_h_total = expected_h_opaque_5r1c + expected_h_w + expected_h_ve;
    // Expected: ~93.10 W/K

    // ── Print full diagnostic ──
    eprintln!("\n╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  Case 600 H_total Hand-Verification Diagnostic         ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");

    eprintln!("\n─── Geometry ───");
    eprintln!("  Floor area:       {expected_floor_area:.1} m²");
    eprintln!("  Volume:           {expected_volume:.1} m³");
    eprintln!("  Total wall area:  {expected_wall_area:.1} m²");
    eprintln!("  Window area:      {expected_window_area:.1} m²");
    eprintln!("  Opaque wall area: {expected_opaque_wall:.1} m²");
    eprintln!("  Roof area:        {expected_roof_area:.1} m²");

    eprintln!("\n─── Construction U-values ───");
    eprintln!("  Wall  U (w/ film):    {:.4} W/m²K", u_wall_ref);
    eprintln!("  Wall  U(None,None):   {:.4} W/m²K", u_wall_none);
    eprintln!("  Roof  U (w/ film):    {:.4} W/m²K", u_roof_ref);
    eprintln!("  Roof  U(None,None):   {:.4} W/m²K", u_roof_none);
    eprintln!("  Floor U (w/ film):    {:.4} W/m²K", u_floor_ref);
    eprintln!(
        "  Window U:            {:.2} W/m²K  SHGC: 0.77",
        u_window_ref
    );

    eprintln!("\n─── 5R1C Conductances ───");
    eprintln!("  ┌─────────────┬──────────────┬──────────────┬───────────┐");
    eprintln!("  │ Parameter   │ Hand-calc    │ Model        │ Δ (%)      │");
    eprintln!("  ├─────────────┼──────────────┼──────────────┼───────────┤");
    print_row("h_tr_is", expected_h_is, h_is);
    print_row("h_tr_ms", expected_h_ms, h_ms);
    print_row("h_tr_em", expected_h_em, h_em);
    print_row("h_tr_w", expected_h_w, h_w);
    print_row("h_ve", expected_h_ve, h_ve);
    print_row("h_tr_floor", expected_h_floor, h_floor);
    eprintln!("  └─────────────┴──────────────┴──────────────┴───────────┘");

    eprintln!("\n─── Thermal Capacitance ───");
    eprintln!("  ┌─────────────┬──────────────┬──────────────┬───────────┐");
    eprintln!("  │ Parameter   │ Hand-calc    │ Model        │ Δ (%)      │");
    eprintln!("  ├─────────────┼──────────────┼──────────────┼───────────┤");
    print_row("Cm (J/K)", expected_cm, cm);
    eprintln!("  └─────────────┴──────────────┴──────────────┴───────────┘");
    eprintln!(
        "  Cm = {:.3} MJ/K (hand) vs {:.3} MJ/K (model)",
        expected_cm / 1e6,
        cm / 1e6
    );

    eprintln!("\n─── H_total Breakdown ───");
    eprintln!("  h_opaque (5R1C series) = 1/(1/h_is + 1/h_ms + 1/h_em)");
    eprintln!(
        "    Hand:  1/(1/{:.1} + 1/{:.1} + 1/{:.2}) = {:.2} W/K",
        expected_h_is, expected_h_ms, expected_h_em, expected_h_opaque_5r1c
    );
    eprintln!(
        "    Model: 1/(1/{:.1} + 1/{:.1} + 1/{:.2}) = {:.2} W/K",
        h_is, h_ms, h_em, h_opaque_5r1c
    );
    eprintln!();
    eprintln!(
        "  H_total = h_opaque + h_tr_w + h_ve = {:.2} + {:.2} + {:.2} = {:.2} W/K (hand)",
        expected_h_opaque_5r1c, expected_h_w, expected_h_ve, expected_h_total
    );
    eprintln!(
        "  H_total = h_opaque + h_tr_w + h_ve = {:.2} + {:.2} + {:.2} = {:.2} W/K (model)",
        h_opaque_5r1c, h_w, h_ve, h_total_model
    );

    // Simple UA comparison
    let h_opaque_simple = u_wall_ref * expected_opaque_wall
        + u_roof_ref * expected_roof_area
        + u_floor_ref * expected_floor_area;
    let h_total_simple = h_opaque_simple + expected_h_w + expected_h_ve;
    eprintln!("\n─── Simple UA Comparison ───");
    eprintln!(
        "  H_total (simple UA):  {:.2} W/K  (Σ U·A + window + ve)",
        h_total_simple
    );
    eprintln!(
        "  H_total (5R1C code):  {:.2} W/K  (series h_is,h_ms,h_em + window + ve)",
        h_total_model
    );
    eprintln!(
        "  Ratio 5R1C/simple:    {:.3}",
        h_total_model / h_total_simple
    );

    // ── Solar distribution ──
    eprintln!("\n─── Solar Distribution ───");
    eprintln!(
        "  solar_beam_to_mass_fraction: {:.2}",
        model.solar.solar_beam_to_mass_fraction
    );
    eprintln!(
        "  solar_distribution_to_air:   {:.2}",
        model.solar.solar_distribution_to_air
    );

    // ── Assertions ──
    // All individual conductances should match hand-calc within 10%
    let tol = CONDUCTANCE_TOL_PCT;

    assert!(
        (h_is - expected_h_is).abs() / expected_h_is < tol,
        "h_tr_is: model={h_is:.2}, expected={expected_h_is:.2}, Δ={:.1}%",
        (h_is - expected_h_is).abs() / expected_h_is * 100.0
    );

    assert!(
        (h_ms - expected_h_ms).abs() / expected_h_ms < tol,
        "h_tr_ms: model={h_ms:.2}, expected={expected_h_ms:.2}, Δ={:.1}%",
        (h_ms - expected_h_ms).abs() / expected_h_ms * 100.0
    );

    assert!(
        (h_em - expected_h_em).abs() / expected_h_em < tol,
        "h_tr_em: model={h_em:.2}, expected={expected_h_em:.2}, Δ={:.1}%",
        (h_em - expected_h_em).abs() / expected_h_em * 100.0
    );

    assert!(
        (h_w - expected_h_w).abs() / expected_h_w < tol,
        "h_tr_w: model={h_w:.2}, expected={expected_h_w:.2}, Δ={:.1}%",
        (h_w - expected_h_w).abs() / expected_h_w * 100.0
    );

    assert!(
        (h_ve - expected_h_ve).abs() / expected_h_ve < tol,
        "h_ve: model={h_ve:.2}, expected={expected_h_ve:.2}, Δ={:.1}%",
        (h_ve - expected_h_ve).abs() / expected_h_ve * 100.0
    );

    assert!(
        (h_floor - expected_h_floor).abs() / expected_h_floor < tol,
        "h_tr_floor: model={h_floor:.2}, expected={expected_h_floor:.2}, Δ={:.1}%",
        (h_floor - expected_h_floor).abs() / expected_h_floor * 100.0
    );

    // Cm should be within 10%
    assert!(
        (cm - expected_cm).abs() / expected_cm < tol,
        "Cm: model={cm:.0}, expected={expected_cm:.0}, Δ={:.1}%",
        (cm - expected_cm).abs() / expected_cm * 100.0
    );

    // H_total should be within 10% of hand-calculated
    assert!(
        (h_total_model - expected_h_total).abs() / expected_h_total < tol,
        "H_total: model={h_total_model:.2}, expected={expected_h_total:.2}, Δ={:.1}%",
        (h_total_model - expected_h_total).abs() / expected_h_total * 100.0
    );

    eprintln!("\n─── All assertions passed (±{:.0}%) ───", tol * 100.0);
}

fn print_row(label: &str, expected: f64, actual: f64) {
    let delta_pct = if expected.abs() > 0.001 {
        (actual - expected).abs() / expected * 100.0
    } else {
        0.0
    };
    let marker = if delta_pct > 10.0 { "⚠️" } else { "✓" };
    eprintln!(
        "  │ {:<11} │ {:>10.2}    │ {:>10.2}    │ {:>6.1}% {:1} │",
        label, expected, actual, delta_pct, marker
    );
}
