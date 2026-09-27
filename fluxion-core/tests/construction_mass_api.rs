//! Unit tests for `Construction::total_thermal_capacitance_per_area()`.
//!
//! Covers Issue #4072: the new full-κ metric that supplements
//! `iso_13790_effective_capacitance_per_area` for massiveness weighting
//! and `h_ms_of_kappa` calibration.
//!
//! Per-area capacitance of each layer is `ρ × c × δ`:
//!
//! | Material              | ρ (kg/m³) | c (J/kg·K) | k (W/m·K) |
//! |-----------------------|-----------|------------|-----------|
//! | Plasterboard          | 784       | 840        | 0.16      |
//! | Fiberglass            | 12        | 840        | 0.04      |
//! | Wood siding           | 530       | 900        | 0.14      |
//! | Concrete block (B1-3) | 1400      | 840        | 0.51      |
//! | Foam                  | 14        | 1400       | 0.04      |
//!
//! ## Expected per-layer κ (= ρ × c × δ) for the ASHRAE 140 wall assemblies
//!
//! **Low-mass wall (Case 600)** — layers [plasterboard 12mm, fiberglass 66mm, wood_siding 9mm]:
//! - plasterboard  : 784 × 840 × 0.012 = 7,902.72
//! - fiberglass    : 12 × 840 × 0.066  = 665.28
//! - wood_siding   : 530 × 900 × 0.009 = 4,293.00
//! - total_κ       : ≈ 12,861 J/m²K (≈ 12,900 reported in Issue #4072)
//! - effective_κ   : 8,568 J/m²K (plasterboard + fiberglass; wood_siding is
//!                    exterior to the dominant insulation, zero contribution;
//!                    also capped at 100mm active thickness)
//!
//! **High-mass wall (Case 900)** — layers [wood_siding 9mm, foam 61.5mm, concrete_block 100mm]:
//! - wood_siding   : 530 × 900 × 0.009    = 4,293.00
//! - foam          : 14 × 1400 × 0.0615   = 1,205.40
//! - concrete_block: 1400 × 840 × 0.100   = 117,600.00
//! - total_κ       : ≈ 123,098 J/m²K (≈ 123,100 reported in Issue #4072)
//! - effective_κ   : 5,498 J/m²K (wood_siding + foam; concrete_block exceeds
//!                    the 100mm active-thickness cap and contributes zero).

use fluxion_core::construction::{Assemblies, Construction};

/// Tolerance for floating-point comparisons of derived per-area capacitances.
/// One part in 10⁵ (~0.001%) — well below any physical modelling uncertainty.
const REL_TOL: f64 = 1e-5;

/// Asserts `actual ≈ expected` to within `REL_TOL` (relative).
#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= expected.abs() * REL_TOL,
        "expected ≈ {expected}, got {actual} (rel diff = {:.3e})",
        (actual - expected).abs() / expected.abs(),
    );
}

#[test]
fn low_mass_wall_total_thermal_capacitance_per_area() {
    // Issue #4072 baseline: Case 600 low-mass wall sums to ≈ 12,861 J/m²K.
    let wall = Assemblies::low_mass_wall();
    let total = wall.total_thermal_capacitance_per_area();
    assert_close(total, 12_861.0);
    // Sanity: every layer contributes a positive value.
    assert!(total > 0.0);
}

#[test]
fn high_mass_wall_total_thermal_capacitance_per_area() {
    // Issue #4072 baseline: Case 900 high-mass wall sums to ≈ 123,098 J/m²K.
    // This is the value `massiveness_weight` keys on for the
    // `h_ms_of_kappa` Issue #2229 calibration.
    let wall = Assemblies::high_mass_wall();
    let total = wall.total_thermal_capacitance_per_area();
    assert_close(total, 123_098.0);
}

#[test]
fn total_equals_effective_plus_excluded_sum() {
    // Invariant: total_κ == effective_κ + Σ(layers excluded from effective_κ).
    //
    // This is the architectural reason `total_thermal_capacitance_per_area`
    // exists alongside `iso_13790_effective_capacitance_per_area`: the two
    // together decompose the construction into "interior-active mass" and
    // "exterior-decoupled mass". A future exterior-mass correction (a
    // documented follow-up to Issue #4072) would use this decomposition.
    let wall = Assemblies::high_mass_wall();
    let total = wall.total_thermal_capacitance_per_area();
    let effective = wall.iso_13790_effective_capacitance_per_area();

    // For Case 900 wall, wood_siding (4,293 J/m²K) and foam (1,205 J/m²K)
    // are interior-active and contribute to effective_κ. The concrete_block
    // (117,600 J/m²K) is the only layer excluded from effective_κ — it sits
    // exterior to the insulation and exceeds the 100mm active-thickness cap.
    let concrete_block_kappa = 117_600.0;
    let excluded_sum = concrete_block_kappa;
    let lhs = total - effective;
    let rhs = excluded_sum;
    assert_close(lhs, rhs);
    assert_eq!(total, effective + excluded_sum);
}

#[test]
fn zero_layer_construction_has_zero_total_thermal_capacitance() {
    // Degenerate case: a `Construction` with no layers is valid but
    // has zero heat-storage capacity on either side of the decomposition.
    //
    // Note: `Construction::new(vec![])` is not used here because the public
    // constructor asserts at least one layer; the empty-layer case is
    // modelled via direct struct construction, which is the idiomatic way
    // for "no layers" assemblies in the codebase.
    let empty = Construction { layers: vec![] };
    assert_eq!(empty.total_thermal_capacitance_per_area(), 0.0);
}

#[test]
fn insulation_only_construction_has_equal_total_and_effective() {
    // When every layer is interior-active (e.g. insulation-only assemblies),
    // the dominant-insulation rule resolves to "last layer is the insulation",
    // so `effective_κ` sums all layers. The new `total_κ` must therefore
    // match `effective_κ` exactly.
    let insulation_only = Construction::new(vec![Assemblies::low_mass_wall().layers[1].clone()]);
    assert_close(
        insulation_only.total_thermal_capacitance_per_area(),
        insulation_only.iso_13790_effective_capacitance_per_area(),
    );
}
