//! ASHRAE 140 material property constants.
//!
//! This module is the **single source of truth** for ALL ASHRAE 140 material
//! properties used throughout fluxion. Both the CTF solver and construction
//! builders **must** import from here — never hardcode these values elsewhere.
//!
//! # Properties Source
//!
//! All material properties are taken directly from:
//! - **ASHRAE Standard 140 Table B1-3** — envelope material properties
//! - **ASHRAE Standard 140 Section 5.2** — surface film coefficients
//! - **ASHRAE Standard 140 Table B1-1** — building geometry and window specs

// ============================================================================
// Surface film coefficients — ASHRAE 140 Section 5.2
// ============================================================================

/// Interior surface film coefficient per ASHRAE 140 Section 5.2.
/// **Value:** 8.29 W/m²K
pub const ASHRAE140_H_INT: f64 = 8.29;

/// Exterior surface film coefficient per ASHRAE 140 Section 5.2 at 6.7 m/s design wind.
/// **Value:** 29.3 W/m²K
pub const ASHRAE140_H_EXT: f64 = 29.3;

/// Interior surface thermal resistance. Value: 1/8.29 ≈ 0.12063 m²K/W
pub const ASHRAE140_R_INT: f64 = 1.0 / ASHRAE140_H_INT;

/// Exterior surface thermal resistance. Value: 1/29.3 ≈ 0.03413 m²K/W
pub const ASHRAE140_R_EXT: f64 = 1.0 / ASHRAE140_H_EXT;

// ============================================================================
// 900-Series heavyweight concrete (ASHRAE 140 Table B1-3)
// ============================================================================

/// Heavyweight concrete k = 0.51 W/mK (medium-density block, NOT normal-weight 1.4)
pub const HW_CONCRETE_K: f64 = 0.51;
/// Heavyweight concrete rho = 1400 kg/m3 (NOT normal-weight 2300)
pub const HW_CONCRETE_RHO: f64 = 1400.0;
/// Heavyweight concrete Cp = 840 J/kgK
pub const HW_CONCRETE_CP: f64 = 840.0;
/// Heavyweight concrete wall thickness for 900-series: 0.200 m
pub const HW_CONCRETE_THICKNESS: f64 = 0.200;
/// Thermal mass per unit area: rho*Cp*d = 1400*840*0.200 = 235,200 J/m2K
pub const HW_CONCRETE_KAPPA: f64 = HW_CONCRETE_RHO * HW_CONCRETE_CP * HW_CONCRETE_THICKNESS;

// ============================================================================
// 900-Series foam board insulation (ASHRAE 140 Table B1-3)
// ============================================================================

/// Foam board k = 0.040 W/mK
pub const FOAM_BOARD_K: f64 = 0.040;
/// Foam board rho = 10 kg/m3
pub const FOAM_BOARD_RHO: f64 = 10.0;
/// Foam board Cp = 1400 J/kgK (higher than fibreglass)
pub const FOAM_BOARD_CP: f64 = 1400.0;
/// Foam board thickness: 0.0615 m
pub const FOAM_BOARD_THICKNESS: f64 = 0.0615;

// ============================================================================
// 600/900-Series wood siding (ASHRAE 140 Table B1-3)
// ============================================================================

/// Wood siding k = 0.14 W/mK
pub const WOOD_SIDING_K: f64 = 0.14;
/// Wood siding rho = 530 kg/m3
pub const WOOD_SIDING_RHO: f64 = 530.0;
/// Wood siding Cp = 900 J/kgK (NOT 840)
pub const WOOD_SIDING_CP: f64 = 900.0;
/// Wood siding thickness: 0.009 m
pub const WOOD_SIDING_THICKNESS: f64 = 0.009;

// ============================================================================
// 600-Series fiberglass batt insulation (ASHRAE 140 Table B1-3)
// ============================================================================

/// Fiberglass batt k = 0.040 W/mK
pub const FIBREGLASS_BATT_K: f64 = 0.040;
/// Fiberglass batt rho = 12 kg/m3
pub const FIBREGLASS_BATT_RHO: f64 = 12.0;
/// Fiberglass batt Cp = 840 J/kgK
pub const FIBREGLASS_BATT_CP: f64 = 840.0;

// ============================================================================
// 600-Series gypsum board (ASHRAE 140 Table B1-3)
// ============================================================================

/// Gypsum board k = 0.16 W/mK
pub const GYPSUM_K: f64 = 0.16;
/// Gypsum board rho = 784 kg/m3 (ASHRAE 140 spec; was 960 — corrected per GH#754)
pub const GYPSUM_RHO: f64 = 784.0;
/// Gypsum board Cp = 840 J/kgK
pub const GYPSUM_CP: f64 = 840.0;
/// Gypsum board thickness: 0.012 m
pub const GYPSUM_THICKNESS: f64 = 0.012;

// ============================================================================
// Building geometry — ASHRAE 140 Table B1-1
// ============================================================================

/// Building E-W width: 8.0 m
pub const BUILDING_WIDTH_M: f64 = 8.0;
/// Building N-S depth: 6.0 m
pub const BUILDING_DEPTH_M: f64 = 6.0;
/// Building height: 2.7 m
pub const BUILDING_HEIGHT_M: f64 = 2.7;
/// Total wall area: 2(8+6)*2.7 = 75.6 m2
pub const TOTAL_WALL_AREA_M2: f64 = 2.0 * (BUILDING_WIDTH_M + BUILDING_DEPTH_M) * BUILDING_HEIGHT_M;
/// South window area per ASHRAE 140 Table B1-1: 12.0 m2
pub const SOUTH_WINDOW_AREA_M2: f64 = 12.0;
/// Net opaque wall area: 75.6 - 12.0 = 63.6 m2
pub const OPAQUE_WALL_AREA_M2: f64 = TOTAL_WALL_AREA_M2 - SOUTH_WINDOW_AREA_M2;
/// Floor/roof area: 8*6 = 48 m2
pub const FLOOR_AREA_M2: f64 = BUILDING_WIDTH_M * BUILDING_DEPTH_M;

// ============================================================================
// Surface optical properties — ASHRAE 140 Table B1-3
// ============================================================================

/// Exterior surface solar absorptance: 0.6 (medium-color)
pub const EXTERIOR_SURFACE_ABSORPTANCE: f64 = 0.6;
/// Surface long-wave emissivity: 0.9
pub const SURFACE_EMISSIVITY: f64 = 0.9;
/// Window SHGC (double-pane clear glass): 0.787
pub const WINDOW_SHGC: f64 = 0.787;
/// Window U-value: 3.0 W/m2K
pub const WINDOW_U_VALUE: f64 = 3.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_concrete_kappa() {
        let expected = 1400.0 * 840.0 * 0.200;
        assert!((HW_CONCRETE_KAPPA - expected).abs() < 1.0);
    }

    #[test]
    fn test_opaque_wall_area() {
        let expected = 2.0 * (8.0 + 6.0) * 2.7 - 12.0;
        assert!((OPAQUE_WALL_AREA_M2 - expected).abs() < 0.01);
    }

    #[test]
    fn test_film_resistances() {
        assert!((ASHRAE140_R_INT - 1.0 / 8.29).abs() < 1e-6);
        assert!((ASHRAE140_R_EXT - 1.0 / 29.3).abs() < 1e-6);
    }

    #[test]
    fn test_hw_concrete_is_not_normal_weight() {
        const { assert!(HW_CONCRETE_K < 1.0, "k=0.51 not 1.4") };
        const { assert!(HW_CONCRETE_RHO < 2000.0, "rho=1400 not 2300") };
    }
}
