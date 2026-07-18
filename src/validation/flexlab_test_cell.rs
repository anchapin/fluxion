//! FLEXLAB Test Cell Model for Empirical Validation (Issue #1807)
//!
//! Constructs a Fluxion building model matching the LBNL FLEXLAB test cell
//! geometry, construction, and schedules. This is the "apples-to-apples" model
//! for empirical validation T10.5.
//!
//! # Test Cell Geometry
//!
//! The model matches FLEXLAB test cell X3A per the Modelica Buildings library
//! (`Buildings.ThermalZones.Detailed.FLEXLAB.Rooms.X3A.TestCell`) and
//! architectural drawings. Key dimensions:
//!
//! | Parameter          | Value          | Source                   |
//! |--------------------|----------------|--------------------------|
//! | Floor area         | 60.97 m²       | Modelica AFlo parameter   |
//! | Room height        | 3.6576 m (12ft)| Modelica hRoo parameter   |
//! | South wall width   | 6.6675 m       | datConExtWin area / hRoo  |
//! | North-South depth  | 9.144 m (30ft) | datConBou area / hRoo     |
//! | Window (south)     | 5.88 m × 1.8288 m (10.75 m²) | datConExtWin |
//! | Window sill height | 0.9144 m (3ft) | Architectural drawings    |
//!
//! # Construction
//!
//! - Exterior walls: R16.8 assembly (Construction10and23 in Modelica) ≈ 0.338 W/m²K
//! - Roof: R20 assembly (ASHRAE 90.1-2010 roof) ≈ 0.284 W/m²K
//! - Floor: On-grade concrete slab with ground coupling
//! - Glazing: ASHRAE 90.1 minimally compliant double clear (ASHRAE901Gla)
//!   - Two panes of ID101 glass, 12.7 mm air gap, frame U = 1.4 W/m²K
//!
//! # Model Differences vs Reference Facility
//!
//! | Aspect               | Actual FLEXLAB                 | Fluxion Model               |
//! |----------------------|--------------------------------|------------------------------|
//! | Location             | Berkeley, CA (37.87°N, 122.27°W)| San Francisco EPW proxy     |
//! | Multi-room           | Test cell + closet + electrical | Single zone (main cell only) |
//! | Dividing walls       | Adiabatic to neighbors          | Exterior walls (conservative)|
//! | Radiant floor        | Embedded tubing, 4 circuits     | Slab-on-grade (conductive)   |
//! | Window frame         | Aluminum frame, U=1.4 W/m²K     | Included in SHGC/U-value     |
//! | Internal gains       | Varies by experiment            | Default office loads         |
//! | HVAC                 | Variable air volume + radiant   | Constant setpoints (20/27°C) |
//! | Weather              | On-site sensors                 | TMY from San Francisco       |

use crate::sim::construction::{Construction, Materials};
use crate::validation::ashrae_140_cases::{
    BuildingType, CaseBuilder, CaseSpec, HvacSchedule, InternalLoads, Orientation, WindowSpec,
};

/// FLEXLAB test cell exterior wall construction (R16.8 assembly).
///
/// Matches the Modelica `Construction10and23` used for exterior walls in
/// FLEXLAB test cells. Total R = 16.8 hr·ft²·°F/BTU (≈ 2.96 m²K/W, U ≈ 0.338 W/m²K)
/// including standard ASHRAE film coefficients.
///
/// Layer stack (interior to exterior):
/// 1. Gypsum board (interior finish)
/// 2. Fiberglass insulation (primary thermal resistance)
/// 3. Exterior rigid foam sheathing
/// 4. Exterior cladding (stucco/cement board)
pub fn flexlab_exterior_wall() -> Construction {
    Construction::new(vec![
        Materials::plasterboard(0.012), // Interior gypsum board 12mm
        Materials::fiberglass(0.052),   // Fiberglass insulation 52mm (R16.8 total incl. films)
        Materials::foam(0.050),         // Rigid foam sheathing 50mm
        Materials::wood_siding(0.025),  // Exterior cladding 25mm
    ])
}

/// FLEXLAB test cell roof construction (R20 assembly).
///
/// Matches the Modelica `ASHRAE_901_2010Roof` used for the ceiling of FLEXLAB
/// test cells. Total R = 20 hr·ft²·°F/BTU (≈ 3.52 m²K/W, U ≈ 0.284 W/m²K)
/// including standard ASHRAE film coefficients.
///
/// Layer stack (interior to exterior):
/// 1. Gypsum board (interior finish)
/// 2. Fiberglass insulation (primary thermal resistance)
/// 3. Roof deck
pub fn flexlab_roof() -> Construction {
    Construction::new(vec![
        Materials::plasterboard(0.010), // Interior gypsum board 10mm
        Materials::fiberglass(0.127),   // Fiberglass insulation 127mm (R20 total incl. films)
        Materials::roof_deck(0.019),    // Roof deck 19mm
    ])
}

/// FLEXLAB test cell floor construction (slab-on-grade).
///
/// The actual FLEXLAB test cell has a radiant floor slab with embedded tubing.
/// For this apples-to-apples model, we use a standard concrete slab-on-grade
/// with ground coupling, which is the simplest representation that captures
/// the dominant thermal behavior.
///
/// Note: The radiant tubing and hydronic system are NOT modeled here.
/// This is documented as a known simplification in the model differences table.
pub fn flexlab_floor() -> Construction {
    Construction::new(vec![
        Materials::concrete_slab(0.100),        // 100mm concrete slab
        Materials::insulation_high_mass(0.050), // 50mm rigid insulation under slab
    ])
}

/// Create the FLEXLAB test cell X3A case specification.
///
/// Returns a `CaseSpec` that matches the FLEXLAB test cell geometry,
/// construction, and default schedules as closely as possible within the
/// single-zone 5R1C framework.
///
/// # Model Choices
///
/// - **Single zone**: The FLEXLAB test cell complex includes an attached closet
///   and electrical room, but these are modeled as boundary conditions in the
///   reference Modelica model. We model only the main test cell volume.
/// - **HVAC**: Constant setpoints (20°C heating, 27°C cooling) representing
///   the general experiment configuration. Actual HVAC varies per experiment.
/// - **Internal loads**: Default office-level gains (200 W lighting + equipment,
///   60% convective, 40% radiant).
/// - **Infiltration**: 0.5 ACH, representing typical sealed test cell conditions
///   with the door closed.
/// - **Window**: South-facing only, matching the Modelica model's `datConExtWin`.
///
/// # Known Differences from Reference Facility
///
/// 1. Single-zone simplification (closet + electrical room omitted)
/// 2. Radiant floor replaced with slab-on-grade conduction model
/// 3. Weather data from San Francisco TMY (not on-site sensors)
/// 4. Dividing walls modeled as exterior (conservative for heat loss)
/// 5. Default HVAC schedule (actual varies per experiment)
/// 6. No shading devices (FLEXLAB has automated shades, not always deployed)
pub fn flexlab_test_cell_spec() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("FLEXLAB-X3A".to_string())
        .with_description(
            "LBNL FLEXLAB test cell X3A - apples-to-apples model for empirical validation (T10.5)"
                .to_string(),
        )
        // Geometry from Modelica AFlo=60.97m², hRoo=3.6576m
        // South wall width 6.6675m, depth 9.144m
        .with_dimensions(6.6675, 9.144, 3.6576)
        // Custom construction matching FLEXLAB assemblies
        .with_construction(
            flexlab_exterior_wall(), // R16.8 wall
            flexlab_roof(),          // R20 roof
            flexlab_floor(),         // Slab-on-grade floor
        )
        // South-facing window: 5.88m × 1.8288m = 10.75 m²
        .with_window(10.75, Orientation::South)
        // ASHRAE 90.1 minimally compliant double clear glass
        // Modelica: UFra=1.4 W/m²K, two ID101 panes, 12.7mm air gap
        // Using ASHRAE 140 double_clear_glass values (U=2.10, SHGC=0.77)
        // as the baseline; the Modelica ASHRAE901Gla has different properties
        // but these are the best available standard values in Fluxion.
        .with_window_properties(WindowSpec::double_clear_glass())
        // Default office internal gains
        .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
        // Constant HVAC setpoints (general experiment configuration)
        .with_hvac(HvacSchedule::constant(20.0, 27.0))
        // Typical sealed test cell infiltration rate
        .with_infiltration(0.5)
        // Opaque absorptance ~0.6 (typical for light-colored exterior)
        .with_opaque_absorptance(0.6)
        // Single zone (main test cell only)
        .with_num_zones(1)
        // Commercial building for furniture factor
        .with_building_type(BuildingType::Commercial)
        .build()
        .expect("FLEXLAB test cell spec should build successfully")
}

/// Model diff summary comparing the Fluxion model to the FLEXLAB reference facility.
///
/// This function returns a human-readable description of the differences
/// between the Fluxion model and the actual FLEXLAB test cell, useful for
/// validation reports and documentation.
pub fn model_diff_summary() -> Vec<&'static str> {
    vec![
        "Geometry: Single main test cell (6.6675m × 9.144m × 3.6576m = 60.97m²). \
         Closet and electrical room omitted (boundary conditions in reference model).",
        "Walls: R16.8 exterior wall assembly (U≈0.338 W/m²K). \
         Dividing walls to adjacent test cells modeled as exterior (conservative).",
        "Roof: R20 assembly (U≈0.284 W/m²K) matching ASHRAE 90.1-2010 specification.",
        "Floor: 100mm concrete slab-on-grade with 50mm insulation. \
         Actual has radiant tubing (4 circuits) - not modeled.",
        "Windows: 10.75m² south-facing double clear glass, ASHRAE 90.1 compliant. \
         Bottom at 0.9144m (3ft) above floor.",
        "HVAC: Constant setpoints 20/27°C. Actual varies per FLEXLAB experiment.",
        "Weather: San Francisco TMY proxy. Actual uses on-site weather station.",
        "Internal gains: 200W default office. Actual depends on experiment setup.",
        "Infiltration: 0.5 ACH assumed. Actual measured via duct blaster testing.",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flexlab_spec_builds_successfully() {
        let spec = flexlab_test_cell_spec();
        assert_eq!(spec.case_id, "FLEXLAB-X3A");
        assert_eq!(spec.num_zones, 1);
        assert!(!spec.is_free_floating());
    }

    #[test]
    fn flexlab_geometry_matches_reference() {
        let spec = flexlab_test_cell_spec();
        let geo = &spec.geometry[0];
        // Verify dimensions match Modelica source
        assert!((geo.width - 6.6675).abs() < 1e-6);
        assert!((geo.depth - 9.144).abs() < 1e-6);
        assert!((geo.height - 3.6576).abs() < 1e-6);
        // Floor area should be 60.97 m²
        let floor_area = geo.width * geo.depth;
        assert!(
            (floor_area - 60.97).abs() < 0.1,
            "Floor area {floor_area} should be ~60.97 m²"
        );
    }

    #[test]
    fn flexlab_window_area_matches_reference() {
        let spec = flexlab_test_cell_spec();
        // Window: 5.88m × 1.8288m ≈ 10.75 m²
        let window_area = spec.total_window_area();
        assert!(
            (window_area - 10.75).abs() < 0.1,
            "Window area {window_area} should be ~10.75 m²"
        );
    }

    #[test]
    fn flexlab_wall_u_value_reasonable() {
        let wall = flexlab_exterior_wall();
        // R16.8 total (incl. films) ≈ 2.96 m²K/W → U ≈ 0.338 W/m²K
        // Film coefficients: interior 8.29 + exterior 25.0 W/m²K
        let u = wall.u_value(None, None);
        assert!(
            (u - 0.338).abs() < 0.02,
            "Wall U-value {u} W/m²K should be ~0.338 W/m²K (R16.8 total)"
        );
    }

    #[test]
    fn flexlab_roof_u_value_reasonable() {
        let roof = flexlab_roof();
        // R20 total (incl. films) ≈ 3.52 m²K/W → U ≈ 0.284 W/m²K
        let u = roof.u_value(None, None);
        assert!(
            (u - 0.284).abs() < 0.02,
            "Roof U-value {u} W/m²K should be ~0.284 W/m²K (R20 total)"
        );
    }

    #[test]
    fn flexlab_model_diff_not_empty() {
        let diffs = model_diff_summary();
        assert!(
            diffs.len() >= 5,
            "Should document at least 5 model differences"
        );
    }
}
