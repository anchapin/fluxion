//! Wall Properties - Slim interface for thermal solver requirements.
//!
//! This module provides `WallProperties` and `LayerProperties` structs that
//! expose only what heat conduction solvers need from `BuildingAssembly`.
//!
//! # Purpose
//!
//! The solvers (CTF, FD) only need thermal properties to compute heat flux.
//! By converting `BuildingAssembly` to `WallProperties` at the adapter seam,
//! we hide `BuildingAssembly` internals from solver implementations.
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::wall_properties::WallProperties;
//! use fluxion::sim::assembly::{AssemblyBuilder, ConcreteMaterial};
//!
//! let assembly = AssemblyBuilder::new("Test Wall".to_string())
//!     .add_layer(Box::new(ConcreteMaterial::new(0.2)))
//!     .build()
//!     .unwrap();
//!
//! let props = WallProperties::from_assembly(&assembly);
//! assert_eq!(props.layers.len(), 1);
//! assert!(props.total_thermal_mass_kj_m2 > 0.0);
//! ```
//!
//! # Architecture
//!
//! ```text
//! SolverManager
//!     |
//!     v
//! BuildingAssembly  ----->  WallProperties (at seam)
//!     |                              |
//!     v                              v
//! CTFSolverWrapper              CTFSolver (internals)
//! FDSolverWrapper               FDSolver (internals)
//! ```
//!
//! The `HeatConductionSolver` trait now takes `WallSpec` directly. The conversion
//! from `BuildingAssembly` to `WallProperties` happens at the `WallSpec` level.

// Issue #1349 (Phase 2 crate split): `BuildingAssembly` moved to
// `fluxion_core::assembly`. We import from there directly so this module no
// longer depends on `crate::sim` — breaking the physics<->sim cycle. The
// `crate::sim::assembly::BuildingAssembly` path remains available via the
// re-export shim in `src/sim/assembly.rs` for callers that haven't migrated.
use fluxion_core::assembly::BuildingAssembly;

/// Layer properties needed by thermal solvers.
///
/// This is a flat, owned struct with no internal structure —
/// just the thermal properties a solver needs to compute heat transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerProperties {
    /// Layer name
    pub name: String,
    /// Thickness [m]
    pub thickness_m: f64,
    /// Thermal conductivity [W/mK]
    pub conductivity_w_mk: f64,
    /// Density [kg/m³]
    pub density_kg_m3: f64,
    /// Specific heat capacity [J/kgK]
    pub specific_heat_j_kgk: f64,
    /// Thermal mass per unit area [kJ/m²K]
    pub thermal_mass_kj_m2: f64,
}

impl LayerProperties {
    /// Create layer properties from a material layer trait object.
    pub fn from_material_layer(layer: &dyn fluxion_core::assembly::MaterialLayer) -> Self {
        let thickness = layer.thickness();
        let density = layer.density();
        let specific_heat = layer.specific_heat();
        let thermal_mass_kj_m2 = density * specific_heat * thickness / 1000.0;

        Self {
            name: layer.name().to_string(),
            thickness_m: thickness,
            conductivity_w_mk: layer.conductivity(),
            density_kg_m3: density,
            specific_heat_j_kgk: specific_heat,
            thermal_mass_kj_m2,
        }
    }
}

/// Wall properties needed by thermal solvers.
///
/// This is a flat, owned struct with surface resistances and layer data.
/// No `BuildingAssembly` internals leak past this interface.
#[derive(Debug, Clone, PartialEq)]
pub struct WallProperties {
    /// All layers (exterior to interior)
    pub layers: Vec<LayerProperties>,
    /// Total thermal mass per unit area [kJ/m²K]
    pub total_thermal_mass_kj_m2: f64,
    /// Interior surface resistance [m²K/W]
    pub surface_resistance_inside: f64,
    /// Exterior surface resistance [m²K/W]
    pub surface_resistance_outside: f64,
}

impl WallProperties {
    /// Surface resistance for interior per ASHRAE 140 Section 5.2 [m²K/W]
    ///
    /// h_int = 8.29 W/m²K → R = 1/8.29 ≈ 0.12063 m²K/W
    pub const R_INT: f64 = 1.0 / 8.29; // ASHRAE 140 Section 5.2: h_int = 8.29 W/m²K

    /// Surface resistance for exterior per ASHRAE 140 Section 5.2 [m²K/W]
    ///
    /// h_ext = 29.3 W/m²K at 6.7 m/s wind speed → R = 1/29.3 ≈ 0.03413 m²K/W
    pub const R_EXT: f64 = 1.0 / 29.3; // ASHRAE 140 Section 5.2: h_ext = 29.3 W/m²K

    /// Create wall properties from a building assembly.
    ///
    /// This conversion happens once at the solver seam. If `BuildingAssembly`
    /// internals change, only this method needs updating.
    pub fn from_assembly(assembly: &BuildingAssembly) -> Self {
        let layers: Vec<LayerProperties> = assembly
            .layers
            .iter()
            .map(|layer| LayerProperties::from_material_layer(layer.as_ref()))
            .collect();

        let total_thermal_mass_kj_m2: f64 = layers.iter().map(|l| l.thermal_mass_kj_m2).sum();

        Self {
            layers,
            total_thermal_mass_kj_m2,
            surface_resistance_inside: Self::R_INT,
            surface_resistance_outside: Self::R_EXT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::method_selector::ThermalMethodSelector;
    use fluxion_core::assembly::{
        AssemblyBuilder, ConcreteMaterial, InsulationMaterial, MaterialLayer,
    };

    /// Test-only material layer that allows specifying exact density and
    /// specific heat — required because `ConcreteMaterial::new(thickness)`
    /// hard-codes `Cp = 840 J/kgK` (the generic normal-weight default), but
    /// ASHRAE 140 Case 900 (Table B1-3) specifies `Cp = 880 J/kgK` for the
    /// stacked concrete layer, and the 13 mm gypsum layer uses
    /// `ρ = 800 kg/m³, Cp = 1090 J/kgK`. Without this seam the regression
    /// test would silently drift 4.8 % on the concrete layer and ~8 % on
    /// the gypsum layer.
    #[derive(Debug, Clone)]
    struct CustomMaterial {
        name: String,
        thickness: f64,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
    }

    impl CustomMaterial {
        fn new(
            name: &str,
            thickness: f64,
            conductivity: f64,
            density: f64,
            specific_heat: f64,
        ) -> Self {
            Self {
                name: name.to_string(),
                thickness,
                conductivity,
                density,
                specific_heat,
            }
        }
    }

    impl MaterialLayer for CustomMaterial {
        fn name(&self) -> &str {
            &self.name
        }
        fn conductivity(&self) -> f64 {
            self.conductivity
        }
        fn thickness(&self) -> f64 {
            self.thickness
        }
        fn density(&self) -> f64 {
            self.density
        }
        fn specific_heat(&self) -> f64 {
            self.specific_heat
        }
        fn absorptance(&self) -> f64 {
            0.5
        }
        fn emissivity(&self) -> f64 {
            0.9
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Build the ASHRAE 140 Case 900 four-layer stacked-concrete wall
    /// (Gypsum 13 mm + Concrete 150 mm + Insulation 50 mm + Brick 100 mm)
    /// using the Table B1-3 material properties. Stacked-concrete reference
    /// `Cm = 468.7 kJ/m²K` per ASHRAE 140-2023 §5.2.
    fn case_900_assembly() -> fluxion_core::assembly::BuildingAssembly {
        AssemblyBuilder::new("ASHRAE 140 Case 900 stacked concrete wall".to_string())
            // interior finish (interior → exterior order)
            .add_layer(Box::new(CustomMaterial::new(
                "Gypsum", 0.013, 0.16, 800.0, 1090.0,
            )))
            .add_layer(Box::new(CustomMaterial::new(
                "Concrete", 0.150, 1.4, 2300.0, 880.0,
            )))
            .add_layer(Box::new(CustomMaterial::new(
                "Insulation",
                0.050,
                0.04,
                50.0,
                840.0,
            )))
            // exterior cladding
            .add_layer(Box::new(CustomMaterial::new(
                "Brick", 0.100, 0.81, 1920.0, 790.0,
            )))
            .build()
            .expect("Case 900 reference wall must build")
    }

    /// Same Case 900 wall but with the **legacy** `Cp = 840 J/kgK` default
    /// (what `ConcreteMaterial::new(0.150)` produces) — used to document the
    /// drift that motivated this regression test.
    fn case_900_assembly_legacy_cp() -> fluxion_core::assembly::BuildingAssembly {
        AssemblyBuilder::new("Case 900 with legacy Cp=840 defaults".to_string())
            .add_layer(Box::new(CustomMaterial::new(
                "Gypsum", 0.013, 0.16, 960.0, 840.0, // legacy generic gypsum
            )))
            .add_layer(Box::new(CustomMaterial::new(
                "Concrete", 0.150, 1.4, 2300.0, 840.0, // legacy Cp=840
            )))
            .add_layer(Box::new(CustomMaterial::new(
                "Insulation",
                0.050,
                0.04,
                50.0,
                840.0,
            )))
            .add_layer(Box::new(CustomMaterial::new(
                "Brick", 0.100, 0.81, 1920.0, 840.0, // legacy generic brick
            )))
            .build()
            .expect("Legacy Cp Case 900 wall must build")
    }

    #[test]
    fn test_layer_properties_from_concrete() {
        let concrete = ConcreteMaterial::new(0.2);
        let layer = LayerProperties::from_material_layer(&concrete);

        assert_eq!(layer.name, "Concrete");
        assert!((layer.thickness_m - 0.2).abs() < 1e-10);
        assert!((layer.conductivity_w_mk - 1.4).abs() < 1e-10);
        assert!((layer.density_kg_m3 - 2300.0).abs() < 1e-10);
        assert!((layer.specific_heat_j_kgk - 840.0).abs() < 1e-10);

        let expected_mass = 2300.0 * 840.0 * 0.2 / 1000.0;
        assert!((layer.thermal_mass_kj_m2 - expected_mass).abs() < 0.01);
    }

    #[test]
    fn test_layer_properties_thermal_mass_calculation() {
        let insulation = InsulationMaterial::new(0.1);
        let layer = LayerProperties::from_material_layer(&insulation);

        let expected_mass = 50.0 * 840.0 * 0.1 / 1000.0;
        assert!((layer.thermal_mass_kj_m2 - expected_mass).abs() < 0.01);
    }

    #[test]
    fn test_wall_properties_from_assembly_single_layer() {
        let assembly = AssemblyBuilder::new("Single Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        let props = WallProperties::from_assembly(&assembly);

        assert_eq!(props.layers.len(), 1);
        assert_eq!(props.layers[0].name, "Concrete");
        assert!((props.surface_resistance_inside - 1.0 / 8.29).abs() < 1e-10);
        assert!((props.surface_resistance_outside - 1.0 / 29.3).abs() < 1e-10);

        let expected_total: f64 = props.layers.iter().map(|l| l.thermal_mass_kj_m2).sum();
        assert!((props.total_thermal_mass_kj_m2 - expected_total).abs() < 0.01);
    }

    #[test]
    fn test_wall_properties_from_assembly_multi_layer() {
        let assembly = AssemblyBuilder::new("Composite Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .add_layer(Box::new(InsulationMaterial::new(0.1)))
            .add_layer(Box::new(ConcreteMaterial::new(0.012)))
            .build()
            .unwrap();

        let props = WallProperties::from_assembly(&assembly);

        assert_eq!(props.layers.len(), 3);
        assert_eq!(props.layers[0].name, "Concrete");
        assert_eq!(props.layers[1].name, "Insulation");
        assert_eq!(props.layers[2].name, "Concrete");

        let expected_total: f64 = props.layers.iter().map(|l| l.thermal_mass_kj_m2).sum();
        assert!((props.total_thermal_mass_kj_m2 - expected_total).abs() < 0.01);
    }

    #[test]
    fn test_wall_properties_surface_resistances() {
        let assembly = AssemblyBuilder::new("Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let props = WallProperties::from_assembly(&assembly);

        assert!((props.surface_resistance_inside - 1.0 / 8.29).abs() < 1e-10);
        assert!((props.surface_resistance_outside - 1.0 / 29.3).abs() < 1e-10);
    }

    #[test]
    fn test_wall_properties_clone() {
        let assembly = AssemblyBuilder::new("Clone Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.15)))
            .build()
            .unwrap();

        let props1 = WallProperties::from_assembly(&assembly);
        let props2 = props1.clone();

        assert_eq!(props1.layers.len(), props2.layers.len());
        assert!((props1.total_thermal_mass_kj_m2 - props2.total_thermal_mass_kj_m2).abs() < 1e-10);
    }

    #[test]
    fn test_layer_properties_clone() {
        let concrete = ConcreteMaterial::new(0.2);
        let layer1 = LayerProperties::from_material_layer(&concrete);
        let layer2 = layer1.clone();

        assert_eq!(layer1.name, layer2.name);
        assert!((layer1.thickness_m - layer2.thickness_m).abs() < 1e-10);
    }

    #[test]
    fn test_layer_properties_debug() {
        let concrete = ConcreteMaterial::new(0.1);
        let layer = LayerProperties::from_material_layer(&concrete);
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("Concrete"));
        assert!(debug_str.contains("thickness_m"));
    }

    #[test]
    fn test_wall_properties_debug() {
        let assembly = AssemblyBuilder::new("Debug Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();
        let props = WallProperties::from_assembly(&assembly);
        let debug_str = format!("{:?}", props);
        assert!(debug_str.contains("WallProperties"));
        assert!(debug_str.contains("layers"));
    }

    // -----------------------------------------------------------------------
    // ASHRAE 140 Case 900 / Case 600 regression tests (Issue #1420)
    //
    // These pin the `WallProperties::from_assembly` converter against the
    // ASHRAE 140 §5.2 stacked-concrete reference construction
    // (`Cm ≈ 468.7 kJ/m²K` per Table B1-3) and against the Case 600 low-mass
    // construction. They guard the converter against silently swapping the
    // ASHRAE-specific Cp / density values for the generic `ConcreteMaterial`
    // defaults (Cp = 840 vs 880 J/kgK, which would drift the concrete layer
    // by 4.8 %).
    // -----------------------------------------------------------------------

    /// ASHRAE 140-2023 Case 900 stacked-concrete wall:
    ///
    /// | Layer      | d (m) | ρ (kg/m³) | Cp (J/kgK) | Cm (kJ/m²K) |
    /// |------------|-------|-----------|------------|-------------|
    /// | Gypsum     | 0.013 |   800     | 1090       | 11.336      |
    /// | Concrete   | 0.150 |  2300     |  880       | 303.600     |
    /// | Insulation | 0.050 |    50     |  840       |   2.100     |
    /// | Brick      | 0.100 |  1920     |  790       | 151.680     |
    /// | **Total**  | 0.313 |           |            | **468.716** |
    ///
    /// Asserts the converter reproduces `468.7 ± 1 %` kJ/m²K (the canonical
    /// reference used by `tests/ctf_coefficient_validation.rs:5-15` and the
    /// Phase-3 `GaugeSolver` validation harness — see ARCHITECTURE.md
    /// §"Module 6 / Phase 3 validation harness").
    #[test]
    fn test_wall_properties_ashrae_140_case_900() {
        let assembly = case_900_assembly();
        let props = WallProperties::from_assembly(&assembly);

        // Layer count — interior gypsum through exterior brick.
        assert_eq!(
            props.layers.len(),
            4,
            "Case 900 reference wall must have 4 layers (Gypsum + Concrete + Insulation + Brick)"
        );
        assert_eq!(props.layers[0].name, "Gypsum");
        assert_eq!(props.layers[1].name, "Concrete");
        assert_eq!(props.layers[2].name, "Insulation");
        assert_eq!(props.layers[3].name, "Brick");

        // Surface film resistances per ASHRAE 140 Section 5.2.
        // h_int = 8.29 W/m²K, h_ext = 29.3 W/m²K.
        assert!(
            (props.surface_resistance_inside - 1.0 / 8.29).abs() < 1e-10,
            "surface_resistance_inside = {:.6} should equal 1/8.29 = {:.6}",
            props.surface_resistance_inside,
            1.0 / 8.29
        );
        assert!(
            (props.surface_resistance_outside - 1.0 / 29.3).abs() < 1e-10,
            "surface_resistance_outside = {:.6} should equal 1/29.3 = {:.6}",
            props.surface_resistance_outside,
            1.0 / 29.3
        );

        // Per-layer Cm sanity (each layer independently pinned).
        // Tolerance 1e-6 kJ/m²K — wall_properties computes
        // `ρ · Cp · d / 1000` in f64 and we hand-computed the reference in
        // f64, so this should be exact to floating-point ULP.
        assert!(
            (props.layers[0].thermal_mass_kj_m2 - 11.336).abs() < 1e-6,
            "Gypsum Cm = {:.6} kJ/m²K, expected 11.336",
            props.layers[0].thermal_mass_kj_m2
        );
        assert!(
            (props.layers[1].thermal_mass_kj_m2 - 303.600).abs() < 1e-6,
            "Concrete Cm = {:.6} kJ/m²K, expected 303.600 (ASHRAE 140 Table B1-3)",
            props.layers[1].thermal_mass_kj_m2
        );
        assert!(
            (props.layers[2].thermal_mass_kj_m2 - 2.100).abs() < 1e-6,
            "Insulation Cm = {:.6} kJ/m²K, expected 2.100",
            props.layers[2].thermal_mass_kj_m2
        );
        assert!(
            (props.layers[3].thermal_mass_kj_m2 - 151.680).abs() < 1e-6,
            "Brick Cm = {:.6} kJ/m²K, expected 151.680 (ASHRAE 140 Table B1-3)",
            props.layers[3].thermal_mass_kj_m2
        );

        // Total Cm — the headline regression assertion.
        // Reference: Σ(ρ·Cp·d) = 468 716 J/m²K = 468.716 kJ/m²K (± 1 %).
        let cm_ref = 468.716;
        let cm_actual = props.total_thermal_mass_kj_m2;
        let lower = 0.99 * cm_ref;
        let upper = 1.01 * cm_ref;
        assert!(
            cm_actual >= lower && cm_actual <= upper,
            "Case 900 total Cm = {:.4} kJ/m²K must lie in [{:.4}, {:.4}] (468.7 ± 1 %)",
            cm_actual,
            lower,
            upper
        );

        // Time constant sanity (helper, not the headline assertion):
        // τ = Cm / (h_int + h_ext) / 3600  ≈ 468 716 / 37.59 / 3600 ≈ 3.46 h
        // → well above the 2-hour `ThermalMethodSelector` threshold so the
        // Case 900 envelope selects FD (not 5R1C).
        let selector = ThermalMethodSelector::default();
        let tau_h = selector.calculate_time_constant(&assembly);
        assert!(
            tau_h > 2.0,
            "Case 900 τ = {:.3} h must exceed the 2 h FD/CTF threshold, got {tau_h:.3}",
            tau_h
        );
    }

    /// ASHRAE 140 Case 600 low-mass wall (50 mm foam-board insulation only):
    ///
    /// | Layer      | d (m) | ρ (kg/m³) | Cp (J/kgK) | Cm (kJ/m²K) |
    /// |------------|-------|-----------|------------|-------------|
    /// | Insulation | 0.050 |    50     |  840       |   2.100     |
    ///
    /// Pins the **600-vs-900** boundary in `ThermalMethodSelector`
    /// (`src/physics/method_selector.rs:566` — `threshold_hours = 2.0`):
    /// `Cm ≈ 2.1 kJ/m²K` and `τ ≈ 0.0155 h ≪ 2 h` → selects **5R1C**.
    #[test]
    fn test_wall_properties_case_600_low_mass() {
        let assembly = AssemblyBuilder::new("Case 600 low-mass insulation".to_string())
            .add_layer(Box::new(InsulationMaterial::new(0.050)))
            .build()
            .expect("Case 600 wall must build");
        let props = WallProperties::from_assembly(&assembly);

        assert_eq!(
            props.layers.len(),
            1,
            "Case 600 reference wall must have a single insulation layer"
        );

        // Cm — 50 mm insulation with default ρ = 50 kg/m³, Cp = 840 J/kgK:
        // Σ(ρ·Cp·d) = 2 100 J/m²K = 2.100 kJ/m²K ± 5 % envelope (wide band
        // because the property table allows for slight material variation).
        let cm_actual = props.total_thermal_mass_kj_m2;
        assert!(
            (cm_actual - 2.1).abs() / 2.1 < 0.05,
            "Case 600 Cm = {cm_actual:.4} kJ/m²K, expected ≈ 2.1 (± 5 %)"
        );

        // τ must sit well below the 2-hour FD/CTF threshold (issue body
        // requirement: τ < 0.5 h so 5R1C is selected).
        let selector = ThermalMethodSelector::default();
        let tau_h = selector.calculate_time_constant(&assembly);
        assert!(
            tau_h < 0.5,
            "Case 600 τ = {tau_h:.4} h must be < 0.5 h (well below 2 h 5R1C threshold), got {tau_h:.4}"
        );
        assert!(
            tau_h > 0.0 && tau_h.is_finite(),
            "Case 600 τ = {tau_h} must be finite and positive"
        );

        // Solver selection side-check — Case 600 must select 5R1C, not FD.
        let method = selector.select_method(&assembly);
        assert_eq!(
            method,
            crate::physics::method_selector::ThermalMethod::FiveR1C,
            "Case 600 (low mass, τ ≪ 2 h) must select FiveR1C"
        );
    }

    /// Documents the **4.8 % Cp drift** on the concrete layer that motivated
    /// this regression test (Issue #1420). The generic
    /// `ConcreteMaterial::new(thickness)` constructor hard-codes
    /// `Cp = 840 J/kgK`; ASHRAE 140 Case 900 Table B1-3 specifies
    /// `Cp = 880 J/kgK` for the stacked concrete layer. Swapping one for
    /// the other would silently propagate through `WallProperties →
    /// WallSpec::from_assembly → CTFSolverWrapper::material_constructor
    /// (src/physics/ctf_solver_wrapper.rs:84-98)` and through
    /// `FDSolverWrapper`. This test pins the converter to use the ASHRAE
    /// values and asserts the **concrete-layer** drift matches the
    /// documented `880/840 − 1 ≈ +4.76 %`.
    ///
    /// Note on drift scope: the headline `Case 900 total Cm` shifts by only
    /// ~3 % overall (because concrete is one of four layers), but the
    /// concrete-layer Cm itself shifts by ~4.76 %, which is the drift that
    /// the converter must not silently absorb.
    #[test]
    fn test_wall_properties_cp_drift_sensitivity() {
        let legacy = WallProperties::from_assembly(&case_900_assembly_legacy_cp());
        let ashrae = WallProperties::from_assembly(&case_900_assembly());

        // Layer order is identical — both walls are the same 4-layer stack.
        assert_eq!(legacy.layers.len(), ashrae.layers.len());
        for (a, b) in legacy.layers.iter().zip(ashrae.layers.iter()) {
            assert_eq!(
                a.name, b.name,
                "layer order must match between legacy and ASHRAE walls"
            );
        }

        // Concrete-layer Cp drift: 880/840 − 1 = +4.7619 %.
        let concrete_idx = 1;
        let cm_legacy = legacy.layers[concrete_idx].thermal_mass_kj_m2;
        let cm_ashrae = ashrae.layers[concrete_idx].thermal_mass_kj_m2;
        let drift_pct = 100.0 * (cm_ashrae - cm_legacy) / cm_legacy;
        let expected_drift_pct = 100.0 * (880.0 / 840.0 - 1.0); // +4.7619 %
        assert!(
            (drift_pct - expected_drift_pct).abs() < 0.01,
            "Concrete-layer Cp drift = {drift_pct:.4} %, expected {expected_drift_pct:.4} % \
             (Cm_legacy = {cm_legacy:.4}, Cm_ashrae = {cm_ashrae:.4})"
        );
        assert!(
            drift_pct > 4.0 && drift_pct < 5.5,
            "Concrete-layer Cp drift = {drift_pct:.4} % must lie in [4.0, 5.5] %"
        );

        // Gypsum-layer drift: legacy (ρ=960, Cp=840) vs ASHRAE (ρ=800,
        // Cp=1090) → 800·1090 / (960·840) − 1 ≈ +8.13 %.
        let gypsum_idx = 0;
        let cm_g_legacy = legacy.layers[gypsum_idx].thermal_mass_kj_m2;
        let cm_g_ashrae = ashrae.layers[gypsum_idx].thermal_mass_kj_m2;
        let g_drift_pct = 100.0 * (cm_g_ashrae - cm_g_legacy) / cm_g_legacy;
        assert!(
            g_drift_pct > 5.0 && g_drift_pct < 12.0,
            "Gypsum-layer density+Cp drift = {g_drift_pct:.4} % must lie in [5, 12] % \
             (Cm_legacy = {cm_g_legacy:.4}, Cm_ashrae = {cm_g_ashrae:.4})"
        );

        // Total Cm — ASHRAE wall must exceed legacy by ~1 % (concrete Cp
        // drift is partly offset by brick Cp dropping 840→790 and gypsum
        // density dropping 960→800).
        let total_drift_pct = 100.0
            * (ashrae.total_thermal_mass_kj_m2 - legacy.total_thermal_mass_kj_m2)
            / legacy.total_thermal_mass_kj_m2;
        assert!(
            total_drift_pct > 0.5 && total_drift_pct < 2.0,
            "Total wall Cm drift = {total_drift_pct:.4} % must lie in [0.5, 2.0] % \
             (legacy = {:.4}, ashrae = {:.4})",
            legacy.total_thermal_mass_kj_m2,
            ashrae.total_thermal_mass_kj_m2
        );

        // The ASHRAE total must land in the 468.7 ± 1 % envelope; the legacy
        // total must NOT (proving the regression test would have caught the
        // silent default-Cp swap that motivated Issue #1420).
        let cm_ref = 468.716;
        assert!(
            ashrae.total_thermal_mass_kj_m2 >= 0.99 * cm_ref
                && ashrae.total_thermal_mass_kj_m2 <= 1.01 * cm_ref,
            "ASHRAE wall Cm = {:.4} must lie in [464.03, 473.40] kJ/m²K",
            ashrae.total_thermal_mass_kj_m2
        );
        assert!(
            legacy.total_thermal_mass_kj_m2 < 0.99 * cm_ref,
            "Legacy Cp=840 wall Cm = {:.4} must fall OUTSIDE the ASHRAE 468.7 ± 1 % envelope \
             (otherwise the regression test would not actually catch the silent Cp swap)",
            legacy.total_thermal_mass_kj_m2
        );
    }
}
