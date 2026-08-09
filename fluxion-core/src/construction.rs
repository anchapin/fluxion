//! Multi-layer construction R-value calculator for building envelopes.
//!
//! This module provides structs and functions for calculating thermal resistance (R-value)
//! and thermal transmittance (U-value) for multi-layer building constructions, following
//! ASHRAE Standard 140 specifications.
//!
//! # ISO 13790 Annex C Implementation
//!
//! This module implements ISO 13790 Annex C methodology for deriving effective thermal
//! mass parameters from multi-layer construction assemblies. The key concepts are:
//!
//! - **Effective Thermal Mass**: Only layers on the interior side of the dominant
//!   insulation layer contribute to effective thermal mass (half-insulation rule).
//! - **Mass Classification**: Constructions are classified by effective specific capacitance
//!   into VeryLight, Light, Medium, Heavy, or VeryHeavy categories.
//! - **Area Multipliers**: Each mass class has an associated effective mass area
//!   multiplier (A_m factor) used in 5R1C thermal network calculations.
//!
//! # Crate split (Issue #2462 — Phase 2 of the crate split)
//!
//! As of #2462, this module was hoisted out of `fluxion::sim::construction` into the
//! workspace leaf crate `fluxion_core::construction` so that the `physics ↔ sim`
//! module cycle (Issue #2462) can close. The original `src/sim/construction.rs`
//! is now a thin re-export shim with `WallSurface` (which depends on
//! `sim::shading::{Overhang, ShadeFin}`) staying put.
//!
//! Constants are inlined at the call sites (these are pure data values from
//! ASHRAE 140-2023 §5.2) — `fluxion-core` cannot depend on `fluxion`'s
//! `physics::constants` module.

#![allow(clippy::approx_constant)] // Allow spec constants like 0.318 (ASHRAE 140 values)

use serde::{Deserialize, Serialize};

// =============================================================================
// Surface film coefficients — ASHRAE 140 Section 5.2
//
// These values are inlined from
// `fluxion::physics::constants::thermal::ashrae_140::{v2023, materials}` (the
// v2023 module is the canonical export — `ashrae_140_v2021` feature flag
// exists for backward compatibility but no test path uses it). Constants
// must be inlined here because `fluxion-core` cannot import from `fluxion`'s
// `physics` module — that would re-introduce the very cycle this file is
// here to break.
// =============================================================================

/// Interior surface film coefficient per ASHRAE 140 Section 5.2.
/// **Value:** 8.29 W/m²K
pub const INTERIOR_FILM_COEFF: f64 = 8.29;

/// Exterior surface film coefficient per ASHRAE 140 Section 5.2.
/// **Value:** 18.3 W/m²K (vertical surfaces, ~3.4 m/s wind)
pub const EXTERIOR_FILM_COEFF: f64 = 18.3;

/// Interior film coefficient for vertical wall surfaces.
/// **Value:** 7.69 W/m²K (R_si = 0.13 m²K/W)
pub const INTERIOR_FILM_COEFF_WALL: f64 = 7.69;

/// Interior film coefficient for ceiling (upward heat flow).
/// **Value:** 10.0 W/m²K (R_si = 0.10 m²K/W)
pub const INTERIOR_FILM_COEFF_CEILING: f64 = 10.0;

/// Interior film coefficient for floor (downward heat flow).
/// **Value:** 5.88 W/m²K (R_si = 0.17 m²K/W)
pub const INTERIOR_FILM_COEFF_FLOOR: f64 = 5.88;

/// Default exterior film coefficient (ASHRAE 140 §5.2).
/// **Value:** 18.3 W/m²K
pub const EXTERIOR_FILM_COEFF_DEFAULT: f64 = 18.3;

/// Air density at sea level under standard conditions.
/// **Value:** 1.225 kg/m³
pub const AIR_DENSITY_SEA_LEVEL: f64 = 1.225;

/// Specific heat capacity of air at constant pressure.
/// **Value:** 1005.0 J/(kg·K)
pub const AIR_SPECIFIC_HEAT: f64 = 1005.0;

/// Surface type for ASHRAE 140 surface-type-specific interior film coefficients.
///
/// ASHRAE 140 specifies different interior resistances for different surface orientations
/// to account for different heat flow directions (vertical, upward, downward).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceType {
    /// Vertical wall surface
    Wall,
    /// Ceiling/roof surface (upward heat flow)
    Ceiling,
    /// Floor surface (downward heat flow)
    Floor,
}

impl SurfaceType {
    /// Returns the interior film coefficient for the specified surface type.
    ///
    /// # Arguments
    /// * `surface_type` - The surface type
    ///
    /// # Returns
    /// Interior film coefficient in W/m²K per ASHRAE 140
    pub fn interior_film_coeff(surface_type: SurfaceType) -> f64 {
        match surface_type {
            SurfaceType::Wall => INTERIOR_FILM_COEFF_WALL,
            SurfaceType::Ceiling => INTERIOR_FILM_COEFF_CEILING,
            SurfaceType::Floor => INTERIOR_FILM_COEFF_FLOOR,
        }
    }
}

/// Returns the exterior film coefficient based on wind speed.
///
/// The exterior film coefficient varies with wind speed due to enhanced convection.
/// This function implements the ASHRAE recommended correlation:
///
/// - Low wind (< 2 m/s): ~21 W/m²K
/// - Moderate wind (2-5 m/s): 21-29 W/m²K
/// - High wind (> 5 m/s): >29 W/m²K
///
/// # Arguments
/// * `wind_speed` - Wind speed in meters per second (m/s)
///
/// # Returns
/// Exterior film coefficient in W/m²K
///
/// # Example
/// ```
/// use fluxion_core::construction::exterior_film_coeff;
///
/// let h_ext = exterior_film_coeff(3.5); // ~24 W/m²K for moderate wind
/// ```
pub fn exterior_film_coeff(wind_speed: f64) -> f64 {
    // ASHRAE correlation: h_ext increases with wind speed
    // Using simplified model: h_ext = 10.0 + 4.0 * v^(0.5)
    // This gives ~21 W/m²K at v=3m/s, ~29 W/m²K at v=9m/s
    10.0 + 4.0 * wind_speed.sqrt()
}

/// Returns the standard interior film coefficient per ASHRAE 140.
///
/// This constant value of 8.29 W/m²K is used for interior surfaces in
/// ASHRAE 140 validation test cases.
///
/// # Returns
/// Interior film coefficient in W/m²K
pub const fn interior_film_coeff() -> f64 {
    INTERIOR_FILM_COEFF
}

/// A single layer in a multi-layer construction assembly.
///
/// Each layer represents a homogeneous material with uniform thermal properties.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConstructionLayer {
    /// Name of the material
    pub name: String,

    /// Thermal conductivity of the material (k) in W/m·K
    ///
    /// This is the rate of heat transfer through the material per unit temperature
    /// gradient. Lower values indicate better insulation.
    pub conductivity: f64,

    /// Density of the material in kg/m³
    ///
    /// Used for calculating thermal mass and heat capacity of the construction.
    pub density: f64,

    /// Specific heat capacity of the material in J/kg·K
    ///
    /// Combined with density and thickness, this determines the thermal mass of the layer.
    pub specific_heat: f64,

    /// Thickness of the layer in meters
    ///
    /// The physical thickness of the material layer in the construction assembly.
    pub thickness: f64,

    /// Surface emissivity (0.0 to 1.0)
    ///
    /// The ratio of radiant energy emitted by the surface compared to a black body.
    /// Used for radiative heat transfer calculations. Defaults to 0.9 for most
    /// building materials.
    pub emissivity: f64,

    /// Surface absorptance (0.0 to 1.0)
    ///
    /// The fraction of incident solar radiation absorbed by the surface.
    /// Used for solar heat gain calculations. Defaults to 0.7 for typical
    /// opaque building materials.
    pub absorptance: f64,
}

impl ConstructionLayer {
    /// Creates a new ConstructionLayer with the specified thermal properties.
    ///
    /// # Arguments
    /// * `conductivity` - Thermal conductivity (k) in W/m·K
    /// * `density` - Material density in kg/m³
    /// * `specific_heat` - Specific heat capacity in J/kg·K
    /// * `thickness` - Layer thickness in meters
    ///
    /// # Returns
    /// A new ConstructionLayer with default emissivity (0.9) and absorptance (0.7)
    ///
    /// # Panics
    /// Panics if conductivity, density, specific_heat, or thickness are non-positive.
    ///
    /// # Example
    /// ```
    /// use fluxion_core::construction::ConstructionLayer;
    ///
    /// let layer = ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, 0.066);
    /// ```
    pub fn new(
        name: impl Into<String>,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        thickness: f64,
    ) -> Self {
        assert!(conductivity > 0.0, "Conductivity must be positive");
        assert!(density > 0.0, "Density must be positive");
        assert!(specific_heat > 0.0, "Specific heat must be positive");
        assert!(thickness > 0.0, "Thickness must be positive");

        Self {
            name: name.into(),
            conductivity,
            density,
            specific_heat,
            thickness,
            emissivity: 0.9,
            absorptance: 0.7,
        }
    }

    /// Creates a new ConstructionLayer with custom surface properties.
    ///
    /// # Arguments
    /// * `conductivity` - Thermal conductivity (k) in W/m·K
    /// * `density` - Material density in kg/m³
    /// * `specific_heat` - Specific heat capacity in J/kg·K
    /// * `thickness` - Layer thickness in meters
    /// * `emissivity` - Surface emissivity (0.0 to 1.0)
    /// * `absorptance` - Surface absorptance (0.0 to 1.0)
    ///
    /// # Returns
    /// A new ConstructionLayer with custom surface properties
    ///
    /// # Panics
    /// Panics if conductivity, density, specific_heat, or thickness are non-positive.
    /// Panics if emissivity or absorptance are outside the range [0.0, 1.0].
    pub fn with_surface_properties(
        name: impl Into<String>,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        thickness: f64,
        emissivity: f64,
        absorptance: f64,
    ) -> Self {
        assert!(conductivity > 0.0, "Conductivity must be positive");
        assert!(density > 0.0, "Density must be positive");
        assert!(specific_heat > 0.0, "Specific heat must be positive");
        assert!(thickness > 0.0, "Thickness must be positive");
        assert!(
            (0.0..=1.0).contains(&emissivity),
            "Emissivity must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&absorptance),
            "Absorptance must be in [0, 1]"
        );

        Self {
            name: name.into(),
            conductivity,
            density,
            specific_heat,
            thickness,
            emissivity,
            absorptance,
        }
    }

    /// Calculates the thermal resistance (R-value) of this single layer.
    ///
    /// The R-value is the ratio of thickness to thermal conductivity:
    /// R = δ / k
    ///
    /// Units: m²K/W
    ///
    /// # Returns
    /// Thermal resistance in m²K/W
    pub fn r_value(&self) -> f64 {
        self.thickness / self.conductivity
    }

    /// Calculates the thermal capacitance per unit area of this layer.
    ///
    /// Thermal capacitance per unit area is calculated as:
    /// C/A = ρ × δ × Cp
    ///
    /// where ρ is density (kg/m³), δ is thickness (m), and Cp is specific heat (J/kg·K).
    ///
    /// Units: J/m²K
    ///
    /// # Returns
    /// Thermal capacitance per unit area in J/m²K
    pub fn thermal_capacitance_per_area(&self) -> f64 {
        self.density * self.thickness * self.specific_heat
    }
}
/// A multi-layer construction assembly.
///
/// Represents a complete building assembly (wall, roof, floor) composed of
/// multiple material layers arranged in series.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Construction {
    /// Ordered list of material layers from interior to exterior.
    ///
    /// Layers are ordered from the interior surface (index 0) to the exterior
    /// surface (last index).
    pub layers: Vec<ConstructionLayer>,
}

impl Construction {
    /// Creates a new Construction from a vector of layers.
    ///
    /// # Arguments
    /// * `layers` - Vector of ConstructionLayer ordered from interior to exterior
    ///
    /// # Returns
    /// A new Construction assembly
    ///
    /// # Panics
    /// Panics if layers is empty
    ///
    /// # Example
    /// ```
    /// use fluxion_core::construction::{Construction, ConstructionLayer};
    ///
    /// let layers = vec![
    ///     ConstructionLayer::new("Plasterboard", 0.16, 950.0, 840.0, 0.012), // Plasterboard
    ///     ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, 0.066),  // Fiberglass
    ///     ConstructionLayer::new("Wood siding", 0.14, 500.0, 1300.0, 0.009), // Wood siding
    /// ];
    /// let wall = Construction::new(layers);
    /// ```
    pub fn new(layers: Vec<ConstructionLayer>) -> Self {
        assert!(
            !layers.is_empty(),
            "Construction must have at least one layer"
        );
        Self { layers }
    }

    /// Calculates the thermal resistance (R-value) of materials only.
    ///
    /// This returns the sum of R-values for all material layers, excluding
    /// interior and exterior film coefficients. This is useful for calculating
    /// conductances where film coefficients are handled separately (e.g., inter-zone
    /// walls where both surfaces are interior).
    ///
    /// # Returns
    /// Materials-only thermal resistance in m²K/W
    ///
    /// # Formula
    /// R_materials = Σ(δ/k) for all layers
    pub fn r_value_materials(&self) -> f64 {
        self.layers.iter().map(|l| l.r_value()).sum()
    }

    /// Calculates the total thermal resistance (R-value) including film coefficients.
    ///
    /// The total R-value is the sum of:
    /// - Interior film resistance: R_film_int = 1 / h_int
    /// - Material layer resistances: R_layer = δ / k (summed for all layers)
    /// - Exterior film resistance: R_film_ext = 1 / h_ext
    ///
    /// R_total = R_film_int + Σ(δ/k) + R_film_ext
    ///
    /// # Arguments
    /// * `surface_type` - Optional surface type for ASHRAE 140 surface-type-specific
    ///   interior film coefficient. If None, uses default INTERIOR_FILM_COEFF.
    /// * `exterior_wind_speed` - Wind speed at exterior surface in m/s
    ///   If not provided, uses default exterior film coefficient
    ///
    /// # Returns
    /// Total thermal resistance in m²K/W
    pub fn r_value_total(
        &self,
        surface_type: Option<SurfaceType>,
        exterior_wind_speed: Option<f64>,
    ) -> f64 {
        let h_int = surface_type
            .map(SurfaceType::interior_film_coeff)
            .unwrap_or_else(interior_film_coeff);
        let h_ext = exterior_wind_speed
            .map(exterior_film_coeff)
            .unwrap_or(EXTERIOR_FILM_COEFF_DEFAULT);

        let r_film_int = 1.0 / h_int;
        let r_film_ext = 1.0 / h_ext;
        let r_materials: f64 = self.layers.iter().map(|l| l.r_value()).sum();

        // Issue #588 Fix: For floor surfaces (SurfaceType::Floor), use ground coupling
        // resistance instead of exterior film coefficient. For slab-on-grade floors,
        // the exterior boundary is ground, not ambient air. The effective exterior
        // resistance for ground-coupled slabs is approximately 0.17 m²K/W (ASHRAE HOAFM),
        // corresponding to a film coefficient of ~6 W/m²K rather than the 25 W/m²K
        // used for above-grade surfaces.
        let r_exterior = if surface_type == Some(SurfaceType::Floor) {
            // Ground coupling resistance for slab-on-grade
            // Approximate value: R_g = 0.17 m²K/W (includes soil resistance)
            0.17
        } else {
            r_film_ext
        };

        r_film_int + r_materials + r_exterior
    }

    /// Calculates the thermal transmittance (U-value) of the construction.
    ///
    /// The U-value is the reciprocal of the total thermal resistance:
    /// U = 1 / R_total
    ///
    /// Units: W/m²K
    ///
    /// # Arguments
    /// * `surface_type` - Optional surface type for ASHRAE 140 surface-type-specific
    ///   interior film coefficient. If None, uses default INTERIOR_FILM_COEFF.
    /// * `exterior_wind_speed` - Wind speed at exterior surface in m/s
    ///   If not provided, uses default exterior film coefficient
    ///
    /// # Returns
    /// Thermal transmittance in W/m²K
    pub fn u_value(
        &self,
        surface_type: Option<SurfaceType>,
        exterior_wind_speed: Option<f64>,
    ) -> f64 {
        let r_total = self.r_value_total(surface_type, exterior_wind_speed);
        assert!(r_total > 0.0, "Total R-value must be positive");
        1.0 / r_total
    }

    /// Calculates the U-value for an internal partition (using two interior film coefficients).
    pub fn u_value_internal(&self) -> f64 {
        let h_int = interior_film_coeff();
        let r_materials: f64 = self.layers.iter().map(|l| l.r_value()).sum();
        let r_total = (1.0 / h_int) + r_materials + (1.0 / h_int);
        1.0 / r_total
    }

    /// Calculates the total thermal mass (capacitance) of the construction.
    ///
    /// Returns the sum of thermal capacitance per unit area for all layers.
    /// This is used to determine the effective thermal mass of the assembly
    /// for thermal network models.
    ///
    /// Units: J/m²K
    ///
    /// # Returns
    /// Total thermal capacitance per unit area in J/m²K
    pub fn thermal_capacitance_per_area(&self) -> f64 {
        self.layers
            .iter()
            .map(|l| l.thermal_capacitance_per_area())
            .sum()
    }

    /// Returns the total thickness of the construction.
    ///
    /// Units: meters
    ///
    /// # Returns
    /// Total thickness in meters
    pub fn total_thickness(&self) -> f64 {
        self.layers.iter().map(|l| l.thickness).sum()
    }

    /// Returns the number of layers in this construction.
    ///
    /// # Returns
    /// Number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Finds the index of the dominant insulation layer.
    ///
    /// The dominant insulation layer is the layer with the highest thermal resistance
    /// (R-value = thickness / conductivity). This layer separates interior thermal
    /// mass from the exterior environment in the ISO 13790 half-insulation rule.
    ///
    /// # Returns
    /// Index of the dominant insulation layer (0-based from interior)
    ///
    /// # ISO 13790 Reference
    /// ISO 13790 Annex C specifies that thermal mass should be calculated
    /// considering only layers on the interior side of the dominant insulation layer.
    pub fn find_dominant_insulation_layer_index(&self) -> usize {
        self.layers
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.r_value()
                    .partial_cmp(&b.r_value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .expect("Construction must have at least one layer")
    }

    /// Calculates the effective thermal capacitance per unit area using ISO 13790 Annex C.
    ///
    /// Layers from the interior surface up to and including the dominant insulation
    /// layer contribute their full thermal capacitance (ρ × c × δ). Layers exterior
    /// to the insulation contribute nothing.
    ///
    /// # Algorithm
    /// 1. Find dominant insulation layer (highest R-value)
    /// 2. For each layer at index `j`:
    ///    - If `j <= ins_idx`: full contribution (ρ × c × δ)
    ///    - If `j > ins_idx`: zero contribution
    /// 3. Sum all contributions (capped at 100 mm active thickness)
    ///
    /// # Returns
    /// Effective specific thermal capacitance (κ) in J/m²K
    ///
    /// # ISO 13790 Reference
    /// ISO 13790 Annex C, Section C.2 specifies the effective thermal capacitance
    /// using layers up to and including the insulation layer.
    ///
    /// # Example
    /// ```
    /// use fluxion_core::construction::{Construction, ConstructionLayer, Assemblies};
    ///
    /// // Case 600 wall: plasterboard + fiberglass + wood siding
    /// // Fiberglass is dominant insulation (R=1.65 m²K/W)
    /// // Only plasterboard (interior side) contributes fully
    /// // Fiberglass contributes half its capacitance
    /// // Wood siding (exterior side) contributes nothing
    /// let wall = Assemblies::low_mass_wall();
    /// let kappa = wall.iso_13790_effective_capacitance_per_area();
    /// // kappa ≈ 9,900 J/m²K (very light mass)
    /// ```
    pub fn iso_13790_effective_capacitance_per_area(&self) -> f64 {
        // ISO 13790 Annex C effective thermal capacitance.
        //
        // Sum the thermal capacitance of layers from the INTERIOR surface up to
        // AND INCLUDING the dominant insulation layer. Each layer contributes its
        // FULL ρ·c·δ. Layers EXTERIOR to the insulation contribute nothing (they
        // are thermally decoupled from the interior by the insulation resistance).
        //
        // The total active thickness is capped at 100 mm from the interior surface
        // per ISO 13790 Annex C.
        let ins_idx = self.find_dominant_insulation_layer_index();

        let mut total_kappa = 0.0;
        let mut active_thickness = 0.0;
        const MAX_ACTIVE_THICKNESS: f64 = 0.10; // ISO 13790: cap at 10cm from interior

        for (j, layer) in self.layers.iter().enumerate() {
            if active_thickness >= MAX_ACTIVE_THICKNESS {
                break;
            }
            if j <= ins_idx {
                // Interior to insulation (inclusive): full contribution
                total_kappa += layer.thermal_capacitance_per_area();
                active_thickness += layer.thickness;
            }
            // Layers exterior to insulation: zero contribution
        }

        total_kappa
    }

    /// Classifies the construction by its effective thermal mass per ISO 13790 Annex C.
    ///
    /// Classification is based on the effective specific thermal capacitance (κ)
    /// calculated using the half-insulation rule. This determines the mass class
    /// used for selecting the effective mass area multiplier (A_m factor) in
    /// 5R1C thermal network calculations.
    ///
    /// # Returns
    /// Mass classification (VeryLight, Light, Medium, Heavy, VeryHeavy)
    ///
    /// # ISO 13790 Reference
    /// ISO 13790 Annex C, Table C.2 defines mass classes based on effective
    /// specific thermal capacitance ranges.
    ///
    /// # Example
    /// ```
    /// use fluxion_core::construction::{Construction, Assemblies};
    ///
    /// let wall = Assemblies::low_mass_wall();
    /// let mass_class = wall.iso_13790_mass_class();
    /// // Returns MassClass::VeryLight (κ ≈ 9,900 J/m²K)
    ///
    /// let heavy_wall = Assemblies::high_mass_wall();
    /// let heavy_class = heavy_wall.iso_13790_mass_class();
    /// // Returns MassClass::Light or Medium (κ ≈ 140,000 J/m²K)
    /// ```
    pub fn iso_13790_mass_class(&self) -> MassClass {
        let kappa = self.iso_13790_effective_capacitance_per_area();

        // Classification per ISO 13790 Table C.2
        if kappa < 80_000.0 {
            MassClass::VeryLight
        } else if kappa < 165_000.0 {
            MassClass::Light
        } else if kappa < 260_000.0 {
            MassClass::Medium
        } else if kappa < 370_000.0 {
            MassClass::Heavy
        } else {
            MassClass::VeryHeavy
        }
    }

    /// Calculates exterior-to-mass conductance (h_tr_em) for 5R1C thermal network.
    ///
    /// This conductance represents heat transfer between exterior environment and
    /// thermal mass, accounting for:
    /// - Window U-value effects on overall envelope conductance
    /// - Thermal bridge effects (edge conditions, corner effects)
    /// - Surface area scaling
    ///
    /// # Arguments
    /// * `window_u_value` - Window thermal transmittance in W/m²K
    /// * `surface_area` - Exterior surface area in m²
    ///
    /// # Returns
    /// Conductance in W/K (thermal conductance, not transmittance)
    ///
    /// # Note
    /// This is a placeholder method that will be implemented in Plan 02.
    /// The final implementation should combine wall U-value, window U-value,
    /// and thermal bridge corrections.
    pub fn calc_h_tr_em(&self, _window_u_value: f64, surface_area: f64) -> f64 {
        // Exterior-to-mass conductance = (U_construction + U_window × window_area_fraction) × surface_area
        // This accounts for both the opaque construction conductance and window conductance
        //
        // For simplified 5R1C model, we use the construction U-value directly
        // The window conductance is handled separately in h_tr_w
        //
        // Units: W/m²K × m² = W/K
        let construction_u_value = self.u_value(None, None);
        construction_u_value * surface_area
    }

    /// Calculates window conductance (h_tr_w) for 5R1C thermal network.
    ///
    /// This conductance represents heat transfer through glazed surfaces
    /// (exterior-to-interior through windows).
    ///
    /// # Arguments
    /// * `window_u_value` - Window thermal transmittance in W/m²K
    /// * `window_area` - Total window area in m²
    ///
    /// # Returns
    /// Conductance in W/K
    ///
    /// # Formula
    /// h_tr_w = U_window × A_window
    pub fn calc_h_tr_w(&self, window_u_value: f64, window_area: f64) -> f64 {
        // Window conductance = U_value × window_area
        // Units: W/m²K × m² = W/K
        window_u_value * window_area
    }

    /// Calculates mass-to-surface conductance (h_tr_ms) for 5R1C thermal network.
    ///
    /// This conductance represents heat transfer between thermal mass
    /// and interior surface of building envelope.
    ///
    /// # Arguments
    /// * `surface_area` - Interior surface area in m²
    ///
    /// # Returns
    /// Conductance in W/K
    ///
    /// The h_ms coefficient depends on mass class per ISO 13790:
    /// - VeryLight/Light: 2.0 W/m²K (furniture/internal mass dominates)
    /// - Medium/Heavy/VeryHeavy: 9.1 W/m²K (envelope mass dominates)
    pub fn calc_h_tr_ms(&self, surface_area: f64) -> f64 {
        let h_ms = self.iso_13790_mass_class().h_ms_coeff();
        h_ms * surface_area
    }

    /// Calculates surface-to-interior conductance (h_tr_is) for 5R1C thermal network.
    ///
    /// This conductance represents heat transfer between interior surface
    /// and zone air, typically dominated by interior film coefficient.
    ///
    /// # Arguments
    /// * `surface_area` - Interior surface area in m²
    ///
    /// # Returns
    /// Conductance in W/K
    ///
    /// # Formula
    /// h_tr_is = h_si × A_si
    ///
    /// Where h_si is interior surface film coefficient and A_si is interior surface area.
    pub fn calc_h_tr_is(&self, surface_area: f64) -> f64 {
        // Surface-to-interior conductance = h_si × A_si
        // Where h_si is interior surface film coefficient
        // For ASHRAE 140 simplified 5R1C model, use h_si = 3.45 W/m²K
        // Units: W/m²K × m² = W/K
        const H_SI: f64 = 3.45; // W/m²K - ASHRAE 140 simplified 5R1C value
        H_SI * surface_area
    }

    /// Calculates exterior-to-mass conductance with thermal bridge correction.
    ///
    /// This variant of calc_h_tr_em includes optional thermal bridge effects
    /// for more accurate modeling of edge conditions and corner effects.
    ///
    /// # Arguments
    /// * `window_u_value` - Window thermal transmittance in W/m²K
    /// * `surface_area` - Exterior surface area in m²
    /// * `include_thermal_bridge` - Whether to apply thermal bridge correction
    ///
    /// # Returns
    /// Conductance in W/K
    pub fn calc_h_tr_em_with_thermal_bridge(
        &self,
        window_u_value: f64,
        surface_area: f64,
        include_thermal_bridge: bool,
    ) -> f64 {
        // Exterior-to-mass conductance with optional thermal bridge correction
        //
        // Thermal bridges represent additional heat transfer paths through
        // edge conditions, corner effects, and structural connections
        //
        // Thermal bridges are modeled using ISO 10211 psi/chi values:
        // - Linear thermal bridges (psi-values): heat flow = psi * length * delta_T
        // - Point thermal bridges (chi-values): heat flow = chi * count * delta_T
        //
        // Units: W/m²K × m² = W/K
        let base_conductance = self.calc_h_tr_em(window_u_value, surface_area);

        if include_thermal_bridge {
            // Calculate thermal bridge contribution using ISO 10211 psi/chi values
            //
            // Typical psi-values (W/mK) per ASHRAE 140 / ISO 10211:
            // - Wall-floor junction (intermediate floor): 0.10-0.30 W/mK
            // - Wall-floor junction (ground floor): 0.05-0.20 W/mK
            // - Wall-roof junction: 0.10-0.30 W/mK
            // - Wall-window frame: 0.05-0.15 W/mK
            // - Corner (external): 0.10-0.20 W/mK
            // - Corner (internal): 0.05-0.10 W/mK
            //
            // Typical chi-values (W/K) per ASHRAE 140 / ISO 10211:
            // - Structural fastener: 0.001-0.005 W/K
            // - Support bracket: 0.002-0.010 W/K
            // - Penetration: 0.001-0.003 W/K
            //
            // For ASHRAE 140 simplified modeling, we use typical values:
            const PSI_EDGE: f64 = 0.15; // W/mK - typical edge linear bridge
            const _PSI_CORNER: f64 = 0.10; // W/mK - typical corner linear bridge
            const CHI_POINT: f64 = 0.002; // W/K - typical point bridge

            // Perimeter-to-area ratio for typical building
            // For a rectangular building: P/A = 2*(L+W)/(L*W) = 2*(1/W + 1/L)
            // With typical aspect ratio 2:1, P/A ≈ 3/L where L is the length
            // This gives ~3m perimeter per m² of surface area
            let perimeter_to_area_ratio = 3.0; // m/m² (typical for rectangular buildings)

            // Calculate linear thermal bridge length per surface area
            let linear_bridge_length = surface_area * perimeter_to_area_ratio;

            // Calculate point thermal bridge count per surface area
            // Typical: 1 fastener per 0.5 m²
            let point_bridge_count = (surface_area / 0.5) as usize;

            // Thermal bridge conductance contribution
            // H_bridge = psi * L + chi * n
            let linear_conductance = PSI_EDGE * linear_bridge_length;
            let point_conductance = CHI_POINT * point_bridge_count as f64;
            let total_bridge_conductance = linear_conductance + point_conductance;

            // Add base conductance and bridge conductance
            base_conductance + total_bridge_conductance
        } else {
            base_conductance
        }
    }
}

/// Thermal mass classification per ISO 13790 Annex C.
///
/// Construction assemblies are classified by their effective specific thermal capacitance
/// (κ, kappa) in J/m²K. This classification determines the effective
/// mass area multiplier (A_m factor) used in 5R1C thermal networks.
///
/// # ISO 13790 Table C.2 Mass Classes
///
/// | Mass Class | κ (J/m²K) | A_m Factor |
/// |-------------|--------------|------------|
/// | VeryLight   | < 80,000     | 2.5        |
/// | Light        | 80,000-165,000 | 2.5        |
/// | Medium      | 165,000-260,000 | 2.5        |
/// | Heavy       | 260,000-370,000 | 3.0        |
/// | VeryHeavy   | ≥ 370,000    | 3.5        |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MassClass {
    /// Very light mass construction (κ < 80,000 J/m²K)
    /// Examples: Timber frame with minimal mass
    VeryLight,
    /// Light mass construction (80,000 ≤ κ < 165,000 J/m²K)
    /// Examples: Light masonry, timber frame with some internal mass
    Light,
    /// Medium mass construction (165,000 ≤ κ < 260,000 J/m²K)
    /// Examples: Concrete walls with internal insulation
    Medium,
    /// Heavy mass construction (260,000 ≤ κ < 370,000 J/m²K)
    /// Examples: Thick concrete walls, brick with minimal insulation
    Heavy,
    /// Very heavy mass construction (κ ≥ 370,000 J/m²K)
    /// Examples: Massive concrete or masonry structures
    VeryHeavy,
}

impl MassClass {
    /// Returns the effective mass area multiplier (A_m factor) per ISO 13790 Table C.2.
    ///
    /// This factor is used to calculate the effective mass area A_m:
    /// A_m = a_m_factor × floor_area
    ///
    /// # Returns
    /// Effective mass area multiplier (dimensionless)
    ///
    /// # ISO 13790 Reference
    /// ISO 13790 Annex C, Table C.2 specifies these multipliers based on
    /// construction thermal mass classification.
    pub fn a_m_factor(&self) -> f64 {
        match self {
            MassClass::VeryLight => 2.5,
            MassClass::Light => 2.5,
            MassClass::Medium => 2.5,
            MassClass::Heavy => 3.0,
            MassClass::VeryHeavy => 3.5,
        }
    }

    /// Returns the effective specific capacitance range for this mass class.
    ///
    /// # Returns
    /// Tuple of (min_kappa, max_kappa) in J/m²K
    pub fn kappa_range(&self) -> (f64, f64) {
        match self {
            MassClass::VeryLight => (0.0, 80_000.0),
            MassClass::Light => (80_000.0, 165_000.0),
            MassClass::Medium => (165_000.0, 260_000.0),
            MassClass::Heavy => (260_000.0, 370_000.0),
            MassClass::VeryHeavy => (370_000.0, f64::INFINITY),
        }
    }

    /// Returns the thermal coupling coefficient (h_ms) for t_i_free calculation.
    ///
    /// For low-mass buildings (VeryLight/Light), the ISO 13790 admittance method
    /// produces h_ms values that are too large, causing t_i_free to be dominated
    /// by mass temperature instead of tracking outdoor conditions.
    ///
    /// Per ISO 13790 Table C.2, the admittance method is calibrated for
    /// medium+ mass classes. For VeryLight/Light, we use a reduced coefficient
    /// that allows proper thermal coupling in the t_i_free formula.
    ///
    /// # Returns
    /// h_ms coefficient in W/(m²·K)
    ///
    /// # Physical Basis
    /// - VeryLight/Light: h_ms = 2.0 W/(m²·K) — furniture and lightweight internal mass
    /// - Medium+: h_ms = 9.1 W/(m²·K) — ISO 13790 full admittance method
    pub fn h_ms_coeff(&self) -> f64 {
        match self {
            MassClass::VeryLight => 2.0,
            MassClass::Light => 2.0,
            MassClass::Medium => 9.1,
            MassClass::Heavy => 9.1,
            MassClass::VeryHeavy => 9.1,
        }
    }
}

/// Pre-defined material properties for common building materials.
///
/// These materials are specified in ASHRAE 140 and other building energy
/// modeling standards.
pub struct Materials;

impl Materials {
    /// Plasterboard (gypsum board)
    pub fn plasterboard(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Plasterboard", 0.16, 784.0, 840.0, thickness)
    }

    /// Fiberglass insulation
    pub fn fiberglass(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, thickness)
    }

    /// Wood siding
    pub fn wood_siding(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Wood Siding", 0.14, 530.0, 900.0, thickness)
    }

    /// Concrete (normal weight)
    pub fn concrete(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Concrete", 1.13, 1400.0, 1000.0, thickness)
    }

    /// Concrete block (ASHRAE 140 Case 900)
    ///
    /// Concrete blocks have lower thermal conductivity (k=0.51 W/mK) than normal concrete (k=1.13 W/mK).
    /// This is specified in ASHRAE 140 Table 7-27 for high-mass construction.
    pub fn concrete_block(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Concrete Block", 0.51, 1400.0, 840.0, thickness)
    }

    /// Foam insulation
    pub fn foam(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Foam", 0.04, 14.0, 1400.0, thickness)
    }

    /// Timber/wood framing
    pub fn timber(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Timber", 0.14, 600.0, 1600.0, thickness)
    }

    /// Roof decking
    pub fn roof_deck(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Roof Deck", 0.14, 500.0, 1300.0, thickness)
    }

    /// Concrete slab (heavy mass)
    ///
    /// **Note:** these are normal-weight slab properties (k=1.13, ρ=1400, cp=1000).
    /// For ASHRAE 140-2023 Case 900-series construction, use
    /// [`Materials::concrete_heavyweight`] instead, which carries the medium-density
    /// values from Table B1-3 (Issue #730).
    pub fn concrete_slab(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Concrete Slab", 1.13, 1400.0, 1000.0, thickness)
    }

    /// Heavyweight (medium-density) concrete per ASHRAE 140-2023 Table B1-3.
    ///
    /// Used for the 900-series high-mass floor slab and (eventually) other
    /// BESTEST heavyweight constructions. Values are the BESTEST medium-density
    /// concrete spec — NOT normal-weight structural concrete.
    ///
    /// | Property | Value | Source |
    /// |----------|-------|--------|
    /// | k        | 0.51 W/m·K | ASHRAE 140-2023 Table B1-3 |
    /// | ρ        | 1400 kg/m³ | ASHRAE 140-2023 Table B1-3 |
    /// | cp       | 840 J/kg·K | ASHRAE 140-2023 Table B1-3 |
    ///
    /// Reference: Judkoff & Neymark (1995), NREL/TP-472-6231, §4.2.2.
    /// Closes Issue #730 (medium-density vs normal-weight concrete confusion).
    pub fn concrete_heavyweight(thickness: f64) -> ConstructionLayer {
        // ASHRAE 140-2023 Table B1-3, BESTEST heavyweight (medium-density) concrete.
        ConstructionLayer::new(
            "Concrete (ASHRAE 140 heavyweight)",
            0.51,
            1400.0,
            840.0,
            thickness,
        )
    }

    /// Insulation for floor/walls
    pub fn insulation_high_mass(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new("Insulation", 0.04, 14.0, 1400.0, thickness)
    }
}

/// Pre-defined construction assemblies from ASHRAE 140 test cases.
///
/// These constructions are used in the ASHRAE Standard 140 validation test cases.
pub struct Assemblies;

impl Assemblies {
    /// Low mass wall construction (ASHRAE 140 Case 600).
    pub fn low_mass_wall() -> Construction {
        Construction::new(vec![
            Materials::plasterboard(0.012),
            Materials::fiberglass(0.066),
            Materials::wood_siding(0.009),
        ])
    }

    /// Low mass roof construction (ASHRAE 140 Case 600).
    pub fn low_mass_roof() -> Construction {
        Construction::new(vec![
            Materials::plasterboard(0.010),
            Materials::fiberglass(0.1118),
            Materials::roof_deck(0.019),
        ])
    }

    /// High mass wall construction (ASHRAE 140 Table 7-27).
    ///
    /// Layers ordered from INTERIOR to EXTERIOR per ASHRAE 140:
    /// - Interior: wood_siding (12mm)
    /// - Middle: foam insulation (61.5mm)
    /// - Exterior: concrete block (100mm)
    ///
    /// The 200mm air gap in Table 7-27 refers to the cavity created by the
    /// concrete block construction method (stacked blocks with mortar).
    pub fn high_mass_wall() -> Construction {
        Construction::new(vec![
            Materials::wood_siding(0.009), // ASHRAE 140: k=0.16 W/mK (interior layer)
            Materials::foam(0.0615), // ASHRAE 140: k=0.04 W/mK, thickness=0.0615m (insulation)
            Materials::concrete_block(0.100), // ASHRAE 140: k=0.51 W/mK (exterior layer)
        ])
    }

    /// Simple concrete wall (for inter-zone partitions).
    pub fn concrete_wall(thickness: f64) -> Construction {
        Construction::new(vec![Materials::concrete(thickness)])
    }

    /// High mass roof construction (ASHRAE 140 Case 900).
    pub fn high_mass_roof() -> Construction {
        Construction::new(vec![
            Materials::concrete(0.080),
            Materials::foam(0.111), // Adjusted for U=0.318
            Materials::roof_deck(0.019),
        ])
    }

    /// Insulated floor construction (ASHRAE 140 Case 600).
    pub fn insulated_floor() -> Construction {
        Construction::new(vec![
            Materials::timber(0.025),
            Materials::fiberglass(0.197), // Adjusted for U=0.190
        ])
    }

    /// High mass wall construction (ASHRAE 140 Case 900) - alias for consistency.
    pub fn high_mass_wall_standard() -> Construction {
        Self::high_mass_wall()
    }

    /// High mass floor construction (ASHRAE 140 Case 900).
    ///
    /// Slab uses [`Materials::concrete_heavyweight`] per ASHRAE 140-2023 Table B1-3
    /// (k=0.51, ρ=1400, cp=840). Closes Issue #730.
    pub fn high_mass_floor() -> Construction {
        Construction::new(vec![
            Materials::concrete_heavyweight(0.080),
            Materials::insulation_high_mass(0.201), // Adjusted for U=0.190
        ])
    }

    /// Calculates ventilation conductance (h_ve) from air change rate.
    ///
    /// This conductance represents heat transfer due to air exchange between
    /// interior and exterior through ventilation and infiltration.
    ///
    /// # Arguments
    /// * `ach` - Air change rate in air changes per hour (ACH)
    /// * `zone_volume` - Zone volume in m³
    ///
    /// # Returns
    /// Conductance in W/K
    ///
    /// # Formula
    /// h_ve = ρ × cp × (ACH / 3600) × V
    ///
    /// Where:
    /// - ρ = air density (≈1.2 kg/m³ at standard conditions)
    /// - cp = air specific heat capacity (≈1005 J/kg·K)
    /// - ACH = air changes per hour
    /// - V = zone volume in m³
    /// - 3600 = seconds per hour conversion
    pub fn calc_h_ve(&self, ach: f64, zone_volume: f64) -> f64 {
        // Ventilation conductance = ρ × cp × (ACH/3600) × V
        // Where:
        // - ρ = air density (kg/m³) = 1.225 kg/m³ at sea level
        // - cp = specific heat of air (J/kg·K) = 1005 J/kg·K
        // - ACH = air changes per hour (1/hr)
        // - V = zone volume (m³)
        // - 3600 = seconds per hour (to convert ACH to per second)
        // Units: kg/m³ × J/kg·K × (1/hr ÷ 3600 s/hr) × m³ = W/K
        AIR_DENSITY_SEA_LEVEL * AIR_SPECIFIC_HEAT * (ach / 3600.0) * zone_volume
    }
}

impl Construction {
    /// Creates a simple single-layer wall construction with the specified material R-value.
    ///
    /// This is a convenience method for quick prototyping. It creates a construction
    /// with a single insulating layer whose thickness is computed to achieve the desired
    /// R-value using a typical insulation conductivity (0.04 W/m·K).
    ///
    /// # Arguments
    /// * `r_value` - Desired thermal resistance of the materials (m²K/W)
    ///
    /// # Returns
    /// A Construction with a single layer achieving the specified R-value.
    pub fn simple_wall(r_value: f64) -> Self {
        let conductivity = 0.04; // W/m·K, typical for fiberglass insulation
        let thickness = r_value * conductivity;
        let layer = ConstructionLayer::new(
            "Simple Wall",
            conductivity,
            30.0,   // density (kg/m³)
            1000.0, // specific heat (J/kg·K)
            thickness,
        );
        Construction::new(vec![layer])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_construction_layer_creation() {
        let layer = ConstructionLayer::new("Test", 0.04, 12.0, 840.0, 0.066);

        assert_eq!(layer.name, "Test");
        assert_eq!(layer.conductivity, 0.04);
        assert_eq!(layer.density, 12.0);
        assert_eq!(layer.specific_heat, 840.0);
        assert_eq!(layer.thickness, 0.066);
        assert_eq!(layer.emissivity, 0.9);
        assert_eq!(layer.absorptance, 0.7);
    }

    #[test]
    fn test_construction_layer_with_custom_surface_properties() {
        let layer = ConstructionLayer::with_surface_properties(
            "Test", 0.04, 12.0, 840.0, 0.066, 0.85, 0.65,
        );

        assert_eq!(layer.name, "Test");
        assert_eq!(layer.emissivity, 0.85);
        assert_eq!(layer.absorptance, 0.65);
    }

    #[test]
    #[should_panic(expected = "Conductivity must be positive")]
    fn test_construction_layer_invalid_conductivity() {
        ConstructionLayer::new("Test", -0.04, 12.0, 840.0, 0.066);
    }

    #[test]
    #[should_panic(expected = "Density must be positive")]
    fn test_construction_layer_invalid_density() {
        ConstructionLayer::new("Test", 0.04, -12.0, 840.0, 0.066);
    }

    #[test]
    #[should_panic(expected = "Thickness must be positive")]
    fn test_construction_layer_invalid_thickness() {
        ConstructionLayer::new("Test", 0.04, 12.0, 840.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Emissivity must be in [0, 1]")]
    fn test_construction_layer_invalid_emissivity() {
        ConstructionLayer::with_surface_properties("Test", 0.04, 12.0, 840.0, 0.066, 1.5, 0.7);
    }

    #[test]
    fn test_layer_r_value() {
        let layer = ConstructionLayer::new("Test", 0.04, 12.0, 840.0, 0.066);

        // R = δ / k = 0.066 / 0.04 = 1.65 m²K/W
        let expected_r = 0.066 / 0.04;
        assert!((layer.r_value() - expected_r).abs() < EPSILON);
    }

    #[test]
    fn test_layer_thermal_capacitance_per_area() {
        let layer = ConstructionLayer::new("Test", 0.04, 12.0, 840.0, 0.066);

        // C/A = ρ × δ × Cp = 12.0 × 0.066 × 840.0 = 665.28 J/m²K
        let expected_c = 12.0 * 0.066 * 840.0;
        assert!((layer.thermal_capacitance_per_area() - expected_c).abs() < EPSILON);
    }

    #[test]
    fn test_construction_creation() {
        let layers = vec![
            ConstructionLayer::new("Layer 1", 0.16, 950.0, 840.0, 0.012),
            ConstructionLayer::new("Layer 2", 0.04, 12.0, 840.0, 0.066),
        ];
        let construction = Construction::new(layers);

        assert_eq!(construction.layer_count(), 2);
    }

    #[test]
    #[should_panic(expected = "Construction must have at least one layer")]
    fn test_construction_empty_layers() {
        Construction::new(vec![]);
    }

    #[test]
    fn test_construction_r_value_total() {
        let construction = Assemblies::low_mass_wall();

        // Calculate expected R-value
        // R_int = 1 / 8.29 ≈ 0.120627 m²K/W (ASHRAE 140 Sec. 5.2)
        // R_plasterboard = 0.012 / 0.16 = 0.075
        // R_fiberglass = 0.066 / 0.04 = 1.65
        // R_siding = 0.009 / 0.14 ≈ 0.064286
        // R_ext = 1 / 18.3 ≈ 0.054645 m²K/W (ASHRAE 140 Sec. 5.2)
        // R_total ≈ 0.120627 + 0.075 + 1.65 + 0.064286 + 0.054645 = 1.964558
        let r_total = construction.r_value_total(None, None);

        let expected_r = 1.0 / 8.29 + 0.012 / 0.16 + 0.066 / 0.04 + 0.009 / 0.14 + 1.0 / 18.3;
        assert!((r_total - expected_r).abs() < EPSILON);

        // Check that U = 1/R
        let u_value = construction.u_value(None, None);
        assert!((u_value - 1.0 / r_total).abs() < EPSILON);
    }

    #[test]
    fn test_construction_u_value() {
        let construction = Assemblies::low_mass_wall();
        let u_value = construction.u_value(None, None);

        // For Case 600 wall: expected U ≈ 0.514 W/m²K
        // This may vary slightly due to different assumptions about film coefficients
        assert!(u_value > 0.5);
        assert!(u_value < 0.6);
    }

    #[test]
    fn test_construction_u_value_with_wind_speed() {
        let construction = Assemblies::low_mass_wall();

        // Test with different wind speeds
        let u_no_wind = construction.u_value(None, Some(0.0));
        let u_low_wind = construction.u_value(None, Some(2.0));
        let u_high_wind = construction.u_value(None, Some(10.0));

        // Higher wind speed → higher exterior film coefficient → lower resistance → higher U
        assert!(u_high_wind > u_low_wind);
        assert!(u_low_wind > u_no_wind);
    }

    #[test]
    fn test_interior_film_coeff() {
        let h_int = interior_film_coeff();
        assert_eq!(h_int, 8.29);
    }

    #[test]
    fn test_exterior_film_coeff() {
        // Test with low wind
        let h_low = exterior_film_coeff(2.0);
        assert!((h_low - (10.0 + 4.0 * 2.0_f64.sqrt())).abs() < EPSILON);

        // Test with high wind
        let h_high = exterior_film_coeff(10.0);
        assert!((h_high - (10.0 + 4.0 * 10.0_f64.sqrt())).abs() < EPSILON);

        // High wind should have higher film coefficient
        assert!(h_high > h_low);

        // Reasonable range check
        assert!(h_low > 15.0 && h_low < 30.0);
        assert!(h_high > 20.0 && h_high < 40.0);
    }

    #[test]
    fn test_materials_plasterboard() {
        let layer = Materials::plasterboard(0.012);
        assert_eq!(layer.conductivity, 0.16);
        assert_eq!(layer.density, 784.0);
        assert_eq!(layer.specific_heat, 840.0);
        assert_eq!(layer.thickness, 0.012);
    }

    #[test]
    fn test_materials_fiberglass() {
        let layer = Materials::fiberglass(0.066);
        assert_eq!(layer.conductivity, 0.04);
        assert_eq!(layer.density, 12.0);
        assert_eq!(layer.specific_heat, 840.0);
        assert_eq!(layer.thickness, 0.066);
    }

    #[test]
    fn test_materials_wood_siding() {
        let layer = Materials::wood_siding(0.009);
        assert_eq!(layer.conductivity, 0.14);
        assert_eq!(layer.density, 530.0);
        assert_eq!(layer.specific_heat, 900.0);
        assert_eq!(layer.thickness, 0.009);
    }

    #[test]
    fn test_materials_concrete() {
        let layer = Materials::concrete(0.100);
        assert_eq!(layer.conductivity, 1.13);
        assert_eq!(layer.density, 1400.0);
        assert_eq!(layer.specific_heat, 1000.0);
        assert_eq!(layer.thickness, 0.100);
    }

    #[test]
    fn test_materials_foam() {
        let layer = Materials::foam(0.0615);
        assert_eq!(layer.conductivity, 0.04);
        assert_eq!(layer.density, 14.0);
        assert_eq!(layer.specific_heat, 1400.0);
        assert_eq!(layer.thickness, 0.0615);
    }

    #[test]
    fn test_assemblies_low_mass_wall() {
        let wall = Assemblies::low_mass_wall();
        assert_eq!(wall.layer_count(), 3);

        // Check layer properties
        assert_eq!(wall.layers[0].thickness, 0.012); // Plasterboard
        assert_eq!(wall.layers[1].thickness, 0.066); // Fiberglass
        assert_eq!(wall.layers[2].thickness, 0.009); // Siding
    }

    #[test]
    fn test_assemblies_low_mass_roof() {
        let roof = Assemblies::low_mass_roof();
        assert_eq!(roof.layer_count(), 3);

        // Check layer properties
        assert_eq!(roof.layers[0].thickness, 0.010); // Plasterboard
        assert_eq!(roof.layers[1].thickness, 0.1118); // Fiberglass
        assert_eq!(roof.layers[2].thickness, 0.019); // Deck
    }

    #[test]
    fn test_assemblies_high_mass_wall() {
        let wall = Assemblies::high_mass_wall();
        assert_eq!(wall.layer_count(), 3);

        // Check layer properties - ordered INTERIOR to EXTERIOR per ASHRAE 140 Table 7-27
        assert_eq!(wall.layers[0].thickness, 0.009); // Wood siding (interior)
        assert_eq!(wall.layers[1].thickness, 0.0615); // Foam (insulation)
        assert_eq!(wall.layers[2].thickness, 0.100); // Concrete block (exterior)
    }

    #[test]
    fn test_assemblies_high_mass_roof() {
        let roof = Assemblies::high_mass_roof();
        assert_eq!(roof.layer_count(), 3);

        // Check layer properties
        assert_eq!(roof.layers[0].thickness, 0.080); // Concrete
        assert_eq!(roof.layers[1].thickness, 0.111); // Foam
        assert_eq!(roof.layers[2].thickness, 0.019); // Deck
    }

    #[test]
    fn test_assemblies_insulated_floor() {
        let floor = Assemblies::insulated_floor();
        assert_eq!(floor.layer_count(), 2);

        // Check layer properties
        assert_eq!(floor.layers[0].thickness, 0.025); // Timber
        assert_eq!(floor.layers[1].thickness, 0.197); // Insulation
    }

    #[test]
    fn test_construction_thermal_capacitance_per_area() {
        let wall = Assemblies::low_mass_wall();
        let c_per_area = wall.thermal_capacitance_per_area();

        // Calculate expected value
        // Plasterboard: 950 × 0.012 × 840 = 9576
        // Fiberglass: 12 × 0.066 × 840 = 665.28
        // Siding: 500 × 0.009 × 1300 = 5850
        // Total: 9576 + 665.28 + 5850 = 16091.28 J/m²K
        let expected_c = 784.0 * 0.012 * 840.0 + 12.0 * 0.066 * 840.0 + 530.0 * 0.009 * 900.0;
        assert!((c_per_area - expected_c).abs() < EPSILON);
    }

    #[test]
    fn test_construction_total_thickness() {
        let wall = Assemblies::low_mass_wall();
        let thickness = wall.total_thickness();

        let expected = 0.012 + 0.066 + 0.009;
        assert!((thickness - expected).abs() < EPSILON);
    }

    #[test]
    fn test_high_mass_vs_low_mass_capacitance() {
        let low_mass = Assemblies::low_mass_wall();
        let high_mass = Assemblies::high_mass_wall();

        let c_low = low_mass.thermal_capacitance_per_area();
        let c_high = high_mass.thermal_capacitance_per_area();

        // High mass should have much higher thermal capacitance
        assert!(c_high > 3.0 * c_low);
    }

    #[test]
    fn test_serialization() {
        let wall = Assemblies::low_mass_wall();

        // Test serialization
        let json = serde_json::to_string(&wall).expect("Failed to serialize");
        assert!(json.contains("conductivity"));
        assert!(json.contains("thickness"));

        // Test deserialization
        let deserialized: Construction =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.layer_count(), wall.layer_count());
    }

    #[test]
    fn test_surface_type_interior_film_coeff() {
        assert_eq!(
            SurfaceType::interior_film_coeff(SurfaceType::Wall),
            INTERIOR_FILM_COEFF_WALL
        );
        assert_eq!(
            SurfaceType::interior_film_coeff(SurfaceType::Ceiling),
            INTERIOR_FILM_COEFF_CEILING
        );
        assert_eq!(
            SurfaceType::interior_film_coeff(SurfaceType::Floor),
            INTERIOR_FILM_COEFF_FLOOR
        );
    }

    #[test]
    fn test_r_value_materials() {
        let wall = Assemblies::low_mass_wall();
        let r_mat = wall.r_value_materials();
        let expected = 0.012 / 0.16 + 0.066 / 0.04 + 0.009 / 0.14;
        assert!((r_mat - expected).abs() < EPSILON);
    }

    #[test]
    fn test_r_value_total_with_surface_type() {
        let wall = Assemblies::low_mass_wall();
        let r_wall = wall.r_value_total(Some(SurfaceType::Wall), None);
        let r_ceiling = wall.r_value_total(Some(SurfaceType::Ceiling), None);
        let r_floor = wall.r_value_total(Some(SurfaceType::Floor), None);
        assert!(r_wall != r_ceiling || r_wall != r_floor);
    }

    #[test]
    fn test_r_value_total_with_wind_speed() {
        let wall = Assemblies::low_mass_wall();
        let r_calm = wall.r_value_total(None, Some(0.0));
        let r_breezy = wall.r_value_total(None, Some(5.0));
        let r_stormy = wall.r_value_total(None, Some(20.0));
        assert!(r_calm > r_breezy);
        assert!(r_breezy > r_stormy);
    }

    #[test]
    fn test_u_value_internal() {
        let wall = Assemblies::low_mass_wall();
        let u_int = wall.u_value_internal();
        let u_ext = wall.u_value(None, None);
        assert!(
            u_int < u_ext,
            "Internal U-value should be lower (more resistance)"
        );
    }

    #[test]
    fn test_find_dominant_insulation_layer() {
        let wall = Assemblies::low_mass_wall();
        assert_eq!(wall.find_dominant_insulation_layer_index(), 1);
    }

    #[test]
    fn test_find_dominant_insulation_layer_high_mass() {
        let wall = Assemblies::high_mass_wall();
        assert_eq!(wall.find_dominant_insulation_layer_index(), 1);
    }

    #[test]
    fn test_find_dominant_insulation_single_layer() {
        let wall = Construction::simple_wall(2.0);
        assert_eq!(wall.find_dominant_insulation_layer_index(), 0);
    }

    #[test]
    fn test_iso_13790_effective_capacitance() {
        let wall = Assemblies::low_mass_wall();
        let kappa = wall.iso_13790_effective_capacitance_per_area();
        // plasterboard (full) + fiberglass (full, dominant insulation); wood_siding excluded
        let expected = 784.0 * 0.012 * 840.0 + 12.0 * 0.066 * 840.0;
        assert!((kappa - expected).abs() < EPSILON);
    }

    #[test]
    fn test_iso_13790_effective_capacitance_high_mass() {
        let wall = Assemblies::high_mass_wall();
        let kappa = wall.iso_13790_effective_capacitance_per_area();
        // wood_siding (full) + foam (full, dominant insulation); concrete_block excluded
        let expected = 530.0 * 0.009 * 900.0 + 14.0 * 0.0615 * 1400.0;
        assert!((kappa - expected).abs() < EPSILON);
    }

    #[test]
    fn test_iso_13790_mass_class_very_light() {
        let wall = Construction::new(vec![Materials::fiberglass(0.05)]);
        assert_eq!(wall.iso_13790_mass_class(), MassClass::VeryLight);
    }

    #[test]
    fn test_iso_13790_mass_class_medium() {
        let layer = ConstructionLayer::with_surface_properties(
            "Concrete", 1.0, 2000.0, 1000.0, 0.1, 0.9, 0.7,
        );
        let wall = Construction::new(vec![layer]);
        assert_eq!(wall.iso_13790_mass_class(), MassClass::Medium);
    }

    #[test]
    fn test_iso_13790_mass_class_heavy() {
        let layer = ConstructionLayer::with_surface_properties(
            "ThickConcrete",
            1.5,
            2400.0,
            1000.0,
            0.15,
            0.9,
            0.7,
        );
        let wall = Construction::new(vec![layer]);
        assert_eq!(wall.iso_13790_mass_class(), MassClass::Heavy);
    }

    #[test]
    fn test_iso_13790_mass_class_very_heavy() {
        let layer = ConstructionLayer::with_surface_properties(
            "MassiveConcrete",
            1.5,
            2400.0,
            1000.0,
            0.2,
            0.9,
            0.7,
        );
        let wall = Construction::new(vec![layer]);
        assert_eq!(wall.iso_13790_mass_class(), MassClass::VeryHeavy);
    }

    #[test]
    fn test_mass_class_a_m_factor() {
        assert_eq!(MassClass::VeryLight.a_m_factor(), 2.5);
        assert_eq!(MassClass::Light.a_m_factor(), 2.5);
        assert_eq!(MassClass::Medium.a_m_factor(), 2.5);
        assert_eq!(MassClass::Heavy.a_m_factor(), 3.0);
        assert_eq!(MassClass::VeryHeavy.a_m_factor(), 3.5);
    }

    #[test]
    fn test_mass_class_kappa_range() {
        assert_eq!(MassClass::VeryLight.kappa_range(), (0.0, 80_000.0));
        assert_eq!(MassClass::Light.kappa_range(), (80_000.0, 165_000.0));
        assert_eq!(MassClass::Medium.kappa_range(), (165_000.0, 260_000.0));
        assert_eq!(MassClass::Heavy.kappa_range(), (260_000.0, 370_000.0));
        assert_eq!(
            MassClass::VeryHeavy.kappa_range(),
            (370_000.0, f64::INFINITY)
        );
    }

    #[test]
    fn test_mass_class_equality() {
        assert_eq!(MassClass::Light, MassClass::Light);
        assert_ne!(MassClass::Light, MassClass::Heavy);
    }

    #[test]
    fn test_calc_h_tr_em() {
        let wall = Assemblies::low_mass_wall();
        let h = wall.calc_h_tr_em(1.5, 48.0);
        assert!(h > 0.0);
        let u = wall.u_value(None, None);
        assert!((h - u * 48.0).abs() < EPSILON);
    }

    #[test]
    fn test_calc_h_tr_w() {
        let wall = Assemblies::low_mass_wall();
        assert_eq!(wall.calc_h_tr_w(3.0, 12.0), 36.0);
    }

    #[test]
    fn test_calc_h_tr_ms() {
        let wall = Assemblies::low_mass_wall();
        assert_eq!(wall.calc_h_tr_ms(48.0), 2.0 * 48.0);
    }

    #[test]
    fn test_calc_h_tr_is() {
        let wall = Assemblies::low_mass_wall();
        assert_eq!(wall.calc_h_tr_is(48.0), 3.45 * 48.0);
    }

    #[test]
    fn test_calc_h_tr_em_with_thermal_bridge() {
        let wall = Assemblies::low_mass_wall();
        let h_no_bridge = wall.calc_h_tr_em_with_thermal_bridge(1.5, 48.0, false);
        let h_with_bridge = wall.calc_h_tr_em_with_thermal_bridge(1.5, 48.0, true);

        // Verify thermal bridge increases conductance
        assert!(
            h_with_bridge > h_no_bridge,
            "Thermal bridge should increase h_tr_em"
        );

        // Calculate expected bridge contribution
        // Linear: PSI_EDGE * (surface_area * 3.0) = 0.15 * (48.0 * 3.0) = 21.6 W/K
        // Point: CHI_POINT * (surface_area / 0.5) = 0.002 * 96 = 0.192 W/K
        // Total bridge conductance: 21.792 W/K
        let surface_area = 48.0;
        let linear_bridge_conductance = 0.15 * surface_area * 3.0;
        let point_bridge_conductance = 0.002 * (surface_area / 0.5);
        let expected_bridge_contribution = linear_bridge_conductance + point_bridge_conductance;

        // Verify bridge contribution is correct
        let actual_bridge_contribution = h_with_bridge - h_no_bridge;
        assert!((actual_bridge_contribution - expected_bridge_contribution).abs() < EPSILON);
    }

    #[test]
    fn test_simple_wall() {
        let wall = Construction::simple_wall(3.0);
        assert_eq!(wall.layer_count(), 1);
        assert_eq!(wall.layers[0].name, "Simple Wall");
        assert!((wall.r_value_materials() - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_concrete_wall_factory() {
        let wall = Assemblies::concrete_wall(0.2);
        assert_eq!(wall.layer_count(), 1);
        assert_eq!(wall.layers[0].name, "Concrete");
        assert_eq!(wall.layers[0].thickness, 0.2);
    }

    #[test]
    fn test_high_mass_wall_standard_alias() {
        let wall1 = Assemblies::high_mass_wall();
        let wall2 = Assemblies::high_mass_wall_standard();
        assert_eq!(wall1.layer_count(), wall2.layer_count());
    }

    #[test]
    fn test_high_mass_floor() {
        let floor = Assemblies::high_mass_floor();
        assert_eq!(floor.layer_count(), 2);
        // Issue #730: ASHRAE 140-2023 Table B1-3 heavyweight (medium-density) concrete.
        assert_eq!(floor.layers[0].name, "Concrete (ASHRAE 140 heavyweight)");
        assert_eq!(floor.layers[1].name, "Insulation");
    }

    #[test]
    fn test_calc_h_ve() {
        let assemblies = Assemblies;
        let h_ve = assemblies.calc_h_ve(0.5, 240.0);
        assert!(h_ve > 0.0);
        let expected = AIR_DENSITY_SEA_LEVEL * AIR_SPECIFIC_HEAT * (0.5 / 3600.0) * 240.0;
        assert!((h_ve - expected).abs() < EPSILON);
    }

    #[test]
    fn test_calc_h_ve_zero_ach() {
        let assemblies = Assemblies;
        assert_eq!(assemblies.calc_h_ve(0.0, 240.0), 0.0);
    }

    #[test]
    fn test_materials_concrete_block() {
        let layer = Materials::concrete_block(0.1);
        assert_eq!(layer.conductivity, 0.51);
        assert_eq!(layer.density, 1400.0);
        assert_eq!(layer.specific_heat, 840.0);
    }

    #[test]
    fn test_materials_timber() {
        let layer = Materials::timber(0.15);
        assert_eq!(layer.conductivity, 0.14);
        assert_eq!(layer.density, 600.0);
        assert_eq!(layer.specific_heat, 1600.0);
    }

    #[test]
    fn test_materials_roof_deck() {
        let layer = Materials::roof_deck(0.02);
        assert_eq!(layer.conductivity, 0.14);
        assert_eq!(layer.density, 500.0);
        assert_eq!(layer.specific_heat, 1300.0);
    }

    #[test]
    fn test_materials_concrete_slab() {
        let layer = Materials::concrete_slab(0.1);
        assert_eq!(layer.conductivity, 1.13);
        assert_eq!(layer.density, 1400.0);
        assert_eq!(layer.specific_heat, 1000.0);
    }

    #[test]
    fn test_materials_concrete_heavyweight_matches_ashrae_140_table_b1_3() {
        // ASHRAE 140-2023 Table B1-3 — BESTEST heavyweight (medium-density) concrete.
        // Reference: Judkoff & Neymark (1995), NREL/TP-472-6231, §4.2.2.
        let layer = Materials::concrete_heavyweight(0.200);
        assert_eq!(layer.conductivity, 0.51, "k must match Table B1-3");
        assert_eq!(layer.density, 1400.0, "rho must match Table B1-3");
        assert_eq!(layer.specific_heat, 840.0, "cp must match Table B1-3");
        assert_eq!(layer.thickness, 0.200);
        // Areal heat capacity per Table B1-3: kappa = rho * cp * d = 1400 * 840 * 0.200
        let kappa = layer.density * layer.specific_heat * layer.thickness;
        assert!(
            (kappa - 235_200.0).abs() < 1.0,
            "kappa must be 235.2 kJ/m^2K"
        );
    }

    #[test]
    fn test_high_mass_floor_uses_ashrae_140_heavyweight_concrete() {
        // Issue #730: high_mass_floor() must source its slab from
        // Materials::concrete_heavyweight, not normal-weight Materials::concrete_slab.
        let floor = Assemblies::high_mass_floor();
        let slab = &floor.layers[0];
        assert_eq!(
            slab.conductivity, 0.51,
            "slab k must be ASHRAE 140 Table B1-3 value"
        );
        assert_eq!(
            slab.density, 1400.0,
            "slab rho must be ASHRAE 140 Table B1-3 value"
        );
        assert_eq!(
            slab.specific_heat, 840.0,
            "slab cp must be ASHRAE 140 Table B1-3 value"
        );
    }

    #[test]
    fn test_materials_insulation_high_mass() {
        let layer = Materials::insulation_high_mass(0.1);
        assert_eq!(layer.conductivity, 0.04);
        assert_eq!(layer.density, 14.0);
        assert_eq!(layer.specific_heat, 1400.0);
    }

    #[test]
    fn test_construction_layer_thermal_capacitance() {
        let layer = ConstructionLayer::new("Test", 0.5, 2000.0, 1000.0, 0.1);
        assert_eq!(layer.thermal_capacitance_per_area(), 200000.0);
    }

    #[test]
    fn test_construction_clone() {
        let wall = Assemblies::low_mass_wall();
        let cloned = wall.clone();
        assert_eq!(cloned.layer_count(), wall.layer_count());
        assert_eq!(cloned.r_value_materials(), wall.r_value_materials());
    }

    #[test]
    fn test_surface_type_equality() {
        assert_eq!(SurfaceType::Wall, SurfaceType::Wall);
        assert_ne!(SurfaceType::Wall, SurfaceType::Ceiling);
    }

    #[test]
    fn test_exterior_film_coeff_zero_wind() {
        assert_eq!(exterior_film_coeff(0.0), 10.0);
    }

    #[test]
    fn test_exterior_film_coeff_various_speeds() {
        assert!((exterior_film_coeff(1.0) - 14.0).abs() < EPSILON);
        assert!((exterior_film_coeff(4.0) - 18.0).abs() < EPSILON);
        assert!((exterior_film_coeff(9.0) - 22.0).abs() < EPSILON);
    }

    #[test]
    fn test_construction_layer_with_surface_properties_boundary() {
        let l1 = ConstructionLayer::with_surface_properties(
            "Boundary", 0.5, 1000.0, 840.0, 0.1, 0.0, 0.0,
        );
        assert_eq!(l1.emissivity, 0.0);
        assert_eq!(l1.absorptance, 0.0);
        let l2 = ConstructionLayer::with_surface_properties(
            "Boundary", 0.5, 1000.0, 840.0, 0.1, 1.0, 1.0,
        );
        assert_eq!(l2.emissivity, 1.0);
        assert_eq!(l2.absorptance, 1.0);
    }

    #[test]
    #[should_panic(expected = "Absorptance must be in [0, 1]")]
    fn test_construction_layer_with_surface_properties_invalid_absorptance() {
        ConstructionLayer::with_surface_properties("Bad", 0.5, 1000.0, 840.0, 0.1, 0.9, 1.5);
    }

    #[test]
    #[should_panic(expected = "Specific heat must be positive")]
    fn test_construction_layer_with_surface_properties_invalid_specific_heat() {
        ConstructionLayer::with_surface_properties("Bad", 0.5, 1000.0, -840.0, 0.1, 0.9, 0.7);
    }

    #[test]
    fn test_all_materials_factory_methods() {
        let t = 0.1;
        assert_eq!(Materials::plasterboard(t).thickness, t);
        assert_eq!(Materials::fiberglass(t).thickness, t);
        assert_eq!(Materials::wood_siding(t).thickness, t);
        assert_eq!(Materials::concrete(t).thickness, t);
        assert_eq!(Materials::concrete_block(t).thickness, t);
        assert_eq!(Materials::foam(t).thickness, t);
        assert_eq!(Materials::timber(t).thickness, t);
        assert_eq!(Materials::roof_deck(t).thickness, t);
        assert_eq!(Materials::concrete_slab(t).thickness, t);
        assert_eq!(Materials::insulation_high_mass(t).thickness, t);
    }

    #[test]
    fn test_assemblies_all_factory_methods() {
        assert_eq!(Assemblies::low_mass_wall().layer_count(), 3);
        assert_eq!(Assemblies::low_mass_roof().layer_count(), 3);
        assert_eq!(Assemblies::high_mass_wall().layer_count(), 3);
        assert_eq!(Assemblies::high_mass_roof().layer_count(), 3);
        assert_eq!(Assemblies::insulated_floor().layer_count(), 2);
        assert_eq!(Assemblies::high_mass_floor().layer_count(), 2);
        assert_eq!(Assemblies::concrete_wall(0.1).layer_count(), 1);
    }

    #[test]
    fn test_mass_class_serialization() {
        let class = MassClass::Heavy;
        let json = serde_json::to_string(&class).unwrap();
        assert!(json.contains("Heavy"));
        let restored: MassClass = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, MassClass::Heavy);
    }

    #[test]
    fn test_construction_layer_equality() {
        let l1 = ConstructionLayer::new("Test", 0.5, 1000.0, 840.0, 0.1);
        let l2 = ConstructionLayer::new("Test", 0.5, 1000.0, 840.0, 0.1);
        assert_eq!(l1, l2);
    }

    #[test]
    fn test_construction_layer_not_equal() {
        let l1 = ConstructionLayer::new("Test", 0.5, 1000.0, 840.0, 0.1);
        let l2 = ConstructionLayer::new("Test", 0.6, 1000.0, 840.0, 0.1);
        assert_ne!(l1, l2);
    }

    #[test]
    fn test_construction_u_value_with_ceiling_surface_type() {
        let roof = Assemblies::low_mass_roof();
        let u_ceiling = roof.u_value(Some(SurfaceType::Ceiling), None);
        let u_default = roof.u_value(None, None);
        assert!(u_ceiling != u_default);
    }

    #[test]
    fn test_construction_u_value_with_floor_surface_type() {
        let floor = Assemblies::insulated_floor();
        let u_floor = floor.u_value(Some(SurfaceType::Floor), None);
        let u_default = floor.u_value(None, None);
        assert!(u_floor != u_default);
    }
}
