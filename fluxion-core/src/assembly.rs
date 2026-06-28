//! Building Assembly Module
//!
//! This module provides trait-based material layer abstraction and building assembly
//! composition with validation. Materials are loaded from YAML configuration files
//! and assemblies are composed using a fluent builder pattern.
//!
//! # Thermal Mass Classification
//!
//! Thermal mass is auto-calculated per ISO 13790 Annex C:
//! - VeryLight: < 50 kJ/m²K
//! - Light: 50-150 kJ/m²K
//! - Medium: 150-260 kJ/m²K
//! - Heavy: 260-370 kJ/m²K
//! - VeryHeavy: > 370 kJ/m²K

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs;

/// Material properties loaded from YAML
#[derive(Debug, Deserialize)]
pub struct MaterialYAML {
    /// Thermal conductivity (W/mK)
    pub conductivity: f64,
    /// Material density (kg/m³)
    pub density: f64,
    /// Specific heat capacity (J/kgK)
    pub specific_heat: f64,
    /// Solar absorptance (0-1)
    pub absorptance: f64,
    /// Thermal emissivity (0-1)
    pub emissivity: f64,
}

/// Layer specification loaded from YAML
#[derive(Debug, Deserialize)]
pub struct LayerYAML {
    /// Material name (key in materials.yaml)
    pub material: String,
    /// Layer thickness (m)
    pub thickness: f64,
}

/// Assembly specification loaded from YAML
#[derive(Debug, Deserialize)]
pub struct AssemblyYAML {
    /// List of material layers (exterior to interior)
    pub layers: Vec<LayerYAML>,
}

/// Load material properties from YAML file
///
/// # Arguments
/// * `path` - Path to materials.yaml file
///
/// # Returns
/// HashMap mapping material names to their properties
///
/// # Errors
/// Returns error if file cannot be read or YAML is invalid
pub fn load_materials(path: &str) -> Result<HashMap<String, MaterialYAML>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read materials file '{}': {}", path, e))?;
    serde_yaml::from_str(&content).map_err(|e| format!("Failed to parse materials YAML: {}", e))
}

/// Load building assemblies from YAML file
///
/// # Arguments
/// * `path` - Path to assemblies.yaml file
///
/// # Returns
/// HashMap mapping assembly names to their specifications
///
/// # Errors
/// Returns error if file cannot be read or YAML is invalid
pub fn load_assemblies(path: &str) -> Result<HashMap<String, AssemblyYAML>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read assemblies file '{}': {}", path, e))?;
    serde_yaml::from_str(&content).map_err(|e| format!("Failed to parse assemblies YAML: {}", e))
}

/// Trait for material layer properties
///
/// Provides thermal and radiative properties for building material layers.
/// All materials implement this trait for consistent property access.
pub trait MaterialLayer: Send + Sync {
    /// Material name/identifier
    fn name(&self) -> &str;

    /// Thermal conductivity (W/mK)
    fn conductivity(&self) -> f64;

    /// Layer thickness (m)
    fn thickness(&self) -> f64;

    /// Material density (kg/m³)
    fn density(&self) -> f64;

    /// Specific heat capacity (J/kgK)
    fn specific_heat(&self) -> f64;

    /// Solar absorptance (0-1, dimensionless)
    fn absorptance(&self) -> f64;

    /// Thermal emissivity (0-1, dimensionless)
    fn emissivity(&self) -> f64;

    /// R-value (thermal resistance, m²K/W)
    ///
    /// Computed as thickness / conductivity
    fn r_value(&self) -> f64 {
        self.thickness() / self.conductivity()
    }

    /// Returns self as Any for downcasting (needed for cloning)
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Concrete material (typical construction material)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcreteMaterial {
    thickness: f64,
    conductivity: f64,
    density: f64,
    specific_heat: f64,
    absorptance: f64,
    emissivity: f64,
}

impl ConcreteMaterial {
    /// Create concrete material with default properties.
    ///
    /// **Default Thermal Conductivity:** 1.4 W/mK
    /// **Units:** W/mK (watts per meter Kelvin)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ISO 10456, Thermal Insulation Products
    /// **Uncertainty:** ±0.1 W/mK (aggregate type and moisture content variation)
    /// **Validity:** Valid for normal-weight concrete (aggregate density 2240-2400 kg/m³)
    /// **Assumptions:** Dry conditions (moisture content < 2%), typical mix design
    /// **Notes:** Conductivity varies with aggregate type: lightweight 0.7-1.0, normal 1.3-1.8, heavy 1.8-2.5 W/mK. Moisture content can increase conductivity by 10-20% at 5% moisture.
    ///
    /// **Default Density:** 2300 kg/m³
    /// **Units:** kg/m³ (kilograms per cubic meter)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C138, Standard Test Method for Density
    /// **Uncertainty:** ±50 kg/m³ (±2%, mix design variation)
    /// **Validity:** Valid for normal-weight concrete with typical aggregate
    /// **Assumptions:** Standard mix design, no air entrainment
    /// **Notes:** Density affects thermal mass: C = Σ(ρ × c_p × t × A). Lightweight concrete: 1600-1920 kg/m³. Heavyweight concrete (with magnetite/barite): 2800-4000 kg/m³.
    ///
    /// **Default Specific Heat:** 840 J/kgK
    /// **Units:** J/kgK (joules per kilogram Kelvin)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C177, Standard Test Method for Steady-State Heat Flux
    /// **Uncertainty:** ±40 J/kgK (±5%, material variation)
    /// **Validity:** Valid for normal-weight concrete at 20°C
    /// **Assumptions:** Temperature-independent (valid for 10-40°C range)
    /// **Notes:** Specific heat relatively constant for building materials (~840 J/kgK). Varies slightly with temperature: c_p(T) = 840 + 0.6(T - 20) J/kgK.
    ///
    /// **Default Absorptance:** 0.7
    /// **Units:** Dimensionless (0-1)
    /// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
    /// **Reference:** ASTM C1371, Standard Test Method for Solar Absorptance
    /// **Uncertainty:** ±0.05 (surface finish variation)
    /// **Validity:** Valid for typical concrete surfaces (gray, weathered)
    /// **Assumptions:** New or recently painted surfaces, clean conditions
    /// **Notes:** Absorptance varies with surface color: white concrete 0.3-0.4, gray 0.6-0.7, dark/red 0.8-0.9. Weathering can increase absorptance by 10-20%.
    ///
    /// **Default Emissivity:** 0.9
    /// **Units:** Dimensionless (0-1)
    /// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
    /// **Reference:** ASTM C1371, Standard Test Method for Emissivity
    /// **Uncertainty:** ±0.05 (surface finish variation)
    /// **Validity:** Valid for typical concrete surfaces (rough, weathered)
    /// **Assumptions:** Non-metallic surface, ambient temperature
    /// **Notes:** Emissivity varies with surface finish: polished concrete 0.8-0.9, rough concrete 0.9-0.95. Low-e coatings can reduce emissivity to 0.1-0.3.
    ///
    /// # Arguments
    /// * `thickness` - Layer thickness in meters
    pub fn new(thickness: f64) -> Self {
        Self {
            thickness,
            conductivity: 1.4,
            density: 2300.0,
            specific_heat: 840.0,
            absorptance: 0.7,
            emissivity: 0.9,
        }
    }

    /// Create ASHRAE 140 heavyweight (medium-density) concrete per Table B1-3.
    ///
/// **Properties per ASHRAE 140 Table B1-3:**
    /// - k = 0.51 W/mK (NOT 1.4 — medium-density block, not normal-weight concrete)
    /// - ρ = 1400 kg/m³ (NOT 2300 — medium-density)
    /// - Cp = 840 J/kgK
    /// - κ = ρ·Cp·d = 1400 × 840 × thickness [J/m²K]
    ///
    /// Use this constructor for all 900-series ASHRAE 140 wall construction.
    /// Use  only for generic normal-weight concrete.
    ///
    /// # Arguments
    /// *  - Layer thickness in meters
    ///
    /// Constants inlined from `fluxion::physics::constants::thermal::ashrae_140::materials`
    /// to keep `fluxion-core` free of dependencies on `fluxion`'s physics constants.
    pub fn ashrae_140_heavyweight(thickness: f64) -> Self {
        // ASHRAE 140 heavyweight concrete (HW_CONCRETE_K=0.51, HW_CONCRETE_RHO=1400,
        // HW_CONCRETE_CP=840) and exterior surface absorptance 0.6 (Table B1-3).
        Self {
            thickness,
            conductivity: 0.51,
            density: 1400.0,
            specific_heat: 840.0,
            absorptance: 0.6, // EXTERIOR_SURFACE_ABSORPTANCE per ASHRAE 140 Table B1-3
            emissivity: 0.9,
        }
    }
}

impl MaterialLayer for ConcreteMaterial {
    fn name(&self) -> &str {
        "Concrete"
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
        self.absorptance
    }

    fn emissivity(&self) -> f64 {
        self.emissivity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Insulation material (thermal barrier)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsulationMaterial {
    thickness: f64,
    conductivity: f64,
    density: f64,
    specific_heat: f64,
    absorptance: f64,
    emissivity: f64,
}

impl InsulationMaterial {
    /// Create insulation material with default properties.
    ///
    /// **Default Thermal Conductivity:** 0.04 W/mK
    /// **Units:** W/mK (watts per meter Kelvin)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C518, Standard Test Method for Steady-State Thermal Transmission
    /// **Uncertainty:** ±0.005 W/mK (±12.5%, material type variation)
    /// **Validity:** Valid for fiberglass or foam board insulation at 24°C mean temperature
    /// **Assumptions:** Dry conditions (moisture content < 1%), typical density
    /// **Notes:** Conductivity varies with material type: fiberglass 0.032-0.040, mineral wool 0.034-0.040, cellulose 0.037-0.042, foam board (EPS) 0.032-0.038, foam board (XPS) 0.029-0.035 W/mK. Moisture can increase conductivity by 20-50% at 5% moisture. Conductivity increases with temperature: k(T) = k_24°C × [1 + 0.0002(T - 24)].
    ///
    /// **Default Density:** 50 kg/m³
    /// **Units:** kg/m³ (kilograms per cubic meter)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C303, Standard Test Method for Density of Loose Fill Insulation
    /// **Uncertainty:** ±10 kg/m³ (±20%, installation variation)
    /// **Validity:** Valid for fiberglass batt or blown-in insulation
    /// **Assumptions:** Proper installation, no compression
    /// **Notes:** Density varies with material type: fiberglass batt 10-15, fiberglass blown 20-40, mineral wool 20-50, cellulose 40-60, EPS foam 20-30, XPS foam 25-40 kg/m³. Compression can increase effective density and reduce thermal performance. Low-density insulation has higher thermal mass contribution per unit thickness.
    ///
    /// **Default Specific Heat:** 840 J/kgK
    /// **Units:** J/kgK (joules per kilogram Kelvin)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C177, Standard Test Method for Steady-State Heat Flux
    /// **Uncertainty:** ±80 J/kgK (±10%, material variation)
    /// **Validity:** Valid for fiberglass or foam insulation at 20°C
    /// **Assumptions:** Temperature-independent (valid for 10-40°C range)
    /// **Notes:** Specific heat relatively constant for organic insulation materials (~840 J/kgK). Mineral wool has lower specific heat (~700 J/kgK). Specific heat affects thermal mass: C = Σ(ρ × c_p × t × A). Insulation typically contributes < 5% to total thermal mass due to low density.
    ///
    /// **Default Absorptance:** 0.5
    /// **Units:** Dimensionless (0-1)
    /// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
    /// **Reference:** ASTM C1371, Standard Test Method for Solar Absorptance
    /// **Uncertainty:** ±0.1 (surface finish variation)
    /// **Validity:** Valid for typical insulation surfaces (kraft paper facing, fiberglass)
    /// **Assumptions:** Typical facing material, no reflective coatings
    /// **Notes:** Absorptance varies with facing material: kraft paper 0.5-0.6, foil-faced 0.1-0.3 (radiant barrier), plastic film 0.4-0.6, unfaced fiberglass 0.6-0.7. Reflective insulation (foil facing) reduces solar heat gain significantly.
    ///
    /// **Default Emissivity:** 0.9
    /// **Units:** Dimensionless (0-1)
    /// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
    /// **Reference:** ASTM C1371, Standard Test Method for Emissivity
    /// **Uncertainty:** ±0.1 (surface finish variation)
    /// **Validity:** Valid for typical insulation surfaces (kraft paper, fiberglass)
    /// **Assumptions:** Non-metallic facing, ambient temperature
    /// **Notes:** Emissivity varies with facing material: kraft paper 0.85-0.90, foil-faced 0.03-0.05 (radiant barrier), plastic film 0.85-0.90, unfaced fiberglass 0.90-0.95. Low-e foil facing reduces radiative heat transfer significantly in air gaps.
    ///
    /// # Arguments
    /// * `thickness` - Layer thickness in meters
    pub fn new(thickness: f64) -> Self {
        Self {
            thickness,
            conductivity: 0.04,
            density: 50.0,
            specific_heat: 840.0,
            absorptance: 0.5,
            emissivity: 0.9,
        }
    }

    /// Create ASHRAE 140 foam board insulation per Table B1-3.
    ///
    /// **Properties per ASHRAE 140 Table B1-3 (900-series outer insulation):**
    /// - k = 0.040 W/mK
    /// - ρ = 10 kg/m³ (very low density foam board)
    /// - Cp = 1400 J/kgK (NOT 840 — foam board has higher specific heat)
    ///
    /// Use this constructor for the insulation layer in 900-series walls.
    ///
    /// # Arguments
    /// *  - Layer thickness in meters (0.0615 m for ASHRAE 140 standard)
    ///
    /// Constants inlined from `fluxion::physics::constants::thermal::ashrae_140::materials`
    /// to keep `fluxion-core` free of dependencies on `fluxion`'s physics constants.
    pub fn ashrae_140_foam_board(thickness: f64) -> Self {
        // ASHRAE 140 foam board: K=0.040, RHO=10, CP=1400.
        Self {
            thickness,
            conductivity: 0.040,
            density: 10.0,
            specific_heat: 1400.0,
            absorptance: 0.5,
            emissivity: 0.9,
        }
    }
}

impl MaterialLayer for InsulationMaterial {
    fn name(&self) -> &str {
        "Insulation"
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
        self.absorptance
    }

    fn emissivity(&self) -> f64 {
        self.emissivity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Gypsum board material (interior finish)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GypsumMaterial {
    thickness: f64,
    conductivity: f64,
    density: f64,
    specific_heat: f64,
    absorptance: f64,
    emissivity: f64,
}

impl GypsumMaterial {
    /// Create gypsum material with default properties.
    ///
    /// **Default Thermal Conductivity:** 0.17 W/mK
    /// **Units:** W/mK (watts per meter Kelvin)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C518, Standard Test Method for Steady-State Thermal Transmission
    /// **Uncertainty:** ±0.02 W/mK (±12%, gypsum type variation)
    /// **Validity:** Valid for standard gypsum board (drywall) at 24°C mean temperature
    /// **Assumptions:** Dry conditions (moisture content < 1%), standard board type
    /// **Notes:** Conductivity varies with gypsum type: standard board 0.16-0.18, type X (fire-resistant) 0.18-0.20, lightweight board 0.15-0.17 W/mK. Moisture can increase conductivity by 5-10% at 2% moisture. Conductivity increases slightly with temperature: k(T) = k_24°C × [1 + 0.0001(T - 24)].
    ///
    /// **Default Density:** 960 kg/m³
    /// **Units:** kg/m³ (kilograms per cubic meter)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C139, Standard Test Method for Density of Gypsum
    /// **Uncertainty:** ±50 kg/m³ (±5%, board type variation)
    /// **Validity:** Valid for standard 1/2" (12.7 mm) gypsum board
    /// **Assumptions:** Standard board type, no additives
    /// **Notes:** Density varies with board type: standard 1/2" 640-800 kg/m³, type X 1/2" 960-1050 kg/m³, 5/8" 800-960 kg/m³, lightweight 1/2" 480-560 kg/m³. Density affects thermal mass: C = Σ(ρ × c_p × t × A). Standard 1/2" board contributes ~8-10 kJ/m²K to thermal mass.
    ///
    /// **Default Specific Heat:** 840 J/kgK
    /// **Units:** J/kgK (joules per kilogram Kelvin)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C177, Standard Test Method for Steady-State Heat Flux
    /// **Uncertainty:** ±40 J/kgK (±5%, gypsum type variation)
    /// **Validity:** Valid for standard gypsum board at 20°C
    /// **Assumptions:** Temperature-independent (valid for 10-40°C range)
    /// **Notes:** Specific heat relatively constant for gypsum-based materials (~840 J/kgK). Gypsum dihydrate (CaSO₄·2H₂O) has specific heat ~1090 J/kgK at 25°C. Specific heat affects thermal mass: C = Σ(ρ × c_p × t × A). Gypsum board typically contributes 5-10% to total wall thermal mass.
    ///
    /// **Default Absorptance:** 0.3
    /// **Units:** Dimensionless (0-1)
    /// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
    /// **Reference:** ASTM C1371, Standard Test Method for Solar Absorptance
    /// **Uncertainty:** ±0.05 (surface finish variation)
    /// **Validity:** Valid for typical gypsum board surfaces (painted, white/light-colored)
    /// **Assumptions:** Typical interior finish, unpainted or light-colored paint
    /// **Notes:** Absorptance varies with surface finish: unpainted gypsum 0.3-0.4, white paint 0.2-0.3, light colors 0.3-0.5, medium colors 0.5-0.7, dark colors 0.7-0.9. Interior surfaces typically have low absorptance due to light-colored paints and finishes.
    ///
    /// **Default Emissivity:** 0.9
    /// **Units:** Dimensionless (0-1)
    /// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
    /// **Reference:** ASTM C1371, Standard Test Method for Emissivity
    /// **Uncertainty:** ±0.05 (surface finish variation)
    /// **Validity:** Valid for typical gypsum board surfaces (painted, interior)
    /// **Assumptions:** Non-metallic surface, ambient temperature
    /// **Notes:** Emissivity varies with surface finish: unpainted gypsum 0.90-0.92, painted (flat/matte) 0.88-0.92, painted (semi-gloss) 0.85-0.90, painted (glossy) 0.80-0.85. Interior surfaces typically have high emissivity for radiative heat transfer with other interior surfaces.
    ///
    /// # Arguments
    /// * `thickness` - Layer thickness in meters
    pub fn new(thickness: f64) -> Self {
        Self {
            thickness,
            conductivity: 0.17,
            density: 960.0,
            specific_heat: 840.0,
            absorptance: 0.3,
            emissivity: 0.9,
        }
    }

    /// Create ASHRAE 140 gypsum board per Table B1-3.
    ///
    /// **Properties per ASHRAE 140 Table B1-3 (600-series interior finish):**
    /// - k = 0.16 W/mK
    /// - ρ = 784 kg/m³ (NOT 960 — ASHRAE 140 specifies standard 12mm board density)
    /// - Cp = 840 J/kgK
    ///
    /// Use this constructor for 600-series ASHRAE 140 wall construction.
    ///
    /// # Arguments
    /// *  - Layer thickness in meters (0.012 m for standard board)
    ///
    /// Constants inlined from `fluxion::physics::constants::thermal::ashrae_140::materials`
    /// to keep `fluxion-core` free of dependencies on `fluxion`'s physics constants.
    pub fn ashrae_140(thickness: f64) -> Self {
        // ASHRAE 140 gypsum board: K=0.16, RHO=784, CP=840.
        Self {
            thickness,
            conductivity: 0.16,
            density: 784.0,
            specific_heat: 840.0,
            absorptance: 0.3,
            emissivity: 0.9,
        }
    }
}

impl MaterialLayer for GypsumMaterial {
    fn name(&self) -> &str {
        "Gypsum"
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
        self.absorptance
    }

    fn emissivity(&self) -> f64 {
        self.emissivity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Brick material (exterior cladding)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrickMaterial {
    thickness: f64,
    conductivity: f64,
    density: f64,
    specific_heat: f64,
    absorptance: f64,
    emissivity: f64,
}

impl BrickMaterial {
    /// Create brick material with default properties.
    ///
    /// **Default Thermal Conductivity:** 0.7 W/mK
    /// **Units:** W/mK (watts per meter Kelvin)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C177, Standard Test Method for Steady-State Heat Flux
    /// **Uncertainty:** ±0.1 W/mK (±14%, brick type and moisture content variation)
    /// **Validity:** Valid for clay brick at 24°C mean temperature
    /// **Assumptions:** Dry conditions (moisture content < 2%), typical clay brick
    /// **Notes:** Conductivity varies with brick type: clay brick 0.6-0.9, concrete brick 0.7-1.1, sand-lime brick 0.8-1.2 W/mK. Moisture can increase conductivity by 15-25% at 5% moisture. Hollow brick (core-filled) has effective conductivity 0.5-0.8 W/mK.
    ///
    /// **Default Density:** 1920 kg/m³
    /// **Units:** kg/m³ (kilograms per cubic meter)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C67, Standard Test Methods for Sampling and Testing Brick
    /// **Uncertainty:** ±80 kg/m³ (±4%, brick type variation)
    /// **Validity:** Valid for solid clay brick (modular size)
    /// **Assumptions:** Standard brick dimensions, solid construction
    /// **Notes:** Density varies with brick type: clay brick 1600-2100 kg/m³, concrete brick 1800-2200 kg/m³, sand-lime brick 1700-2000 kg/m³. Hollow brick (60% solid) has effective density 1150-1260 kg/m³. Density affects thermal mass: C = Σ(ρ × c_p × t × A).
    ///
    /// **Default Specific Heat:** 840 J/kgK
    /// **Units:** J/kgK (joules per kilogram Kelvin)
    /// **Source:** ASHRAE Handbook of Fundamentals, Chapter 26, Building Envelope
    /// **Reference:** ASTM C177, Standard Test Method for Steady-State Heat Flux
    /// **Uncertainty:** ±40 J/kgK (±5%, brick type variation)
    /// **Validity:** Valid for clay brick at 20°C
    /// **Assumptions:** Temperature-independent (valid for 10-40°C range)
    /// **Notes:** Specific heat relatively constant for ceramic materials (~840 J/kgK). Clay minerals (kaolinite, illite) have specific heat ~900-1000 J/kgK at 25°C. Specific heat affects thermal mass: C = Σ(ρ × c_p × t × A). Brick typically contributes 30-50% to wall thermal mass.
    ///
    /// **Default Absorptance:** 0.9
    /// **Units:** Dimensionless (0-1)
    /// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
    /// **Reference:** ASTM C1371, Standard Test Method for Solar Absorptance
    /// **Uncertainty:** ±0.05 (surface finish variation)
    /// **Validity:** Valid for typical brick surfaces (red/brown clay brick)
    /// **Assumptions:** Uncoated brick, typical weathered surface
    /// **Notes:** Absorptance varies with brick color: white/light tan 0.5-0.6, tan/cream 0.6-0.7, red 0.8-0.9, brown/dark 0.85-0.95. Weathering can increase absorptance by 5-10%. Painted brick follows paint color absorptance values.
    ///
    /// **Default Emissivity:** 0.9
    /// **Units:** Dimensionless (0-1)
    /// **Source:** ASHRAE Standard 140-2023, Table X, Surface Properties
    /// **Reference:** ASTM C1371, Standard Test Method for Emissivity
    /// **Uncertainty:** ±0.05 (surface finish variation)
    /// **Validity:** Valid for typical brick surfaces (rough, weathered clay)
    /// **Assumptions:** Non-metallic surface, ambient temperature
    /// **Notes:** Emissivity varies with surface finish: smooth glazed brick 0.85-0.90, rough unglazed brick 0.90-0.95, weathered brick 0.90-0.95. Brick surfaces typically have high emissivity for radiative heat transfer with outdoor environment.
    ///
    /// # Arguments
    /// * `thickness` - Layer thickness in meters
    pub fn new(thickness: f64) -> Self {
        Self {
            thickness,
            conductivity: 0.7,
            density: 1920.0,
            specific_heat: 840.0,
            absorptance: 0.9,
            emissivity: 0.9,
        }
    }
}

impl MaterialLayer for BrickMaterial {
    fn name(&self) -> &str {
        "Brick"
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
        self.absorptance
    }

    fn emissivity(&self) -> f64 {
        self.emissivity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Assembly validation errors
#[derive(Debug)]
pub enum AssemblyError {
    /// No layers provided in assembly
    NoLayers,
    /// Invalid layer thickness (must be > 0)
    InvalidThickness { layer_name: String, thickness: f64 },
    /// Invalid layer conductivity (must be > 0)
    InvalidConductivity {
        layer_name: String,
        conductivity: f64,
    },
    /// Invalid layer density (must be > 0)
    InvalidDensity { layer_name: String, density: f64 },
    /// Invalid layer specific heat (must be > 0)
    InvalidSpecificHeat {
        layer_name: String,
        specific_heat: f64,
    },
    /// Invalid emissivity (must be in [0, 1])
    InvalidEmissivity { layer_name: String, emissivity: f64 },
    /// Invalid absorptance (must be in [0, 1])
    InvalidAbsorptance {
        layer_name: String,
        absorptance: f64,
    },
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssemblyError::NoLayers => write!(f, "Assembly must have at least one layer"),
            AssemblyError::InvalidThickness {
                layer_name,
                thickness,
            } => {
                write!(
                    f,
                    "Layer '{}' has invalid thickness: {} (must be > 0)",
                    layer_name, thickness
                )
            }
            AssemblyError::InvalidConductivity {
                layer_name,
                conductivity,
            } => {
                write!(
                    f,
                    "Layer '{}' has invalid conductivity: {} (must be > 0)",
                    layer_name, conductivity
                )
            }
            AssemblyError::InvalidDensity {
                layer_name,
                density,
            } => {
                write!(
                    f,
                    "Layer '{}' has invalid density: {} (must be > 0)",
                    layer_name, density
                )
            }
            AssemblyError::InvalidSpecificHeat {
                layer_name,
                specific_heat,
            } => {
                write!(
                    f,
                    "Layer '{}' has invalid specific heat: {} (must be > 0)",
                    layer_name, specific_heat
                )
            }
            AssemblyError::InvalidEmissivity {
                layer_name,
                emissivity,
            } => {
                write!(
                    f,
                    "Layer '{}' has invalid emissivity: {} (must be in [0, 1])",
                    layer_name, emissivity
                )
            }
            AssemblyError::InvalidAbsorptance {
                layer_name,
                absorptance,
            } => {
                write!(
                    f,
                    "Layer '{}' has invalid absorptance: {} (must be in [0, 1])",
                    layer_name, absorptance
                )
            }
        }
    }
}

impl std::error::Error for AssemblyError {}

/// Thermal mass classification per ISO 13790 Annex C
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalMassClassification {
    /// VeryLight: < 50 kJ/m²K
    VeryLight,
    /// Light: 50-150 kJ/m²K
    Light,
    /// Medium: 150-260 kJ/m²K
    Medium,
    /// Heavy: 260-370 kJ/m²K
    Heavy,
    /// VeryHeavy: > 370 kJ/m²K
    VeryHeavy,
}

impl fmt::Display for ThermalMassClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThermalMassClassification::VeryLight => write!(f, "VeryLight"),
            ThermalMassClassification::Light => write!(f, "Light"),
            ThermalMassClassification::Medium => write!(f, "Medium"),
            ThermalMassClassification::Heavy => write!(f, "Heavy"),
            ThermalMassClassification::VeryHeavy => write!(f, "VeryHeavy"),
        }
    }
}

/// Building assembly composed of material layers
pub struct BuildingAssembly {
    /// Assembly name/identifier
    pub name: String,
    /// Material layers in order (exterior to interior)
    pub layers: Vec<Box<dyn MaterialLayer>>,
}

impl fmt::Debug for BuildingAssembly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuildingAssembly")
            .field("name", &self.name)
            .field("num_layers", &self.layers.len())
            .finish()
    }
}

impl Clone for BuildingAssembly {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    // Clone based on material type
                    if let Some(concrete) = layer.as_any().downcast_ref::<ConcreteMaterial>() {
                        Box::new(concrete.clone()) as Box<dyn MaterialLayer>
                    } else if let Some(insulation) =
                        layer.as_any().downcast_ref::<InsulationMaterial>()
                    {
                        Box::new(insulation.clone()) as Box<dyn MaterialLayer>
                    } else if let Some(gypsum) = layer.as_any().downcast_ref::<GypsumMaterial>() {
                        Box::new(gypsum.clone()) as Box<dyn MaterialLayer>
                    } else if let Some(brick) = layer.as_any().downcast_ref::<BrickMaterial>() {
                        Box::new(brick.clone()) as Box<dyn MaterialLayer>
                    } else {
                        panic!("Unsupported material type for cloning")
                    }
                })
                .collect(),
        }
    }
}

impl BuildingAssembly {
    /// Calculate total R-value of assembly (sum of layer R-values)
    pub fn total_r_value(&self) -> f64 {
        self.layers.iter().map(|layer| layer.r_value()).sum()
    }

    /// Calculate total thickness of assembly
    pub fn total_thickness(&self) -> f64 {
        self.layers.iter().map(|layer| layer.thickness()).sum()
    }

    /// Calculate thermal mass per unit area (kJ/m²K)
    ///
    /// Thermal mass = Σ(density × specific_heat × thickness × area)
    /// For unit area (1 m²): Σ(density × specific_heat × thickness)
    /// Result in kJ/m²K
    pub fn thermal_mass(&self) -> f64 {
        let capacitance_per_area: f64 = self
            .layers
            .iter()
            .map(|layer| layer.density() * layer.specific_heat() * layer.thickness())
            .sum(); // J/m²K

        capacitance_per_area / 1000.0 // Convert to kJ/m²K
    }

    /// Get thermal mass classification per ISO 13790 Annex C
    pub fn classification(&self) -> ThermalMassClassification {
        let thermal_mass = self.thermal_mass();

        // ISO 13790 Annex C thresholds (kJ/m²K)
        if thermal_mass < 50.0 {
            ThermalMassClassification::VeryLight
        } else if thermal_mass < 150.0 {
            ThermalMassClassification::Light
        } else if thermal_mass < 260.0 {
            ThermalMassClassification::Medium
        } else if thermal_mass < 370.0 {
            ThermalMassClassification::Heavy
        } else {
            ThermalMassClassification::VeryHeavy
        }
    }
}

/// Builder for constructing building assemblies with validation
pub struct AssemblyBuilder {
    layers: Vec<Box<dyn MaterialLayer>>,
    name: String,
}

impl AssemblyBuilder {
    /// Create a new assembly builder
    pub fn new(name: String) -> Self {
        Self {
            layers: Vec::new(),
            name,
        }
    }

    /// Add a material layer to the assembly
    ///
    /// # Arguments
    /// * `layer` - Material layer to add (exterior to interior order)
    pub fn add_layer(mut self, layer: Box<dyn MaterialLayer>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Build the assembly with validation
    ///
    /// # Errors
    /// Returns `AssemblyError` if any layer has invalid properties
    pub fn build(self) -> Result<BuildingAssembly, AssemblyError> {
        // Validate at least one layer
        if self.layers.is_empty() {
            return Err(AssemblyError::NoLayers);
        }

        // Validate each layer
        for layer in &self.layers {
            let layer_name = layer.name().to_string();

            // Validate thickness
            if layer.thickness() <= 0.0 {
                return Err(AssemblyError::InvalidThickness {
                    layer_name,
                    thickness: layer.thickness(),
                });
            }

            // Validate conductivity
            if layer.conductivity() <= 0.0 {
                return Err(AssemblyError::InvalidConductivity {
                    layer_name,
                    conductivity: layer.conductivity(),
                });
            }

            // Validate density
            if layer.density() <= 0.0 {
                return Err(AssemblyError::InvalidDensity {
                    layer_name,
                    density: layer.density(),
                });
            }

            // Validate specific heat
            if layer.specific_heat() <= 0.0 {
                return Err(AssemblyError::InvalidSpecificHeat {
                    layer_name,
                    specific_heat: layer.specific_heat(),
                });
            }

            // Validate emissivity in [0, 1]
            if !(0.0..=1.0).contains(&layer.emissivity()) {
                return Err(AssemblyError::InvalidEmissivity {
                    layer_name,
                    emissivity: layer.emissivity(),
                });
            }

            // Validate absorptance in [0, 1]
            if !(0.0..=1.0).contains(&layer.absorptance()) {
                return Err(AssemblyError::InvalidAbsorptance {
                    layer_name,
                    absorptance: layer.absorptance(),
                });
            }
        }

        Ok(BuildingAssembly {
            name: self.name,
            layers: self.layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_layer_properties() {
        // Test concrete material
        let concrete = ConcreteMaterial::new(0.1);
        assert_eq!(concrete.name(), "Concrete");
        assert_eq!(concrete.conductivity(), 1.4);
        assert_eq!(concrete.thickness(), 0.1);
        assert_eq!(concrete.density(), 2300.0);
        assert_eq!(concrete.specific_heat(), 840.0);
        assert_eq!(concrete.absorptance(), 0.7);
        assert_eq!(concrete.emissivity(), 0.9);
        assert_eq!(concrete.r_value(), 0.1 / 1.4);

        // Test insulation material
        let insulation = InsulationMaterial::new(0.05);
        assert_eq!(insulation.name(), "Insulation");
        assert_eq!(insulation.conductivity(), 0.04);
        assert_eq!(insulation.thickness(), 0.05);
        assert_eq!(insulation.density(), 50.0);
        assert_eq!(insulation.specific_heat(), 840.0);
        assert_eq!(insulation.absorptance(), 0.5);
        assert_eq!(insulation.emissivity(), 0.9);
        assert_eq!(insulation.r_value(), 0.05 / 0.04);

        // Test gypsum material
        let gypsum = GypsumMaterial::new(0.012);
        assert_eq!(gypsum.name(), "Gypsum");
        assert_eq!(gypsum.conductivity(), 0.17);
        assert_eq!(gypsum.thickness(), 0.012);
        assert_eq!(gypsum.density(), 960.0);
        assert_eq!(gypsum.specific_heat(), 840.0);
        assert_eq!(gypsum.absorptance(), 0.3);
        assert_eq!(gypsum.emissivity(), 0.9);
        assert_eq!(gypsum.r_value(), 0.012 / 0.17);

        // Test brick material
        let brick = BrickMaterial::new(0.2);
        assert_eq!(brick.name(), "Brick");
        assert_eq!(brick.conductivity(), 0.7);
        assert_eq!(brick.thickness(), 0.2);
        assert_eq!(brick.density(), 1920.0);
        assert_eq!(brick.specific_heat(), 840.0);
        assert_eq!(brick.absorptance(), 0.9);
        assert_eq!(brick.emissivity(), 0.9);
        assert_eq!(brick.r_value(), 0.2 / 0.7);
    }

    #[test]
    fn test_assembly_builder_validation() {
        // Test successful assembly build
        let assembly = AssemblyBuilder::new("test_wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .add_layer(Box::new(InsulationMaterial::new(0.05)))
            .add_layer(Box::new(GypsumMaterial::new(0.012)))
            .build()
            .unwrap();

        assert_eq!(assembly.name, "test_wall");
        assert_eq!(assembly.layers.len(), 3);

        // Test total R-value calculation
        let total_r = assembly.total_r_value();
        let expected_r = 0.1 / 1.4 + 0.05 / 0.04 + 0.012 / 0.17;
        assert!((total_r - expected_r).abs() < 1e-10);

        // Test total thickness calculation
        assert!((assembly.total_thickness() - 0.162).abs() < 1e-10);

        // Test no layers error
        let result = AssemblyBuilder::new("empty".to_string()).build();
        assert!(matches!(result, Err(AssemblyError::NoLayers)));

        // Test invalid thickness error
        let result = AssemblyBuilder::new("invalid_thickness".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(-0.1)))
            .build();
        assert!(matches!(
            result,
            Err(AssemblyError::InvalidThickness { .. })
        ));

        // Test invalid conductivity error (would need custom material, so skip for now)
        // Test invalid density error (would need custom material, so skip for now)
        // Test invalid specific heat error (would need custom material, so skip for now)
        // Test invalid emissivity error (would need custom material, so skip for now)
        // Test invalid absorptance error (would need custom material, so skip for now)
    }

    #[test]
    fn test_yaml_loading() {
        // Issue #1349: after moving to fluxion-core, the test cwd is `fluxion-core/`
        // (not the repo root). Use `CARGO_MANIFEST_DIR` to resolve `../data/...`.
        let data_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
        // Test materials.yaml loading
        let materials = load_materials(&format!("{data_dir}/materials.yaml")).unwrap();
        assert!(materials.contains_key("Concrete"));
        assert!(materials.contains_key("Insulation"));
        assert!(materials.contains_key("Gypsum"));
        assert!(materials.contains_key("Brick"));

        let concrete = &materials["Concrete"];
        assert_eq!(concrete.conductivity, 1.4);
        assert_eq!(concrete.density, 2300.0);
        assert_eq!(concrete.specific_heat, 840.0);
        assert_eq!(concrete.absorptance, 0.7);
        assert_eq!(concrete.emissivity, 0.9);

        // Test assemblies.yaml loading
        let assemblies = load_assemblies(&format!("{data_dir}/assemblies.yaml")).unwrap();
        assert!(assemblies.contains_key("light_mass_wall"));
        assert!(assemblies.contains_key("heavy_mass_wall"));

        let light_wall = &assemblies["light_mass_wall"];
        assert_eq!(light_wall.layers.len(), 3);
        assert_eq!(light_wall.layers[0].material, "Concrete");
        assert_eq!(light_wall.layers[0].thickness, 0.1);
        assert_eq!(light_wall.layers[1].material, "Insulation");
        assert_eq!(light_wall.layers[1].thickness, 0.05);
        assert_eq!(light_wall.layers[2].material, "Gypsum");
        assert_eq!(light_wall.layers[2].thickness, 0.012);

        let heavy_wall = &assemblies["heavy_mass_wall"];
        assert_eq!(heavy_wall.layers.len(), 4);
        assert_eq!(heavy_wall.layers[0].material, "Brick");
        assert_eq!(heavy_wall.layers[0].thickness, 0.2);
        assert_eq!(heavy_wall.layers[1].material, "Concrete");
        assert_eq!(heavy_wall.layers[1].thickness, 0.15);
        assert_eq!(heavy_wall.layers[2].material, "Insulation");
        assert_eq!(heavy_wall.layers[2].thickness, 0.1);
        assert_eq!(heavy_wall.layers[3].material, "Gypsum");
        assert_eq!(heavy_wall.layers[3].thickness, 0.013);
    }

    #[test]
    fn test_thermal_mass_classification() {
        // Test thermal mass calculation for light mass wall
        let light_wall = AssemblyBuilder::new("light_mass_wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .add_layer(Box::new(InsulationMaterial::new(0.05)))
            .add_layer(Box::new(GypsumMaterial::new(0.012)))
            .build()
            .unwrap();

        // Expected thermal mass:
        // Concrete: 2300 * 840 * 0.1 = 193,200 J/m²K = 193.2 kJ/m²K
        // Insulation: 50 * 840 * 0.05 = 2,100 J/m²K = 2.1 kJ/m²K
        // Gypsum: 960 * 840 * 0.012 = 9,676.8 J/m²K = 9.6768 kJ/m²K
        // Total: ~204.9768 kJ/m²K → Medium (150-260)
        let thermal_mass = light_wall.thermal_mass();
        assert!((thermal_mass - 204.98).abs() < 0.01);
        assert_eq!(
            light_wall.classification(),
            ThermalMassClassification::Medium
        );

        // Test thermal mass calculation for heavy mass wall
        let heavy_wall = AssemblyBuilder::new("heavy_mass_wall".to_string())
            .add_layer(Box::new(BrickMaterial::new(0.2)))
            .add_layer(Box::new(ConcreteMaterial::new(0.15)))
            .add_layer(Box::new(InsulationMaterial::new(0.1)))
            .add_layer(Box::new(GypsumMaterial::new(0.013)))
            .build()
            .unwrap();

        // Expected thermal mass:
        // Brick: 1920 * 840 * 0.2 = 322,560 J/m²K = 322.56 kJ/m²K
        // Concrete: 2300 * 840 * 0.15 = 289,800 J/m²K = 289.8 kJ/m²K
        // Insulation: 50 * 840 * 0.1 = 4,200 J/m²K = 4.2 kJ/m²K
        // Gypsum: 960 * 840 * 0.013 = 10,483.2 J/m²K = 10.4832 kJ/m²K
        // Total: ~627.0432 kJ/m²K → VeryHeavy (> 370)
        let thermal_mass = heavy_wall.thermal_mass();
        assert!((thermal_mass - 627.04).abs() < 0.01);
        assert_eq!(
            heavy_wall.classification(),
            ThermalMassClassification::VeryHeavy
        );

        // Test VeryLight classification (< 50 kJ/m²K)
        let very_light = AssemblyBuilder::new("very_light".to_string())
            .add_layer(Box::new(GypsumMaterial::new(0.01))) // ~8 kJ/m²K
            .build()
            .unwrap();
        assert!(very_light.thermal_mass() < 50.0);
        assert_eq!(
            very_light.classification(),
            ThermalMassClassification::VeryLight
        );

        // Test Light classification (50-150 kJ/m²K)
        let light = AssemblyBuilder::new("light".to_string())
            .add_layer(Box::new(GypsumMaterial::new(0.05))) // ~40 kJ/m²K
            .add_layer(Box::new(ConcreteMaterial::new(0.02))) // ~38.6 kJ/m²K
            .build()
            .unwrap();
        let thermal_mass = light.thermal_mass();
        assert!(thermal_mass >= 50.0 && thermal_mass < 150.0);
        assert_eq!(light.classification(), ThermalMassClassification::Light);

        // Test Heavy classification (260-370 kJ/m²K)
        let heavy = AssemblyBuilder::new("heavy".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1))) // ~193.2 kJ/m²K
            .add_layer(Box::new(BrickMaterial::new(0.05))) // ~80.6 kJ/m²K
            .build()
            .unwrap();
        let thermal_mass = heavy.thermal_mass();
        assert!(thermal_mass >= 260.0 && thermal_mass < 370.0);
        assert_eq!(heavy.classification(), ThermalMassClassification::Heavy);
    }

    #[test]
    fn test_assembly_clone() {
        let assembly = AssemblyBuilder::new("clone_test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .add_layer(Box::new(InsulationMaterial::new(0.05)))
            .build()
            .unwrap();
        let cloned = assembly.clone();
        assert_eq!(cloned.name, "clone_test");
        assert_eq!(cloned.layers.len(), 2);
        assert_eq!(cloned.total_r_value(), assembly.total_r_value());
    }

    #[test]
    fn test_assembly_clone_gypsum() {
        let assembly = AssemblyBuilder::new("gypsum_clone".to_string())
            .add_layer(Box::new(GypsumMaterial::new(0.012)))
            .build()
            .unwrap();
        let cloned = assembly.clone();
        assert_eq!(cloned.name, "gypsum_clone");
        assert_eq!(cloned.layers.len(), 1);
    }

    #[test]
    fn test_assembly_clone_brick() {
        let assembly = AssemblyBuilder::new("brick_clone".to_string())
            .add_layer(Box::new(BrickMaterial::new(0.1)))
            .build()
            .unwrap();
        let cloned = assembly.clone();
        assert_eq!(cloned.name, "brick_clone");
        assert_eq!(cloned.layers.len(), 1);
    }

    #[test]
    fn test_assembly_debug() {
        let assembly = AssemblyBuilder::new("debug_test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();
        let debug_str = format!("{:?}", assembly);
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("num_layers"));
    }

    #[test]
    fn test_assembly_error_display() {
        let err = AssemblyError::NoLayers;
        let msg = format!("{}", err);
        assert!(msg.contains("at least one layer"));
    }

    #[test]
    fn test_assembly_error_invalid_thickness_display() {
        let err = AssemblyError::InvalidThickness {
            layer_name: "TestLayer".to_string(),
            thickness: -0.5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("TestLayer"));
        assert!(msg.contains("-0.5"));
    }

    #[test]
    fn test_assembly_error_invalid_conductivity_display() {
        let err = AssemblyError::InvalidConductivity {
            layer_name: "BadCond".to_string(),
            conductivity: 0.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("BadCond"));
        assert!(msg.contains("conductivity"));
    }

    #[test]
    fn test_assembly_error_invalid_density_display() {
        let err = AssemblyError::InvalidDensity {
            layer_name: "NoDensity".to_string(),
            density: -10.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("NoDensity"));
        assert!(msg.contains("density"));
    }

    #[test]
    fn test_assembly_error_invalid_specific_heat_display() {
        let err = AssemblyError::InvalidSpecificHeat {
            layer_name: "NoHeat".to_string(),
            specific_heat: 0.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("NoHeat"));
        assert!(msg.contains("specific heat"));
    }

    #[test]
    fn test_assembly_error_invalid_emissivity_display() {
        let err = AssemblyError::InvalidEmissivity {
            layer_name: "BadEmiss".to_string(),
            emissivity: 1.5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("BadEmiss"));
        assert!(msg.contains("emissivity"));
    }

    #[test]
    fn test_assembly_error_invalid_absorptance_display() {
        let err = AssemblyError::InvalidAbsorptance {
            layer_name: "BadAbsorb".to_string(),
            absorptance: -0.1,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("BadAbsorb"));
        assert!(msg.contains("absorptance"));
    }

    #[test]
    fn test_assembly_error_is_error() {
        let err = AssemblyError::NoLayers;
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_load_materials_invalid_path() {
        let result = load_materials("/nonexistent/path/materials.yaml");
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("Failed to read"));
    }

    #[test]
    fn test_load_assemblies_invalid_path() {
        let result = load_assemblies("/nonexistent/path/assemblies.yaml");
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("Failed to read"));
    }

    #[test]
    fn test_material_layer_downcast() {
        let concrete = ConcreteMaterial::new(0.15);
        let layer: &dyn MaterialLayer = &concrete;
        assert!(layer.as_any().downcast_ref::<ConcreteMaterial>().is_some());
        assert!(layer
            .as_any()
            .downcast_ref::<InsulationMaterial>()
            .is_none());
        assert!(layer.as_any().downcast_ref::<GypsumMaterial>().is_none());
        assert!(layer.as_any().downcast_ref::<BrickMaterial>().is_none());
    }

    #[test]
    fn test_thermal_mass_classification_display() {
        assert_eq!(
            format!("{}", ThermalMassClassification::VeryLight),
            "VeryLight"
        );
        assert_eq!(format!("{}", ThermalMassClassification::Light), "Light");
        assert_eq!(format!("{}", ThermalMassClassification::Medium), "Medium");
        assert_eq!(format!("{}", ThermalMassClassification::Heavy), "Heavy");
        assert_eq!(
            format!("{}", ThermalMassClassification::VeryHeavy),
            "VeryHeavy"
        );
    }

    #[test]
    fn test_thermal_mass_classification_copy() {
        let c1 = ThermalMassClassification::Heavy;
        let c2 = c1;
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_assembly_builder_add_multiple_layers() {
        let builder = AssemblyBuilder::new("multi".to_string());
        let layers = vec![
            Box::new(BrickMaterial::new(0.1)) as Box<dyn MaterialLayer>,
            Box::new(ConcreteMaterial::new(0.15)),
            Box::new(InsulationMaterial::new(0.05)),
            Box::new(GypsumMaterial::new(0.012)),
        ];
        let builder = layers.into_iter().fold(builder, |b, l| b.add_layer(l));
        let assembly = builder.build().unwrap();
        assert_eq!(assembly.layers.len(), 4);
    }

    #[test]
    fn test_assembly_r_value_single_layer() {
        let assembly = AssemblyBuilder::new("single".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();
        let expected_r = 0.2 / 1.4;
        assert!((assembly.total_r_value() - expected_r).abs() < 1e-10);
    }

    #[test]
    fn test_assembly_total_thickness_single_layer() {
        let assembly = AssemblyBuilder::new("single".to_string())
            .add_layer(Box::new(BrickMaterial::new(0.15)))
            .build()
            .unwrap();
        assert!((assembly.total_thickness() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_material_properties_ranges() {
        let concrete = ConcreteMaterial::new(0.1);
        assert!((0.0..=1.0).contains(&concrete.absorptance()));
        assert!((0.0..=1.0).contains(&concrete.emissivity()));
        assert!(concrete.conductivity() > 0.0);
        assert!(concrete.density() > 0.0);
        assert!(concrete.specific_heat() > 0.0);
        assert!(concrete.thickness() > 0.0);
    }

    #[test]
    fn test_insulation_material_properties() {
        let insulation = InsulationMaterial::new(0.1);
        assert_eq!(insulation.name(), "Insulation");
        assert_eq!(insulation.conductivity(), 0.04);
        assert_eq!(insulation.density(), 50.0);
        assert_eq!(insulation.specific_heat(), 840.0);
        assert_eq!(insulation.absorptance(), 0.5);
        assert_eq!(insulation.emissivity(), 0.9);
        assert_eq!(insulation.r_value(), 0.1 / 0.04);
    }

    #[test]
    fn test_gypsum_material_properties() {
        let gypsum = GypsumMaterial::new(0.013);
        assert_eq!(gypsum.name(), "Gypsum");
        assert_eq!(gypsum.conductivity(), 0.17);
        assert_eq!(gypsum.density(), 960.0);
        assert_eq!(gypsum.specific_heat(), 840.0);
        assert_eq!(gypsum.absorptance(), 0.3);
        assert_eq!(gypsum.emissivity(), 0.9);
        assert_eq!(gypsum.r_value(), 0.013 / 0.17);
    }

    #[test]
    fn test_brick_material_properties() {
        let brick = BrickMaterial::new(0.09);
        assert_eq!(brick.name(), "Brick");
        assert_eq!(brick.conductivity(), 0.7);
        assert_eq!(brick.density(), 1920.0);
        assert_eq!(brick.specific_heat(), 840.0);
        assert_eq!(brick.absorptance(), 0.9);
        assert_eq!(brick.emissivity(), 0.9);
        assert_eq!(brick.r_value(), 0.09 / 0.7);
    }

    #[test]
    fn test_load_materials_from_data() {
        let data_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
        let materials = load_materials(&format!("{data_dir}/materials.yaml"));
        assert!(materials.is_ok());
        let materials = materials.unwrap();
        assert!(!materials.is_empty());
    }

    #[test]
    fn test_load_assemblies_from_data() {
        let data_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
        let assemblies = load_assemblies(&format!("{data_dir}/assemblies.yaml"));
        assert!(assemblies.is_ok());
        let assemblies = assemblies.unwrap();
        assert!(!assemblies.is_empty());
    }
}
