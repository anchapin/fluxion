//! HVAC Equipment Models — moved from `fluxion::sim::hvac::equipment`.
//!
//! Issue #4157: These types are moved to `fluxion_core` to break the
//! `sim ↔ validation` dependency cycle. `CaseSpec` (which lives in
//! `fluxion::validation::ashrae_140_cases`) had a field
//! `pub hvac_equipment: Option<crate::sim::hvac::AnyEquipment>`, creating
//! an upward dependency from `validation` into `sim`. By hoisting the equipment
//! types into the leaf crate, `CaseSpec` can reference them without pulling in
//! the entire `sim` module hierarchy.
//!
//! ## Simplified types
//!
//! The versions here are simplified compared to the main-crate originals:
//! - `VAVTerminal` omits the nested `VAVTerminalUnit` psychrometric state
//! - `CAVSystem` omits the nested `CavTerminalUnit` psychrometric state
//! - These simplifications are sufficient for the `VariableCapacityEquipment`
//!   trait implementations used by the physics solvers
//!
//! The main crate (`src/sim/hvac/mod.rs`) retains the full types with
//! psychrometric state for NAPI/Python bindings and advanced use cases.
//! Re-exports via `pub use fluxion_core::hvac_equipment::*` in
//! `src/sim/hvac/equipment.rs` maintain backward compatibility.

#![allow(clippy::option_as_ref_deref)]

use serde::{Deserialize, Serialize};

// =============================================================================
// Efficiency Curves (from efficiency_curves.rs)
// =============================================================================

/// Polynomial efficiency curve coefficients.
///
/// Uses cubic polynomial to model COP as function of part-load ratio (PLR):
/// COP(PLR) = a + b*PLR + c*PLR² + d*PLR³
///
/// Combined with temperature degradation:
/// COP(PLR, T) = COP(PLR) * (1 - temp_coeff * |T - T_design|)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyCurve {
    /// Cubic polynomial coefficients: [a, b, c, d] for COP = a + b*PLR + c*PLR² + d*PLR³
    pub plr_coefficients: [f64; 4],
    /// Temperature coefficient (COP degrades per degree from design temperature)
    pub temp_coefficient: f64,
    /// Design outdoor temperature (°C)
    pub design_temp: f64,
}

impl EfficiencyCurve {
    /// Create a new efficiency curve from coefficients.
    pub fn new(plr_coefficients: [f64; 4], temp_coefficient: f64, design_temp: f64) -> Self {
        Self {
            plr_coefficients,
            temp_coefficient,
            design_temp,
        }
    }

    /// Calculate COP at given PLR and outdoor temperature.
    pub fn cop_at(&self, plr: f64, outdoor_temp: f64) -> f64 {
        let plr_cop = ((self.plr_coefficients[3] * plr + self.plr_coefficients[2]) * plr
            + self.plr_coefficients[1])
            * plr
            + self.plr_coefficients[0];

        let temp_diff = (self.design_temp - outdoor_temp).abs();
        let temp_factor = 1.0 - self.temp_coefficient * temp_diff;

        plr_cop * temp_factor.max(0.3)
    }
}

/// Curve coefficients for a single equipment type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveCoefficients {
    pub plr: [f64; 4],
    pub temp_coefficient: f64,
    pub design_temp: f64,
}

/// AHRI efficiency curve configuration for multiple equipment types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyCurveConfig {
    pub heatpump_heating: CurveCoefficients,
    pub heatpump_cooling: CurveCoefficients,
    pub chiller: CurveCoefficients,
    pub boiler: CurveCoefficients,
}

impl From<&CurveCoefficients> for EfficiencyCurve {
    fn from(coeffs: &CurveCoefficients) -> Self {
        EfficiencyCurve::new(coeffs.plr, coeffs.temp_coefficient, coeffs.design_temp)
    }
}

/// Create default AHRI coefficients (placeholder values).
pub fn default_ahri_coefficients() -> EfficiencyCurveConfig {
    EfficiencyCurveConfig {
        heatpump_heating: CurveCoefficients {
            plr: [3.5, 0.0, 0.0, 0.0],
            temp_coefficient: 0.02,
            design_temp: -5.0,
        },
        heatpump_cooling: CurveCoefficients {
            plr: [4.5, 0.0, 0.0, 0.0],
            temp_coefficient: 0.022,
            design_temp: 35.0,
        },
        chiller: CurveCoefficients {
            plr: [1.0, 0.0, 0.0, 0.0],
            temp_coefficient: 0.0,
            design_temp: 35.0,
        },
        boiler: CurveCoefficients {
            plr: [0.85, 0.05, -0.03, 0.01],
            temp_coefficient: 0.001,
            design_temp: -5.0,
        },
    }
}

// =============================================================================
// HVAC Modes
// =============================================================================

/// HVAC operating mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HVACMode {
    /// Heating mode
    Heating,
    /// Cooling mode
    Cooling,
    /// Off
    Off,
}

/// Heat pump operating mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeatPumpMode {
    /// Heating mode
    Heating,
    /// Cooling mode
    Cooling,
    /// Off
    Off,
}

// =============================================================================
// VariableCapacityEquipment trait
// =============================================================================

/// Trait for variable-capacity HVAC equipment.
///
/// This trait provides a unified interface for HVAC equipment that can
/// modulate continuously from 0-100% capacity, enabling accurate simulation
/// of part-load performance and energy consumption.
pub trait VariableCapacityEquipment: Send + Sync + Clone {
    /// Calculate equipment capacity at given part-load ratio and outdoor temperature.
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64;

    /// Calculate equipment efficiency at given operating conditions.
    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64;

    /// Calculate power consumption for a given load.
    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64;

    /// Get rated capacity at design conditions.
    fn rated_capacity(&self) -> f64;

    /// Get rated efficiency at design conditions.
    fn rated_efficiency(&self, mode: HVACMode) -> f64;

    /// Get current part-load ratio.
    fn current_plr(&self) -> f64;

    /// Update equipment state based on current load and conditions.
    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode);
}

// =============================================================================
// Chiller
// =============================================================================

/// Chiller equipment model with polynomial efficiency curves.
///
/// Chillers provide chilled water for cooling coils in large commercial buildings.
/// Uses cubic polynomial curves for realistic part-load efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chiller {
    /// Equipment identifier
    pub id: String,
    /// Rated cooling capacity at design conditions (W)
    pub cooling_capacity: f64,
    /// Rated cooling COP at design conditions
    pub cooling_cop: f64,
    /// Design outdoor temperature for cooling (°C)
    pub design_temp: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Minimum outdoor temperature (°C)
    pub min_outdoor_temp: f64,
    /// Maximum outdoor temperature (°C)
    pub max_outdoor_temp: f64,
    /// Polynomial efficiency curve for cooling mode
    pub efficiency_curve_cooling: EfficiencyCurve,
    /// When true, bypass polynomial curves and use rated COP at all conditions.
    #[serde(default = "default_use_constant_cop")]
    pub use_constant_cop: bool,
}

fn default_use_constant_cop() -> bool {
    true
}

impl Chiller {
    /// Create a new chiller with default parameters
    pub fn new(id: String, cooling_capacity: f64, cooling_cop: f64, design_temp: f64) -> Self {
        let default_coeffs = default_ahri_coefficients();

        Self {
            id,
            cooling_capacity,
            cooling_cop,
            design_temp,
            current_plr: 0.0,
            min_outdoor_temp: 5.0,
            max_outdoor_temp: 45.0,
            efficiency_curve_cooling: (&default_coeffs.chiller).into(),
            use_constant_cop: true,
        }
    }

    fn capacity_at_temperature(&self, outdoor_temp: f64) -> f64 {
        if outdoor_temp < self.min_outdoor_temp || outdoor_temp > self.max_outdoor_temp {
            self.cooling_capacity * 0.3
        } else {
            let temp_diff = (outdoor_temp - self.design_temp).abs();
            let capacity_factor = 1.0 - (temp_diff * 0.005);
            self.cooling_capacity * capacity_factor.max(0.3)
        }
    }

    fn normalize_polynomial_cop(
        &self,
        curve: &EfficiencyCurve,
        plr: f64,
        outdoor_temp: f64,
        design_temp: f64,
        rated_cop: f64,
    ) -> f64 {
        let poly_cop = curve.cop_at(plr, outdoor_temp);
        let poly_cop_at_rated = curve.cop_at(1.0, design_temp);
        if poly_cop_at_rated > 0.0 && rated_cop > 0.0 {
            (poly_cop / poly_cop_at_rated) * rated_cop
        } else {
            poly_cop
        }
    }
}

impl VariableCapacityEquipment for Chiller {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        let base_capacity = self.capacity_at_temperature(outdoor_temp);
        base_capacity * plr
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Cooling => {
                if self.use_constant_cop {
                    self.cooling_cop
                } else {
                    self.normalize_polynomial_cop(
                        &self.efficiency_curve_cooling,
                        plr,
                        outdoor_temp,
                        self.design_temp,
                        self.cooling_cop,
                    )
                }
            }
            HVACMode::Heating | HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        let efficiency =
            self.calculate_efficiency(load / self.rated_capacity(), outdoor_temp, mode);
        if efficiency > 0.0 {
            load / efficiency
        } else {
            0.0
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.cooling_capacity
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Cooling => self.cooling_cop,
            HVACMode::Heating | HVACMode::Off => 0.0,
        }
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        if mode != HVACMode::Cooling {
            self.current_plr = 0.0;
            return;
        }

        let capacity = self.capacity_at_temperature(outdoor_temp);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

// =============================================================================
// Boiler
// =============================================================================

/// Boiler equipment model with polynomial efficiency curves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boiler {
    /// Equipment identifier
    pub id: String,
    /// Rated heating capacity at design conditions (W)
    pub heating_capacity: f64,
    /// Rated efficiency (AFUE) at design conditions (0.0 to 1.0)
    pub efficiency: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Minimum outdoor temperature (°C)
    pub min_outdoor_temp: f64,
    /// Design outdoor temperature for heating (°C)
    pub design_temp: f64,
    /// Polynomial efficiency curve for heating mode
    pub efficiency_curve_heating: EfficiencyCurve,
    /// Standby power consumption for controls and ignition (W)
    pub standby_power: f64,
    /// Electrical power consumption factor when firing (W per W of thermal output)
    pub electrical_power_factor: f64,
}

impl Boiler {
    /// Create a new boiler with default parameters
    pub fn new(id: String, heating_capacity: f64, efficiency: f64, design_temp: f64) -> Self {
        let default_coeffs = default_ahri_coefficients();

        Self {
            id,
            heating_capacity,
            efficiency,
            current_plr: 0.0,
            min_outdoor_temp: -20.0,
            design_temp,
            efficiency_curve_heating: (&default_coeffs.boiler).into(),
            standby_power: 5.0,
            electrical_power_factor: 0.08,
        }
    }

    fn capacity_at_temperature(&self, outdoor_temp: f64) -> f64 {
        if outdoor_temp < self.min_outdoor_temp {
            self.heating_capacity * 0.5
        } else {
            let temp_diff = (self.design_temp - outdoor_temp).abs();
            let capacity_factor = 1.0 - (temp_diff * 0.001);
            self.heating_capacity * capacity_factor.max(0.5)
        }
    }
}

impl VariableCapacityEquipment for Boiler {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        let base_capacity = self.capacity_at_temperature(outdoor_temp);
        base_capacity * plr
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => {
                self.efficiency_curve_heating.cop_at(plr, outdoor_temp)
            }
            HVACMode::Cooling | HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => {
                if load > 0.0 {
                    let plr = if self.current_plr > 0.0 {
                        self.current_plr
                    } else {
                        let capacity = self.capacity_at_temperature(outdoor_temp);
                        if capacity > 0.0 {
                            (load / capacity).clamp(0.0, 1.0)
                        } else {
                            0.0
                        }
                    };
                    if plr > 0.0 {
                        load / self.efficiency + load * self.electrical_power_factor
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
            HVACMode::Cooling | HVACMode::Off => 0.0,
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.heating_capacity
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => self.efficiency,
            HVACMode::Cooling | HVACMode::Off => 0.0,
        }
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        if mode != HVACMode::Heating {
            self.current_plr = 0.0;
            return;
        }

        let capacity = self.capacity_at_temperature(outdoor_temp);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

// =============================================================================
// VAVTerminal (simplified)
// =============================================================================

/// Represents a VAV (Variable Air Volume) terminal unit (simplified).
///
/// This is a simplified version without the nested psychrometric state
/// (CavTerminalUnit) that exists in the main crate. It contains only the
/// fields needed for VariableCapacityEquipment trait implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAVTerminal {
    /// Terminal unit identifier
    pub id: String,
    /// Zone served by this terminal
    pub zone_id: usize,
    /// Maximum air flow rate (m³/s)
    pub max_airflow: f64,
    /// Minimum air flow rate (m³/s)
    pub min_airflow: f64,
    /// Reheat coil capacity (W)
    pub reheat_capacity: f64,
    /// Current airflow setpoint (m³/s)
    pub airflow_setpoint: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
}

impl VAVTerminal {
    /// Create a new VAV terminal unit
    pub fn new(id: String, zone_id: usize, max_airflow: f64) -> Self {
        Self {
            id,
            zone_id,
            max_airflow,
            min_airflow: max_airflow * 0.3,
            reheat_capacity: 5000.0,
            airflow_setpoint: max_airflow,
            current_plr: 0.0,
        }
    }
}

impl VariableCapacityEquipment for VAVTerminal {
    fn calculate_capacity(&self, plr: f64, _outdoor_temp: f64) -> f64 {
        self.reheat_capacity * plr
    }

    fn calculate_efficiency(&self, _plr: f64, _outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => 0.8,
            HVACMode::Cooling => 3.0,
            HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, _outdoor_temp: f64, mode: HVACMode) -> f64 {
        let efficiency = self.calculate_efficiency(load / self.rated_capacity(), 20.0, mode);
        if efficiency > 0.0 {
            load / efficiency
        } else {
            0.0
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.reheat_capacity
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        self.calculate_efficiency(1.0, 20.0, mode)
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, _outdoor_temp: f64, _mode: HVACMode) {
        let capacity = self.calculate_capacity(1.0, 20.0);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

// =============================================================================
// CAVSystem (simplified)
// =============================================================================

/// Represents a CAV (Constant Air Volume) system (simplified).
///
/// This is a simplified version without the nested CavTerminalUnit
/// psychrometric state. It contains only the fields needed for
/// VariableCapacityEquipment trait implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CAVSystem {
    /// System identifier
    pub id: String,
    /// Design air flow rate (m³/s)
    pub design_airflow: f64,
    /// Fan power consumption (W)
    pub fan_power: f64,
    /// Fan efficiency (0-1)
    pub fan_efficiency: f64,
    /// Heating coil capacity (W)
    pub heating_capacity: f64,
    /// Cooling coil capacity (W)
    pub cooling_capacity: f64,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
}

impl CAVSystem {
    /// Create a new CAV system.
    pub fn new(id: String, design_airflow: f64) -> Self {
        Self {
            id,
            design_airflow,
            fan_power: design_airflow * 500.0,
            fan_efficiency: 0.7,
            heating_capacity: 10000.0,
            cooling_capacity: 10000.0,
            current_plr: 0.0,
        }
    }
}

impl VariableCapacityEquipment for CAVSystem {
    fn calculate_capacity(&self, plr: f64, _outdoor_temp: f64) -> f64 {
        let max_capacity = self.heating_capacity.max(self.cooling_capacity);
        max_capacity * plr
    }

    fn calculate_efficiency(&self, _plr: f64, _outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => 0.85,
            HVACMode::Cooling => 3.2,
            HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, _outdoor_temp: f64, mode: HVACMode) -> f64 {
        let fan_power = self.fan_power / self.fan_efficiency;
        let thermal_power = {
            let efficiency = self.calculate_efficiency(load / self.rated_capacity(), 20.0, mode);
            if efficiency > 0.0 {
                load / efficiency
            } else {
                0.0
            }
        };
        fan_power + thermal_power
    }

    fn rated_capacity(&self) -> f64 {
        self.heating_capacity.max(self.cooling_capacity)
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        self.calculate_efficiency(1.0, 20.0, mode)
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, _outdoor_temp: f64, _mode: HVACMode) {
        let capacity = self.calculate_capacity(1.0, 20.0);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

// =============================================================================
// HeatPump
// =============================================================================

/// Represents a heat pump system with COP curves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatPump {
    /// System identifier
    pub id: String,
    /// Rated heating capacity at design conditions (W)
    pub heating_capacity: f64,
    /// Rated cooling capacity at design conditions (W)
    pub cooling_capacity: f64,
    /// Rated heating COP at design conditions
    pub heating_cop: f64,
    /// Rated cooling COP (EER) at design conditions
    pub cooling_cop: f64,
    /// Design outdoor temperature for heating (°C)
    pub design_temp_heating: f64,
    /// Design outdoor temperature for cooling (°C)
    pub design_temp_cooling: f64,
    /// Current operating mode
    pub mode: HeatPumpMode,
    /// Current part-load ratio (0.0 to 1.0)
    pub current_plr: f64,
    /// Polynomial efficiency curve for heating mode
    pub efficiency_curve_heating: EfficiencyCurve,
    /// Polynomial efficiency curve for cooling mode
    pub efficiency_curve_cooling: EfficiencyCurve,
}

impl HeatPump {
    /// Create a new heat pump
    pub fn new(
        id: String,
        heating_capacity: f64,
        cooling_capacity: f64,
        heating_cop: f64,
        cooling_cop: f64,
    ) -> Self {
        let default_coeffs = default_ahri_coefficients();

        Self {
            id,
            heating_capacity,
            cooling_capacity,
            heating_cop,
            cooling_cop,
            design_temp_heating: -5.0,
            design_temp_cooling: 35.0,
            mode: HeatPumpMode::Off,
            current_plr: 0.0,
            efficiency_curve_heating: (&default_coeffs.heatpump_heating).into(),
            efficiency_curve_cooling: (&default_coeffs.heatpump_cooling).into(),
        }
    }

    fn normalize_polynomial_cop(
        &self,
        curve: &EfficiencyCurve,
        plr: f64,
        outdoor_temp: f64,
        design_temp: f64,
        rated_cop: f64,
    ) -> f64 {
        let poly_cop = curve.cop_at(plr, outdoor_temp);
        let poly_cop_at_rated = curve.cop_at(1.0, design_temp);
        if poly_cop_at_rated > 0.0 && rated_cop > 0.0 {
            (poly_cop / poly_cop_at_rated) * rated_cop
        } else {
            poly_cop
        }
    }
}

impl VariableCapacityEquipment for HeatPump {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        let (base_capacity, design_temp) = match self.mode {
            HeatPumpMode::Heating => (self.heating_capacity, self.design_temp_heating),
            HeatPumpMode::Cooling => (self.cooling_capacity, self.design_temp_cooling),
            HeatPumpMode::Off => (self.heating_capacity, self.design_temp_heating),
        };

        let temp_diff = (design_temp - outdoor_temp).abs();
        let capacity_factor = 1.0 - (temp_diff * 0.01);
        base_capacity * capacity_factor.max(0.3) * plr
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => self.normalize_polynomial_cop(
                &self.efficiency_curve_heating,
                plr,
                outdoor_temp,
                self.design_temp_heating,
                self.heating_cop,
            ),
            HVACMode::Cooling => self.normalize_polynomial_cop(
                &self.efficiency_curve_cooling,
                plr,
                outdoor_temp,
                self.design_temp_cooling,
                self.cooling_cop,
            ),
            HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        let efficiency =
            self.calculate_efficiency(load / self.rated_capacity(), outdoor_temp, mode);
        if efficiency > 0.0 {
            load / efficiency
        } else {
            0.0
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.heating_capacity.max(self.cooling_capacity)
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => self.heating_cop,
            HVACMode::Cooling => self.cooling_cop,
            HVACMode::Off => 0.0,
        }
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        let capacity = self.calculate_capacity(1.0, outdoor_temp);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.mode = match mode {
            HVACMode::Heating => HeatPumpMode::Heating,
            HVACMode::Cooling => HeatPumpMode::Cooling,
            HVACMode::Off => HeatPumpMode::Off,
        };
    }
}

// =============================================================================
// AnyEquipment
// =============================================================================

/// Enum wrapper for all variable-capacity HVAC equipment types.
///
/// This enum enables dynamic equipment selection while maintaining Clone compatibility
/// for ThermalModel.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyEquipment {
    /// Chiller equipment (cooling-only)
    Chiller(Chiller),
    /// Boiler equipment (heating-only)
    Boiler(Boiler),
    /// VAV terminal unit with reheat
    VAVTerminal(VAVTerminal),
    /// CAV system with constant airflow
    CAVSystem(CAVSystem),
    /// Heat pump system (heating and cooling)
    HeatPump(HeatPump),
}

impl VariableCapacityEquipment for AnyEquipment {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.calculate_capacity(plr, outdoor_temp),
            AnyEquipment::Boiler(e) => e.calculate_capacity(plr, outdoor_temp),
            AnyEquipment::VAVTerminal(e) => e.calculate_capacity(plr, outdoor_temp),
            AnyEquipment::CAVSystem(e) => e.calculate_capacity(plr, outdoor_temp),
            AnyEquipment::HeatPump(e) => e.calculate_capacity(plr, outdoor_temp),
        }
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
            AnyEquipment::Boiler(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
            AnyEquipment::VAVTerminal(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
            AnyEquipment::CAVSystem(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
            AnyEquipment::HeatPump(e) => e.calculate_efficiency(plr, outdoor_temp, mode),
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.calculate_power(load, outdoor_temp, mode),
            AnyEquipment::Boiler(e) => e.calculate_power(load, outdoor_temp, mode),
            AnyEquipment::VAVTerminal(e) => e.calculate_power(load, outdoor_temp, mode),
            AnyEquipment::CAVSystem(e) => e.calculate_power(load, outdoor_temp, mode),
            AnyEquipment::HeatPump(e) => e.calculate_power(load, outdoor_temp, mode),
        }
    }

    fn rated_capacity(&self) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.rated_capacity(),
            AnyEquipment::Boiler(e) => e.rated_capacity(),
            AnyEquipment::VAVTerminal(e) => e.rated_capacity(),
            AnyEquipment::CAVSystem(e) => e.rated_capacity(),
            AnyEquipment::HeatPump(e) => e.rated_capacity(),
        }
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.rated_efficiency(mode),
            AnyEquipment::Boiler(e) => e.rated_efficiency(mode),
            AnyEquipment::VAVTerminal(e) => e.rated_efficiency(mode),
            AnyEquipment::CAVSystem(e) => e.rated_efficiency(mode),
            AnyEquipment::HeatPump(e) => e.rated_efficiency(mode),
        }
    }

    fn current_plr(&self) -> f64 {
        match self {
            AnyEquipment::Chiller(e) => e.current_plr(),
            AnyEquipment::Boiler(e) => e.current_plr(),
            AnyEquipment::VAVTerminal(e) => e.current_plr(),
            AnyEquipment::CAVSystem(e) => e.current_plr(),
            AnyEquipment::HeatPump(e) => e.current_plr(),
        }
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        match self {
            AnyEquipment::Chiller(e) => e.update_state(current_load, outdoor_temp, mode),
            AnyEquipment::Boiler(e) => e.update_state(current_load, outdoor_temp, mode),
            AnyEquipment::VAVTerminal(e) => e.update_state(current_load, outdoor_temp, mode),
            AnyEquipment::CAVSystem(e) => e.update_state(current_load, outdoor_temp, mode),
            AnyEquipment::HeatPump(e) => e.update_state(current_load, outdoor_temp, mode),
        }
    }
}

impl AnyEquipment {
    /// Returns the ventilation airflow rate in m³/s for economizer free-cooling calculations.
    pub fn ventilation_airflow_m3_per_s(&self) -> f64 {
        match self {
            AnyEquipment::Chiller(_) => 0.0,
            AnyEquipment::Boiler(_) => 0.0,
            AnyEquipment::VAVTerminal(e) => e.max_airflow,
            AnyEquipment::CAVSystem(e) => e.design_airflow,
            AnyEquipment::HeatPump(_) => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chiller_variable_capacity() {
        let chiller = Chiller::new(
            "CH-1".to_string(),
            100000.0,
            4.5,
            35.0,
        );
        assert_eq!(chiller.rated_capacity(), 100000.0);
        assert_eq!(chiller.rated_efficiency(HVACMode::Cooling), 4.5);

        let capacity_design = chiller.calculate_capacity(1.0, 35.0);
        assert!((capacity_design - 100000.0).abs() < 1.0);

        let capacity_hot = chiller.calculate_capacity(1.0, 45.0);
        assert!(capacity_hot < 100000.0);
        assert!(capacity_hot > 30000.0);
    }

    #[test]
    fn test_vav_terminal() {
        let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);
        assert_eq!(vav.max_airflow, 0.5);
        assert_eq!(vav.min_airflow, 0.15);
        assert_eq!(vav.reheat_capacity, 5000.0);
    }

    #[test]
    fn test_cav_system() {
        let cav = CAVSystem::new("CAV-1".to_string(), 1.0);
        assert_eq!(cav.design_airflow, 1.0);
        assert!(cav.fan_power > 0.0);
    }

    #[test]
    fn test_heat_pump_cop() {
        let hp = HeatPump::new(
            "HP-1".to_string(),
            12000.0,
            10000.0,
            3.5,
            3.0,
        );

        assert_eq!(hp.rated_capacity(), 12000.0);
        assert_eq!(hp.heating_cop, 3.5);
        assert_eq!(hp.cooling_cop, 3.0);
    }

    #[test]
    fn test_any_equipment_dispatch() {
        let chiller = AnyEquipment::Chiller(Chiller::new("CH-1".to_string(), 100000.0, 4.5, 35.0));
        let boiler = AnyEquipment::Boiler(Boiler::new("BO-1".to_string(), 50000.0, 0.85, -5.0));
        let vav = AnyEquipment::VAVTerminal(VAVTerminal::new("VAV-1".to_string(), 0, 0.5));
        let cav = AnyEquipment::CAVSystem(CAVSystem::new("CAV-1".to_string(), 1.0));
        let hp = AnyEquipment::HeatPump(HeatPump::new("HP-1".to_string(), 12000.0, 10000.0, 3.5, 3.0));

        assert_eq!(chiller.rated_capacity(), 100000.0);
        assert_eq!(boiler.rated_capacity(), 50000.0);
        assert_eq!(vav.rated_capacity(), 5000.0);
        assert_eq!(cav.rated_capacity(), 10000.0);
        assert_eq!(hp.rated_capacity(), 12000.0);

        assert_eq!(chiller.ventilation_airflow_m3_per_s(), 0.0);
        assert_eq!(vav.ventilation_airflow_m3_per_s(), 0.5);
        assert_eq!(cav.ventilation_airflow_m3_per_s(), 1.0);
    }
}
