//! Psychrometric state and supply-flow types for airside HVAC coupling.
//!
//! Properties use ASHRAE Handbook—Fundamentals (2021), Chapter 1 in SI
//! units. Construction rejects non-finite, negative-humidity, and
//! supersaturated states before they reach the thermal solver.

use fluxion_core::weather::psychrometrics::{
    calculate_enthalpy, calculate_humidity_ratio, calculate_wet_bulb, moist_air_density,
    partial_vapor_pressure, saturation_vapor_pressure,
};
use thiserror::Error;

pub(crate) const CP_WATER_VAPOR_KJ_PER_KG_K: f64 = 1.86;
pub(crate) const LATENT_HEAT_0C_KJ_PER_KG: f64 = 2501.0;
const CP_DRY_AIR_KJ_PER_KG_K: f64 = 1.006;
const SATURATION_TOLERANCE_PERCENT: f64 = 1.0e-8;

/// Maximum timestep covered by the issue #1767 annual stability validation.
pub const MAX_VALIDATED_TIMESTEP_SECONDS: f64 = 360.0;

/// Absolute First-Law tolerance at the airside/envelope interface.
///
/// This matches the `1e-7 W` tolerance used by `MultiNodeSolver` for its
/// backward-Euler mass-node balance.
pub const DEFAULT_ENERGY_BALANCE_TOLERANCE_W: f64 = 1.0e-7;

/// Errors returned without committing a partially advanced coupled state.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum AirsideCouplingError {
    #[error("invalid airside coupling input `{field}`: {value}")]
    InvalidInput { field: &'static str, value: f64 },
    #[error("derived psychrometric property `{property}` is non-finite")]
    NonFinitePsychrometricProperty { property: &'static str },
    #[error(
        "air state is supersaturated: relative humidity {relative_humidity_percent:.9}% exceeds 100%"
    )]
    SupersaturatedAirState { relative_humidity_percent: f64 },
    #[error(
        "timestep {dt_seconds} s exceeds the validated six-minute maximum of {MAX_VALIDATED_TIMESTEP_SECONDS} s"
    )]
    TimestepExceedsValidatedMaximum { dt_seconds: f64 },
    #[error("9R4C envelope state is non-finite or has non-positive thermal coefficients")]
    InvalidEnvelopeState,
    #[error("coupled energy residual {residual_w:.6e} W exceeds tolerance {tolerance_w:.6e} W")]
    EnergyBalanceViolation { residual_w: f64, tolerance_w: f64 },
}

/// Thermodynamically consistent moist-air state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoistAirState {
    pub dry_bulb_c: f64,
    pub relative_humidity_percent: f64,
    pub pressure_pa: f64,
    pub humidity_ratio_kg_per_kg_dry_air: f64,
    pub enthalpy_kj_per_kg_dry_air: f64,
    pub wet_bulb_c: f64,
    pub density_kg_per_m3: f64,
    pub partial_vapor_pressure_pa: f64,
}

impl MoistAirState {
    /// Construct from dry-bulb temperature, relative humidity, and pressure.
    pub fn try_new(
        dry_bulb_c: f64,
        relative_humidity_percent: f64,
        pressure_pa: f64,
    ) -> Result<Self, AirsideCouplingError> {
        validate_finite("dry_bulb_c", dry_bulb_c)?;
        validate_finite("relative_humidity_percent", relative_humidity_percent)?;
        validate_finite("pressure_pa", pressure_pa)?;
        if dry_bulb_c <= -273.15 {
            return Err(AirsideCouplingError::InvalidInput {
                field: "dry_bulb_c",
                value: dry_bulb_c,
            });
        }
        if !(0.0..=100.0).contains(&relative_humidity_percent) {
            return Err(AirsideCouplingError::InvalidInput {
                field: "relative_humidity_percent",
                value: relative_humidity_percent,
            });
        }
        if pressure_pa <= 0.0 {
            return Err(AirsideCouplingError::InvalidInput {
                field: "pressure_pa",
                value: pressure_pa,
            });
        }

        let humidity_ratio_kg_per_kg_dry_air =
            calculate_humidity_ratio(dry_bulb_c, relative_humidity_percent, pressure_pa);
        let enthalpy_kj_per_kg_dry_air =
            calculate_enthalpy(dry_bulb_c, relative_humidity_percent, pressure_pa);
        let wet_bulb_c = calculate_wet_bulb(dry_bulb_c, relative_humidity_percent, pressure_pa);
        let density_kg_per_m3 =
            moist_air_density(dry_bulb_c, humidity_ratio_kg_per_kg_dry_air, pressure_pa);
        let partial_vapor_pressure_pa =
            partial_vapor_pressure(humidity_ratio_kg_per_kg_dry_air, pressure_pa);

        let state = Self {
            dry_bulb_c,
            relative_humidity_percent,
            pressure_pa,
            humidity_ratio_kg_per_kg_dry_air,
            enthalpy_kj_per_kg_dry_air,
            wet_bulb_c,
            density_kg_per_m3,
            partial_vapor_pressure_pa,
        };
        state.validate_derived()?;
        Ok(state)
    }

    pub(crate) fn from_humidity_ratio(
        dry_bulb_c: f64,
        humidity_ratio_kg_per_kg_dry_air: f64,
        pressure_pa: f64,
    ) -> Result<Self, AirsideCouplingError> {
        validate_nonnegative(
            "humidity_ratio_kg_per_kg_dry_air",
            humidity_ratio_kg_per_kg_dry_air,
        )?;
        let vapor_pressure_pa =
            partial_vapor_pressure(humidity_ratio_kg_per_kg_dry_air, pressure_pa);
        let saturation_pressure_pa = saturation_vapor_pressure(dry_bulb_c);
        if !vapor_pressure_pa.is_finite()
            || !saturation_pressure_pa.is_finite()
            || saturation_pressure_pa <= 0.0
        {
            return Err(AirsideCouplingError::NonFinitePsychrometricProperty {
                property: "relative_humidity",
            });
        }
        let relative_humidity_percent = 100.0 * vapor_pressure_pa / saturation_pressure_pa;
        if relative_humidity_percent > 100.0 + SATURATION_TOLERANCE_PERCENT {
            return Err(AirsideCouplingError::SupersaturatedAirState {
                relative_humidity_percent,
            });
        }
        Self::try_new(
            dry_bulb_c,
            relative_humidity_percent.clamp(0.0, 100.0),
            pressure_pa,
        )
    }

    pub(crate) fn validate_derived(&self) -> Result<(), AirsideCouplingError> {
        let properties = [
            ("humidity_ratio", self.humidity_ratio_kg_per_kg_dry_air),
            ("enthalpy", self.enthalpy_kj_per_kg_dry_air),
            ("wet_bulb", self.wet_bulb_c),
            ("density", self.density_kg_per_m3),
            ("partial_vapor_pressure", self.partial_vapor_pressure_pa),
        ];
        for (property, value) in properties {
            if !value.is_finite() {
                return Err(AirsideCouplingError::NonFinitePsychrometricProperty { property });
            }
        }
        if self.humidity_ratio_kg_per_kg_dry_air < 0.0
            || self.density_kg_per_m3 <= 0.0
            || self.partial_vapor_pressure_pa < 0.0
            || self.partial_vapor_pressure_pa >= self.pressure_pa
            || self.wet_bulb_c > self.dry_bulb_c + 1.0e-7
        {
            return Err(AirsideCouplingError::InvalidInput {
                field: "derived_moist_air_state",
                value: self.humidity_ratio_kg_per_kg_dry_air,
            });
        }
        Ok(())
    }

    pub fn is_finite(&self) -> bool {
        [
            self.dry_bulb_c,
            self.relative_humidity_percent,
            self.pressure_pa,
            self.humidity_ratio_kg_per_kg_dry_air,
            self.enthalpy_kj_per_kg_dry_air,
            self.wet_bulb_c,
            self.density_kg_per_m3,
            self.partial_vapor_pressure_pa,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    pub(crate) fn dry_air_specific_heat_j_per_kg_k(&self) -> f64 {
        1000.0
            * (CP_DRY_AIR_KJ_PER_KG_K
                + CP_WATER_VAPOR_KJ_PER_KG_K * self.humidity_ratio_kg_per_kg_dry_air)
    }
}

/// Supply-air state produced by a VAV, DOAS, or other airside component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AirsideFlow {
    supply_air: MoistAirState,
    volumetric_flow_m3_per_s: f64,
    dry_air_mass_flow_kg_per_s: f64,
}

impl AirsideFlow {
    pub fn new(
        supply_air: MoistAirState,
        volumetric_flow_m3_per_s: f64,
    ) -> Result<Self, AirsideCouplingError> {
        supply_air.validate_derived()?;
        validate_nonnegative("volumetric_flow_m3_per_s", volumetric_flow_m3_per_s)?;
        let dry_air_mass_flow_kg_per_s = supply_air.density_kg_per_m3 * volumetric_flow_m3_per_s
            / (1.0 + supply_air.humidity_ratio_kg_per_kg_dry_air);
        validate_nonnegative("dry_air_mass_flow_kg_per_s", dry_air_mass_flow_kg_per_s)?;
        Ok(Self {
            supply_air,
            volumetric_flow_m3_per_s,
            dry_air_mass_flow_kg_per_s,
        })
    }

    pub fn supply_air(&self) -> &MoistAirState {
        &self.supply_air
    }

    pub fn volumetric_flow_m3_per_s(&self) -> f64 {
        self.volumetric_flow_m3_per_s
    }

    pub fn dry_air_mass_flow_kg_per_s(&self) -> f64 {
        self.dry_air_mass_flow_kg_per_s
    }
}

pub(crate) fn validate_finite(field: &'static str, value: f64) -> Result<(), AirsideCouplingError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AirsideCouplingError::InvalidInput { field, value })
    }
}

pub(crate) fn validate_nonnegative(
    field: &'static str,
    value: f64,
) -> Result<(), AirsideCouplingError> {
    validate_finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(AirsideCouplingError::InvalidInput { field, value })
    }
}

pub(crate) fn validate_positive(
    field: &'static str,
    value: f64,
) -> Result<(), AirsideCouplingError> {
    validate_finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(AirsideCouplingError::InvalidInput { field, value })
    }
}
