//! Stable coupling between airside supply-air states and the 9R4C envelope.
//!
//! The airside component issues (#1761-#1765) produce supply temperature,
//! humidity and flow. This module is the integration boundary: it deliberately
//! contains no fan or coil correlations.
//!
//! # Coupling scheme
//!
//! Each timestep uses a sequential implicit operator split:
//!
//! 1. Advance the 9R4C mass nodes by `dt / 2` with backward Euler.
//! 2. Solve the algebraic zone-air node implicitly with ventilation and supply.
//! 3. Repeat the half envelope step and implicit air solve.
//! 4. Advance zone humidity ratio with backward Euler.
//!
//! The implicit kernels are unconditionally stable for positive capacitances
//! and conductances. Validation covers `dt <= 360 s` (six minutes); larger
//! timesteps are rejected because their splitting error is not accepted here.

use super::airside_state::{
    validate_finite, validate_nonnegative, validate_positive, AirsideCouplingError, AirsideFlow,
    MoistAirState, CP_WATER_VAPOR_KJ_PER_KG_K, DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
    LATENT_HEAT_0C_KJ_PER_KG, MAX_VALIDATED_TIMESTEP_SECONDS,
};
use crate::physics::multi_node_solver::{h_series, MultiNodeSolver, SurfaceExteriorTemperatures};
use fluxion_core::multi_node::MassAirCouplingMode;

const MIN_POSITIVE: f64 = 1.0e-12;

/// Envelope, weather, ventilation, and internal-gain forcing for one timestep.
#[derive(Debug, Clone)]
pub struct CoupledStepForcing {
    pub exterior_temperatures: SurfaceExteriorTemperatures,
    pub outdoor_air: MoistAirState,
    pub ventilation_conductance_w_per_k: f64,
    pub convective_gain_w: f64,
    /// `[wall, roof, floor, internal]` heat-gain rates in watts.
    pub envelope_gains_w: [f64; 4],
}

/// Diagnostics from one accepted coupled timestep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledStepResult {
    pub zone_air: MoistAirState,
    pub envelope_node_temperatures_c: [f64; 4],
    pub supply_dry_air_mass_flow_kg_per_s: f64,
    pub supply_sensible_heat_w: f64,
    pub supply_latent_heat_w: f64,
    pub supply_total_heat_w: f64,
    pub ventilation_total_heat_w: f64,
    pub latent_storage_rate_w: f64,
    pub sensible_balance_residual_w: f64,
    pub latent_balance_residual_w: f64,
    pub energy_balance_residual_w: f64,
    pub moisture_balance_residual_kg_per_s: f64,
}

impl CoupledStepResult {
    pub fn is_finite(&self) -> bool {
        self.zone_air.is_finite()
            && self
                .envelope_node_temperatures_c
                .into_iter()
                .all(f64::is_finite)
            && [
                self.supply_dry_air_mass_flow_kg_per_s,
                self.supply_sensible_heat_w,
                self.supply_latent_heat_w,
                self.supply_total_heat_w,
                self.ventilation_total_heat_w,
                self.latent_storage_rate_w,
                self.sensible_balance_residual_w,
                self.latent_balance_residual_w,
                self.energy_balance_residual_w,
                self.moisture_balance_residual_kg_per_s,
            ]
            .into_iter()
            .all(f64::is_finite)
    }
}

/// Transactional, stability-preserving airside/9R4C coupling controller.
#[derive(Debug, Clone)]
pub struct AirsideEnvelopeCoupler {
    envelope: MultiNodeSolver,
    zone_air: MoistAirState,
    zone_dry_air_mass_kg: f64,
}

impl AirsideEnvelopeCoupler {
    pub fn new(
        mut envelope: MultiNodeSolver,
        initial_zone_air: MoistAirState,
        zone_volume_m3: f64,
    ) -> Result<Self, AirsideCouplingError> {
        validate_positive("zone_volume_m3", zone_volume_m3)?;
        initial_zone_air.validate_derived()?;
        validate_envelope(&envelope)?;
        let zone_dry_air_mass_kg = initial_zone_air.density_kg_per_m3 * zone_volume_m3
            / (1.0 + initial_zone_air.humidity_ratio_kg_per_kg_dry_air);
        validate_positive("zone_dry_air_mass_kg", zone_dry_air_mass_kg)?;
        envelope.set_zone_temperature(initial_zone_air.dry_bulb_c);
        Ok(Self {
            envelope,
            zone_air: initial_zone_air,
            zone_dry_air_mass_kg,
        })
    }

    pub fn envelope(&self) -> &MultiNodeSolver {
        &self.envelope
    }

    pub fn zone_air(&self) -> &MoistAirState {
        &self.zone_air
    }

    /// Advance one coupled timestep and commit only after all balance checks pass.
    pub fn step(
        &mut self,
        dt_seconds: f64,
        forcing: &CoupledStepForcing,
        airside: &AirsideFlow,
    ) -> Result<CoupledStepResult, AirsideCouplingError> {
        validate_positive("dt_seconds", dt_seconds)?;
        if dt_seconds > MAX_VALIDATED_TIMESTEP_SECONDS {
            return Err(AirsideCouplingError::TimestepExceedsValidatedMaximum { dt_seconds });
        }
        validate_forcing(forcing)?;
        validate_envelope(&self.envelope)?;

        // Transactional update prevents a failed finite-state check from
        // poisoning the last accepted envelope and zone-air state.
        let mut envelope = self.envelope.clone();
        envelope.set_surface_exterior_temperatures(forcing.exterior_temperatures.clone());

        let supply_conductance_w_per_k = airside.dry_air_mass_flow_kg_per_s()
            * airside.supply_air().dry_air_specific_heat_j_per_kg_k();
        validate_nonnegative("supply_conductance_w_per_k", supply_conductance_w_per_k)?;

        let half_dt = dt_seconds / 2.0;
        let mut zone_temperature_c = self.zone_air.dry_bulb_c;
        for _ in 0..2 {
            envelope.set_zone_temperature(zone_temperature_c);
            envelope.step_with_gains(
                half_dt,
                forcing.envelope_gains_w[0],
                forcing.envelope_gains_w[1],
                forcing.envelope_gains_w[2],
                forcing.envelope_gains_w[3],
                0.0,
                forcing.outdoor_air.dry_bulb_c,
            );
            validate_envelope(&envelope)?;
            zone_temperature_c = solve_air_node_temperature(
                &envelope,
                forcing,
                airside,
                supply_conductance_w_per_k,
            )?;
        }
        envelope.set_zone_temperature(zone_temperature_c);

        let outdoor_cp = forcing.outdoor_air.dry_air_specific_heat_j_per_kg_k();
        let ventilation_dry_air_mass_flow_kg_per_s =
            forcing.ventilation_conductance_w_per_k / outdoor_cp;
        let moisture_denominator = self.zone_dry_air_mass_kg / dt_seconds
            + airside.dry_air_mass_flow_kg_per_s()
            + ventilation_dry_air_mass_flow_kg_per_s;
        let new_humidity_ratio = (self.zone_dry_air_mass_kg / dt_seconds
            * self.zone_air.humidity_ratio_kg_per_kg_dry_air
            + airside.dry_air_mass_flow_kg_per_s()
                * airside.supply_air().humidity_ratio_kg_per_kg_dry_air
            + ventilation_dry_air_mass_flow_kg_per_s
                * forcing.outdoor_air.humidity_ratio_kg_per_kg_dry_air)
            / moisture_denominator;
        let zone_air = MoistAirState::from_humidity_ratio(
            zone_temperature_c,
            new_humidity_ratio,
            self.zone_air.pressure_pa,
        )?;

        let result = coupled_diagnostics(
            &envelope,
            &self.zone_air,
            zone_air,
            self.zone_dry_air_mass_kg,
            dt_seconds,
            forcing,
            airside,
            ventilation_dry_air_mass_flow_kg_per_s,
            supply_conductance_w_per_k,
        )?;
        if result.energy_balance_residual_w.abs() > DEFAULT_ENERGY_BALANCE_TOLERANCE_W {
            return Err(AirsideCouplingError::EnergyBalanceViolation {
                residual_w: result.energy_balance_residual_w,
                tolerance_w: DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
            });
        }

        self.envelope = envelope;
        self.zone_air = zone_air;
        Ok(result)
    }
}

fn solve_air_node_temperature(
    envelope: &MultiNodeSolver,
    forcing: &CoupledStepForcing,
    airside: &AirsideFlow,
    supply_conductance_w_per_k: f64,
) -> Result<f64, AirsideCouplingError> {
    let source_conductance = forcing.ventilation_conductance_w_per_k + supply_conductance_w_per_k;
    let source_temperature = if source_conductance > MIN_POSITIVE {
        (forcing.ventilation_conductance_w_per_k * forcing.outdoor_air.dry_bulb_c
            + supply_conductance_w_per_k * airside.supply_air().dry_bulb_c)
            / source_conductance
    } else {
        forcing.outdoor_air.dry_bulb_c
    };
    let temperature = envelope.compute_zone_air_temperature(
        source_temperature,
        source_conductance,
        0.0,
        forcing.convective_gain_w,
    );
    validate_finite("zone_temperature_c", temperature)?;
    Ok(temperature)
}

#[allow(clippy::too_many_arguments)]
fn coupled_diagnostics(
    envelope: &MultiNodeSolver,
    old_zone_air: &MoistAirState,
    new_zone_air: MoistAirState,
    zone_dry_air_mass_kg: f64,
    dt_seconds: f64,
    forcing: &CoupledStepForcing,
    airside: &AirsideFlow,
    ventilation_dry_air_mass_flow_kg_per_s: f64,
    supply_conductance_w_per_k: f64,
) -> Result<CoupledStepResult, AirsideCouplingError> {
    let t_zone = new_zone_air.dry_bulb_c;
    let q_envelope_to_air_w = envelope_air_heat_w(envelope, t_zone);
    let q_ventilation_sensible_w =
        forcing.ventilation_conductance_w_per_k * (forcing.outdoor_air.dry_bulb_c - t_zone);
    let supply_sensible_heat_w =
        supply_conductance_w_per_k * (airside.supply_air().dry_bulb_c - t_zone);
    let sensible_balance_residual_w = q_envelope_to_air_w
        + q_ventilation_sensible_w
        + supply_sensible_heat_w
        + forcing.convective_gain_w;

    let supply_total_heat_w = airside.dry_air_mass_flow_kg_per_s()
        * (airside.supply_air().enthalpy_kj_per_kg_dry_air
            - new_zone_air.enthalpy_kj_per_kg_dry_air)
        * 1000.0;
    let ventilation_total_heat_w = ventilation_dry_air_mass_flow_kg_per_s
        * (forcing.outdoor_air.enthalpy_kj_per_kg_dry_air
            - new_zone_air.enthalpy_kj_per_kg_dry_air)
        * 1000.0;
    let supply_latent_heat_w = supply_total_heat_w - supply_sensible_heat_w;
    let ventilation_latent_heat_w = ventilation_total_heat_w - q_ventilation_sensible_w;
    let latent_storage_rate_w = zone_dry_air_mass_kg / dt_seconds
        * (new_zone_air.humidity_ratio_kg_per_kg_dry_air
            - old_zone_air.humidity_ratio_kg_per_kg_dry_air)
        * (LATENT_HEAT_0C_KJ_PER_KG + CP_WATER_VAPOR_KJ_PER_KG_K * new_zone_air.dry_bulb_c)
        * 1000.0;
    let latent_balance_residual_w =
        supply_latent_heat_w + ventilation_latent_heat_w - latent_storage_rate_w;
    let energy_balance_residual_w = sensible_balance_residual_w + latent_balance_residual_w;

    let moisture_sources_kg_per_s = airside.dry_air_mass_flow_kg_per_s()
        * (airside.supply_air().humidity_ratio_kg_per_kg_dry_air
            - new_zone_air.humidity_ratio_kg_per_kg_dry_air)
        + ventilation_dry_air_mass_flow_kg_per_s
            * (forcing.outdoor_air.humidity_ratio_kg_per_kg_dry_air
                - new_zone_air.humidity_ratio_kg_per_kg_dry_air);
    let moisture_storage_kg_per_s = zone_dry_air_mass_kg / dt_seconds
        * (new_zone_air.humidity_ratio_kg_per_kg_dry_air
            - old_zone_air.humidity_ratio_kg_per_kg_dry_air);
    let moisture_balance_residual_kg_per_s = moisture_sources_kg_per_s - moisture_storage_kg_per_s;

    let result = CoupledStepResult {
        zone_air: new_zone_air,
        envelope_node_temperatures_c: envelope.snapshot_temperatures(),
        supply_dry_air_mass_flow_kg_per_s: airside.dry_air_mass_flow_kg_per_s(),
        supply_sensible_heat_w,
        supply_latent_heat_w,
        supply_total_heat_w,
        ventilation_total_heat_w,
        latent_storage_rate_w,
        sensible_balance_residual_w,
        latent_balance_residual_w,
        energy_balance_residual_w,
        moisture_balance_residual_kg_per_s,
    };
    if !result.is_finite() {
        return Err(AirsideCouplingError::NonFinitePsychrometricProperty {
            property: "coupled_step_result",
        });
    }
    Ok(result)
}

fn envelope_air_heat_w(envelope: &MultiNodeSolver, t_zone_c: f64) -> f64 {
    match envelope.coupling_mode {
        MassAirCouplingMode::AdditiveSum => {
            let h_total = envelope.mass.wall.h_tr_ms
                + envelope.mass.roof.h_tr_ms
                + envelope.mass.floor.h_tr_ms;
            let t_surface = if h_total > MIN_POSITIVE {
                (envelope.mass.wall.h_tr_ms * envelope.mass.wall.temperature
                    + envelope.mass.roof.h_tr_ms * envelope.mass.roof.temperature
                    + envelope.mass.floor.h_tr_ms * envelope.mass.floor.temperature)
                    / h_total
            } else {
                envelope.envelope_temperature()
            };
            envelope.h_tr_is * (t_surface - t_zone_c)
        }
        MassAirCouplingMode::ParallelResistance => {
            h_series(envelope.mass.wall.h_tr_ms, envelope.h_tr_is)
                * (envelope.mass.wall.temperature - t_zone_c)
                + h_series(envelope.mass.roof.h_tr_ms, envelope.h_tr_is)
                    * (envelope.mass.roof.temperature - t_zone_c)
                + h_series(envelope.mass.floor.h_tr_ms, envelope.h_tr_is)
                    * (envelope.mass.floor.temperature - t_zone_c)
        }
    }
}

fn validate_forcing(forcing: &CoupledStepForcing) -> Result<(), AirsideCouplingError> {
    forcing.outdoor_air.validate_derived()?;
    for (field, value) in [
        ("t_ext_wall", forcing.exterior_temperatures.t_ext_wall),
        ("t_ext_roof", forcing.exterior_temperatures.t_ext_roof),
        ("t_ext_floor", forcing.exterior_temperatures.t_ext_floor),
        ("convective_gain_w", forcing.convective_gain_w),
    ] {
        validate_finite(field, value)?;
    }
    validate_nonnegative(
        "ventilation_conductance_w_per_k",
        forcing.ventilation_conductance_w_per_k,
    )?;
    for value in forcing.envelope_gains_w {
        validate_finite("envelope_gain_w", value)?;
    }
    Ok(())
}

fn validate_envelope(envelope: &MultiNodeSolver) -> Result<(), AirsideCouplingError> {
    let nodes = [
        &envelope.mass.wall,
        &envelope.mass.roof,
        &envelope.mass.floor,
        &envelope.mass.internal,
    ];
    let valid = envelope.h_tr_is.is_finite()
        && envelope.h_tr_is > 0.0
        && envelope.zone_temperature.is_finite()
        && envelope.surface_temperature.is_finite()
        && nodes.iter().all(|node| {
            node.temperature.is_finite()
                && node.capacitance.is_finite()
                && node.capacitance > 0.0
                && node.h_tr_ms.is_finite()
                && node.h_tr_ms >= 0.0
                && node.h_tr_em.is_finite()
                && node.h_tr_em >= 0.0
                && node.h_tr_me.is_finite()
                && node.h_tr_me >= 0.0
        })
        && envelope.mass.wall.h_tr_ms > 0.0
        && envelope.mass.roof.h_tr_ms > 0.0
        && envelope.mass.floor.h_tr_ms > 0.0;
    if valid {
        Ok(())
    } else {
        Err(AirsideCouplingError::InvalidEnvelopeState)
    }
}
