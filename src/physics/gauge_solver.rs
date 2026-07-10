use crate::physics::cta::VectorField;
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::units::{FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_spec::WallSpec;

/// Per-zone gauge-connection bookkeeping for shadow mode.
///
/// This is a private shadow-mode buffer held inside [`GaugeSolver`]. It is
/// intentionally **not** aliased with the public `ThermalManifold` from
/// `geometry_tensor` (Phase 1a, #1461): that structure is the real continuous
/// Riemannian manifold carrying `metric_tensor` + `scalar_field` + a 4-D
/// [`nalgebra::Vector4`] `gauge_connection`. Shadow mode needs only an
/// ordered pair `[solar_irradiance, outside_air_temp]` per zone, so we keep a
/// minimal per-zone `Vec<VectorField>` here and resolve the name collision by
/// keeping both as private module items.
const GAUGE_CONNECTION_COMPONENTS: usize = 2;
const GAUGE_CONNECTION_SOLAR_INDEX: usize = 0;
const GAUGE_CONNECTION_OUTDOOR_TEMP_INDEX: usize = 1;

#[derive(Debug, Clone)]
struct ThermalManifold {
    gauge_connection: Vec<VectorField>,
}

impl ThermalManifold {
    fn new(num_zones: usize) -> Self {
        assert!(
            num_zones <= crate::physics::geometry_tensor::MAX_ZONES,
            "ThermalManifold zone count exceeds MAX_ZONES"
        );
        let gauge_connection = (0..num_zones)
            .map(|_| VectorField::from_scalar(0.0, GAUGE_CONNECTION_COMPONENTS))
            .collect();

        Self { gauge_connection }
    }

    fn num_zones(&self) -> usize {
        self.gauge_connection.len()
    }

    fn set_gauge_connection(
        &mut self,
        zone_index: usize,
        solar_irradiance_wm2: f64,
        outside_air_temp_c: f64,
    ) -> Result<(), String> {
        if zone_index >= self.gauge_connection.len() {
            return Err(format!(
                "zone_index {} out of bounds for {} zones",
                zone_index,
                self.gauge_connection.len()
            ));
        }

        self.gauge_connection[zone_index] =
            VectorField::new(vec![solar_irradiance_wm2, outside_air_temp_c]);
        Ok(())
    }

    #[cfg(test)]
    fn gauge_connection(&self, zone_index: usize) -> Option<&VectorField> {
        self.gauge_connection.get(zone_index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaugeBoundaryConditions {
    pub solar_irradiance_wm2: f64,
    pub outside_air_temp_c: f64,
}

impl GaugeBoundaryConditions {
    pub fn new(solar_irradiance_wm2: f64, outside_air_temp_c: f64) -> Self {
        Self {
            solar_irradiance_wm2,
            outside_air_temp_c,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GaugeSolver {
    manifold: ThermalManifold,
    zone_index: usize,
    r_total: f64,
    q_flux: f64,
    energy_storage_rate: f64,
    initialized: bool,
}

impl GaugeSolver {
    fn new(manifold: ThermalManifold) -> Self {
        Self {
            manifold,
            zone_index: 0,
            r_total: 0.0,
            q_flux: 0.0,
            energy_storage_rate: 0.0,
            initialized: false,
        }
    }

    pub fn with_zone_index(mut self, zone_index: usize) -> Self {
        self.zone_index = zone_index;
        self
    }

    #[cfg(test)]
    fn manifold(&self) -> &ThermalManifold {
        &self.manifold
    }

    pub fn translate_boundary_conditions(boundary: GaugeBoundaryConditions) -> VectorField {
        VectorField::new(vec![
            boundary.solar_irradiance_wm2,
            boundary.outside_air_temp_c,
        ])
    }

    pub fn effective_exterior_temperature(
        gauge_connection: &VectorField,
        h_exterior: f64,
    ) -> Result<f64, SolverError> {
        if gauge_connection.len() != GAUGE_CONNECTION_COMPONENTS {
            return Err(SolverError::InvalidConfig(format!(
                "gauge_connection has {} components, expected {}",
                gauge_connection.len(),
                GAUGE_CONNECTION_COMPONENTS
            )));
        }

        let solar_irradiance_wm2 = gauge_connection[GAUGE_CONNECTION_SOLAR_INDEX];
        let outside_air_temp_c = gauge_connection[GAUGE_CONNECTION_OUTDOOR_TEMP_INDEX];
        if solar_irradiance_wm2 != 0.0 && (h_exterior <= 0.0 || !h_exterior.is_finite()) {
            return Err(SolverError::InvalidConfig(
                "h_exterior must be positive and finite when solar forcing is present".to_string(),
            ));
        }

        if solar_irradiance_wm2 == 0.0 {
            Ok(outside_air_temp_c)
        } else {
            Ok(outside_air_temp_c + solar_irradiance_wm2 / h_exterior)
        }
    }

    pub fn step_with_boundary_conditions(
        &mut self,
        _timestep: Time,
        T_interior: Temperature,
        h_exterior: HeatTransferCoefficient,
        boundary: GaugeBoundaryConditions,
    ) -> Result<HeatFlux, SolverError> {
        if !self.initialized {
            return Err(SolverError::InvalidConfig(
                "Solver not initialized. Call initialize() first.".to_string(),
            ));
        }
        if self.r_total <= 0.0 || !self.r_total.is_finite() {
            return Err(SolverError::ConstructionError(
                "Invalid wall resistance (must be positive and finite)".to_string(),
            ));
        }

        let gauge_connection = Self::translate_boundary_conditions(boundary);
        self.manifold
            .set_gauge_connection(
                self.zone_index,
                gauge_connection[GAUGE_CONNECTION_SOLAR_INDEX],
                gauge_connection[GAUGE_CONNECTION_OUTDOOR_TEMP_INDEX],
            )
            .map_err(SolverError::InvalidConfig)?;

        let t_ext = Self::effective_exterior_temperature(&gauge_connection, h_exterior.to_value())?;
        self.q_flux = (t_ext - T_interior.to_value()) / self.r_total;
        self.energy_storage_rate = 0.0;
        Ok(HeatFlux::from_value(self.q_flux))
    }
}

impl Default for GaugeSolver {
    fn default() -> Self {
        Self::new(ThermalManifold::new(1))
    }
}

impl HeatConductionSolver for GaugeSolver {
    fn name(&self) -> &str {
        "Gauge"
    }

    fn initialize(&mut self, wall: &WallSpec) -> Result<(), SolverError> {
        self.r_total = wall.total_r_value();
        if self.r_total <= 0.0 || !self.r_total.is_finite() {
            return Err(SolverError::ConstructionError(
                "Invalid wall resistance (must be positive and finite)".to_string(),
            ));
        }
        if self.zone_index >= self.manifold.num_zones() {
            return Err(SolverError::InvalidConfig(format!(
                "zone_index {} out of bounds for {} zones",
                self.zone_index,
                self.manifold.num_zones()
            )));
        }

        self.q_flux = 0.0;
        self.energy_storage_rate = 0.0;
        self.initialized = true;
        Ok(())
    }

    fn step(
        &mut self,
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        _h_interior: HeatTransferCoefficient,
        h_exterior: HeatTransferCoefficient,
    ) -> Result<HeatFlux, SolverError> {
        self.step_with_boundary_conditions(
            timestep,
            T_interior,
            h_exterior,
            GaugeBoundaryConditions::new(0.0, T_exterior.to_value()),
        )
    }

    fn energy_storage_rate(&self) -> f64 {
        self.energy_storage_rate
    }

    fn steady_state_flux(
        &self,
        T_interior: Temperature,
        T_exterior: Temperature,
    ) -> Result<HeatFlux, SolverError> {
        if !self.initialized {
            return Err(SolverError::InvalidConfig(
                "Solver not initialized. Call initialize() first.".to_string(),
            ));
        }
        if self.r_total <= 0.0 || !self.r_total.is_finite() {
            return Err(SolverError::ConstructionError(
                "Invalid wall resistance (must be positive and finite)".to_string(),
            ));
        }

        Ok(HeatFlux::from_value(
            (T_exterior.to_value() - T_interior.to_value()) / self.r_total,
        ))
    }

    fn is_valid(&self) -> bool {
        self.initialized && self.r_total > 0.0 && self.r_total.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::wall_spec::WallSpec;

    #[test]
    fn test_gauge_solver_implements_heat_conduction_solver() {
        let wall = WallSpec::single_layer("Test", 0.2, 1.0, 1000.0, 1000.0);
        let mut solver = GaugeSolver::default();
        solver.initialize(&wall).unwrap();

        let flux = solver
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();

        assert_eq!(solver.name(), "Gauge");
        assert!(solver.is_valid());
        assert_eq!(flux.to_value(), -75.0);
    }

    #[test]
    fn test_boundary_translation_preserves_raw_values() {
        let connection = GaugeSolver::translate_boundary_conditions(GaugeBoundaryConditions::new(
            250_000.0, -80.0,
        ));

        assert_eq!(connection.as_slice(), &[250_000.0, -80.0]);
        assert_eq!(
            GaugeSolver::effective_exterior_temperature(&connection, 25.0).unwrap(),
            9920.0
        );
    }

    #[test]
    fn test_gauge_solver_uses_solar_boundary_without_clamping() {
        let wall = WallSpec::single_layer("Test", 0.2, 1.0, 1000.0, 1000.0);
        let mut solver = GaugeSolver::default();
        solver.initialize(&wall).unwrap();

        let flux = solver
            .step_with_boundary_conditions(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                HeatTransferCoefficient::from_value(25.0),
                GaugeBoundaryConditions::new(800.0, 20.0),
            )
            .unwrap();

        assert_eq!(flux.to_value(), 160.0);
        assert_eq!(
            solver.manifold().gauge_connection(0).unwrap().as_slice(),
            &[800.0, 20.0]
        );
    }
}
