use crate::physics::five_r1c_solver::FiveR1CSolver;
use crate::physics::gauge_solver::{GaugeBoundaryConditions, GaugeSolver};
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::units::{HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_spec::WallSpec;
pub use crate::thermal::thermal_model::ThermalModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrimarySolver {
    #[default]
    Baseline,
    Gauge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhysicsAdapterConfig {
    pub primary_solver: PrimarySolver,
    /// When true, gauge solver runs in shadow mode (records but doesn't affect output)
    pub gauge_shadow_mode: bool,
}

impl PhysicsAdapterConfig {
    pub fn baseline_only() -> Self {
        Self {
            primary_solver: PrimarySolver::Baseline,
            gauge_shadow_mode: false,
        }
    }

    pub fn gauge_primary() -> Self {
        Self {
            primary_solver: PrimarySolver::Gauge,
            gauge_shadow_mode: false,
        }
    }

    /// Deprecated: gauge shadow mode. Use `baseline_only()` or `gauge_primary()`.
    #[deprecated(since = "0.2.0", note = "Use baseline_only() or gauge_primary()")]
    pub fn gauge_shadow() -> Self {
        Self {
            primary_solver: PrimarySolver::Baseline,
            gauge_shadow_mode: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaugeShadowRecord {
    pub baseline_flux_wm2: f64,
    pub gauge_flux_wm2: Option<f64>,
    pub delta_wm2: Option<f64>,
    pub gauge_connection: Vec<f64>,
    pub error: Option<String>,
}

pub struct PhysicsAdapter {
    baseline_solver: Box<dyn HeatConductionSolver>,
    gauge_solver: Option<GaugeSolver>,
    shadow_records: Vec<GaugeShadowRecord>,
    config: PhysicsAdapterConfig,
}

impl PhysicsAdapter {
    pub fn new(config: PhysicsAdapterConfig) -> Self {
        Self::with_baseline(Box::new(FiveR1CSolver::new()), config)
    }

    pub fn with_baseline(
        baseline_solver: Box<dyn HeatConductionSolver>,
        config: PhysicsAdapterConfig,
    ) -> Self {
        let gauge_solver =
            if config.gauge_shadow_mode || config.primary_solver == PrimarySolver::Gauge {
                Some(GaugeSolver::default())
            } else {
                None
            };

        Self {
            baseline_solver,
            gauge_solver,
            shadow_records: Vec::new(),
            config,
        }
    }

    /// Returns which solver is currently acting as primary.
    pub fn primary_solver(&self) -> PrimarySolver {
        self.config.primary_solver
    }

    pub fn initialize(&mut self, wall: &WallSpec) -> Result<(), SolverError> {
        self.baseline_solver.initialize(wall)?;
        if let Some(gauge_solver) = &mut self.gauge_solver {
            gauge_solver.initialize(wall)?;
        }
        Ok(())
    }

    pub fn step(
        &mut self,
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        h_interior: HeatTransferCoefficient,
        h_exterior: HeatTransferCoefficient,
        solar_irradiance_wm2: f64,
    ) -> Result<HeatFlux, SolverError> {
        let baseline_flux = self
            .baseline_solver
            .step(timestep, T_interior, T_exterior, h_interior, h_exterior)?;

        let boundary = GaugeBoundaryConditions::new(solar_irradiance_wm2, T_exterior.to_value());
        let mut gauge_flux_result: Result<HeatFlux, SolverError> = Err(SolverError::InvalidConfig(
            "Gauge solver not available".to_string(),
        ));

        if let Some(gauge_solver) = &mut self.gauge_solver {
            let gauge_connection = GaugeSolver::translate_boundary_conditions(boundary)
                .as_slice()
                .to_vec();
            gauge_flux_result = gauge_solver
                .step_with_boundary_conditions(timestep, T_interior, h_exterior, boundary);
            let baseline_flux_wm2 = baseline_flux.to_value();
            let record = match gauge_flux_result {
                Ok(gauge_flux) => {
                    let gauge_flux_wm2 = gauge_flux.to_value();
                    GaugeShadowRecord {
                        baseline_flux_wm2,
                        gauge_flux_wm2: Some(gauge_flux_wm2),
                        delta_wm2: Some(gauge_flux_wm2 - baseline_flux_wm2),
                        gauge_connection,
                        error: None,
                    }
                }
                Err(ref error) => GaugeShadowRecord {
                    baseline_flux_wm2,
                    gauge_flux_wm2: None,
                    delta_wm2: None,
                    gauge_connection,
                    error: Some(error.to_string()),
                },
            };
            self.shadow_records.push(record);
        }

        // Return gauge flux if it's the primary solver, otherwise baseline
        if self.primary_solver() == PrimarySolver::Gauge {
            gauge_flux_result
        } else {
            Ok(baseline_flux)
        }
    }

    pub fn shadow_records(&self) -> &[GaugeShadowRecord] {
        &self.shadow_records
    }

    pub fn last_shadow_record(&self) -> Option<&GaugeShadowRecord> {
        self.shadow_records.last()
    }

    pub fn gauge_shadow_enabled(&self) -> bool {
        self.gauge_solver.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::units::{FromF64, ToF64};

    fn test_wall() -> WallSpec {
        WallSpec::single_layer("Test", 0.2, 1.0, 1000.0, 1000.0)
    }

    // These tests deliberately exercise the deprecated `gauge_shadow` config.
    #[test]
    #[allow(deprecated)]
    fn test_shadow_mode_does_not_change_primary_flux() {
        let wall = test_wall();
        let mut direct = FiveR1CSolver::new();
        direct.initialize(&wall).unwrap();
        let direct_flux = direct
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(20.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();

        let mut adapter = PhysicsAdapter::new(PhysicsAdapterConfig::gauge_shadow());
        adapter.initialize(&wall).unwrap();
        let adapter_flux = adapter
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(20.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
                800.0,
            )
            .unwrap();

        assert_eq!(adapter_flux.to_value(), direct_flux.to_value());
        assert_eq!(adapter.shadow_records().len(), 1);
        let record = adapter.last_shadow_record().unwrap();
        assert_eq!(record.baseline_flux_wm2, direct_flux.to_value());
        assert_eq!(record.gauge_flux_wm2, Some(160.0));
        assert_eq!(record.delta_wm2, Some(160.0));
        assert_eq!(record.gauge_connection, vec![800.0, 20.0]);
    }

    #[test]
    fn test_shadow_mode_disabled_records_nothing() {
        let wall = test_wall();
        let mut adapter = PhysicsAdapter::new(PhysicsAdapterConfig::default());
        adapter.initialize(&wall).unwrap();

        let flux = adapter
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
                800.0,
            )
            .unwrap();

        assert_eq!(flux.to_value(), -75.0);
        assert!(!adapter.gauge_shadow_enabled());
        assert!(adapter.shadow_records().is_empty());
    }

    // Deliberately exercises the deprecated `gauge_shadow` config.
    #[test]
    #[allow(deprecated)]
    fn test_shadow_mode_gauge_error_is_nonfatal() {
        let wall = test_wall();
        let mut adapter = PhysicsAdapter::new(PhysicsAdapterConfig::gauge_shadow());
        adapter.initialize(&wall).unwrap();

        let flux = adapter
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(20.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(0.0),
                800.0,
            )
            .unwrap();

        assert_eq!(flux.to_value(), 0.0);
        let record = adapter.last_shadow_record().unwrap();
        assert_eq!(record.baseline_flux_wm2, 0.0);
        assert_eq!(record.gauge_flux_wm2, None);
        assert_eq!(record.delta_wm2, None);
        assert!(record.error.as_ref().unwrap().contains("h_exterior"));
    }

    /// A5.3: Integration test verifying that when primary_solver == Gauge,
    /// the step() method returns the gauge flux (not the baseline flux).
    /// This confirms the routing branch fires correctly in production mode.
    #[test]
    fn test_gauge_primary_returns_gauge_flux() {
        let wall = test_wall();

        // First, get the baseline flux for comparison
        let mut baseline = FiveR1CSolver::new();
        baseline.initialize(&wall).unwrap();
        let baseline_flux = baseline
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(20.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();

        // With gauge_primary(), the returned flux should be the gauge flux
        let mut adapter = PhysicsAdapter::new(PhysicsAdapterConfig::gauge_primary());
        adapter.initialize(&wall).unwrap();
        let gauge_flux = adapter
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(20.0), // T_exterior
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
                800.0,
            )
            .unwrap();

        // Gauge flux with solar boundary should differ from baseline
        // baseline = (20-20)/R = 0, gauge = (T_ext - T_int)/R where T_ext includes solar
        assert_eq!(gauge_flux.to_value(), 160.0); // Sol-air effect: 20 + 800/25 = 52°C ext
        assert_ne!(baseline_flux.to_value(), gauge_flux.to_value());

        // Shadow records should still be populated for observability
        assert_eq!(adapter.shadow_records().len(), 1);
        let record = adapter.last_shadow_record().unwrap();
        assert_eq!(record.baseline_flux_wm2, baseline_flux.to_value());
        assert_eq!(record.gauge_flux_wm2, Some(160.0));
        assert_eq!(record.delta_wm2, Some(160.0 - baseline_flux.to_value()));
    }
}
