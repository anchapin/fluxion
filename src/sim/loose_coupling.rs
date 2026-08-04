//! Loose (Quasi-Dynamic) Coupling for BES-FFD Co-Simulation
//!
//! This module implements loose coupling between the Building Energy Simulation (BES)
//! engine and the Fast Fluid Dynamics (FFD) solver.
//!
//! ## Coupling Strategy
//!
//! - **BES Time-Step (Macro):** Typically 15-60 minutes for whole-building energy simulation
//! - **FFD Time-Step (Micro):** Typically seconds to capture transient airflow events
//!
//! ```text
//! Macro Step N: [t₀ → t₁]
//!   ├── BES: calculates zone loads for [t₀ → t₁]
//!   ├── FFD: receives BCs at t₀, runs micro-steps internally to reach t₁
//!   └── Data Exchange: averaged results returned to BES at t₁
//! ```
//!
//! ## Key Design Decisions
//!
//! - **Quasi-dynamic (loose) coupling**: No iterations within macro step
//! - **FFD runs autonomously** between exchange points
//! - **Results are time-averaged** over the macro step before exchange
//! - **Adaptive macro step sizes** are supported
//!
//! ## References
//!
//! - Zuo et al. (2016) on BES-CFD coupling strategies
//! - Clarke & Hensen (2017) on co-simulation synchronization

use thiserror::Error;

/// Errors that can occur during loose coupling operations.
#[derive(Debug, Clone, Error)]
pub enum LooseCouplingError {
    #[error("FFD solver error: {0}")]
    FfdSolver(String),

    #[error("Invalid timestep configuration: {0}")]
    InvalidTimestep(String),

    #[error("Boundary condition error: {0}")]
    BoundaryCondition(String),

    #[error("Averaging error: {0}")]
    Averaging(String),
}

/// Result type for loose coupling operations.
pub type LooseCouplingResult<T> = Result<T, LooseCouplingError>;

/// Boundary conditions passed from BES to FFD at the start of a macro timestep.
///
/// All temperatures are in Kelvin to match standard engineering practice for fluid dynamics.
#[derive(Debug, Clone, Default)]
pub struct BesToFfdBoundaryConditions {
    /// Outdoor dry-bulb temperature [K].
    pub outdoor_temperature: f64,
    /// Surface temperatures for each zone surface [K].
    /// Order: walls, roof, floor (matching zone surface indexing).
    pub surface_temperatures: Vec<f64>,
    /// HVAC supply air temperature [K].
    pub hvac_supply_temperature: f64,
    /// HVAC supply air flow rate [m³/s].
    pub hvac_supply_flow: f64,
    /// Wind pressure on each facade [Pa].
    /// Positive = pushing into building, negative = pulling out.
    pub wind_pressure: Vec<f64>,
    /// Internal gains from occupants, equipment, lighting [W].
    pub internal_gains: f64,
    /// Simulation time at start of macro step [s].
    pub time_start: f64,
    /// Duration of macro step [s].
    pub macro_timestep: f64,
}

/// Results returned from FFD to BES at the end of a macro timestep.
///
/// All temperatures are in Kelvin. Flow rates are in m³/s.
/// Heat fluxes are in W/m².
#[derive(Debug, Clone, Default)]
pub struct FfdToBesResults {
    /// Convective heat transfer coefficients for each surface [W/m²K].
    pub chtc: Vec<f64>,
    /// Zone air temperatures (can be stratified) [K].
    pub zone_temperatures: Vec<f64>,
    /// Surface heat fluxes (averaged over macro step) [W/m²].
    pub surface_heat_flux: Vec<f64>,
    /// Infiltration flow rates [m³/s].
    pub infiltration_flow: Vec<f64>,
    /// Zone mixing flow rates [m³/s].
    pub mixing_flow: Vec<f64>,
    /// Number of micro steps FFD took during this macro step.
    pub micro_step_count: usize,
    /// Actual simulation time covered [s].
    pub simulation_time_covered: f64,
}

impl FfdToBesResults {
    /// Returns true if no results have been computed.
    pub fn is_empty(&self) -> bool {
        self.chtc.is_empty()
            && self.zone_temperatures.is_empty()
            && self.surface_heat_flux.is_empty()
    }
}

/// Trait for FFD solvers that can be used with loose coupling.
///
/// This trait defines the interface for Fast Fluid Dynamics solvers that
/// can be coupled to the BES engine using loose (quasi-dynamic) coupling.
pub trait FfdSolver: Send + Sync {
    /// Get the solver name.
    fn name(&self) -> &str;

    /// Initialize the FFD solver with zone geometry.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones
    /// * `zone_volumes` - Volume of each zone [m³]
    /// * `surface_areas` - Surface areas for each zone [m²]
    /// * `num_surfaces` - Total number of surfaces across all zones
    fn initialize(
        &mut self,
        num_zones: usize,
        zone_volumes: &[f64],
        surface_areas: &[f64],
        num_surfaces: usize,
    ) -> LooseCouplingResult<()>;

    /// Set boundary conditions and advance the FFD solver by one micro timestep.
    ///
    /// This method runs a single micro timestep (typically 0.1-1.0 seconds)
    /// and returns the instantaneous results.
    ///
    /// # Arguments
    /// * `bc` - Boundary conditions from BES
    /// * `dt` - Micro timestep duration [s]
    ///
    /// # Returns
    /// Instantaneous FFD results at the end of the micro timestep.
    fn step_micro(
        &mut self,
        bc: &BesToFfdBoundaryConditions,
        dt: f64,
    ) -> LooseCouplingResult<FfdMicroResults>;

    /// Get the recommended micro timestep for this solver.
    ///
    /// Returns the nominal micro timestep that the FFD solver uses internally.
    fn recommended_micro_timestep(&self) -> f64;

    /// Check if the solver is valid and ready for simulation.
    fn is_valid(&self) -> bool;
}

/// Instantaneous results from a single FFD micro step.
#[derive(Debug, Clone, Default)]
pub struct FfdMicroResults {
    /// Surface convective heat transfer coefficients [W/m²K].
    pub chtc: Vec<f64>,
    /// Zone air temperatures [K].
    pub zone_temperatures: Vec<f64>,
    /// Surface heat fluxes [W/m²].
    pub surface_heat_flux: Vec<f64>,
    /// Infiltration flow rates [m³/s].
    pub infiltration_flow: Vec<f64>,
    /// Zone mixing flow rates [m³/s].
    pub mixing_flow: Vec<f64>,
}

/// Time-averaged accumulator for FFD micro step results.
///
/// This struct accumulates results over multiple micro steps and computes
/// time-averaged values at the end of a macro timestep.
#[derive(Debug, Clone, Default)]
pub struct FfdAccumulator {
    /// Running sum of CHTC × dt products.
    chtc_weighted_sum: Vec<f64>,
    /// Running sum of temperatures × dt products.
    temperature_weighted_sum: Vec<f64>,
    /// Running sum of heat fluxes × dt products.
    flux_weighted_sum: Vec<f64>,
    /// Running sum of infiltration flow × dt.
    infiltration_weighted_sum: Vec<f64>,
    /// Running sum of mixing flow × dt.
    mixing_weighted_sum: Vec<f64>,
    /// Total accumulated time.
    total_time: f64,
    /// Number of micro steps accumulated.
    step_count: usize,
    /// Number of surfaces.
    num_surfaces: usize,
    /// Number of zones.
    num_zones: usize,
}

impl FfdAccumulator {
    /// Create a new accumulator with the specified number of surfaces and zones.
    pub fn new(num_surfaces: usize, num_zones: usize) -> Self {
        Self {
            chtc_weighted_sum: vec![0.0; num_surfaces],
            temperature_weighted_sum: vec![0.0; num_zones],
            flux_weighted_sum: vec![0.0; num_surfaces],
            infiltration_weighted_sum: vec![0.0; num_zones],
            mixing_weighted_sum: vec![0.0; num_zones],
            total_time: 0.0,
            step_count: 0,
            num_surfaces,
            num_zones,
        }
    }

    /// Accumulate results from a single micro step.
    ///
    /// # Arguments
    /// * `results` - Instantaneous results from FFD micro step
    /// * `dt` - Duration of the micro step [s]
    pub fn accumulate(&mut self, results: &FfdMicroResults, dt: f64) -> LooseCouplingResult<()> {
        if results.chtc.len() != self.num_surfaces {
            return Err(LooseCouplingError::Averaging(format!(
                "CHTC length mismatch: expected {}, got {}",
                self.num_surfaces,
                results.chtc.len()
            )));
        }
        if results.zone_temperatures.len() != self.num_zones {
            return Err(LooseCouplingError::Averaging(format!(
                "Temperature length mismatch: expected {}, got {}",
                self.num_zones,
                results.zone_temperatures.len()
            )));
        }

        for (i, &val) in results.chtc.iter().enumerate() {
            self.chtc_weighted_sum[i] += val * dt;
        }
        for (i, &val) in results.zone_temperatures.iter().enumerate() {
            self.temperature_weighted_sum[i] += val * dt;
        }
        for (i, &val) in results.surface_heat_flux.iter().enumerate() {
            self.flux_weighted_sum[i] += val * dt;
        }
        for (i, &val) in results.infiltration_flow.iter().enumerate() {
            self.infiltration_weighted_sum[i] += val * dt;
        }
        for (i, &val) in results.mixing_flow.iter().enumerate() {
            self.mixing_weighted_sum[i] += val * dt;
        }

        self.total_time += dt;
        self.step_count += 1;
        Ok(())
    }

    /// Compute the time-averaged results.
    ///
    /// Returns `None` if no data has been accumulated.
    pub fn compute_averages(&self) -> Option<FfdToBesResults> {
        if self.step_count == 0 || self.total_time <= 0.0 {
            return None;
        }

        let inv_total = 1.0 / self.total_time;
        Some(FfdToBesResults {
            chtc: self
                .chtc_weighted_sum
                .iter()
                .map(|&v| v * inv_total)
                .collect(),
            zone_temperatures: self
                .temperature_weighted_sum
                .iter()
                .map(|&v| v * inv_total)
                .collect(),
            surface_heat_flux: self
                .flux_weighted_sum
                .iter()
                .map(|&v| v * inv_total)
                .collect(),
            infiltration_flow: self
                .infiltration_weighted_sum
                .iter()
                .map(|&v| v * inv_total)
                .collect(),
            mixing_flow: self
                .mixing_weighted_sum
                .iter()
                .map(|&v| v * inv_total)
                .collect(),
            micro_step_count: self.step_count,
            simulation_time_covered: self.total_time,
        })
    }

    /// Reset the accumulator for a new macro timestep.
    pub fn reset(&mut self) {
        for v in &mut self.chtc_weighted_sum {
            *v = 0.0;
        }
        for v in &mut self.temperature_weighted_sum {
            *v = 0.0;
        }
        for v in &mut self.flux_weighted_sum {
            *v = 0.0;
        }
        for v in &mut self.infiltration_weighted_sum {
            *v = 0.0;
        }
        for v in &mut self.mixing_weighted_sum {
            *v = 0.0;
        }
        self.total_time = 0.0;
        self.step_count = 0;
    }
}

/// Loose coupling coordinator for BES-FFD co-simulation.
///
/// This struct manages the data exchange between the BES and FFD engines,
/// handling:
/// - Boundary condition transfer at the start of each macro timestep
/// - FFD micro-step execution
/// - Time-averaging of results
/// - Dynamic timestep ratio handling
pub struct LooseCoupling {
    /// The FFD solver.
    ffd_solver: Box<dyn FfdSolver>,
    /// Accumulator for time-averaged results.
    accumulator: FfdAccumulator,
    /// Current macro timestep duration [s].
    macro_timestep: f64,
    /// Current simulation time [s].
    current_time: f64,
    #[allow(dead_code)]
    num_zones: usize,
    #[allow(dead_code)]
    num_surfaces: usize,
    /// Last boundary conditions received from BES.
    last_bc: Option<BesToFfdBoundaryConditions>,
}

impl LooseCoupling {
    /// Create a new loose coupling coordinator.
    ///
    /// # Arguments
    /// * `ffd_solver` - The FFD solver to use
    /// * `num_zones` - Number of thermal zones
    /// * `num_surfaces` - Total number of surfaces
    /// * `macro_timestep` - BES macro timestep duration [s]
    pub fn new(
        ffd_solver: Box<dyn FfdSolver>,
        num_zones: usize,
        num_surfaces: usize,
        macro_timestep: f64,
    ) -> LooseCouplingResult<Self> {
        if macro_timestep <= 0.0 {
            return Err(LooseCouplingError::InvalidTimestep(
                "Macro timestep must be positive".to_string(),
            ));
        }
        if num_zones == 0 {
            return Err(LooseCouplingError::InvalidTimestep(
                "Number of zones must be positive".to_string(),
            ));
        }

        Ok(Self {
            ffd_solver,
            accumulator: FfdAccumulator::new(num_surfaces, num_zones),
            macro_timestep,
            current_time: 0.0,
            num_zones,
            num_surfaces,
            last_bc: None,
        })
    }

    /// Set boundary conditions from BES and run FFD micro steps.
    ///
    /// This is called at the start of each BES macro timestep. The FFD solver
    /// runs micro steps internally until it catches up to the end of the
    /// macro timestep, accumulating time-averaged results.
    ///
    /// # Arguments
    /// * `bc` - Boundary conditions from BES
    ///
    /// # Returns
    /// Time-averaged FFD results for use by BES at the end of the macro step.
    pub fn exchange_and_step(
        &mut self,
        bc: BesToFfdBoundaryConditions,
    ) -> LooseCouplingResult<FfdToBesResults> {
        self.last_bc = Some(bc.clone());

        let micro_dt = self.ffd_solver.recommended_micro_timestep();
        let mut time_elapsed = 0.0;

        self.accumulator.reset();

        while time_elapsed < self.macro_timestep {
            let remaining = self.macro_timestep - time_elapsed;
            let step_dt = if micro_dt > remaining {
                remaining
            } else {
                micro_dt
            };

            let micro_results = self.ffd_solver.step_micro(&bc, step_dt)?;
            self.accumulator.accumulate(&micro_results, step_dt)?;

            time_elapsed += step_dt;
        }

        let results = self
            .accumulator
            .compute_averages()
            .ok_or_else(|| LooseCouplingError::Averaging("No results accumulated".to_string()))?;

        self.current_time += self.macro_timestep;

        Ok(results)
    }

    /// Update the macro timestep duration.
    ///
    /// This allows for adaptive macro step sizes during simulation.
    pub fn set_macro_timestep(&mut self, macro_timestep: f64) -> LooseCouplingResult<()> {
        if macro_timestep <= 0.0 {
            return Err(LooseCouplingError::InvalidTimestep(
                "Macro timestep must be positive".to_string(),
            ));
        }
        self.macro_timestep = macro_timestep;
        Ok(())
    }

    /// Get the current simulation time.
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Get the current macro timestep.
    pub fn macro_timestep(&self) -> f64 {
        self.macro_timestep
    }

    /// Get the FFD-to-BES timestep ratio.
    ///
    /// Returns the number of micro steps per macro step.
    pub fn timestep_ratio(&self) -> f64 {
        if self.ffd_solver.recommended_micro_timestep() > 0.0 {
            self.macro_timestep / self.ffd_solver.recommended_micro_timestep()
        } else {
            0.0
        }
    }

    /// Get the last boundary conditions received from BES.
    pub fn last_boundary_conditions(&self) -> Option<&BesToFfdBoundaryConditions> {
        self.last_bc.as_ref()
    }

    /// Check if the underlying FFD solver is valid.
    pub fn is_valid(&self) -> bool {
        self.ffd_solver.is_valid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockFfdSolver {
        num_zones: usize,
        num_surfaces: usize,
        micro_timestep: f64,
        valid: bool,
    }

    impl MockFfdSolver {
        fn new(num_zones: usize, num_surfaces: usize) -> Self {
            Self {
                num_zones,
                num_surfaces,
                micro_timestep: 1.0,
                valid: true,
            }
        }
    }

    impl FfdSolver for MockFfdSolver {
        fn name(&self) -> &str {
            "MockFfdSolver"
        }

        fn initialize(
            &mut self,
            num_zones: usize,
            _zone_volumes: &[f64],
            _surface_areas: &[f64],
            num_surfaces: usize,
        ) -> LooseCouplingResult<()> {
            self.num_zones = num_zones;
            self.num_surfaces = num_surfaces;
            Ok(())
        }

        fn step_micro(
            &mut self,
            bc: &BesToFfdBoundaryConditions,
            dt: f64,
        ) -> LooseCouplingResult<FfdMicroResults> {
            let _ = (bc, dt);
            Ok(FfdMicroResults {
                chtc: vec![10.0; self.num_surfaces],
                zone_temperatures: vec![293.15; self.num_zones],
                surface_heat_flux: vec![50.0; self.num_surfaces],
                infiltration_flow: vec![0.1; self.num_zones],
                mixing_flow: vec![0.05; self.num_zones],
            })
        }

        fn recommended_micro_timestep(&self) -> f64 {
            self.micro_timestep
        }

        fn is_valid(&self) -> bool {
            self.valid
        }
    }

    #[test]
    fn test_loose_coupling_creation() {
        let ffd = MockFfdSolver::new(2, 8);
        let coupling = LooseCoupling::new(Box::new(ffd), 2, 8, 3600.0);
        assert!(coupling.is_ok());
        let coupling = coupling.unwrap();
        assert_eq!(coupling.macro_timestep(), 3600.0);
        assert_eq!(coupling.timestep_ratio(), 3600.0);
        assert!(coupling.is_valid());
    }

    #[test]
    fn test_loose_coupling_invalid_timestep() {
        let ffd = MockFfdSolver::new(1, 4);
        let result = LooseCoupling::new(Box::new(ffd), 1, 4, 0.0);
        assert!(result.is_err());
        if let Err(LooseCouplingError::InvalidTimestep(_)) = result {
            // Expected error type
        } else {
            panic!("Expected InvalidTimestep error");
        }
    }

    #[test]
    fn test_loose_coupling_exchange_and_step() {
        let ffd = MockFfdSolver::new(2, 8);
        let mut coupling = LooseCoupling::new(Box::new(ffd), 2, 8, 10.0).unwrap();

        let bc = BesToFfdBoundaryConditions {
            outdoor_temperature: 280.0,
            surface_temperatures: vec![295.15; 8],
            hvac_supply_temperature: 293.15,
            hvac_supply_flow: 0.5,
            wind_pressure: vec![0.0; 4],
            internal_gains: 500.0,
            time_start: 0.0,
            macro_timestep: 10.0,
        };

        let results = coupling.exchange_and_step(bc);
        assert!(results.is_ok());
        let results = results.unwrap();

        assert_eq!(results.chtc.len(), 8);
        assert_eq!(results.zone_temperatures.len(), 2);
        assert_eq!(results.surface_heat_flux.len(), 8);
        assert!(results.micro_step_count > 0);
        assert!((results.simulation_time_covered - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_loose_coupling_timestep_ratio() {
        let ffd = MockFfdSolver::new(1, 4);
        let coupling = LooseCoupling::new(Box::new(ffd), 1, 4, 3600.0).unwrap();
        assert!((coupling.timestep_ratio() - 3600.0).abs() < 1e-9);
    }

    #[test]
    fn test_accumulator_creation() {
        let acc = FfdAccumulator::new(8, 2);
        assert!(acc.compute_averages().is_none());
    }

    #[test]
    fn test_accumulator_single_step() {
        let mut acc = FfdAccumulator::new(2, 1);
        let results = FfdMicroResults {
            chtc: vec![10.0, 20.0],
            zone_temperatures: vec![293.15],
            surface_heat_flux: vec![50.0, 60.0],
            infiltration_flow: vec![0.1],
            mixing_flow: vec![0.05],
        };

        acc.accumulate(&results, 1.0).unwrap();
        let averages = acc.compute_averages().unwrap();

        assert_eq!(averages.chtc, vec![10.0, 20.0]);
        assert_eq!(averages.zone_temperatures, vec![293.15]);
        assert_eq!(averages.surface_heat_flux, vec![50.0, 60.0]);
        assert_eq!(averages.micro_step_count, 1);
        assert!((averages.simulation_time_covered - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_accumulator_time_averaging() {
        let mut acc = FfdAccumulator::new(1, 1);

        let results1 = FfdMicroResults {
            chtc: vec![10.0],
            zone_temperatures: vec![290.0],
            surface_heat_flux: vec![100.0],
            infiltration_flow: vec![0.1],
            mixing_flow: vec![0.05],
        };

        let results2 = FfdMicroResults {
            chtc: vec![20.0],
            zone_temperatures: vec![300.0],
            surface_heat_flux: vec![200.0],
            infiltration_flow: vec![0.2],
            mixing_flow: vec![0.1],
        };

        acc.accumulate(&results1, 0.5).unwrap();
        acc.accumulate(&results2, 0.5).unwrap();

        let averages = acc.compute_averages().unwrap();

        assert!((averages.chtc[0] - 15.0).abs() < 1e-9);
        assert!((averages.zone_temperatures[0] - 295.0).abs() < 1e-9);
        assert!((averages.surface_heat_flux[0] - 150.0).abs() < 1e-9);
        assert_eq!(averages.micro_step_count, 2);
        assert!((averages.simulation_time_covered - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = FfdAccumulator::new(1, 1);
        let results = FfdMicroResults {
            chtc: vec![10.0],
            zone_temperatures: vec![293.15],
            surface_heat_flux: vec![50.0],
            infiltration_flow: vec![0.1],
            mixing_flow: vec![0.05],
        };

        acc.accumulate(&results, 1.0).unwrap();
        assert!(acc.compute_averages().is_some());

        acc.reset();
        assert!(acc.compute_averages().is_none());
    }

    #[test]
    fn test_ffd_to_bes_results_is_empty() {
        let empty = FfdToBesResults::default();
        assert!(empty.is_empty());

        let populated = FfdToBesResults {
            chtc: vec![10.0],
            zone_temperatures: vec![293.15],
            surface_heat_flux: vec![50.0],
            infiltration_flow: vec![0.1],
            mixing_flow: vec![0.05],
            micro_step_count: 1,
            simulation_time_covered: 1.0,
        };
        assert!(!populated.is_empty());
    }

    #[test]
    fn test_boundary_conditions_default() {
        let bc = BesToFfdBoundaryConditions::default();
        assert_eq!(bc.outdoor_temperature, 0.0);
        assert!(bc.surface_temperatures.is_empty());
        assert_eq!(bc.hvac_supply_temperature, 0.0);
        assert_eq!(bc.hvac_supply_flow, 0.0);
        assert!(bc.wind_pressure.is_empty());
        assert_eq!(bc.internal_gains, 0.0);
        assert_eq!(bc.time_start, 0.0);
        assert_eq!(bc.macro_timestep, 0.0);
    }
}
