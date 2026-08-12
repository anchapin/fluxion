//! Thermal model core module
//!
//! ISO 13790-compliant 5R1C/6R2C thermal network implementation.
//! Contains the core thermal model types, struct, and implementations.

use log::{debug, trace, warn};

use crate::physics::constants::thermal::ashrae_140::INTERIOR_FILM_COEFF;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::physics::ctf_coefficients::{CTFCalculator, CTFMaterial};
use crate::physics::ctf_solver::{CTFSolver, CTFSolverConfig};
use crate::sim::adaptive_timestep::TimestepMode;
use crate::sim::schedule::DailySchedule;
use crate::sim::thermal_model_core::{ThermalModel, ThermalModelType};

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Updates derived physical parameters based on geometry and constants.
    pub(crate) fn update_derived_parameters(&mut self) {
        // Geometry Calculations
        let width = self
            .0
            .zone_area
            .zip_with(&self.0.aspect_ratio, |a, ar| (a * ar).sqrt());
        let depth = self.0.zone_area.zip_with(&width, |a, w| a / w);
        let perimeter = (width.clone() + depth) * 2.0;
        let gross_wall_area = perimeter * self.0.ceiling_height.clone();

        let window_area = gross_wall_area.clone() * self.0.window_ratio.clone();
        // Opaque wall area: Gross - Window (used for h_tr_em calculations)
        // Note: Floor and roof are handled separately
        #[allow(unused_variables)]
        let opaque_wall_area = gross_wall_area.zip_with(&window_area, |g, w| g - w);

        let volume = self.0.zone_area.clone() * self.0.ceiling_height.clone();

        // Update Conductances
        // h_tr_w = U_win * Window Area
        self.0.h_tr_w = window_area * self.0.window_u_value;

        // NOTE: h_tr_em is NOT updated here - it is calculated in from_spec() using
        // physics-based k*A/d formula and should not be overwritten with U*A.
        // The k*A/d formula correctly models conductance from exterior to thermal mass node,
        // while U*A represents overall heat transfer (different physical meaning).
        // See from_spec() around line 1800 for the correct h_tr_em calculation.

        // h_tr_floor = U_floor * Floor Area
        // Issue #375: Use construction U-value from ThermalModel field
        self.0.h_tr_floor = self.0.zone_area.clone() * self.0.floor_u_value;

        // h_tr_is = Surface-to-air conductance for ASHRAE 140 simplified 5R1C model
        // Issue #714 Fix: Use H_SI = 3.45 W/m²K × floor_area (ASHRAE 140 simplified method)
        // instead of detailed surface-specific film coefficients
        const H_SI: f64 = 3.45; // W/m²K - ASHRAE 140 simplified 5R1C value
        self.0.h_tr_is = self.0.zone_area.clone() * H_SI;

        // Ventilation
        // h_ve = (infiltration_rate * volume * density * cp) / 3600
        // infiltration_rate is in ACH (1/hr)
        let air_cap = volume * self.0.air_density.clone() * self.0.heat_capacity.clone();
        self.0.h_ve = (air_cap.clone() * self.0.infiltration_rate.clone()) / 3600.0;

        // Issue #821: thermal_capacitance is set in `from_spec()` using actual construction
        // layers (Issue #585) and must NOT be overwritten here. The previous hardcoded
        // overwrite (200,000 J/m²K × zone_area) was a ~15× overestimate for low-mass
        // construction and biased peak air temperatures 10-20 °C low for FF cases.

        // Update optimization cache (computes derived_h_tr_3, derived_h_ext, etc.)
        self.update_optimization_cache();
        self.update_optimization_cache();
    }

    /// Pre-computes derived values used in the inner simulation loop to avoid redundant calculations.
    ///
    /// This should be called whenever physical parameters (conductances) are modified.
    pub fn update_optimization_cache(&mut self) {
        // Calculate the series conductance of h_tr_is and h_tr_ms
        // This represents the thermal resistance from interior air through interior surface to mass
        let _h_tr_is_ms_series = self.0.h_tr_is.zip_with(&self.0.h_tr_ms, |is_val, ms_val| {
            (is_val * ms_val) / (is_val + ms_val)
        });

        // h_ext = h_tr_w + h_ve + non-south opaque envelope
        //
        // Issue #917: derived_h_ext must NOT include h_tr_em_non_south.
        //
        // In the ISO 13790 5R1C network, the opaque envelope conductance (h_tr_em)
        // connects the MASS node to outdoor (used in the backward Euler / Crank-
        // Nicolson mass update). Adding it to h_ext (the AIR-to-outdoor path) double-
        // counts the opaque envelope, creating ~49 W/K of phantom air-to-outdoor
        // conductance that drains heat from the zone and suppresses free-floating
        // temperatures by ~2-5 °C.
        //
        // The correct air-to-outdoor conductance is:
        //   h_ext = h_tr_w (windows) + h_ve (ventilation/infiltration)
        //
        // The south-wall bypass (h_south_series) was already removed by Issue #715.
        // The dedicated south-wall vectors are kept for the 9R4C / CTF paths.
        self.0.derived_h_ext = self.0.h_tr_w.zip_with(&self.0.h_ve, |w, v| w + v);

        // term_rest_1 = h_tr_ms + h_tr_is + h_tr_me
        // Note: h_tr_me is 0 for 5R1C, non-zero for 6R2C (envelope↔internal mass coupling)
        self.0.derived_term_rest_1 = self
            .0
            .h_tr_ms
            .zip_with(&self.0.h_tr_is, |ms, is_val| ms + is_val)
            .zip_with(&self.0.h_tr_me, |ms_is, me| ms_is + me);

        // h_ms_is_prod = h_tr_ms * h_tr_is
        self.0.derived_h_ms_is_prod = self
            .0
            .h_tr_ms
            .zip_with(&self.0.h_tr_is, |ms, is_val| ms * is_val);

        // ground_coeff = term_rest_1 * h_tr_floor
        // Physics-based: No ground coupling multiplier
        // ground_coeff = term_rest_1 * h_tr_floor
        // Physics-based: No ground coupling multiplier
        self.0.derived_ground_coeff = self
            .0
            .derived_term_rest_1
            .zip_with(&self.0.h_tr_floor, |rest, floor| rest * floor);

        // For multi-zone buildings, include inter-zone conductance in sensitivity calculation
        // Issue #351: Update thermal network for inter-zone coupling
        // den = h_ms_is_prod + term_rest_1 * (h_ext + h_tr_floor + h_tr_iz)
        // Fix: Remove ground term from h_total to avoid over-damping
        let h_total = if self.0.num_zones > 1 {
            // Include both conductive and radiative inter-zone conductance
            self.0.derived_h_ext.clone() + self.0.h_tr_iz.clone() + self.0.h_tr_iz_rad.clone()
        } else {
            self.0.derived_h_ext.clone()
        };

        // Factor out term_rest_1: den = h_ms_is_prod + term_rest_1 * h_total
        // Issue #588 Fix: Include derived_ground_coeff in the base denominator
        // so ground coupling is always active (not just during night ventilation).
        // This makes the static denominator consistent with the dynamic denominator
        // used at runtime in step_physics functions.
        self.0.derived_den = self.0.derived_h_ms_is_prod.clone()
            + self.0.derived_term_rest_1.clone() * h_total.clone()
            + self.0.derived_ground_coeff.clone();

        // ISO 13790 §C.6-C.8: Combined conductances for Crank-Nicolson mass update
        // H_tr_1 = 1 / (1/h_ve + 1/h_tr_is) = h_ve * h_tr_is / (h_ve + h_tr_is)
        self.0.derived_h_tr_1 = self
            .0
            .h_ve
            .zip_with(&self.0.h_tr_is, |ve, is_val| (ve * is_val) / (ve + is_val));

        // H_tr_2 = H_tr_1 + h_tr_w
        self.0.derived_h_tr_2 = self
            .0
            .derived_h_tr_1
            .zip_with(&self.0.h_tr_w, |tr1, w| tr1 + w);

        // H_tr_3 = 1 / (1/H_tr_2 + 1/h_tr_ms) = H_tr_2 * h_tr_ms / (H_tr_2 + h_tr_ms)
        self.0.derived_h_tr_3 = self
            .0
            .derived_h_tr_2
            .zip_with(&self.0.h_tr_ms, |tr2, ms| (tr2 * ms) / (tr2 + ms));

        // sensitivity = 1 / h_total (thermal resistance in K/W)
        // This represents the temperature change per Watt of HVAC power
        // HVAC power formula: P = (T_sp - T_free) / sensitivity
        //
        // h_total = h_ext + h_is_m = (h_tr_em + h_tr_w + h_ve) + (h_tr_is // h_tr_ms)
        // This includes both exterior conductance and internal mass coupling
        // Heat applied to interior air must go through both paths:
        // 1. Through h_ext to exterior environment
        // 2. Through h_is_m to thermal mass (which acts as heat sink/source)
        //
        // The series combination of these paths gives the total thermal resistance
        // that the HVAC system "sees" when trying to control air temperature.
        // Note (#872): derived_sensitivity has been removed. HVAC demand now uses
        // the physics-based h_loss × (T_sp - T_free) formula in step_physics_9r4c,
        // and ideal loads formula in step_physics_5r1c.
    }

    /// Configures the model to use the 6R2C thermal network with two mass nodes.
    ///
    /// This method sets up the 6R2C model by:
    /// 1. Splitting thermal capacitance into envelope and internal components
    /// 2. Setting up conductance between the two mass nodes
    /// 3. Initializing mass temperatures appropriately
    ///
    /// # Arguments
    /// * `envelope_mass_fraction` - Fraction of total thermal mass that is envelope (walls, roof, floor)
    ///   - Typical values: 0.7-0.8 for high-mass buildings
    /// * `h_tr_me_value` - Conductance between envelope and internal mass (W/K)
    ///   - Typical values: 50-200 W/K depending on construction
    /// * `h_tr_ms_value` - Optional override for mass-to-surface conductance (W/K)
    ///   - If None, uses ISO 13790 value (9.1 × A_m ≈ 1092 W/K)
    ///   - For 6R2C, lower values may be more appropriate
    pub fn configure_6r2c_model(
        &mut self,
        envelope_mass_fraction: f64,
        _h_tr_me_value: f64,
        h_tr_ms_value: Option<f64>,
    ) {
        self.0.thermal_model_type = ThermalModelType::SixRTwoC;

        // Split thermal capacitance
        // Envelope: walls, roof, floor (typically 70-80% of total mass)
        // Internal: furniture, partitions (typically 20-30% of total mass)
        let total_cap = self.0.thermal_capacitance.clone();
        self.0.envelope_thermal_capacitance = total_cap.clone() * envelope_mass_fraction;
        self.0.internal_thermal_capacitance = total_cap * (1.0 - envelope_mass_fraction);

        // h_tr_me is now set from physics in from_spec() - do not overwrite here
        // Previously: self.0.h_tr_me = self.0.zone_area.clone().map(|_| h_tr_me_value);
        // The physics-based h_tr_me (≈432 W/K for 48m² zone) provides stronger thermal
        // coupling than the old hardcoded 100.0 W/K, addressing Issue 692.

        // Override h_tr_ms if provided (for 6R2C tuning)
        if let Some(h_tr_ms) = h_tr_ms_value {
            self.0.h_tr_ms = self.0.zone_area.clone().map(|_| h_tr_ms);
        }

        // Initialize mass temperatures from current single mass temperature
        // For 6R2C model, envelope and internal masses should have different time constants
        self.0.envelope_mass_temperatures = self.0.mass_temperatures.clone();
        self.0.internal_mass_temperatures = self.0.mass_temperatures.clone();
    }

    /// Returns true if the model is configured for 6R2C mode.
    pub fn is_6r2c_model(&self) -> bool {
        self.0.thermal_model_type == ThermalModelType::SixRTwoC
    }

    /// Check if this is an 8R3C thermal model (Phase 20 evaluation).
    pub fn is_8r3c_model(&self) -> bool {
        self.0.thermal_model_type == ThermalModelType::EightRThreeC
    }

    /// Check if this is a 9R4C thermal model (Phase 6, Issue #715).
    pub fn is_nine_r4c_model(&self) -> bool {
        self.0.thermal_model_type == ThermalModelType::NineRFourC
    }

    /// Reset to 5R1C thermal model (disable 6R2C and 8R3C).
    ///
    /// This reverts the thermal model to the default ISO 13790 5R1C configuration
    /// with a single thermal mass node.
    pub fn reset_to_5r1c(&mut self) {
        self.0.thermal_model_type = ThermalModelType::FiveROneC;
    }

    /// Enable 9R4C thermal model for high-mass buildings (Phase 6).
    ///
    /// The 9R4C model uses 4 thermal mass nodes (wall, roof, floor, internal)
    /// to properly capture thermal inertia in heavy-mass buildings (Case 900+ series).
    ///
    /// This method should be called during model construction for high-mass buildings.
    /// The per-surface conductances and MultiNodeSolver instances must already be
    /// initialized in `from_spec()` via the `is_9r4c_model` path.
    pub fn enable_9r4c_model(&mut self) {
        self.0.thermal_model_type = ThermalModelType::NineRFourC;
    }

    /// Disable 6R2C model and revert to 5R1C with single thermal mass node.
    pub fn disable_6r2c(&mut self) {
        self.0.thermal_model_type = ThermalModelType::FiveROneC;
    }

    /// Enable CTF (Conduction Transfer Function) solver for high-mass wall conduction.
    ///
    /// This method precomputes CTF coefficients for the wall construction and initializes
    /// CTF solvers for each thermal zone. The CTF solver will be used instead of 5R1C
    /// for calculating heat conduction through opaque surfaces.
    ///
    /// # Arguments
    /// * `wall_layers` - Wall construction layers (interior to exterior) with thermal properties
    /// * `timestep` - Simulation timestep in seconds (typically 3600 for 1-hour)
    /// * `history_size` - Number of history elements to retain (typically 50)
    ///
    /// # Example
    /// ```rust
    /// let mut model = ThermalModel::new(1);
    /// let layers = vec![
    ///     CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
    ///     CTFMaterial::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
    ///     CTFMaterial::new("Insulation", 0.050, 0.04, 50.0, 840.0),
    ///     CTFMaterial::new("Brick", 0.100, 0.81, 1920.0, 790.0),
    /// ];
    /// model.enable_ctf(&layers, 3600.0, 50);
    /// ```
    pub fn enable_ctf(&mut self, wall_layers: &[CTFMaterial], timestep: f64, history_size: usize) {
        // Precompute CTF coefficients for the wall construction
        let calculator = CTFCalculator::with_defaults(wall_layers, timestep);
        let coefficients = calculator.compute_coefficients();

        // Create a solver for each zone
        let mut solvers = Vec::with_capacity(self.0.num_zones);
        for i in 0..self.0.num_zones {
            let mut config = CTFSolverConfig::new(timestep, history_size);

            // Issue #1152: Calculate the actual opaque wall surface area for this zone.
            // The CTF flux is per m² of this surface, so it must be the actual opaque
            // wall area (not floor area). Each zone has 6 surfaces (S, W, N, E, Up, Down).
            let opaque_wall_area: f64 = self
                .0
                .surfaces
                .get(i)
                .map(|zone_surfaces| {
                    zone_surfaces
                        .iter()
                        .filter(|s| {
                            // Only vertical walls (S, W, N, E) - exclude roof/floor
                            !matches!(
                                s.orientation,
                                crate::validation::ashrae_140_cases::Orientation::Up
                                    | crate::validation::ashrae_140_cases::Orientation::Down
                            )
                        })
                        .map(|s| s.area - s.window_area)
                        .sum()
                })
                .unwrap_or(20.0);

            config.surface_area = opaque_wall_area;
            config.h_interior = INTERIOR_FILM_COEFF;
            config.h_exterior =
                crate::physics::constants::thermal::ashrae_140::v2023::EXTERIOR_FILM_COEFF;
            solvers.push(CTFSolver::new(coefficients.clone(), config));
        }

        self.0.conduction.ctf_coefficients = Some(coefficients);
        self.0.conduction.ctf_solvers = solvers;
        self.0.conduction.ctf_enabled = true;
        self.0.conduction.ctf_timestep = timestep;

        // === SESSION 77: CTF-Zone Air Coupling Solver ===
        // DISABLED: The iterative coupling solver creates an explicit feedback loop
        // between T_zone and T_si, causing exponential divergence (oscillation with
        // growing amplitude ~3.1x per timestep). The non-iterative solver.step()
        // path is stable and provides correct CTF flux without coupling instability.
        // The coupling solver would need implicit (simultaneous) solving of T_zone
        // and T_si to be stable, which is a future enhancement.
        // self.0.conduction.ctf_zone_coupling_solver = Some(CtfZoneCouplingSolver::new());
    }

    /// Disable CTF solver and revert to 5R1C conduction calculation.
    pub fn disable_ctf(&mut self) {
        self.0.conduction.ctf_enabled = false;
        self.0.conduction.ctf_coefficients = None;
        self.0.conduction.ctf_solvers.clear();
    }

    /// Check if CTF solver is enabled.
    pub fn ctf_is_enabled(&self) -> bool {
        self.0.conduction.ctf_enabled
    }

    /// Enable Finite Difference (FD) solver for high-mass walls.
    ///
    /// This method creates FD solvers for each zone using the wall layer discretization.
    /// FD solver uses implicit BTCS scheme with Thomas algorithm for tridiagonal system solving.
    ///
    /// # Arguments
    ///
    /// * `wall_layers` - Wall construction layers (used for FD discretization)
    /// * `timestep` - Simulation timestep in seconds (typically 3600 for 1-hour)
    /// * `nodes_per_layer` - Number of nodes per material layer (default: 5-10 for accuracy)
    /// * `initial_temp` - Initial wall temperature [°C] (default: 20°C)
    ///
    /// # Example
    ///
    /// ```rust
    /// let layers = vec![
    ///     MaterialLayer::new("Concrete", 0.200, 1.4, 2300.0, 880.0),
    /// ];
    /// model.enable_fd(&layers, 3600.0, 5, 20.0);
    /// ```
    pub fn enable_fd(
        &mut self,
        wall_layers: &[crate::physics::fd_discretization::MaterialLayer],
        timestep: f64,
        nodes_per_layer: usize,
        initial_temp: f64,
    ) {
        use crate::physics::fd_discretization::WallDiscretization;
        use crate::physics::fd_solver::ImplicitFDSolver;

        // Create discretization for the wall
        let discretization = WallDiscretization::from_layers(wall_layers, nodes_per_layer);

        // Create a solver for each zone
        let mut solvers = Vec::with_capacity(self.0.num_zones);
        for _ in 0..self.0.num_zones {
            solvers.push(ImplicitFDSolver::new(discretization.clone(), initial_temp));
        }

        self.0.conduction.fd_solvers = solvers;
        self.0.conduction.fd_enabled = true;
        self.0.conduction.fd_timestep = timestep;
    }

    /// Disable FD solver and revert to 5R1C conduction calculation.
    pub fn disable_fd(&mut self) {
        self.0.conduction.fd_enabled = false;
        self.0.conduction.fd_solvers.clear();
    }

    /// Check if FD solver is enabled.
    pub fn fd_is_enabled(&self) -> bool {
        self.0.conduction.fd_enabled
    }

    /// Enable the unified solver manager with explicit solver selection.
    ///
    /// The solver manager provides automatic method selection (5R1C/CTF/FD) based on
    /// thermal mass, but with explicit configuration control per Issue #502 requirements.
    ///
    /// # Arguments
    ///
    /// * `selection_config` - Solver selection configuration (Automatic, ForceMethod, PerSurface)
    ///
    /// # Example
    ///
    /// ```rust
    /// use fluxion::physics::method_selector::{
    ///     ThermalMethodSelector, SolverSelectionConfig, SurfaceSolverConfig, ThermalMethod
    /// };
    ///
    /// // Automatic selection based on thermal mass
    /// model.enable_solver_manager(SolverSelectionConfig::Automatic);
    ///
    /// // Force all surfaces to use CTF
    /// model.enable_solver_manager(SolverSelectionConfig::ForceMethod(ThermalMethod::CTF));
    ///
    /// // Per-surface explicit selection
    /// model.enable_solver_manager(SolverSelectionConfig::PerSurface(vec![
    ///     SurfaceSolverConfig::wall(ThermalMethod::FiveR1C),
    ///     SurfaceSolverConfig::roof(ThermalMethod::CTF),
    /// ]));
    /// ```
    pub fn enable_solver_manager(
        &mut self,
        selection_config: crate::physics::method_selector::SolverSelectionConfig,
    ) {
        use crate::physics::method_selector::ThermalMethodSelector;

        let mut selector = ThermalMethodSelector::default();
        selector.set_selection_config(selection_config);

        self.0.conduction.solver_manager =
            Some(crate::physics::solver_manager::SolverManager::new(selector));
    }

    /// Get a reference to the solver manager if it exists.
    pub fn get_solver_manager(&self) -> Option<&crate::physics::solver_manager::SolverManager> {
        self.0.conduction.solver_manager.as_ref()
    }

    /// Get a mutable reference to the solver manager if it exists.
    pub fn get_solver_manager_mut(
        &mut self,
    ) -> Option<&mut crate::physics::solver_manager::SolverManager> {
        self.0.conduction.solver_manager.as_mut()
    }

    /// Disable the solver manager and revert to default 5R1C/CTF/FD behavior.
    pub fn disable_solver_manager(&mut self) {
        self.0.conduction.solver_manager = None;
    }

    /// Check if solver manager is enabled.
    pub fn solver_manager_is_enabled(&self) -> bool {
        self.0.conduction.solver_manager.is_some()
    }

    /// Enable CTF with automatic fallback to FD if coefficients are invalid.
    ///
    /// This method attempts to enable CTF solver, but if coefficient calculation fails
    /// or produces invalid results, it automatically falls back to FD solver.
    ///
    /// # Arguments
    ///
    /// * `wall_layers` - Wall construction layers for both CTF and FD
    /// * `timestep` - Simulation timestep in seconds
    /// * `history_size` - CTF history buffer size (default: 50)
    /// * `fd_nodes` - FD nodes per layer (default: 5)
    ///
    /// # Returns
    ///
    /// `true` if CTF was enabled, `false` if fell back to FD
    pub fn enable_ctf_with_fd_fallback(
        &mut self,
        wall_layers: &[crate::physics::fd_discretization::MaterialLayer],
        timestep: f64,
        history_size: usize,
        fd_nodes: usize,
    ) -> bool {
        use crate::physics::ctf_coefficients::CTFCalculator;
        use crate::physics::method_selector::ThermalMethodSelector;

        // Convert wall layers to CTF materials
        let ctf_materials: Vec<crate::physics::ctf_coefficients::CTFMaterial> = wall_layers
            .iter()
            .map(|l| {
                crate::physics::ctf_coefficients::CTFMaterial::new(
                    &l.name,
                    l.thickness,
                    l.conductivity,
                    l.density,
                    l.specific_heat,
                )
            })
            .collect();

        // Try to compute CTF coefficients
        let calculator = CTFCalculator::with_defaults(&ctf_materials, timestep);
        let coefficients = calculator.compute_coefficients();

        // Validate coefficients
        if !ThermalMethodSelector::validate_ctf_coefficients(&coefficients) {
            log::warn!("CTF coefficients invalid, falling back to FD solver");
            self.enable_fd(wall_layers, timestep, fd_nodes, 20.0);
            return false;
        }

        // CTF coefficients are valid, enable CTF
        self.enable_ctf(&ctf_materials, timestep, history_size);
        true
    }

    /// Updates model parameters based on a gene vector from an optimizer.
    ///
    /// This method maps optimization variables (genes) to physical parameters of the thermal model.
    ///
    /// # Arguments
    /// * `params` - Parameter vector from optimizer:
    ///   - `params[0]`: Window U-value (W/m²K, range: 0.5-3.0)
    ///   - `params[1]`: Heating setpoint (°C, range: 15-25)
    ///   - `params[2]`: Cooling setpoint (°C, range: 22-32)
    ///
    /// # Notes
    /// - If heating_setpoint >= cooling_setpoint, the values will be swapped to maintain valid deadband.
    ///
    /// Applies building design parameters to the thermal model.
    ///
    /// This method validates all parameters for NaN/Inf values and physical constraints
    /// before applying them. Invalid parameters will cause a panic with a descriptive message.
    ///
    /// # Arguments
    /// * `params` - Parameter vector where:
    ///   - `params[0]`: Window U-value (W/m²K)
    ///   - `params[1]`: Heating setpoint (°C)
    ///   - `params[2]`: Cooling setpoint (°C)
    ///
    /// # Panics
    /// Panics if any parameter is NaN or infinite with a message like:
    /// "Window U-value (index 0) is NaN (value: nan W/m²K). Cannot use in simulation."
    ///
    /// # Example
    /// ```no_run
    /// model.apply_parameters(&[1.5, 20.0, 27.0]);
    /// ```
    /// Applies design parameters to the thermal model.
    ///
    /// Maps gene vector elements to model fields and broadcasts 5R1C/6R2C conductances.
    /// Updates derived parameters (conductances, schedules) after applying values.
    ///
    /// # Parameter Vector Semantics
    /// - Element 0: Window U-value (range: 0.1–5.0 W/m²K)
    /// - Element 1: Heating setpoint (range: 15–30°C)
    /// - Element 2: Cooling setpoint (range: 15–30°C, must be > heating setpoint)
    /// - Future elements: Thermal mass, infiltration rates, etc.
    ///
    /// # Errors
    /// Panics if parameters are invalid (NaN, Inf, or heating setpoint >= cooling setpoint).
    /// Use `validate_parameters()` before calling this method for graceful error handling.
    ///
    /// # Example
    /// ```rust,no_run
    /// model.apply_parameters(&[1.5, 20.0, 22.0]);
    /// // Applies: window_u_value=1.5, heating_setpoint=20.0, cooling_setpoint=22.0
    /// ```
    pub fn apply_parameters(&mut self, params: &[f64]) {
        debug!("Applying parameters: {:?}", params);

        // Validate all parameters for NaN/Inf before applying
        if let Some(&u_value) = params.first() {
            if !u_value.is_finite() {
                let error_type = if u_value.is_nan() { "NaN" } else { "infinite" };
                panic!(
                    "Window U-value (index 0) is {} (value: {:.2} W/m²K). Cannot use in simulation.",
                    error_type, u_value
                );
            }
        }
        if let Some(&heating_setpoint) = params.get(1) {
            if !heating_setpoint.is_finite() {
                let error_type = if heating_setpoint.is_nan() {
                    "NaN"
                } else {
                    "infinite"
                };
                panic!(
                    "Heating setpoint (index 1) is {} (value: {:.2}°C). Cannot use in simulation.",
                    error_type, heating_setpoint
                );
            }
        }
        if let Some(&cooling_setpoint) = params.get(2) {
            if !cooling_setpoint.is_finite() {
                let error_type = if cooling_setpoint.is_nan() {
                    "NaN"
                } else {
                    "infinite"
                };
                panic!(
                    "Cooling setpoint (index 2) is {} (value: {:.2}°C). Cannot use in simulation.",
                    error_type, cooling_setpoint
                );
            }
        }

        // Apply parameters
        if !params.is_empty() {
            self.0.window_u_value = params[0];
            debug!("Set window U-value to {} W/m²K", self.0.window_u_value);
            // Surfaces update for metadata/consistency
            for zone_surfaces in &mut self.0.surfaces {
                for surface in zone_surfaces {
                    surface.u_value = self.0.window_u_value;
                }
            }
        }
        if params.len() >= 2 {
            self.0.heating_setpoint = params[1];
            self.0.heating_schedule = DailySchedule::constant(self.0.heating_setpoint);
            debug!("Set heating setpoint to {}°C", self.0.heating_setpoint);
        }
        if params.len() >= 3 {
            self.0.cooling_setpoint = params[2];

            // Ensure heating < cooling for valid deadband
            if self.0.heating_setpoint >= self.0.cooling_setpoint {
                warn!(
                    "Heating setpoint ({}) >= cooling setpoint ({}), swapping to maintain valid deadband",
                    self.0.heating_setpoint, self.0.cooling_setpoint
                );
                std::mem::swap(&mut self.0.heating_setpoint, &mut self.0.cooling_setpoint);
            }
            self.0.heating_schedule = DailySchedule::constant(self.0.heating_setpoint);
            self.0.cooling_schedule = DailySchedule::constant(self.0.cooling_setpoint);
            debug!("Set cooling setpoint to {}°C", self.0.cooling_setpoint);
        }

        // Recalculate derived conductances (h_tr_w, etc.) using new U-values and fixed geometry
        self.update_derived_parameters();
        trace!("Derived parameters updated after applying parameters");
    }

    /// Set the timestep mode for adaptive timestep simulation.
    ///
    /// This enables automatic detection of high-mass buildings (Case 900 series)
    /// and uses finer timesteps (6 minutes) for improved numerical accuracy.
    ///
    /// # Arguments
    /// * `mode` - TimestepMode: either `Fixed` with a specific dt, or `Adaptive`
    ///   with base_dt, min_dt, and threshold_tau for automatic detection.
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluxion::sim::adaptive_timestep::TimestepMode;
    /// use std::time::Duration;
    ///
    /// let mut model = ThermalModel::<VectorField>::new(1);
    ///
    /// // Enable adaptive timestep (6-min for high-mass, 1-hr for low-mass)
    /// model.set_timestep_mode(TimestepMode::adaptive(
    ///     Duration::from_secs(360),   // 6-minute base timestep
    ///     Duration::from_secs(60),    // 1-minute minimum
    ///     2.0,                         // 2-hour threshold
    /// ));
    /// ```
    pub fn set_timestep_mode(&mut self, mode: TimestepMode) {
        self.0.timestep_mode = mode;
        // Issue #2523: demoted from `info!` to `trace!`. Although this
        // setter is config-time (not per-timestep), it is invoked once per
        // model build inside `BatchOracle::evaluate_population` and was
        // listed in #2523. `trace!` keeps it available under verbose
        // tracing while removing it from the default release log stream.
        trace!("Timestep mode set to {:?}", self.0.timestep_mode);
    }

    /// Get the current timestep mode.
    pub fn get_timestep_mode(&self) -> &TimestepMode {
        &self.0.timestep_mode
    }
}
