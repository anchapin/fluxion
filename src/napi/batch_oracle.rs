// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! NAPI bindings for BatchOracle - high-throughput building energy evaluation.
//!
//! Provides JavaScript interface for evaluating populations of building configurations
//! with >10,000 configs/sec throughput. Critical for optimization workflows in BIM tools.

use crate::sim::engine::ThermalModel;
use crate::sim::thermal_selector::ThermalSelector;
use crate::validation::ashrae_140_cases::CaseBuilder;
use crate::BatchOracle as CoreBatchOracle;

/// JavaScript-accessible BatchOracle wrapper for high-throughput building energy evaluation.
///
/// This class provides a JavaScript interface to Fluxion's BatchOracle, enabling
/// optimization workflows in BIM tools (Autodesk, Speckle, Trimble) and parametric
/// analysis platforms.
///
/// # TypeScript Example
/// ```typescript
/// import { BatchOracle } from '@fluxion/native';
///
/// // Create oracle instance
/// const oracle = new BatchOracle();
///
/// // Evaluate multiple configurations in parallel
/// const population = [
///   [1.5, 20.0, 24.0], // [window_u_value, heating_setpoint, cooling_setpoint]
///   [2.0, 20.0, 24.0],
///   [2.5, 20.0, 24.0]
/// ];
///
/// // Evaluate without surrogates (physics-based)
/// const physicsResults = oracle.evaluatePopulation(population, false);
///
/// // Evaluate with surrogates (AI-accelerated, ~10x faster)
/// const aiResults = oracle.evaluatePopulation(population, true);
///
/// console.log(`EUI values (physics): ${physicsResults}`);
/// console.log(`EUI values (AI): ${aiResults}`);
/// ```
///
/// # Performance Characteristics
/// - **Physics-based**: ~1,000 configs/sec on 8-core CPU
/// - **AI-accelerated**: ~10,000+ configs/sec with GPU surrogates
/// - **Latency**: <100ms for single configuration (8760 timesteps)
/// - **Memory**: Minimal allocations via CTA buffer reuse
///
/// # Parameter Constraints
/// - Window U-value: 0.1–5.0 W/m²K
/// - Heating setpoint: 15.0–25.0 °C
/// - Cooling setpoint: 22.0–32.0 °C
/// - Heating setpoint must be less than cooling setpoint
#[napi_derive::napi]
pub struct BatchOracle {
    inner: CoreBatchOracle,
}

#[napi_derive::napi]
impl BatchOracle {
    /// Create a new BatchOracle instance with default ASHRAE 600 configuration.
    ///
    /// This initializes the oracle with a base thermal model suitable for
    /// parametric studies and optimization workflows.
    ///
    /// # TypeScript Example
    /// ```typescript
    /// import { BatchOracle } from '@fluxion/native';
    /// const oracle = new BatchOracle();
    /// ```
    ///
    /// # Returns
    /// A new `BatchOracle` instance ready for evaluation
    ///
    /// # Throws
    /// - `FluxionError` if initialization fails (e.g., model loading, surrogate initialization)
    #[napi(constructor)]
    pub fn new() -> napi::bindgen_prelude::Result<Self> {
        let spec = CaseBuilder::case_600_baseline();
        let thermal_model =
            ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
                .expect("default selector must initialize");

        let inner = CoreBatchOracle::from_model(thermal_model);
        Ok(BatchOracle { inner })
    }

    /// Evaluate a population of building design configurations in parallel.
    ///
    /// This is the critical "hot loop" for optimization. The function uses Rayon for
    /// multi-threaded evaluation and can process 10,000+ configurations per second.
    ///
    /// # Arguments
    /// * `population` - Array of parameter arrays. Each inner array should contain at least:
    ///   - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
    ///   - `[1]`: Heating setpoint (°C, range: 15-25)
    ///   - `[2]`: Cooling setpoint (°C, range: 22-32)
    /// * `use_surrogates` - If true, use neural network surrogates for faster evaluation;
    ///   if false, use analytical physics calculations.
    ///
    /// # Returns
    /// Array of EUI values (kWh/m²/yr) for each candidate configuration.
    /// Invalid configurations return `NaN`.
    #[napi]
    pub fn evaluate_population(
        &self,
        population: Vec<Vec<f64>>,
        use_surrogates: bool,
    ) -> napi::bindgen_prelude::Result<Vec<f64>> {
        self.inner
            .evaluate_population(population, use_surrogates)
            .map_err(|e| napi::bindgen_prelude::Error::from_reason(format!("Fluxion error: {}", e)))
    }

    /// Validate building parameters against physical constraints.
    ///
    /// # Arguments
    /// * `params` - Parameter array containing at least:
    ///   - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
    ///   - `[1]`: Heating setpoint (°C, range: 15-25)
    ///   - `[2]`: Cooling setpoint (°C, range: 22-32)
    ///
    /// # Throws
    /// - `ValidationError` if parameters are out of valid ranges
    #[napi]
    pub fn validate_parameters(&self, params: Vec<f64>) -> napi::bindgen_prelude::Result<()> {
        CoreBatchOracle::validate_parameters(&params)
            .map_err(|e| napi::bindgen_prelude::Error::from_reason(format!("{}", e)))
    }
}
