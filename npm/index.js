// @fluxion/native - High-performance native Node.js bindings for Fluxion

/**
 * Fluxion native Node.js bindings for building energy modeling.
 *
 * This module provides high-performance native bindings to the Fluxion building energy
 * modeling engine, enabling JavaScript/TypeScript applications to evaluate building
 * design configurations at >10,000 configs/sec throughput.
 *
 * @module @fluxion/native
 * @example
 * ```javascript
 * const { BatchOracle, BuildingParameters } = require('@fluxion/native');
 *
 * // Create oracle instance
 * const oracle = new BatchOracle();
 *
 * // Define building parameters
 * const params = new BuildingParameters(1.5, 20.0, 24.0);
 *
 * // Evaluate population (high-throughput optimization)
 * const population = [
 *   [1.5, 20.0, 24.0],
 *   [2.0, 20.0, 24.0],
 *   [2.5, 20.0, 24.0]
 * ];
 *
 * const results = oracle.evaluatePopulation(population, false);
 * console.log(`EUI values: ${results}`); // [120.5, 115.2, 110.8]
 * ```
 */

// Native bindings
const native = require('./fluxion.node');

// Export error classes for proper error handling
module.exports = {
  // Main classes
  BatchOracle: native.BatchOracle,
  BuildingParameters: native.BuildingParameters,
  StateExtractor: native.StateExtractor,
  StateMatrices: native.StateMatrices,
  OsmExporter: native.OsmExporter,
  GbXmlExporter: native.GbXmlExporter,
  FmiExporter: native.FmiExporter,

  // Error classes
  FluxionError: native.FluxionError,
  ValidationError: native.ValidationError,
  SimulationError: native.SimulationError,
  SurrogateError: native.SurrogateError,

  // Version info
  version: '1.0.0',
};

/**
 * Create a BatchOracle instance with default configuration.
 *
 * @function createBatchOracle
 * @returns {BatchOracle} A new BatchOracle instance
 * @example
 * ```javascript
 * const oracle = createBatchOracle();
 * const results = oracle.evaluatePopulation([[1.5, 20.0, 24.0]], false);
 * ```
 */
module.exports.createBatchOracle = () => new native.BatchOracle();

/**
 * Create BuildingParameters with validation.
 *
 * @function createBuildingParameters
 * @param {number} windowUValue - Window U-value in W/m²K (range: 0.1-5.0)
 * @param {number} heatingSetpoint - Heating setpoint in °C (range: 15.0-25.0)
 * @param {number} coolingSetpoint - Cooling setpoint in °C (range: 22.0-32.0)
 * @returns {BuildingParameters} Validated building parameters
 * @throws {ValidationError} If parameters are out of valid ranges
 * @example
 * ```javascript
 * const params = createBuildingParameters(1.5, 20.0, 24.0);
 * console.log(params.windowUValue); // 1.5
 * ```
 */
module.exports.createBuildingParameters = (windowUValue, heatingSetpoint, coolingSetpoint) =>
  new native.BuildingParameters(windowUValue, heatingSetpoint, coolingSetpoint);
