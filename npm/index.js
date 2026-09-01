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

/**
 * Hidden gate for experimental zone solvers (Issue #3282).
 *
 * `new StateExtractor({ zoneSolver: '6r2c' | '8r3c' })` is rejected unless
 * the `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` environment variable is set in
 * the Node process. Even with the env var set, those identifiers stay
 * rejected until the `fluxion-experimental-zone-solvers` cargo feature
 * ships (issue #3291) — the env var widens no doors the build cannot back.
 *
 * @name FLUXION_EXPERIMENTAL_ZONE_SOLVERS
 * @type {string}
 */

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
  NineR4CConfig: native.NineR4CConfig,
  transferMatrix: native.transferMatrix,

  // Issue #1800 (T9.6): sub-hourly 9R4C nodal temperature trace
  NineR4CNodalTracer: native.NineR4CNodalTracer,

  // Issue #1798 (T9.3): HVAC configuration classes
  HvacVavTerminal: native.HvacVavTerminal,
  HvacCavSystem: native.HvacCavSystem,
  HvacHeatPump: native.HvacHeatPump,
  HvacChiller: native.HvacChiller,
  HvacBoiler: native.HvacBoiler,
  ZoneSetpoints: native.ZoneSetpoints,
  HvacDailySchedule: native.HvacDailySchedule,
  HvacSchedule: native.HvacSchedule,
  ZoneController: native.ZoneController,

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

/**
 * Create a NineR4CNodalTracer with default 9R4C parameters (issue #1800).
 *
 * Use this to extract sub-hourly nodal temperature traces from the
 * 9R4C multi-node solver. The returned trace agrees bit-for-bit with
 * the canonical solver trace used by the Python binding (T9.5).
 *
 * @function createNineR4CNodalTracer
 * @returns {NineR4CNodalTracer} A new tracer instance
 * @example
 * ```javascript
 * const tracer = createNineR4CNodalTracer();
 * const trace = tracer.runSubHourlyTrace({
 *   dtSeconds: 300.0,
 *   timesteps: 288,
 *   couplingMode: 'additive_sum',
 * });
 * console.log(`Wall[0] = ${trace.wall[0]} °C`);
 * ```
 */
module.exports.createNineR4CNodalTracer = () => new native.NineR4CNodalTracer();

/**
 * Experimental zone-solver identifiers, mirrored from
 * `EXPERIMENTAL_ZONE_SOLVERS` in src/sim/thermal_selector.rs (issue #3282).
 * They are rejected unless the `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1`
 * environment variable is set — and even then they stay rejected by the
 * Rust layer until the `fluxion-experimental-zone-solvers` cargo feature
 * ships (issue #3291).
 *
 * @private
 */
const EXPERIMENTAL_ZONE_SOLVERS = new Set(['6r2c', '8r3c']);

/**
 * Exact copy of the `parse_zone_solver` rejection message for
 * experimental identifiers without the env gate (src/sim/thermal_selector.rs),
 * so JS fast-failures carry the same wording as the Rust layer.
 *
 * @private
 * @param {string} identifier - Normalized experimental zone-solver identifier
 * @returns {string} The shared experimental-gate rejection message
 */
function experimentalZoneSolverMessage(identifier) {
  return (
    `experimental zone solver '${identifier}' requires ` +
    'FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1 to be set (and even then ' +
    'it stays unavailable until the `fluxion-experimental-zone-solvers` ' +
    'cargo feature ships; issue #3291)'
  );
}

/**
 * Normalize a solver identifier the same way the Rust parsers do
 * (trim + ASCII lowercase).
 *
 * @private
 * @param {string} value - Raw identifier
 * @param {string} optionName - Option name for error messages
 * @returns {string} Normalized identifier
 */
function normalizeSolverIdentifier(value, optionName) {
  if (typeof value !== 'string') {
    throw new Error(`runSimulation: ${optionName} must be a string when provided`);
  }
  return value.trim().toLowerCase();
}

/**
 * Run a one-shot simulation with an explicit thermal solver selection
 * (issue #3306) — the ergonomic one-call wrapper that issue #3282's Node
 * example assumed: construct `StateExtractor` → `configure` →
 * `runSimulation` → plain serializable data, with the
 * `{ zoneSolver, conductionSolver }` selector wired end-to-end.
 *
 * The simulation runs the native `StateExtractor` surface, which is built
 * on the ASHRAE 600 baseline configuration (1 zone); `caseSpec` / `schema`
 * inputs are therefore **not** accepted yet and fail closed when provided.
 *
 * Zone-solver values are pre-validated against the shared vocabulary:
 * experimental identifiers (`'6r2c'`, `'8r3c'`) are rejected without the
 * `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` environment variable using the
 * exact shared gate message from the Rust `parse_zone_solver` parser.
 * Unknown values, gate-set values, and conduction-solver values are passed
 * through to the authoritative Rust parser (surface their errors verbatim).
 *
 * @function runSimulation
 * @param {object} [options] - Simulation options
 * @param {number} [options.years=1] - Number of years to simulate (8760
 *   timesteps per year)
 * @param {string} [options.zoneSolver='gauge'] - Zone solver:
 *   `'gauge'` (default) | `'5r1c'` | `'9r4c'`; `'6r2c'` / `'8r3c'` are
 *   experimental and gated
 * @param {string} [options.conductionSolver='default'] - Conduction
 *   algorithm: `'default'` (default) | `'ctf'` | `'fd'`
 * @param {boolean} [options.useSurrogates=false] - Use AI surrogates for
 *   faster evaluation when available
 * @param {*} [options.caseSpec] - Reserved; not accepted by the native
 *   `StateExtractor` surface yet — providing it throws
 * @param {*} [options.schema] - Reserved; not accepted by the native
 *   `StateExtractor` surface yet — providing it throws
 * @returns {{years: number, timesteps: number, zoneSolver: string,
 *   conductionSolver: string, useSurrogates: boolean, zoneTemperatures:
 *   number[], massTemperatures: number[], heatingLoads: number[],
 *   coolingLoads: number[], solarGains: number[]}} Plain serializable
 *   result; `zoneSolver` / `conductionSolver` echo the effective lowercase
 *   labels of the selector that was wired through `StateExtractor`
 * @throws {Error} If `caseSpec` / `schema` are provided, `years` is not a
 *   positive integer, an experimental zone solver is selected without
 *   `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1`, or the Rust layer rejects the
 *   selector or simulation inputs
 * @example
 * ```javascript
 * const { runSimulation } = require('@fluxion/native');
 *
 * // Default path: gauge zone solver, default conduction
 * const baseline = runSimulation({ years: 1 });
 * console.log(baseline.zoneSolver); // 'gauge'
 *
 * // Explicit 5R1C selection (issue #3282 ergonomics)
 * const legacy = runSimulation({ years: 1, zoneSolver: '5r1c' });
 * console.log(legacy.zoneSolver); // '5r1c'
 * ```
 */
module.exports.runSimulation = function runSimulation(options = {}) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new Error('runSimulation: expected an options object');
  }

  const {
    caseSpec,
    schema,
    years = 1,
    zoneSolver,
    conductionSolver,
    useSurrogates = false,
  } = options;

  // The native StateExtractor surface simulates the built-in ASHRAE 600
  // baseline only — fail closed instead of silently ignoring inputs.
  if (caseSpec !== undefined || schema !== undefined) {
    throw new Error(
      "runSimulation: 'caseSpec' / 'schema' inputs are not yet accepted by the " +
        'native StateExtractor surface, which currently simulates the built-in ' +
        'ASHRAE 600 baseline configuration only; omit them to run the baseline ' +
        'case (issue #3306)'
    );
  }

  if (!Number.isInteger(years) || years < 1) {
    throw new Error('runSimulation: years must be a positive integer');
  }

  const effectiveZoneSolver =
    zoneSolver === undefined || zoneSolver === null
      ? 'gauge'
      : normalizeSolverIdentifier(zoneSolver, 'zoneSolver');
  const effectiveConductionSolver =
    conductionSolver === undefined || conductionSolver === null
      ? 'default'
      : normalizeSolverIdentifier(conductionSolver, 'conductionSolver');

  // Fast-fail on experimental zone solvers without the shared env gate,
  // reusing the exact Rust `parse_zone_solver` wording. Everything else
  // (unknown values; gate-set experimental values, which stay unavailable
  // until the cargo feature ships) is left to the authoritative Rust
  // parser invoked by the StateExtractor constructor.
  if (
    EXPERIMENTAL_ZONE_SOLVERS.has(effectiveZoneSolver) &&
    process.env.FLUXION_EXPERIMENTAL_ZONE_SOLVERS !== '1'
  ) {
    throw new Error(experimentalZoneSolverMessage(effectiveZoneSolver));
  }

  const extractorOptions = {};
  if (zoneSolver !== undefined && zoneSolver !== null) {
    extractorOptions.zoneSolver = zoneSolver;
  }
  if (conductionSolver !== undefined && conductionSolver !== null) {
    extractorOptions.conductionSolver = conductionSolver;
  }

  const extractor = new native.StateExtractor(extractorOptions);
  extractor.configure(1);
  const matrices = extractor.runSimulation(years, useSurrogates === true);

  return {
    years,
    timesteps: years * 8760,
    zoneSolver: effectiveZoneSolver,
    conductionSolver: effectiveConductionSolver,
    useSurrogates: useSurrogates === true,
    zoneTemperatures: Array.from(matrices.zoneTemperatures),
    massTemperatures: Array.from(matrices.massTemperatures),
    heatingLoads: Array.from(matrices.heatingLoads),
    coolingLoads: Array.from(matrices.coolingLoads),
    solarGains: Array.from(matrices.solarGains),
  };
};
