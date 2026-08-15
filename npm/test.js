// Test suite for Fluxion native Node.js bindings
// Run with: node --test test.js

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert');

// Note: These tests require the native module to be built first
// Run: npm run build before running tests

describe('@fluxion/native', () => {
  let fluxion, BatchOracle, BuildingParameters, NineR4CConfig, ValidationError;

  before(() => {
    // Load the module
    fluxion = require('./index.js');
    BatchOracle = fluxion.BatchOracle;
    BuildingParameters = fluxion.BuildingParameters;
    NineR4CConfig = fluxion.NineR4CConfig;
    ValidationError = fluxion.ValidationError;
  });

  describe('BuildingParameters', () => {
    it('should create valid building parameters', () => {
      const params = new BuildingParameters(1.5, 20.0, 24.0);
      assert.strictEqual(params.windowUValue, 1.5);
      assert.strictEqual(params.heatingSetpoint, 20.0);
      assert.strictEqual(params.coolingSetpoint, 24.0);
    });

    it('should convert to array format', () => {
      const params = new BuildingParameters(2.0, 19.0, 23.0);
      const array = params.toVec();
      assert.deepStrictEqual(array, [2.0, 19.0, 23.0]);
    });

    it('should reject invalid U-value (too low)', () => {
      assert.throws(
        () => new BuildingParameters(0.05, 20.0, 24.0),
        (error) => /Validation error|Parameter validation error/.test(error.message)
      );
    });

    it('should reject invalid U-value (too high)', () => {
      assert.throws(
        () => new BuildingParameters(6.0, 20.0, 24.0),
        (error) => /Validation error|Parameter validation error/.test(error.message)
      );
    });

    it('should reject invalid heating setpoint (too low)', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 14.0, 24.0),
        (error) => /Validation error|Parameter validation error/.test(error.message)
      );
    });

    it('should reject invalid heating setpoint (too high)', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 26.0, 24.0),
        (error) => /Validation error|Parameter validation error/.test(error.message)
      );
    });

    it('should reject invalid cooling setpoint (too low)', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 20.0, 21.0),
        (error) => /Validation error|Parameter validation error/.test(error.message)
      );
    });

    it('should reject invalid cooling setpoint (too high)', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 20.0, 33.0),
        (error) => /Validation error|Parameter validation error/.test(error.message)
      );
    });

    it('should reject heating >= cooling', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 24.0, 22.0),
        (error) => /Validation error|Parameter validation error/.test(error.message)
      );
    });
  });

  describe('BatchOracle', () => {
    let oracle;

    it('should create BatchOracle instance', () => {
      oracle = new BatchOracle();
      assert.ok(oracle);
      assert.strictEqual(typeof oracle.evaluatePopulation, 'function');
      assert.strictEqual(typeof oracle.validateParameters, 'function');
    });

    it('should validate valid parameters', () => {
      assert.doesNotThrow(() => {
        oracle.validateParameters([1.5, 20.0, 24.0]);
      });
    });

    it('should reject invalid parameters via validateParameters', () => {
      assert.throws(
        () => oracle.validateParameters([6.0, 20.0, 24.0]),
        (error) => /Validation error|Parameter validation error/.test(error.message)
      );
    });

    it('should evaluate single configuration', () => {
      const population = [[1.5, 20.0, 24.0]];
      const results = oracle.evaluatePopulation(population, false);
      assert.strictEqual(results.length, 1);
      assert.strictEqual(typeof results[0], 'number');
      assert.ok(isFinite(results[0])); // Should be a valid number, not NaN or Infinity
    });

    it('should evaluate multiple configurations', () => {
      const population = [
        [1.5, 20.0, 24.0],
        [2.0, 20.0, 24.0],
        [2.5, 20.0, 24.0],
      ];
      const results = oracle.evaluatePopulation(population, false);
      assert.strictEqual(results.length, 3);
      results.forEach(result => {
        assert.strictEqual(typeof result, 'number');
        assert.ok(isFinite(result));
      });
    });

    it('should return NaN for invalid configurations', () => {
      const population = [
        [1.5, 20.0, 24.0],  // Valid
        [6.0, 20.0, 24.0],  // Invalid: U-value too high
        [2.0, 20.0, 24.0],  // Valid
      ];
      const results = oracle.evaluatePopulation(population, false);
      assert.strictEqual(results.length, 3);
      assert.ok(isFinite(results[0]));  // Valid
      assert.ok(isNaN(results[1]));     // Invalid -> NaN
      assert.ok(isFinite(results[2]));  // Valid
    });

    it('should handle large population efficiently', () => {
      const population = Array.from({ length: 100 }, () => [
        1.5 + Math.random() * 2.0,
        18.0 + Math.random() * 4.0,
        22.0 + Math.random() * 4.0,
      ]);

      const startTime = Date.now();
      const results = oracle.evaluatePopulation(population, false);
      const duration = Date.now() - startTime;

      assert.strictEqual(results.length, 100);
      results.forEach(result => {
        assert.ok(isFinite(result));
      });

      // Should complete in reasonable time (< 15 seconds for 100 configs on CI runners).
      // The 5s threshold is flaky on Windows GitHub Actions runners under transient load;
      // see #3001 CI retry. 15s still verifies that the BatchOracle is functionally responsive
      // and catches real performance regressions (which would be orders of magnitude slower).
      assert.ok(duration < 15000, `Evaluation took ${duration}ms, expected < 15000ms`);
    });

    it('should produce consistent results for same inputs', () => {
      const population = [[1.5, 20.0, 24.0]];
      const results1 = oracle.evaluatePopulation(population, false);
      const results2 = oracle.evaluatePopulation(population, false);

      assert.strictEqual(results1[0], results2[0]);
    });

    it('should respect parameter constraints in evaluation', () => {
      const population = [
        [1.5, 20.0, 24.0],  // Lower U-value -> higher EUI expected
        [3.0, 20.0, 24.0],  // Higher U-value -> lower EUI expected
      ];
      const results = oracle.evaluatePopulation(population, false);

      // Results should be finite numbers
      assert.ok(isFinite(results[0]));
      assert.ok(isFinite(results[1]));

      // Different inputs should produce different results
      assert.notStrictEqual(results[0], results[1]);
    });
  });

  describe('Integration', () => {
    let oracle;

    before(() => {
      oracle = new BatchOracle();
    });

    it('should work with BuildingParameters and BatchOracle together', () => {
      const params = new BuildingParameters(1.5, 20.0, 24.0);
      const paramArray = params.toVec();
      const population = [paramArray];
      const results = oracle.evaluatePopulation(population, false);

      assert.strictEqual(results.length, 1);
      assert.ok(isFinite(results[0]));
    });

    it('should handle error scenarios gracefully', () => {
      const population = [
        [1.5, 20.0, 24.0],  // Valid
        [NaN, 20.0, 24.0],   // Invalid: NaN
        [1.5, 20.0, 24.0],  // Valid
      ];
      const results = oracle.evaluatePopulation(population, false);

      assert.strictEqual(results.length, 3);
      assert.ok(isFinite(results[0]));  // Valid
      assert.ok(isNaN(results[1]));     // Invalid -> NaN
      assert.ok(isFinite(results[2]));  // Valid
    });
  });

  describe('NineR4CConfig', () => {
    it('should create with default constructor', () => {
      const config = new NineR4CConfig();
      assert.strictEqual(config.hTrIs, 10.0);
      assert.strictEqual(config.zoneTemperature, 20.0);
      assert.strictEqual(config.surfaceTemperature, 20.0);
      assert.strictEqual(config.exteriorTemperature, 10.0);
      assert.strictEqual(config.couplingMode, 'additive_sum');
      assert.strictEqual(config.wall.temperature, 20.0);
      assert.strictEqual(config.roof.temperature, 20.0);
      assert.strictEqual(config.floor.temperature, 20.0);
      assert.strictEqual(config.internal.temperature, 20.0);
    });

    it('should round-trip a 9R4C config object', () => {
      // Create a custom config
      const config = new NineR4CConfig({
        hTrIs: 15.0,
        wall: { temperature: 22.0, capacitance: 5e6, hTrMs: 50.0, hTrEm: 20.0 },
        roof: { temperature: 22.0, capacitance: 3e6, hTrMs: 30.0, hTrEm: 15.0 },
        floor: { temperature: 20.0, capacitance: 2e6, hTrMs: 20.0, hTrEm: 10.0 },
        internal: { temperature: 21.0, capacitance: 1e6, hTrMs: 0.0, hTrEm: 0.0, hTrMe: 100.0 },
        couplingMode: 'parallel_resistance',
      });

      // Verify parameters round-trip correctly
      assert.strictEqual(config.hTrIs, 15.0);
      assert.strictEqual(config.couplingMode, 'parallel_resistance');
      assert.strictEqual(config.wall.temperature, 22.0);
      assert.strictEqual(config.wall.capacitance, 5e6);
      assert.strictEqual(config.wall.hTrMs, 50.0);
      assert.strictEqual(config.wall.hTrEm, 20.0);
      assert.strictEqual(config.roof.temperature, 22.0);
      assert.strictEqual(config.roof.capacitance, 3e6);
      assert.strictEqual(config.roof.hTrMs, 30.0);
      assert.strictEqual(config.roof.hTrEm, 15.0);
      assert.strictEqual(config.floor.temperature, 20.0);
      assert.strictEqual(config.floor.capacitance, 2e6);
      assert.strictEqual(config.floor.hTrMs, 20.0);
      assert.strictEqual(config.floor.hTrEm, 10.0);
      assert.strictEqual(config.internal.temperature, 21.0);
      assert.strictEqual(config.internal.capacitance, 1e6);
      assert.strictEqual(config.internal.hTrMe, 100.0);
    });

    it('should step forward in time and update temperatures', () => {
      const config = new NineR4CConfig();
      config.zoneTemperature = 25.0;
      config.exteriorTemperature = 5.0;
      config.surfaceTemperature = 18.0;

      const wallTempBefore = config.wallTemperature;
      config.step(3600.0);
      const wallTempAfter = config.wall.temperature;

      // Wall should cool toward exterior temperature
      assert.ok(config.wallTemperature < wallTempBefore,
        `Wall temperature (${config.wallTemperature}) should decrease from ${wallTempBefore}`);
      assert.ok(config.wallTemperature > 5.0,
        `Wall temperature (${config.wallTemperature}) should stay above exterior (5.0)`);
    });

    it('should set and get per-surface exterior temperatures', () => {
      const config = new NineR4CConfig();
      config.setSurfaceExteriorTemperatures(30.0, 35.0, 15.0);

      assert.strictEqual(config.tExtWall, 30.0);
      assert.strictEqual(config.tExtRoof, 35.0);
      assert.strictEqual(config.tExtFloor, 15.0);
    });

    it('should compute zone air temperature', () => {
      const config = new NineR4CConfig();
      // All nodes at 20°C, outdoor at 20°C -> T_air ≈ 20°C
      const tAir = config.computeZoneAirTemperature(20.0, 5.0, 0.0, 0.0);
      assert.ok(Math.abs(tAir - 20.0) < 1.0,
        `Zone air temperature (${tAir}) should be near 20°C`);
    });

    it('should compute HVAC demand', () => {
      const config = new NineR4CConfig();
      // T_air_free < heating setpoint -> positive Q (heating needed)
      const q = config.computeHvacDemand(15.0, 20.0, 26.0);
      assert.ok(q > 0.0, `Heating demand should be positive, got ${q}`);
      // Q = h_tr_is * (20 - 15) = 10 * 5 = 50 W
      assert.ok(Math.abs(q - 50.0) < 1.0, `Expected ~50W, got ${q}`);
    });

    it('should update conductances via setters', () => {
      const config = new NineR4CConfig();
      config.setWallConductances(25.0, 55.0);
      assert.strictEqual(config.wall.hTrEm, 25.0);
      assert.strictEqual(config.wall.hTrMs, 55.0);

      config.setInternalConductance(150.0);
      assert.strictEqual(config.internal.hTrMe, 150.0);
    });

    it('should update capacitances via setters', () => {
      const config = new NineR4CConfig();
      config.setWallCapacitance(1e7);
      config.setRoofCapacitance(2e7);
      config.setFloorCapacitance(3e7);
      config.setInternalCapacitance(4e6);

      assert.strictEqual(config.wall.capacitance, 1e7);
      assert.strictEqual(config.roof.capacitance, 2e7);
      assert.strictEqual(config.floor.capacitance, 3e7);
      assert.strictEqual(config.internal.capacitance, 4e6);
    });

    it('should step with gains', () => {
      const config = new NineR4CConfig();
      config.zoneTemperature = 20.0;
      config.exteriorTemperature = 10.0;
      config.surfaceTemperature = 18.0;

      const wallTempBefore = config.wallTemperature;
      config.stepWithGains(3600.0, 1000.0, 500.0, 0.0, 0.0);

      // Wall with gains should be hotter than without
      assert.ok(config.wallTemperature > wallTempBefore,
        `Wall with gains (${config.wallTemperature}) should be hotter than without (${wallTempBefore})`);
    });

    it('should expose effective time constant', () => {
      const config = new NineR4CConfig();
      const tau = config.effectiveTimeConstant;
      assert.ok(tau > 0.0, `Time constant (${tau}) should be positive`);
      assert.ok(tau < 1e8, `Time constant (${tau}) should be finite`);
    });
  });

  // Issue #1800 (T9.6): Node parity with T9.5 — sub-hourly nodal
  // temperature trace for the 9R4C multi-node solver.
  describe('NineR4CNodalTracer (issue #1800)', () => {
    let NineR4CNodalTracer;
    before(() => {
      NineR4CNodalTracer = fluxion.NineR4CNodalTracer;
    });

    it('should expose the tracer constructor', () => {
      assert.strictEqual(typeof NineR4CNodalTracer, 'function',
        'NineR4CNodalTracer should be exported');
      const tracer = new NineR4CNodalTracer();
      assert.ok(tracer);
      assert.strictEqual(typeof tracer.runSubHourlyTrace, 'function');
    });

    it('should return five Float64Array series of equal length', () => {
      const tracer = new NineR4CNodalTracer();
      const trace = tracer.runSubHourlyTrace({
        dtSeconds: 300.0,
        timesteps: 48,
        couplingMode: 'additive_sum',
        initialZoneTemperature: 20.0,
        surfaceExteriorTemperatures: {
          tExtWall: 5.0,
          tExtRoof: 5.0,
          tExtFloor: 5.0,
        },
        hTrIs: 10.0,
        gains: Array.from({ length: 48 }, () => [0.0, 0.0, 0.0, 0.0]),
      });

      assert.strictEqual(trace.timesteps, 48);
      assert.strictEqual(trace.dtSeconds, 300.0);
      assert.strictEqual(trace.couplingMode, 'additive_sum');
      assert.strictEqual(trace.wall.length, 48);
      assert.strictEqual(trace.roof.length, 48);
      assert.strictEqual(trace.floor.length, 48);
      assert.strictEqual(trace.internal.length, 48);
      assert.strictEqual(trace.zone.length, 48);
      for (let i = 0; i < 48; i++) {
        assert.ok(Number.isFinite(trace.wall[i]), `wall[${i}] must be finite`);
        assert.ok(Number.isFinite(trace.zone[i]), `zone[${i}] must be finite`);
      }
    });

    it('should be deterministic across repeated runs', () => {
      const tracer = new NineR4CNodalTracer();
      const params = {
        dtSeconds: 60.0,
        timesteps: 10,
        couplingMode: 'additive_sum',
        initialZoneTemperature: 22.0,
        surfaceExteriorTemperatures: {
          tExtWall: 0.0, tExtRoof: 0.0, tExtFloor: 0.0,
        },
        hTrIs: 10.0,
      };
      const a = tracer.runSubHourlyTrace(params);
      const b = tracer.runSubHourlyTrace(params);
      for (let i = 0; i < 10; i++) {
        assert.strictEqual(a.wall[i], b.wall[i]);
        assert.strictEqual(a.zone[i], b.zone[i]);
      }
    });

    it('should reject non-positive dtSeconds', () => {
      const tracer = new NineR4CNodalTracer();
      assert.throws(
        () => tracer.runSubHourlyTrace({
          dtSeconds: 0.0,
          timesteps: 4,
        }),
        /dt_seconds/,
      );
    });

    it('should reject mismatched gains length', () => {
      const tracer = new NineR4CNodalTracer();
      assert.throws(
        () => tracer.runSubHourlyTrace({
          dtSeconds: 60.0,
          timesteps: 10,
          gains: [[0, 0, 0, 0], [0, 0, 0, 0]], // length 2, expected 10
        }),
        /gains vector length/,
      );
    });

    it('should accept parallel-resistance coupling mode', () => {
      const tracer = new NineR4CNodalTracer();
      const trace = tracer.runSubHourlyTrace({
        dtSeconds: 60.0,
        timesteps: 5,
        couplingMode: 'parallel_resistance',
        initialZoneTemperature: 20.0,
        surfaceExteriorTemperatures: {
          tExtWall: 0.0, tExtRoof: 0.0, tExtFloor: 0.0,
        },
        hTrIs: 10.0,
      });
      assert.strictEqual(trace.couplingMode, 'parallel_resistance');
      assert.strictEqual(trace.zone.length, 5);
    });
  });

  describe('zero-copy matrix transfer (issue #1802)', () => {
    it('should expose transferMatrix', () => {
      assert.strictEqual(typeof fluxion.transferMatrix, 'function');
    });

    it('should preserve the typed array and backing buffer', () => {
      const matrix = new Float64Array([1.0, 2.0, 3.0, 4.0]);
      const transferred = fluxion.transferMatrix(matrix);

      assert.strictEqual(transferred, matrix);
      assert.strictEqual(transferred.buffer, matrix.buffer);
      assert.strictEqual(transferred.byteOffset, matrix.byteOffset);
      assert.strictEqual(transferred.byteLength, matrix.byteLength);
    });

    it('should preserve subarray offsets without copying', () => {
      const allocation = new Float64Array([0.0, 1.0, 2.0, 3.0, 4.0]);
      const matrix = allocation.subarray(1, 4);
      const transferred = fluxion.transferMatrix(matrix);

      assert.strictEqual(transferred, matrix);
      assert.strictEqual(transferred.buffer, allocation.buffer);
      assert.strictEqual(transferred.byteOffset, Float64Array.BYTES_PER_ELEMENT);
      assert.deepStrictEqual(Array.from(transferred), [1.0, 2.0, 3.0]);
    });
  });

  describe('HVAC configuration (issue #1798)', () => {
    let HvacVavTerminal, HvacCavSystem, HvacHeatPump, HvacChiller, HvacBoiler;
    let ZoneSetpoints, HvacDailySchedule, HvacSchedule, ZoneController;

    before(() => {
      HvacVavTerminal = fluxion.HvacVavTerminal;
      HvacCavSystem = fluxion.HvacCavSystem;
      HvacHeatPump = fluxion.HvacHeatPump;
      HvacChiller = fluxion.HvacChiller;
      HvacBoiler = fluxion.HvacBoiler;
      ZoneSetpoints = fluxion.ZoneSetpoints;
      HvacDailySchedule = fluxion.HvacDailySchedule;
      HvacSchedule = fluxion.HvacSchedule;
      ZoneController = fluxion.ZoneController;
    });

    it('should round-trip a VAV terminal system from Node', () => {
      // Acceptance criterion: build a VAV system from Node and round-trip it.
      const vav = new HvacVavTerminal('VAV-1', 0, 0.5);
      assert.strictEqual(vav.id, 'VAV-1');
      assert.strictEqual(vav.zoneId, 0);
      assert.strictEqual(vav.maxAirflow, 0.5);
      // min defaults to 30% of max
      assert.ok(Math.abs(vav.minAirflow - 0.15) < 1e-9);
      assert.ok(Math.abs(vav.reheatCapacity - 5000.0) < 1e-9);

      vav.reheatCapacity = 7500.0;
      vav.airflowSetpoint = 0.4;
      assert.ok(Math.abs(vav.reheatCapacity - 7500.0) < 1e-9);
      assert.ok(Math.abs(vav.airflowSetpoint - 0.4) < 1e-9);

      // Reheat delivered when zone is below comfort threshold.
      const demand = vav.reheatDemand(20.0, 18.0);
      assert.ok(demand > 0.0);
      assert.strictEqual(vav.reheatDemand(20.0, 22.0), 0.0);
    });

    it('should configure a CAV system', () => {
      const cav = new HvacCavSystem('CAV-1', 1.0);
      assert.strictEqual(cav.id, 'CAV-1');
      assert.strictEqual(cav.designAirflow, 1.0);
      cav.fanEfficiency = 0.8;
      assert.ok(Math.abs(cav.fanPowerConsumption() - 500.0 / 0.8) < 1e-6);
    });

    it('should configure a heat pump with mode selection', () => {
      const hp = new HvacHeatPump('HP-1', 12000.0, 10000.0, 3.5, 3.0);
      assert.strictEqual(hp.mode, 'off');
      // COP at design temp ~ rated
      assert.ok(Math.abs(hp.heatingCopAtTemperature(-5.0) - 3.5) < 0.1);
      hp.setMode(18.0, 20.0, 27.0);
      assert.strictEqual(hp.mode, 'heating');
      assert.ok(hp.heatingPower(-5.0) > 0.0);
      hp.setMode(28.0, 20.0, 27.0);
      assert.strictEqual(hp.mode, 'cooling');
      assert.ok(hp.coolingPower(35.0) > 0.0);
    });

    it('should expose chiller and boiler capacity curves', () => {
      const chiller = new HvacChiller('CH-1', 50000.0, 4.0, 35.0);
      assert.ok(Math.abs(chiller.ratedCapacity() - 50000.0) < 1e-9);
      const cap = chiller.calculateCapacity(1.0, 35.0);
      assert.ok(cap > 0.0);
      assert.ok(chiller.calculatePower(cap, 35.0, 'cooling') > 0.0);

      const boiler = new HvacBoiler('BL-1', 40000.0, 0.9, -5.0);
      assert.ok(Math.abs(boiler.ratedCapacity() - 40000.0) < 1e-9);
      // Boilers only heat: cooling mode => no power
      assert.strictEqual(boiler.calculatePower(boiler.calculateCapacity(1.0, -5.0), -5.0, 'cooling'), 0.0);
    });

    it('should manage zone setpoints and deadband', () => {
      const sp = new ZoneSetpoints(2);
      assert.strictEqual(sp.numZones, 2);
      sp.setHeatingSetpoint(0, 21.0);
      sp.setCoolingSetpoint(0, 25.0);
      assert.ok(Math.abs(sp.getHeatingSetpoint(0) - 21.0) < 1e-9);
      assert.ok(Math.abs(sp.getCoolingSetpoint(0) - 25.0) < 1e-9);
      assert.doesNotThrow(() => sp.validate());
      // Out-of-range temperature rejected
      assert.throws(() => sp.setHeatingSetpoint(0, 5.0));
      // Bad zone rejected
      assert.throws(() => sp.getDeadband(9));
      assert.throws(() => new ZoneSetpoints(0));
    });

    it('should build daily and HVAC schedules', () => {
      const ds = new HvacDailySchedule('occ', 'DailyCycle');
      ds.fillRange(8, 18, 21.0);
      assert.ok(Math.abs(ds.value(12) - 21.0) < 1e-9);
      assert.strictEqual(ds.value(2), 0.0);
      assert.strictEqual(ds.name, 'occ');
      assert.strictEqual(ds.scheduleType, 'DailyCycle');

      const constant = HvacDailySchedule.constant(24.0);
      assert.ok(Math.abs(constant.value(0) - 24.0) < 1e-9);

      assert.throws(() => new HvacDailySchedule('x', 'Bogus'));

      const sched = HvacSchedule.constantSchedule(20.0, 24.0);
      assert.strictEqual(sched.isFreeFloating(), false);
      assert.ok(Math.abs(sched.heatingSetpoint(5) - 20.0) < 1e-9);

      const setback = HvacSchedule.setbackSchedule(20.0, 15.0, 25.0, 22, 6);
      assert.ok(Math.abs(setback.heatingSetpoint(2) - 15.0) < 1e-9);
      assert.ok(Math.abs(setback.heatingSetpoint(10) - 20.0) < 1e-9);

      const occ = HvacSchedule.withOperatingHours(20.0, 24.0, 8, 18);
      assert.ok(Math.abs(occ.heatingSetpoint(2) - (-100.0)) < 1e-9);

      assert.strictEqual(HvacSchedule.freeFloating().isFreeFloating(), true);

      const heat = sched.getHeatingSchedule();
      assert.ok(Math.abs(heat.value(0) - 20.0) < 1e-9);
    });

    it('should select control strategies and report HVAC status', () => {
      const ctrl = new ZoneController(2);
      assert.strictEqual(ctrl.getZoneStrategy(0), 'ideal_loads');
      ctrl.setZoneStrategy(0, 'staged_equipment');
      assert.strictEqual(ctrl.getZoneStrategy(0), 'staged_equipment');
      ctrl.setZoneStrategy(0, 'schedule_aware');
      assert.strictEqual(ctrl.getZoneStrategy(0), 'schedule_aware');
      assert.throws(() => ctrl.setZoneStrategy(0, 'bogus'));

      const energy = ctrl.updateControls([15.0, 30.0]);
      assert.strictEqual(energy.length, 2);
      assert.strictEqual(ctrl.getZoneStatus(0), 'heating');
      assert.strictEqual(ctrl.getZoneStatus(1), 'cooling');

      assert.throws(() => new ZoneController(0));
    });
  });
});
