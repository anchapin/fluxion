// Test suite for Fluxion native Node.js bindings
// Run with: node --test test.js

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert');

// Note: These tests require the native module to be built first
// Run: npm run build before running tests

describe('@fluxion/native', () => {
  let BatchOracle, BuildingParameters, ValidationError;

  before(() => {
    // Load the module
    const fluxion = require('./index.js');
    BatchOracle = fluxion.BatchOracle;
    BuildingParameters = fluxion.BuildingParameters;
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
        (error) => error instanceof ValidationError
      );
    });

    it('should reject invalid U-value (too high)', () => {
      assert.throws(
        () => new BuildingParameters(6.0, 20.0, 24.0),
        (error) => error instanceof ValidationError
      );
    });

    it('should reject invalid heating setpoint (too low)', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 14.0, 24.0),
        (error) => error instanceof ValidationError
      );
    });

    it('should reject invalid heating setpoint (too high)', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 26.0, 24.0),
        (error) => error instanceof ValidationError
      );
    });

    it('should reject invalid cooling setpoint (too low)', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 20.0, 21.0),
        (error) => error instanceof ValidationError
      );
    });

    it('should reject invalid cooling setpoint (too high)', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 20.0, 33.0),
        (error) => error instanceof ValidationError
      );
    });

    it('should reject heating >= cooling', () => {
      assert.throws(
        () => new BuildingParameters(1.5, 24.0, 22.0),
        (error) => error instanceof ValidationError
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
        (error) => error instanceof ValidationError
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

      // Should complete in reasonable time (< 5 seconds for 100 configs)
      assert.ok(duration < 5000, `Evaluation took ${duration}ms, expected < 5000ms`);
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
});
