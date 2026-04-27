// Example usage of Fluxion native Node.js bindings
// Run with: node example.js

const { BatchOracle, BuildingParameters, ValidationError } = require('./index.js');

console.log('=== Fluxion Native Node.js Bindings Example ===\n');

// Create oracle instance
console.log('1. Creating BatchOracle instance...');
const oracle = new BatchOracle();
console.log('✓ BatchOracle created\n');

// Create building parameters
console.log('2. Creating BuildingParameters...');
const params = new BuildingParameters(1.5, 20.0, 24.0);
console.log(`✓ BuildingParameters created:`);
console.log(`   Window U-value: ${params.windowUValue} W/m²K`);
console.log(`   Heating setpoint: ${params.heatingSetpoint}°C`);
console.log(`   Cooling setpoint: ${params.coolingSetpoint}°C\n`);

// Convert to array
console.log('3. Converting parameters to array...');
const paramArray = params.toVec();
console.log(`✓ Array representation: ${paramArray}\n`);

// Validate parameters
console.log('4. Validating parameters...');
try {
  oracle.validateParameters(paramArray);
  console.log('✓ Parameters are valid\n');
} catch (error) {
  console.error(`✗ Validation failed: ${error.message}\n`);
}

// Test invalid parameters
console.log('5. Testing invalid parameters...');
try {
  const invalidParams = [6.0, 20.0, 24.0]; // U-value too high
  oracle.validateParameters(invalidParams);
  console.log('✗ Should have thrown ValidationError\n');
} catch (error) {
  console.log(`✓ Correctly rejected invalid params: ${error.message}\n`);
}

// Evaluate small population
console.log('6. Evaluating small population (physics-based)...');
const smallPopulation = [
  [1.5, 20.0, 24.0],
  [2.0, 20.0, 24.0],
  [2.5, 20.0, 24.0],
];

console.time('Physics evaluation');
const physicsResults = oracle.evaluatePopulation(smallPopulation, false);
console.timeEnd('Physics evaluation');

console.log('Physics-based EUI values:');
physicsResults.forEach((eui, i) => {
  console.log(`   Config ${i + 1}: ${eui.toFixed(2)} kWh/m²/yr`);
});

console.log('');

// Performance benchmark
console.log('7. Performance benchmark (1000 configs)...');
const largePopulation = Array.from({ length: 1000 }, () => [
  1.5 + Math.random() * 2.0,  // U-value: 1.5-3.5
  18.0 + Math.random() * 4.0,  // Heating: 18-22
  22.0 + Math.random() * 4.0,  // Cooling: 22-26
]);

console.time('1000 configs evaluation');
const benchmarkResults = oracle.evaluatePopulation(largePopulation, false);
console.timeEnd('1000 configs evaluation');

const avgEUI = benchmarkResults.reduce((a, b) => a + b, 0) / benchmarkResults.length;
const minEUI = Math.min(...benchmarkResults);
const maxEUI = Math.max(...benchmarkResults);

console.log(`Average EUI: ${avgEUI.toFixed(2)} kWh/m²/yr`);
console.log(`Min EUI: ${minEUI.toFixed(2)} kWh/m²/yr`);
console.log(`Max EUI: ${maxEUI.toFixed(2)} kWh/m²/yr`);
console.log('');

// Optimization example
console.log('8. Simple optimization example...');
const uValues = [1.5, 2.0, 2.5, 3.0, 3.5];
const heatingSetpoints = [18.0, 20.0, 22.0];
const coolingSetpoints = [22.0, 24.0, 26.0];

const optimizationPopulation = [];
for (const uValue of uValues) {
  for (const heating of heatingSetpoints) {
    for (const cooling of coolingSetpoints) {
      try {
        oracle.validateParameters([uValue, heating, cooling]);
        optimizationPopulation.push([uValue, heating, cooling]);
      } catch (error) {
        // Skip invalid combinations
      }
    }
  }
}

console.time('Optimization evaluation');
const optimizationResults = oracle.evaluatePopulation(optimizationPopulation, false);
console.timeEnd('Optimization evaluation');

const optimalIndex = optimizationResults.indexOf(Math.min(...optimizationResults));
const optimalParams = optimizationPopulation[optimalIndex];
const optimalEUI = optimizationResults[optimalIndex];

console.log(`\n✓ Optimal configuration found:`);
console.log(`   Window U-value: ${optimalParams[0]} W/m²K`);
console.log(`   Heating setpoint: ${optimalParams[1]}°C`);
console.log(`   Cooling setpoint: ${optimalParams[2]}°C`);
console.log(`   EUI: ${optimalEUI.toFixed(2)} kWh/m²/yr`);
console.log('\n=== Example completed successfully ===');
