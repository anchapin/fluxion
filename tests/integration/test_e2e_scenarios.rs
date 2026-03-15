//! End-to-end integration tests for full system workflows
//!
//! Tests validate complete workflows from input to output using real implementations
//! (not mocks) to catch wiring issues and integration bugs.

// TODO: Implement E2E scenario tests using fixtures from src/testing/integration/fixtures.rs
// Tests should cover:
// - Batch oracle throughput (population evaluation)
// - Python API (BatchOracle, Model classes)
// - CLI commands (validate, simulation)
// - Surrogate integration (AI surrogate calls)
// - HVAC equipment (VAV, CAV, HeatPump, Chiller, Boiler)
// - Psychrometrics (dew point, humidity ratio, enthalpy, wet-bulb)
// - Internal loads (lighting, equipment, occupancy with schedules)
// - Multi-zone physics (inter-zone conductance, zonal coupling)
