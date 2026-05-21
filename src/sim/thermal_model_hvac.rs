//! Thermal model HVAC integration.
//!
//! This module contains HVAC demand calculation logic that bridges
//! the thermal model with the HVAC system.
//!
//! ## Module Responsibilities
//! - HVAC demand calculation using IdealLoadsSystem thermodynamic formulas
//! - Free cooling capacity calculation
//! - Economizer logic integration
//!
//! ## Current Status
//! The hvac_demand_from_ideal_loads function is currently in thermal_model_physics.rs
//! due to tight coupling with ThermalModel internal state (access to ideal_loads_system,
//! hvac_enabled, hvac_heating_capacity, hvac_cooling_capacity). This module is a marker
//! for future extraction.
//!
//! ## Design Considerations for Future Extraction
//! 1. Requires access to ThermalModel's ideal_loads_system field
//! 2. The function uses zone-specific HVAC capacity limits
//! 3. The function signature matches VariableCapacityEquipment::calculate_power_demand_vector
//! 4. The physics (mass_flow × cp × ΔT) should be preserved exactly
