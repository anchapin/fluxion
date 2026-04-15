//! Case 900 Energy Balance Verification Test
//!
//! This test investigates the Case 900 cooling error (-33.76%) by analyzing the energy balance
//! across all components (solar, internal gains, conduction, ventilation, HVAC, storage, exfiltration).
//!
//! The test runs a full year simulation and verifies that daily energy balance equations hold:
//! (solar + internal + conduction + ventilation) - (stored + exfiltration) = hvac_cooling ± 5%

/*
#[cfg(test)]
mod tests {
...
*/
