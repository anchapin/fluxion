//! Per-surface incident solar radiation tracking.
//!
//! Tracks annual incident solar energy (kWh/m²) and peak irradiance (W/m²)
//! for each surface. Per ASHRAE 140-2023 Section 8.2.3.

#[derive(Clone, Debug, Default)]
pub struct IncidentSolarAccumulator {
    pub annual_kwh_m2: f64,
    pub peak_wm2: f64,
}

impl IncidentSolarAccumulator {
    pub fn new() -> Self {
        Self {
            annual_kwh_m2: 0.0,
            peak_wm2: 0.0,
        }
    }

    pub fn accumulate(&mut self, irradiance_wm2: f64, _area_m2: f64, dt_seconds: f64) {
        self.annual_kwh_m2 += irradiance_wm2 * dt_seconds / 3_600_000.0;
        self.peak_wm2 = self.peak_wm2.max(irradiance_wm2);
    }
}
