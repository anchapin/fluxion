//! Solar state — gains, windows, weather, per-zone wall surfaces.
//!
//! Extracted from `ThermalModelData` (Issue #2878). Owns the per-zone solar
//! gains, the per-zone window U-value/properties/orientations, the weather
//! and site-location inputs, the solar-position cache, and the
//! `Vec<Vec<WallSurface>>` geometry used by the solar/distribution hot loops.
//! The Clone impl drops the per-step `sun_pos_cache` (Issue #1970) — a fresh
//! cache is built on the first step after clone, exactly like the legacy
//! `ThermalModelData::clone` path.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::construction::WallSurface;
use crate::sim::solar::{SolarPosition, WindowProperties};
use crate::weather::HourlyWeatherData;
use fluxion_core::ashrae_cases::Orientation;
use std::collections::HashMap;

pub struct SolarState<T: ContinuousTensor<f64>> {
    // Per-zone solar gains (W).
    pub solar_gains: T,
    pub opaque_solar_gains: T,

    // Window properties.
    pub window_u_value: f64,
    pub window_properties: Vec<WindowProperties>,
    pub window_orientations: Vec<Vec<Orientation>>,

    // Solar distribution fractions.
    pub solar_distribution_to_air: f64,
    pub solar_beam_to_mass_fraction: f64,
    pub convective_fraction: f64,

    // Weather + location.
    pub weather: Option<HourlyWeatherData>,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    /// Issue #1416: explicit EPW LOCATION time-zone offset (decimal hours).
    pub utc_offset_hours: Option<f64>,

    /// Issue #1212 — solar position cache keyed by `(timestep, hour_slot)`.
    pub sun_pos_cache: HashMap<(usize, i32), SolarPosition>,
    /// Issue #1968 — cached zero vector to eliminate per-timestep allocations.
    pub zero_vector: VectorField,

    /// Per-zone wall surfaces (orientation, area, U-value) used by the
    /// solar/distribution hot loops.
    pub surfaces: Vec<Vec<WallSurface>>,

    /// Internal radiative heat gains to thermal mass (Plan 17-04).
    pub internal_radiative_to_mass: f64,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for SolarState<T> {
    fn clone(&self) -> Self {
        Self {
            solar_gains: self.solar_gains.clone(),
            opaque_solar_gains: self.opaque_solar_gains.clone(),

            window_u_value: self.window_u_value,
            window_properties: self.window_properties.clone(),
            window_orientations: self.window_orientations.clone(),

            solar_distribution_to_air: self.solar_distribution_to_air,
            solar_beam_to_mass_fraction: self.solar_beam_to_mass_fraction,
            convective_fraction: self.convective_fraction,

            weather: self.weather.clone(),
            latitude_deg: self.latitude_deg,
            longitude_deg: self.longitude_deg,
            utc_offset_hours: self.utc_offset_hours,

            // Issue #1970 — sun_pos_cache is per-step scratch; drop on clone.
            sun_pos_cache: HashMap::new(),
            zero_vector: self.zero_vector.clone(),

            surfaces: self.surfaces.clone(),
            internal_radiative_to_mass: self.internal_radiative_to_mass,
        }
    }
}
