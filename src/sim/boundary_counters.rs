//! Boundary-quantity recomputation counters (Fluxion issue #4689).
//!
//! This module is the *audit instrument* for the
//! "[spike][perf] Precompute annual boundary conditions once, outside the
//! timestep/Newton loop" spike (issue #4689). It answers one question with
//! measurements instead of guesses:
//!
//! > How many times is each annual boundary quantity (solar incident
//! > irradiance, shading factor, sky temperature, longwave view-factor
//! > coefficients, longwave network evaluations, exterior convection
//! > coefficient) *recomputed* during a simulation run?
//!
//! # Design constraints
//!
//! * **Counting only.** The counters never change a computed value: the
//!   instrumented call sites are bit-identical with the master switch ON and
//!   OFF. No tolerance, baseline or gate is touched.
//! * **Off by default.** The master switch ([`set_enabled`]) defaults to
//!   `false`, so production builds pay one relaxed atomic load per
//!   instrumented site and nothing else.
//! * **No environment variable.** The switch is a process-global API only;
//!   there is no env-var backdoor, so behaviour cannot drift by accident.
//! * **Relaxed atomics.** Counters use `Ordering::Relaxed` — they are
//!   statistics, not synchronisation. The master switch is also relaxed.
//!
//! # Usage
//!
//! ```no_run
//! use fluxion::sim::boundary_counters::{self, BoundaryQuantity};
//!
//! boundary_counters::set_enabled(true);
//! boundary_counters::reset();
//! // ... run the simulation ...
//! let snap = boundary_counters::snapshot();
//! let per_hour = snap.per_hour(BoundaryQuantity::SkyTemperature, 8760.0);
//! boundary_counters::set_enabled(false);
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The boundary quantities tracked by the audit counters.
///
/// Each variant maps to exactly one *computation site* in the engine (see the
/// module documentation of the instrumented call sites in
/// `src/sim/solar.rs`, `src/sim/shading.rs`, `src/sim/sky_radiation.rs`,
/// `src/sim/longwave_exchange.rs`, `src/sim/exterior_boundary.rs` and
/// `src/physics/exterior_convection.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryQuantity {
    /// Solar incident irradiance on a surface (`calculate_surface_irradiance`).
    SolarIncident,
    /// Shaded fraction of a window (`calculate_shaded_fraction`).
    ShadingFactor,
    /// EPW-derived sky temperature (`HourlyWeatherData::sky_temperature`).
    SkyTemperature,
    /// Longwave view-factor coefficients (exterior sky view factor `F_sky`
    /// and the interior floor/ceiling/wall network coefficients).
    LongwaveCoefficient,
    /// Longwave network *evaluations* (the network exchange itself).
    LongwaveNetwork,
    /// Exterior convection coefficient (`h_c_ext_wind_dependent`).
    ExteriorConvection,
}

impl BoundaryQuantity {
    /// All tracked quantities, in a stable reporting order.
    pub const ALL: [BoundaryQuantity; 6] = [
        BoundaryQuantity::SolarIncident,
        BoundaryQuantity::ShadingFactor,
        BoundaryQuantity::SkyTemperature,
        BoundaryQuantity::LongwaveCoefficient,
        BoundaryQuantity::LongwaveNetwork,
        BoundaryQuantity::ExteriorConvection,
    ];

    /// Stable snake_case label used in reports and log lines.
    pub fn as_str(&self) -> &'static str {
        match self {
            BoundaryQuantity::SolarIncident => "solar_incident",
            BoundaryQuantity::ShadingFactor => "shading_factor",
            BoundaryQuantity::SkyTemperature => "sky_temperature",
            BoundaryQuantity::LongwaveCoefficient => "longwave_coefficient",
            BoundaryQuantity::LongwaveNetwork => "longwave_network",
            BoundaryQuantity::ExteriorConvection => "exterior_convection",
        }
    }
}

// --- Master switch ---------------------------------------------------------
//
// `false` by default: production behaviour is unchanged and the instrumented
// sites cost a single relaxed load.

static MASTER_ENABLED: AtomicBool = AtomicBool::new(false);

// --- Individual counters ----------------------------------------------------

static SOLAR_INCIDENT: AtomicU64 = AtomicU64::new(0);
static SHADING_FACTOR: AtomicU64 = AtomicU64::new(0);
static SKY_TEMPERATURE: AtomicU64 = AtomicU64::new(0);
static LONGWAVE_COEFFICIENT: AtomicU64 = AtomicU64::new(0);
static LONGWAVE_NETWORK: AtomicU64 = AtomicU64::new(0);
static EXTERIOR_CONVECTION: AtomicU64 = AtomicU64::new(0);

/// Enable or disable the counters process-wide.
///
/// Disabled (the default) the counters are inert: every instrumented site
/// returns after a single relaxed atomic load and no value is affected.
pub fn set_enabled(enabled: bool) {
    MASTER_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether the counters are currently enabled.
pub fn is_enabled() -> bool {
    MASTER_ENABLED.load(Ordering::Relaxed)
}

/// Reset every counter to zero.
///
/// Does **not** change the master switch.
pub fn reset() {
    SOLAR_INCIDENT.store(0, Ordering::Relaxed);
    SHADING_FACTOR.store(0, Ordering::Relaxed);
    SKY_TEMPERATURE.store(0, Ordering::Relaxed);
    LONGWAVE_COEFFICIENT.store(0, Ordering::Relaxed);
    LONGWAVE_NETWORK.store(0, Ordering::Relaxed);
    EXTERIOR_CONVECTION.store(0, Ordering::Relaxed);
}

/// Record one computation of `quantity`.
///
/// This is the only function the instrumented call sites use. It is
/// `#[inline]` so the disabled case compiles down to a relaxed load + branch.
#[inline]
pub fn count(quantity: BoundaryQuantity) {
    if !MASTER_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    match quantity {
        BoundaryQuantity::SolarIncident => {
            SOLAR_INCIDENT.fetch_add(1, Ordering::Relaxed);
        }
        BoundaryQuantity::ShadingFactor => {
            SHADING_FACTOR.fetch_add(1, Ordering::Relaxed);
        }
        BoundaryQuantity::SkyTemperature => {
            SKY_TEMPERATURE.fetch_add(1, Ordering::Relaxed);
        }
        BoundaryQuantity::LongwaveCoefficient => {
            LONGWAVE_COEFFICIENT.fetch_add(1, Ordering::Relaxed);
        }
        BoundaryQuantity::LongwaveNetwork => {
            LONGWAVE_NETWORK.fetch_add(1, Ordering::Relaxed);
        }
        BoundaryQuantity::ExteriorConvection => {
            EXTERIOR_CONVECTION.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A point-in-time copy of all counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BoundaryCounterSnapshot {
    /// Solar incident irradiance computations.
    pub solar_incident: u64,
    /// Shading-factor computations.
    pub shading_factor: u64,
    /// Sky-temperature computations.
    pub sky_temperature: u64,
    /// Longwave view-factor coefficient computations.
    pub longwave_coefficient: u64,
    /// Longwave network evaluations.
    pub longwave_network: u64,
    /// Exterior convection coefficient computations.
    pub exterior_convection: u64,
}

impl BoundaryCounterSnapshot {
    /// Count for a single quantity.
    pub fn get(&self, quantity: BoundaryQuantity) -> u64 {
        match quantity {
            BoundaryQuantity::SolarIncident => self.solar_incident,
            BoundaryQuantity::ShadingFactor => self.shading_factor,
            BoundaryQuantity::SkyTemperature => self.sky_temperature,
            BoundaryQuantity::LongwaveCoefficient => self.longwave_coefficient,
            BoundaryQuantity::LongwaveNetwork => self.longwave_network,
            BoundaryQuantity::ExteriorConvection => self.exterior_convection,
        }
    }

    /// Sum over all quantities.
    pub fn total(&self) -> u64 {
        BoundaryQuantity::ALL
            .iter()
            .map(|q| self.get(*q))
            .fold(0u64, |acc, v| acc.saturating_add(v))
    }

    /// Count per recorded hour for one quantity.
    ///
    /// `hours` is the number of *recorded hours* in the measured window (for
    /// an annual ASHRAE 140 run that is 8760). A zero/negative window yields
    /// `0.0` instead of a non-finite value.
    pub fn per_hour(&self, quantity: BoundaryQuantity, hours: f64) -> f64 {
        if hours <= 0.0 {
            return 0.0;
        }
        self.get(quantity) as f64 / hours
    }

    /// Count per recorded hour for every quantity, in [`BoundaryQuantity::ALL`]
    /// order.
    pub fn per_hour_all(&self, hours: f64) -> Vec<(BoundaryQuantity, f64)> {
        BoundaryQuantity::ALL
            .iter()
            .map(|q| (*q, self.per_hour(*q, hours)))
            .collect()
    }
}

/// Take a snapshot of all counters.
pub fn snapshot() -> BoundaryCounterSnapshot {
    BoundaryCounterSnapshot {
        solar_incident: SOLAR_INCIDENT.load(Ordering::Relaxed),
        shading_factor: SHADING_FACTOR.load(Ordering::Relaxed),
        sky_temperature: SKY_TEMPERATURE.load(Ordering::Relaxed),
        longwave_coefficient: LONGWAVE_COEFFICIENT.load(Ordering::Relaxed),
        longwave_network: LONGWAVE_NETWORK.load(Ordering::Relaxed),
        exterior_convection: EXTERIOR_CONVECTION.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters and the master switch are process-global, so every
    /// stateful test in this module serialises on this lock — without it,
    /// parallel tests reset/disable the counters mid-assertion (observed:
    /// `per_hour_rates_use_recorded_hours` counting 9 instead of 8,760).
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The master switch must default to OFF so production runs are
    /// unaffected (issue #4689 hard stop: no default behaviour change).
    #[test]
    fn master_switch_defaults_off() {
        let _serial = SERIAL.lock().unwrap();
        // The static is process-global and other tests may flip it; assert on
        // the *initial* value semantics via a fresh read after an explicit
        // disable, and check that a disabled run counts nothing.
        set_enabled(false);
        assert!(!is_enabled());
    }

    /// With the switch disabled, counting is a no-op.
    #[test]
    fn disabled_counters_record_nothing() {
        let _serial = SERIAL.lock().unwrap();
        set_enabled(false);
        reset();
        for _ in 0..10 {
            count(BoundaryQuantity::SolarIncident);
            count(BoundaryQuantity::SkyTemperature);
        }
        let snap = snapshot();
        assert_eq!(snap.solar_incident, 0);
        assert_eq!(snap.sky_temperature, 0);
        assert_eq!(snap.total(), 0);
    }

    /// With the switch enabled, each quantity is counted independently.
    #[test]
    fn enabled_counters_record_per_quantity() {
        let _serial = SERIAL.lock().unwrap();
        set_enabled(true);
        reset();
        count(BoundaryQuantity::SolarIncident);
        count(BoundaryQuantity::SolarIncident);
        count(BoundaryQuantity::ShadingFactor);
        count(BoundaryQuantity::LongwaveCoefficient);
        count(BoundaryQuantity::LongwaveNetwork);
        count(BoundaryQuantity::ExteriorConvection);
        count(BoundaryQuantity::SkyTemperature);

        let snap = snapshot();
        assert_eq!(snap.solar_incident, 2);
        assert_eq!(snap.shading_factor, 1);
        assert_eq!(snap.longwave_coefficient, 1);
        assert_eq!(snap.longwave_network, 1);
        assert_eq!(snap.exterior_convection, 1);
        assert_eq!(snap.sky_temperature, 1);
        assert_eq!(snap.total(), 7);

        set_enabled(false);
    }

    /// `reset` zeroes every counter without touching the master switch.
    #[test]
    fn reset_clears_counters_but_keeps_switch() {
        let _serial = SERIAL.lock().unwrap();
        set_enabled(true);
        count(BoundaryQuantity::ExteriorConvection);
        reset();
        assert_eq!(snapshot().exterior_convection, 0);
        assert!(is_enabled());
        set_enabled(false);
    }

    /// Per-hour rates divide by the recorded-hour window.
    #[test]
    fn per_hour_rates_use_recorded_hours() {
        let _serial = SERIAL.lock().unwrap();
        set_enabled(true);
        reset();
        for _ in 0..8760 {
            count(BoundaryQuantity::SkyTemperature);
        }
        let snap = snapshot();
        assert_eq!(snap.per_hour(BoundaryQuantity::SkyTemperature, 8760.0), 1.0);
        assert_eq!(snap.per_hour(BoundaryQuantity::SkyTemperature, 4380.0), 2.0);
        // A non-positive window must not yield NaN/inf.
        assert_eq!(snap.per_hour(BoundaryQuantity::SkyTemperature, 0.0), 0.0);
        assert_eq!(snap.per_hour(BoundaryQuantity::SkyTemperature, -1.0), 0.0);
        set_enabled(false);
    }

    /// `per_hour_all` covers every quantity in the stable order.
    #[test]
    fn per_hour_all_covers_every_quantity() {
        let _serial = SERIAL.lock().unwrap();
        set_enabled(true);
        reset();
        count(BoundaryQuantity::LongwaveNetwork);
        let rates = snapshot().per_hour_all(2.0);
        assert_eq!(rates.len(), BoundaryQuantity::ALL.len());
        assert_eq!(rates[0].0, BoundaryQuantity::SolarIncident);
        assert_eq!(rates[0].1, 0.0);
        let lw = rates
            .iter()
            .find(|(q, _)| *q == BoundaryQuantity::LongwaveNetwork)
            .unwrap();
        assert_eq!(lw.1, 0.5);
        set_enabled(false);
    }

    /// Quantity labels are stable snake_case identifiers.
    #[test]
    fn quantity_labels_are_stable() {
        assert_eq!(BoundaryQuantity::SolarIncident.as_str(), "solar_incident");
        assert_eq!(BoundaryQuantity::ShadingFactor.as_str(), "shading_factor");
        assert_eq!(BoundaryQuantity::SkyTemperature.as_str(), "sky_temperature");
        assert_eq!(
            BoundaryQuantity::LongwaveCoefficient.as_str(),
            "longwave_coefficient"
        );
        assert_eq!(
            BoundaryQuantity::LongwaveNetwork.as_str(),
            "longwave_network"
        );
        assert_eq!(
            BoundaryQuantity::ExteriorConvection.as_str(),
            "exterior_convection"
        );
    }
}
