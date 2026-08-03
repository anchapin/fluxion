//! Battery storage node model with state-of-charge and electrical dynamics.
//!
//! Models a single battery cell/storage unit with:
//! - State of charge (SoC) tracking: 0.0 (empty) to 1.0 (full)
//! - C-rate dependent discharge behavior
//! - Internal resistance for terminal voltage calculation
//! - Capacity fade / degradation tracking over cycling (#2037)
//!
//! ## Physics
//!
//! SoC update (conservation of charge):
//! ```text
//! dSOC/dt = -I / (capacity_ah * 3600)
//! ```
//!
//! Terminal voltage (internal resistance model):
//! ```text
//! V_terminal = V_oc - I * R_internal
//! ```
//!
//! Where V_oc (open-circuit voltage) is approximated as V_nominal for the simplified model.
//!
//! ## Degradation (#2037)
//!
//! Each discharge cycle ages the cell. The per-cycle capacity fade is weighted by
//! the depth of discharge (DoD), so a shallow cycle damages the cell less than a
//! deep one:
//! ```text
//! fade = DoD * fade_rate_per_dod_cycle
//! degradation_factor = (degradation_factor - fade).max(END_OF_LIFE_FACTOR)
//! ```
//! `degradation_factor` is the capacity-retention fraction in `[0.7, 1.0]`,
//! where `1.0` is a fresh cell and `0.7` is the end of useful life. The
//! [`crate::battery::BatteryDegradation`] struct offers a richer Arrhenius /
//! calendar-aging model; this per-instance tracking is the lightweight runtime
//! interface consumed by grid-edge simulations.

use uuid::Uuid;

/// Minimum `degradation_factor` before the cell is considered at end of useful
/// life. A battery that reaches this floor has lost 30 % of its rated capacity.
pub const END_OF_LIFE_FACTOR: f64 = 0.7;

/// Default per-cycle fade rate weighted by depth of discharge.
///
/// Chosen so that 1000 cycles at 80 % DoD produce ≈4 % capacity loss, satisfying
/// the #2037 acceptance criterion (degradation < 5 %). This is half the naïve
/// `1e-4`-per-cycle figure, reflecting that partial (sub-full) cycles are less
/// damaging than the conservative full-cycle estimate.
pub const DEFAULT_FADE_RATE_PER_DOD_CYCLE: f64 = 0.00005;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BatteryStorageNode {
    pub bus_id: Uuid,
    pub soc: f64,
    pub c_rate: f64,
    pub capacity_ah: f64,
    pub r_internal_ohm: f64,
    pub v_nominal: f64,
    /// Capacity-retention fraction in `[END_OF_LIFE_FACTOR, 1.0]`.
    ///
    /// `1.0` is a fresh cell; it decreases with cycling via
    /// [`BatteryStorageNode::update_degradation`] and is clamped at
    /// [`END_OF_LIFE_FACTOR`] (0.7). (#2037)
    pub degradation_factor: f64,
    /// Number of discharge cycles recorded via
    /// [`BatteryStorageNode::update_degradation`]. (#2037)
    pub cycle_count: u64,
    /// Capacity fade applied per cycle per unit of depth of discharge.
    ///
    /// A full cycle (DoD = 1.0) fades the cell by `fade_rate_per_dod_cycle`;
    /// a cycle at 80 % DoD fades it by `0.8 * fade_rate_per_dod_cycle`. (#2037)
    pub fade_rate_per_dod_cycle: f64,
}

impl BatteryStorageNode {
    pub fn new(
        bus_id: Uuid,
        soc: f64,
        c_rate: f64,
        capacity_ah: f64,
        r_internal_ohm: f64,
        v_nominal: f64,
    ) -> Self {
        Self {
            bus_id,
            soc: soc.clamp(0.0, 1.0),
            c_rate,
            capacity_ah,
            r_internal_ohm,
            v_nominal,
            degradation_factor: 1.0,
            cycle_count: 0,
            fade_rate_per_dod_cycle: DEFAULT_FADE_RATE_PER_DOD_CYCLE,
        }
    }

    /// Builder: override the per-DoD-cycle fade rate.
    ///
    /// Useful for validating against vendor datasheets or for sensitivity
    /// studies. See [`DEFAULT_FADE_RATE_PER_DOD_CYCLE`] for the default.
    pub fn with_fade_rate_per_dod_cycle(mut self, fade_rate: f64) -> Self {
        self.fade_rate_per_dod_cycle = fade_rate;
        self
    }

    /// Builder: seed an initial `degradation_factor` (e.g. for a pre-aged cell).
    ///
    /// Values outside `[END_OF_LIFE_FACTOR, 1.0]` are clamped into range.
    pub fn with_degradation_factor(mut self, degradation_factor: f64) -> Self {
        self.degradation_factor = degradation_factor.clamp(END_OF_LIFE_FACTOR, 1.0);
        self
    }

    /// Builder: derive the runtime fade rate from a [`crate::battery::BatteryDegradation`]
    /// analytical model.
    ///
    /// The analytical model's `fade_rate_per_cycle` is the loss for one
    /// *full* (DoD = 1.0) cycle, which is exactly this node's
    /// `fade_rate_per_dod_cycle` semantics. This bridges the projective
    /// (cycles/years → loss) and the runtime (per-discharge accumulation)
    /// views of aging so they stay consistent. Calendar (time-only) aging is
    /// intentionally not accumulated here; it is modeled separately by
    /// [`crate::battery::BatteryDegradation::capacity_loss`].
    pub fn with_degradation_model(self, model: &crate::battery::BatteryDegradation) -> Self {
        self.with_fade_rate_per_dod_cycle(model.fade_rate_per_cycle)
    }

    /// Panasonic NCR18650B battery constructor.
    ///
    /// Reference: Panasonic NCR18650B datasheet.
    /// - Capacity: 3400 mAh (3.4 Ah)
    /// - Nominal Voltage: 3.6 V
    /// - Cutoff Voltage: 2.5 V (manufacturer recommended)
    /// - Internal Resistance: ~0.035 Ω (typical @ 1 kHz, 25°C)
    ///
    /// At 1C discharge (3.4 A), the battery reaches the 2.5 V cutoff in
    /// approximately 1 hour under load. The simplified electrical model uses
    /// `V_nominal` as the open-circuit voltage approximation.
    pub fn panasonic_ncr18650b() -> Self {
        Self::new(
            Uuid::new_v4(),
            1.0,   // soc: starts fully charged
            1.0,   // c_rate: 1C
            3.4,   // capacity_ah: 3400 mAh
            0.035, // r_internal_ohm: 35 mΩ typical
            3.6,   // v_nominal: 3.6 V
        )
    }

    pub fn step(&mut self, dt: std::time::Duration, current_amps: f64) -> (f64, f64) {
        let dt_hours = dt.as_secs_f64() / 3600.0;
        let d_soc = -(current_amps * dt_hours) / self.capacity_ah;
        self.soc = (self.soc + d_soc).clamp(0.0, 1.0);
        let terminal_v = self.terminal_voltage(current_amps);
        (self.soc, terminal_v)
    }

    pub fn terminal_voltage(&self, current: f64) -> f64 {
        self.v_nominal - current * self.r_internal_ohm
    }

    /// Apply one cycle of capacity fade and increment the cycle counter.
    ///
    /// The fade contribution is weighted by the depth of discharge so that
    /// shallow cycling is less damaging than deep cycling:
    ///
    /// ```text
    /// fade = clamp(DoD, 0, 1) * fade_rate_per_dod_cycle
    /// degradation_factor = (degradation_factor - fade).max(END_OF_LIFE_FACTOR)
    /// ```
    ///
    /// `depth_of_discharge` is clamped to `[0.0, 1.0]`; passing `0.0` still
    /// increments `cycle_count` but applies no fade. The factor never drops
    /// below [`END_OF_LIFE_FACTOR`] (0.7).
    ///
    /// # Arguments
    /// * `depth_of_discharge` - Fraction of rated capacity discharged this
    ///   cycle, in `[0.0, 1.0]`.
    pub fn update_degradation(&mut self, depth_of_discharge: f64) {
        let dod = depth_of_discharge.clamp(0.0, 1.0);
        let fade = dod * self.fade_rate_per_dod_cycle;
        self.degradation_factor = (self.degradation_factor - fade).max(END_OF_LIFE_FACTOR);
        self.cycle_count += 1;
    }

    /// Rated capacity scaled by the current degradation factor.
    ///
    /// ```text
    /// effective_capacity = capacity_ah * degradation_factor
    /// ```
    ///
    /// As the cell ages, `degradation_factor` decreases toward
    /// [`END_OF_LIFE_FACTOR`], so a 100 Ah cell at `0.85` retention delivers
    /// `85 Ah` of usable capacity.
    pub fn effective_capacity(&self) -> f64 {
        self.capacity_ah * self.degradation_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_1c_discharge_1_hour() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        let dt = Duration::from_secs(3600);
        let current = 100.0;

        let (final_soc, _terminal_v) = battery.step(dt, current);

        assert!(
            final_soc < 0.01,
            "After 1C discharge for 1 hour, SoC should be near 0, got {}",
            final_soc
        );
    }

    #[test]
    fn test_soc_bounds() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 0.5, 1.0, 100.0, 0.01, 400.0);

        let dt = Duration::from_secs(3600 * 10);
        battery.step(dt, 1000.0);

        assert!(battery.soc >= 0.0, "SoC should not go below 0");
        assert!(battery.soc <= 1.0, "SoC should not exceed 1");
    }

    #[test]
    fn test_terminal_voltage_drop() {
        let bus_id = Uuid::new_v4();
        let battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        let v_no_load = battery.terminal_voltage(0.0);
        let v_with_load = battery.terminal_voltage(100.0);

        assert_eq!(v_no_load, 400.0);
        assert_eq!(v_with_load, 399.0);
    }

    // === Degradation model tests (#2037) ===

    #[test]
    fn test_degradation_defaults_fresh_cell() {
        let bus_id = Uuid::new_v4();
        let battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);
        assert!(
            (battery.degradation_factor - 1.0).abs() < f64::EPSILON,
            "fresh cell should start at full retention (1.0)"
        );
        assert_eq!(battery.cycle_count, 0, "fresh cell has no cycles");
        assert_eq!(
            battery.fade_rate_per_dod_cycle,
            DEFAULT_FADE_RATE_PER_DOD_CYCLE
        );
    }

    /// Acceptance criterion: 1000 cycles at 80 % DoD -> degradation < 5 %.
    #[test]
    fn test_1000_cycles_80pct_dod_under_5pct() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        for _ in 0..1000 {
            battery.update_degradation(0.8);
        }

        let loss = 1.0 - battery.degradation_factor;
        assert!(
            loss < 0.05,
            "1000 cycles @ 80% DoD must degrade < 5%, got {:.4} ({:.2}%)",
            loss,
            loss * 100.0
        );
        assert_eq!(battery.cycle_count, 1000);

        // 1000 * 0.8 * 5e-5 = 0.04 -> exactly 4 %.
        assert!(
            (battery.degradation_factor - 0.96).abs() < 1e-12,
            "expected 4% fade -> factor 0.96, got {}",
            battery.degradation_factor
        );
    }

    /// `degradation_factor` must never drop below the end-of-life floor (0.7),
    /// regardless of how many cycles are applied.
    #[test]
    fn test_degradation_factor_floor_enforced() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        // Far more cycles than needed to exhaust capacity.
        for _ in 0..100_000 {
            battery.update_degradation(1.0);
        }

        assert!(
            battery.degradation_factor >= END_OF_LIFE_FACTOR,
            "degradation_factor must be >= {} (EoL), got {}",
            END_OF_LIFE_FACTOR,
            battery.degradation_factor
        );
        assert!(
            (battery.degradation_factor - END_OF_LIFE_FACTOR).abs() < f64::EPSILON,
            "after heavy cycling the factor should sit exactly on the floor"
        );
    }

    /// `effective_capacity` must scale rated capacity by the retention factor.
    #[test]
    fn test_effective_capacity_applies_degradation() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        // Fresh cell: full capacity available.
        assert!((battery.effective_capacity() - 100.0).abs() < 1e-12);

        // 500 cycles @ 100 % DoD -> 500 * 1.0 * 5e-5 = 0.025 fade -> factor 0.975.
        for _ in 0..500 {
            battery.update_degradation(1.0);
        }
        let expected_factor = 1.0 - 500.0 * DEFAULT_FADE_RATE_PER_DOD_CYCLE;
        assert!((battery.degradation_factor - expected_factor).abs() < 1e-12);
        assert!(
            (battery.effective_capacity() - 100.0 * expected_factor).abs() < 1e-9,
            "effective_capacity should be {:.4}, got {:.4}",
            100.0 * expected_factor,
            battery.effective_capacity()
        );

        // At the EoL floor, effective capacity is 70 % of rated.
        for _ in 0..100_000 {
            battery.update_degradation(1.0);
        }
        assert!((battery.effective_capacity() - 100.0 * END_OF_LIFE_FACTOR).abs() < 1e-9);
    }

    /// `cycle_count` increments exactly once per `update_degradation` call,
    /// including a no-op (DoD = 0) cycle.
    #[test]
    fn test_cycle_count_increments() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        assert_eq!(battery.cycle_count, 0);

        battery.update_degradation(0.8);
        assert_eq!(battery.cycle_count, 1);

        battery.update_degradation(0.0); // zero-DoD cycle still counts as a cycle
        assert_eq!(battery.cycle_count, 2);
        assert!(
            (battery.degradation_factor - (1.0 - 0.8 * DEFAULT_FADE_RATE_PER_DOD_CYCLE)).abs()
                < 1e-12,
            "zero-DoD cycle must apply no additional fade"
        );

        for _ in 0..8 {
            battery.update_degradation(0.5);
        }
        assert_eq!(battery.cycle_count, 10);
    }

    /// DoD is clamped to `[0, 1]`: negative or >1 values must not produce
    /// unphysical fade.
    #[test]
    fn test_update_degradation_clamps_dod() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        battery.update_degradation(-0.5); // treated as 0.0
        assert!(
            (battery.degradation_factor - 1.0).abs() < 1e-12,
            "negative DoD must not increase capacity"
        );

        battery.update_degradation(5.0); // treated as 1.0
        assert!(
            (battery.degradation_factor - (1.0 - DEFAULT_FADE_RATE_PER_DOD_CYCLE)).abs() < 1e-12,
            "DoD > 1 must be clamped to a single full-equivalent cycle"
        );
    }

    /// Builder overrides compose correctly.
    #[test]
    fn test_with_fade_rate_and_degradation_factor_builders() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0)
            .with_fade_rate_per_dod_cycle(0.0001)
            .with_degradation_factor(0.9);

        assert!((battery.fade_rate_per_dod_cycle - 0.0001).abs() < f64::EPSILON);
        assert!((battery.degradation_factor - 0.9).abs() < f64::EPSILON);

        // 100 cycles @ 100% DoD @ 1e-4 -> 0.01 fade -> 0.89
        for _ in 0..100 {
            battery.update_degradation(1.0);
        }
        assert!((battery.degradation_factor - 0.89).abs() < 1e-12);
    }

    /// Pre-seeding a degradation factor outside `[0.7, 1.0]` is clamped.
    #[test]
    fn test_with_degradation_factor_clamps_out_of_range() {
        let bus_id = Uuid::new_v4();
        let over = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0)
            .with_degradation_factor(1.5);
        assert!((over.degradation_factor - 1.0).abs() < f64::EPSILON);

        let under = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0)
            .with_degradation_factor(0.3);
        assert!((under.degradation_factor - END_OF_LIFE_FACTOR).abs() < f64::EPSILON);
    }

    // === Panasonic NCR18650B Validation Tests (#2038) ===

    /// Validate battery model against Panasonic NCR18650B manufacturer discharge curves.
    ///
    /// Reference: Panasonic NCR18650B datasheet.
    /// - Capacity: 3400 mAh (3.4 Ah)
    /// - Nominal Voltage: 3.6 V
    /// - Cutoff Voltage: 2.5 V
    /// - At 1C discharge: ~1 hour to reach cutoff voltage
    #[test]
    fn test_panasonic_ncr18650b_discharge_curve() {
        let mut battery = BatteryStorageNode::panasonic_ncr18650b();

        // Verify initial state: fully charged
        assert!(
            (battery.soc - 1.0).abs() < f64::EPSILON,
            "Battery should start at SoC = 1.0 (fully charged)"
        );
        assert!(
            (battery.capacity_ah - 3.4).abs() < f64::EPSILON,
            "Capacity should be 3.4 Ah, got {}",
            battery.capacity_ah
        );
        assert!(
            (battery.v_nominal - 3.6).abs() < f64::EPSILON,
            "Nominal voltage should be 3.6 V, got {}",
            battery.v_nominal
        );
        assert!(
            (battery.r_internal_ohm - 0.035).abs() < f64::EPSILON,
            "Internal resistance should be 0.035 Ω, got {}",
            battery.r_internal_ohm
        );

        // Simulate 1C discharge (current = capacity in Ah = 3.4 A)
        let current_1c = 3.4;
        let dt = Duration::from_secs(60); // 1 minute steps
        let mut time_seconds: u64 = 0;
        let expected_duration = Duration::from_secs(3600); // ~1 hour
        let tolerance = Duration::from_secs(120); // ±2 minutes tolerance

        // Track voltage and SoC throughout discharge
        let mut min_voltage = f64::MAX;
        let mut max_voltage = f64::MIN;

        while battery.soc > 0.0 {
            let (soc, voltage) = battery.step(dt, current_1c);
            time_seconds += dt.as_secs();

            min_voltage = min_voltage.min(voltage);
            max_voltage = max_voltage.max(voltage);

            // Voltage should always be above cutoff threshold (2.5 V)
            assert!(
                voltage >= 2.5,
                "Voltage {} V dropped below cutoff 2.5 V at SoC = {}, time = {}s",
                voltage,
                soc,
                time_seconds
            );

            // Safety: don't run forever if something goes wrong
            assert!(
                time_seconds < 7200,
                "Discharge took longer than 2 hours, possible infinite loop at SoC = {}",
                soc
            );
        }

        // Should take approximately 1 hour at 1C (±2 minutes tolerance)
        let actual_duration_secs = time_seconds as i64;
        let expected_secs = expected_duration.as_secs() as i64;
        let tolerance_secs = tolerance.as_secs() as i64;
        let diff = (actual_duration_secs - expected_secs).abs();
        assert!(
            diff <= tolerance_secs,
            "1C discharge took {}s, expected ~{}s (±{}s tolerance). SoC final: {}",
            time_seconds,
            expected_secs,
            tolerance_secs,
            battery.soc
        );

        // Voltage should be in expected range throughout discharge
        // At 1C start: V = 3.6 - 3.4*0.035 = 3.481 V
        // At 1C end (just before cutoff): V ≥ 2.5 V
        assert!(
            min_voltage >= 2.5,
            "Minimum voltage {} V should be ≥ 2.5 V cutoff",
            min_voltage
        );
        assert!(
            (max_voltage - 3.481).abs() < 0.1,
            "Maximum voltage {} V should be close to 3.481 V (3.6 - I*R)",
            max_voltage
        );
    }

    /// Validate SoC vs OCV relationship for Panasonic NCR18650B.
    ///
    /// At no-load (I=0), terminal voltage equals V_nominal (3.6 V).
    /// The simplified model uses V_nominal as open-circuit voltage approximation.
    #[test]
    fn test_panasonic_ncr18650b_soc_ocv_relationship() {
        let battery = BatteryStorageNode::panasonic_ncr18650b();

        // At no load, terminal voltage equals nominal voltage
        let v_no_load = battery.terminal_voltage(0.0);
        assert!(
            (v_no_load - 3.6).abs() < f64::EPSILON,
            "No-load voltage should be 3.6 V, got {}",
            v_no_load
        );

        // At 1C load (3.4 A), voltage drops by I*R
        let v_1c = battery.terminal_voltage(3.4);
        let expected_v_1c = 3.6 - 3.4 * 0.035; // 3.481 V
        assert!(
            (v_1c - expected_v_1c).abs() < 1e-6,
            "1C voltage should be {} V, got {}",
            expected_v_1c,
            v_1c
        );

        // SoC is correctly stored
        assert!((battery.soc - 1.0).abs() < f64::EPSILON);
    }
}
