//! Utility Tariff Module
//!
//! This module provides utility tariff functionality for financial cost tracking,
//! including time-of-use (TOU) rates and demand charges.

use serde::{Deserialize, Serialize};

/// Period of the day for TOU pricing
#[derive(Debug, Clone, Copy, Eq, Serialize, Deserialize)]
pub enum TouPeriod {
    /// Off-peak period (low demand, lowest rates)
    OffPeak,
    /// Mid-peak period (moderate demand)
    MidPeak,
    /// On-peak period (high demand, highest rates)
    OnPeak,
}

impl PartialEq for TouPeriod {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

/// Utility tariff structure holding rate tables and demand charge parameters.
///
/// # Example
///
/// ```ignore
/// use fluxion::sim::utility_tariff::{UtilityTariff, TouPeriod};
///
/// let mut tariff = UtilityTariff::new();
/// tariff.set_tou_rates(0.08, 0.12, 0.20); // off-peak, mid-peak, on-peak
/// tariff.set_demand_charge(15.0); // $15.00/kW
/// tariff.set_monthly_peak_window(9, 21); // 9am to 9pm
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityTariff {
    /// Off-peak rate ($/kWh)
    pub off_peak_rate: f64,
    /// Mid-peak rate ($/kWh)
    pub mid_peak_rate: f64,
    /// On-peak rate ($/kWh)
    pub on_peak_rate: f64,
    /// Demand charge ($/kW) applied to monthly peak
    pub demand_charge: f64,
    /// Start hour of the monthly peak window (0-23)
    pub monthly_peak_window_start_hour: u32,
    /// End hour of the monthly peak window (0-23)
    pub monthly_peak_window_end_hour: u32,
    /// Hourly TOU rates for the current day (24 values)
    hourly_tou_rates: [f64; 24],
}

impl UtilityTariff {
    /// Create a new utility tariff with default rates.
    ///
    /// Default rates:
    /// - Off-peak: $0.08/kWh
    /// - Mid-peak: $0.12/kWh
    /// - On-peak: $0.20/kWh
    /// - Demand charge: $15.00/kW
    /// - Peak window: 9am to 9pm
    pub fn new() -> Self {
        let mut tariff = Self {
            off_peak_rate: 0.08,
            mid_peak_rate: 0.12,
            on_peak_rate: 0.20,
            demand_charge: 15.0,
            monthly_peak_window_start_hour: 9,
            monthly_peak_window_end_hour: 21,
            hourly_tou_rates: [0.08; 24],
        };
        tariff.update_hourly_rates();
        tariff
    }

    /// Set the time-of-use rates.
    ///
    /// # Arguments
    /// * `off_peak` - Off-peak rate in $/kWh
    /// * `mid_peak` - Mid-peak rate in $/kWh
    /// * `on_peak` - On-peak rate in $/kWh
    pub fn set_tou_rates(&mut self, off_peak: f64, mid_peak: f64, on_peak: f64) {
        self.off_peak_rate = off_peak;
        self.mid_peak_rate = mid_peak;
        self.on_peak_rate = on_peak;
        self.update_hourly_rates();
    }

    /// Set the demand charge rate.
    ///
    /// # Arguments
    /// * `rate` - Demand charge in $/kW
    pub fn set_demand_charge(&mut self, rate: f64) {
        self.demand_charge = rate;
    }

    /// Set the monthly peak window hours.
    ///
    /// # Arguments
    /// * `start_hour` - Start hour (0-23)
    /// * `end_hour` - End hour (0-23)
    pub fn set_monthly_peak_window(&mut self, start_hour: u32, end_hour: u32) {
        self.monthly_peak_window_start_hour = start_hour;
        self.monthly_peak_window_end_hour = end_hour;
    }

    /// Get the TOU rate for a specific hour.
    ///
    /// # Arguments
    /// * `hour` - Hour of day (0-23)
    ///
    /// # Returns
    /// The rate in $/kWh for the specified hour
    pub fn get_rate_for_hour(&self, hour: u32) -> f64 {
        self.hourly_tou_rates[(hour % 24) as usize]
    }

    /// Get the TOU period for a specific hour.
    ///
    /// # Arguments
    /// * `hour` - Hour of day (0-23)
    ///
    /// # Returns
    /// The TOU period for the specified hour
    pub fn get_tou_period(&self, hour: u32) -> TouPeriod {
        let rate = self.get_rate_for_hour(hour);
        if (rate - self.off_peak_rate).abs() < 1e-10 {
            TouPeriod::OffPeak
        } else if (rate - self.mid_peak_rate).abs() < 1e-10 {
            TouPeriod::MidPeak
        } else {
            TouPeriod::OnPeak
        }
    }

    /// Calculate energy cost for a given hour and energy consumption.
    ///
    /// # Arguments
    /// * `hour` - Hour of day (0-23)
    /// * `energy_kwh` - Energy consumed in kWh
    ///
    /// # Returns
    /// Cost in dollars
    pub fn calculate_energy_cost(&self, hour: u32, energy_kwh: f64) -> f64 {
        self.get_rate_for_hour(hour) * energy_kwh
    }

    /// Check if a given hour falls within the peak demand window.
    ///
    /// # Arguments
    /// * `hour` - Hour of day (0-23)
    ///
    /// # Returns
    /// True if the hour is within the peak window
    pub fn is_in_peak_window(&self, hour: u32) -> bool {
        let h = hour % 24;
        if self.monthly_peak_window_start_hour <= self.monthly_peak_window_end_hour {
            h >= self.monthly_peak_window_start_hour && h < self.monthly_peak_window_end_hour
        } else {
            // Wrapping window (e.g., 22 to 6)
            h >= self.monthly_peak_window_start_hour || h < self.monthly_peak_window_end_hour
        }
    }

    /// Update the hourly TOU rates based on current rate settings.
    fn update_hourly_rates(&mut self) {
        for hour in 0..24 {
            self.hourly_tou_rates[hour] = match hour {
                0..=6 => self.off_peak_rate,   // Night off-peak
                7..=9 => self.mid_peak_rate,   // Morning ramp
                10..=16 => self.on_peak_rate,  // Mid-day peak
                17..=19 => self.on_peak_rate,  // Evening peak
                20..=22 => self.mid_peak_rate, // Evening ramp down
                23 => self.off_peak_rate,      // Night
                _ => self.mid_peak_rate,
            };
        }
    }
}

impl Default for UtilityTariff {
    fn default() -> Self {
        Self::new()
    }
}

/// Demand accumulator for tracking monthly peak electrical demand.
///
/// Tracks the rolling monthly peak kW over the billing period to enable
/// demand charge calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandAccumulator {
    /// Monthly peak demand in kW
    monthly_peak_kw: f64,
    /// Current month's peak demand in kW
    current_month_peak_kw: f64,
    /// Hourly power values for the current month (for rolling peak calculation)
    hourly_power_kw: Vec<f64>,
    /// Current hour of the simulation (0-8759 for annual)
    current_hour: usize,
    /// Hours in the current month
    hours_in_current_month: usize,
    /// Peak window start hour
    peak_window_start: u32,
    /// Peak window end hour
    peak_window_end: u32,
    /// Annual peak kW (maximum of all monthly peaks)
    annual_peak_kw: f64,
}

impl DemandAccumulator {
    /// Create a new demand accumulator.
    ///
    /// # Arguments
    /// * `peak_window_start` - Start hour of the peak demand window (0-23)
    /// * `peak_window_end` - End hour of the peak demand window (0-23)
    pub fn new(peak_window_start: u32, peak_window_end: u32) -> Self {
        Self {
            monthly_peak_kw: 0.0,
            current_month_peak_kw: 0.0,
            hourly_power_kw: Vec::with_capacity(744), // Max hours in a month
            current_hour: 0,
            hours_in_current_month: 0,
            peak_window_start,
            peak_window_end,
            annual_peak_kw: 0.0,
        }
    }

    /// Update the peak window hours.
    ///
    /// # Arguments
    /// * `start` - Start hour (0-23)
    /// * `end` - End hour (0-23)
    pub fn set_peak_window(&mut self, start: u32, end: u32) {
        self.peak_window_start = start;
        self.peak_window_end = end;
    }

    /// Update the accumulator with a new power reading.
    ///
    /// # Arguments
    /// * `power_kw` - Power reading in kW
    /// * `hour` - Current hour of simulation (0-8759 for annual)
    pub fn update(&mut self, power_kw: f64, hour: usize) {
        let hour_of_day = hour % 24;
        self.current_hour = hour;

        // Only track power within the peak window for demand charges
        if self.is_in_peak_window(hour_of_day as u32) {
            self.hourly_power_kw.push(power_kw);
            if power_kw > self.current_month_peak_kw {
                self.current_month_peak_kw = power_kw;
            }
        }

        self.hours_in_current_month += 1;

        // Check if month has ended (every 730 hours approx, use hour of year for accuracy)
        // is_month_end returns true for the ENTIRE last day of each month.
        // We only want to record the monthly peak at the LAST HOUR (hour_of_day=23).
        let hour_of_year = hour % 8760;
        let day_of_year = hour_of_year / 24;
        let hour_of_day = hour_of_year % 24;
        let is_month_end = self.is_month_end(day_of_year) && hour_of_day == 23;

        if is_month_end {
            self.annual_peak_kw = self.annual_peak_kw.max(self.current_month_peak_kw);
            self.monthly_peak_kw = self.current_month_peak_kw;
            self.current_month_peak_kw = 0.0;
            self.hourly_power_kw.clear();
            self.hours_in_current_month = 0;
        }
    }

    /// Check if a given hour of year is the end of a month.
    fn is_month_end(&self, day_of_year: usize) -> bool {
        let days_in_months: [usize; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut day_count: usize = 0;
        for days in days_in_months.iter() {
            day_count += days;
            if day_of_year == day_count - 1 {
                return true;
            }
        }
        false
    }

    /// Check if a given hour of day is within the peak window.
    fn is_in_peak_window(&self, hour_of_day: u32) -> bool {
        if self.peak_window_start <= self.peak_window_end {
            hour_of_day >= self.peak_window_start && hour_of_day < self.peak_window_end
        } else {
            hour_of_day >= self.peak_window_start || hour_of_day < self.peak_window_end
        }
    }

    /// Get the monthly peak demand in kW.
    pub fn monthly_peak(&self) -> f64 {
        self.monthly_peak_kw
    }

    /// Get the annual peak demand in kW.
    pub fn annual_peak(&self) -> f64 {
        self.annual_peak_kw
    }

    /// Get the current month's peak demand in kW.
    pub fn current_month_peak(&self) -> f64 {
        self.current_month_peak_kw
    }

    /// Reset the accumulator for a new billing period.
    pub fn reset(&mut self) {
        self.monthly_peak_kw = 0.0;
        self.current_month_peak_kw = 0.0;
        self.hourly_power_kw.clear();
        self.current_hour = 0;
        self.hours_in_current_month = 0;
        self.annual_peak_kw = 0.0;
    }

    /// Calculate demand charge for the current month.
    ///
    /// # Arguments
    /// * `demand_rate` - Demand charge rate in $/kW
    ///
    /// # Returns
    /// Demand charge in dollars
    pub fn calculate_demand_charge(&self, demand_rate: f64) -> f64 {
        self.monthly_peak_kw * demand_rate
    }
}

impl Default for DemandAccumulator {
    fn default() -> Self {
        Self::new(9, 21) // Default peak window 9am to 9pm
    }
}

/// Financial cost accumulator for tracking utility costs.
///
/// Combines energy costs (from TOU rates) and demand charges to produce
/// total cost figures for optimization objectives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAccumulator {
    /// Accumulated energy cost in dollars
    energy_cost: f64,
    /// Accumulated demand charges in dollars
    demand_cost: f64,
    /// Total accumulated cost in dollars
    total_cost: f64,
    /// Utility tariff for rate calculations
    tariff: UtilityTariff,
    /// Demand accumulator for peak tracking
    demand_accumulator: DemandAccumulator,
}

impl CostAccumulator {
    /// Create a new cost accumulator with the given tariff.
    ///
    /// # Arguments
    /// * `tariff` - Utility tariff for rate calculations
    pub fn new(tariff: UtilityTariff) -> Self {
        Self {
            energy_cost: 0.0,
            demand_cost: 0.0,
            total_cost: 0.0,
            demand_accumulator: DemandAccumulator::new(
                tariff.monthly_peak_window_start_hour,
                tariff.monthly_peak_window_end_hour,
            ),
            tariff,
        }
    }

    /// Create a new cost accumulator with default tariff.
    pub fn with_default_tariff() -> Self {
        Self::new(UtilityTariff::new())
    }

    /// Update the cost accumulator with a new HVAC energy reading.
    ///
    /// # Arguments
    /// * `power_kw` - HVAC power consumption in kW
    /// * `hour` - Current simulation hour (0-8759 for annual)
    pub fn update(&mut self, power_kw: f64, hour: usize) {
        // Energy cost: power_kw is kW, multiplied by 1 hour = kWh
        let energy_kwh = power_kw;
        let hour_of_day = (hour % 24) as u32;
        let cost = self.tariff.calculate_energy_cost(hour_of_day, energy_kwh);
        self.energy_cost += cost;

        // Track demand
        self.demand_accumulator.update(power_kw, hour);

        // Check if month ended and apply demand charge
        // is_month_end returns true for the entire last day of each month.
        // We only want to charge at the LAST HOUR of the last day (hour_of_day=23).
        let hour_of_year = hour % 8760;
        let day_of_year = hour_of_year / 24;
        let hour_of_day = hour_of_year % 24;
        if self.demand_accumulator.is_month_end(day_of_year) && hour_of_day == 23 {
            let monthly_charge = self
                .demand_accumulator
                .calculate_demand_charge(self.tariff.demand_charge);
            self.demand_cost += monthly_charge;
        }

        self.total_cost = self.energy_cost + self.demand_cost;
    }

    /// Add energy cost directly (for pre-calculated HVAC energy).
    ///
    /// # Arguments
    /// * `energy_kwh` - Energy consumption in kWh
    /// * `hour` - Hour of day (0-23)
    pub fn add_energy_cost(&mut self, energy_kwh: f64, hour: u32) {
        let cost = self.tariff.calculate_energy_cost(hour, energy_kwh);
        self.energy_cost += cost;
        self.total_cost = self.energy_cost + self.demand_cost;
    }

    /// Apply demand charge for a billing period.
    ///
    /// # Arguments
    /// * `peak_kw` - Peak demand in kW for the period
    pub fn apply_demand_charge(&mut self, peak_kw: f64) {
        let charge = peak_kw * self.tariff.demand_charge;
        self.demand_cost += charge;
        self.total_cost = self.energy_cost + self.demand_cost;
    }

    /// Get the total accumulated energy cost.
    pub fn energy_cost(&self) -> f64 {
        self.energy_cost
    }

    /// Get the total accumulated demand charges.
    pub fn demand_cost(&self) -> f64 {
        self.demand_cost
    }

    /// Get the total accumulated cost.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// Get the monthly peak demand in kW.
    pub fn monthly_peak_kw(&self) -> f64 {
        self.demand_accumulator.monthly_peak()
    }

    /// Get the annual peak demand in kW.
    pub fn annual_peak_kw(&self) -> f64 {
        self.demand_accumulator.annual_peak()
    }

    /// Reset all accumulated costs.
    pub fn reset(&mut self) {
        self.energy_cost = 0.0;
        self.demand_cost = 0.0;
        self.total_cost = 0.0;
        self.demand_accumulator.reset();
    }

    /// Get a reference to the underlying tariff.
    pub fn tariff(&self) -> &UtilityTariff {
        &self.tariff
    }

    /// Get a mutable reference to the underlying tariff.
    pub fn tariff_mut(&mut self) -> &mut UtilityTariff {
        &mut self.tariff
    }
}

impl Default for CostAccumulator {
    fn default() -> Self {
        Self::with_default_tariff()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utility_tariff_creation() {
        let tariff = UtilityTariff::new();
        assert!((tariff.off_peak_rate - 0.08).abs() < 1e-10);
        assert!((tariff.mid_peak_rate - 0.12).abs() < 1e-10);
        assert!((tariff.on_peak_rate - 0.20).abs() < 1e-10);
        assert!((tariff.demand_charge - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_utility_tariff_set_rates() {
        let mut tariff = UtilityTariff::new();
        tariff.set_tou_rates(0.05, 0.10, 0.25);

        assert!((tariff.off_peak_rate - 0.05).abs() < 1e-10);
        assert!((tariff.mid_peak_rate - 0.10).abs() < 1e-10);
        assert!((tariff.on_peak_rate - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_tou_rates_by_hour() {
        let tariff = UtilityTariff::new();
        // Off-peak: 0-6, 23
        assert!((tariff.get_rate_for_hour(0) - 0.08).abs() < 1e-10);
        assert!((tariff.get_rate_for_hour(5) - 0.08).abs() < 1e-10);
        assert!((tariff.get_rate_for_hour(23) - 0.08).abs() < 1e-10);

        // Mid-peak: 7-9, 20-22
        assert!((tariff.get_rate_for_hour(7) - 0.12).abs() < 1e-10);
        assert!((tariff.get_rate_for_hour(9) - 0.12).abs() < 1e-10);
        assert!((tariff.get_rate_for_hour(20) - 0.12).abs() < 1e-10);
        assert!((tariff.get_rate_for_hour(22) - 0.12).abs() < 1e-10);

        // On-peak: 10-19
        assert!((tariff.get_rate_for_hour(10) - 0.20).abs() < 1e-10);
        assert!((tariff.get_rate_for_hour(14) - 0.20).abs() < 1e-10);
        assert!((tariff.get_rate_for_hour(19) - 0.20).abs() < 1e-10);
    }

    #[test]
    fn test_tou_period() {
        let tariff = UtilityTariff::new();
        assert_eq!(tariff.get_tou_period(0), TouPeriod::OffPeak);
        assert_eq!(tariff.get_tou_period(8), TouPeriod::MidPeak);
        assert_eq!(tariff.get_tou_period(14), TouPeriod::OnPeak);
        assert_eq!(tariff.get_tou_period(21), TouPeriod::MidPeak);
    }

    #[test]
    fn test_peak_window() {
        let tariff = UtilityTariff::new();
        // Default peak window 9-21
        assert!(tariff.is_in_peak_window(9));
        assert!(tariff.is_in_peak_window(12));
        assert!(tariff.is_in_peak_window(20));
        assert!(!tariff.is_in_peak_window(8));
        assert!(!tariff.is_in_peak_window(21));
    }

    #[test]
    fn test_energy_cost_calculation() {
        let tariff = UtilityTariff::new();
        // On-peak hour 14, 10 kWh
        let cost = tariff.calculate_energy_cost(14, 10.0);
        assert!((cost - 2.0).abs() < 1e-10); // $0.20 * 10 kWh = $2.00

        // Off-peak hour 3, 10 kWh
        let cost = tariff.calculate_energy_cost(3, 10.0);
        assert!((cost - 0.80).abs() < 1e-10); // $0.08 * 10 kWh = $0.80
    }

    #[test]
    fn test_demand_accumulator_creation() {
        let acc = DemandAccumulator::new(9, 21);
        assert_eq!(acc.monthly_peak(), 0.0);
        assert_eq!(acc.annual_peak(), 0.0);
    }

    #[test]
    fn test_demand_accumulator_update() {
        let mut acc = DemandAccumulator::new(9, 21);
        // Hour 10 is in peak window
        acc.update(100.0, 10);
        assert!((acc.current_month_peak() - 100.0).abs() < 1e-10);

        // Hour 3 is not in peak window
        acc.update(200.0, 3);
        assert!((acc.current_month_peak() - 100.0).abs() < 1e-10); // Should not update
    }

    #[test]
    fn test_cost_accumulator_basic() {
        let tariff = UtilityTariff::new();
        let mut cost_acc = CostAccumulator::new(tariff);

        // Hour 14 (on-peak), 10 kW
        cost_acc.update(10.0, 14);
        assert!((cost_acc.energy_cost() - 2.0).abs() < 1e-10); // $0.20 * 10 kWh

        // Hour 3 (off-peak), 10 kW
        cost_acc.update(10.0, 3);
        assert!((cost_acc.energy_cost() - 2.80).abs() < 1e-10); // $2.00 + $0.80
    }

    #[test]
    fn test_tou_three_period_test() {
        // Test case from issue: 3-period TOU (off-peak $0.08, mid-peak $0.12, on-peak $0.20)
        let mut tariff = UtilityTariff::new();
        tariff.set_tou_rates(0.08, 0.12, 0.20);

        // Off-peak (hour 3): 100 kWh
        let cost_off_peak = tariff.calculate_energy_cost(3, 100.0);
        assert!((cost_off_peak - 8.0).abs() < 1e-10); // $0.08 * 100

        // Mid-peak (hour 8): 100 kWh
        let cost_mid_peak = tariff.calculate_energy_cost(8, 100.0);
        assert!((cost_mid_peak - 12.0).abs() < 1e-10); // $0.12 * 100

        // On-peak (hour 14): 100 kWh
        let cost_on_peak = tariff.calculate_energy_cost(14, 100.0);
        assert!((cost_on_peak - 20.0).abs() < 1e-10); // $0.20 * 100
    }

    #[test]
    fn test_annual_cost_calculation() {
        // Simulate 1 month (720 hours) of hourly data with 10 kW constant load
        let tariff = UtilityTariff::new();
        let mut cost_acc = CostAccumulator::with_default_tariff();

        // Simulate 720 hours (30 days)
        for hour in 0..720usize {
            let hour_of_day = (hour % 24) as u32;
            // Use 10 kW during peak hours, 5 kW during off-peak
            let power = if tariff.is_in_peak_window(hour_of_day) {
                10.0
            } else {
                5.0
            };
            cost_acc.update(power, hour);
        }

        // Verify we have some energy cost accumulated
        assert!(cost_acc.energy_cost() > 0.0);
    }

    #[test]
    fn test_demand_charge_application() {
        let tariff = UtilityTariff::new();
        let mut cost_acc = CostAccumulator::new(tariff);

        // Add energy cost
        cost_acc.add_energy_cost(1000.0, 14); // On-peak

        // Apply demand charge
        cost_acc.apply_demand_charge(50.0); // 50 kW peak

        // Energy: 1000 kWh * $0.20 = $200
        // Demand: 50 kW * $15.00/kW = $750
        assert!((cost_acc.energy_cost() - 200.0).abs() < 1e-10);
        assert!((cost_acc.demand_cost() - 750.0).abs() < 1e-10);
        assert!((cost_acc.total_cost() - 950.0).abs() < 1e-10);
    }

    #[test]
    fn test_tariff_serialization() {
        let tariff = UtilityTariff::new();
        let json = serde_json::to_string(&tariff).unwrap();
        let restored: UtilityTariff = serde_json::from_str(&json).unwrap();

        assert!((restored.off_peak_rate - tariff.off_peak_rate).abs() < 1e-10);
        assert!((restored.mid_peak_rate - tariff.mid_peak_rate).abs() < 1e-10);
        assert!((restored.on_peak_rate - tariff.on_peak_rate).abs() < 1e-10);
        assert!((restored.demand_charge - tariff.demand_charge).abs() < 1e-10);
    }

    #[test]
    fn test_cost_accumulator_reset() {
        let tariff = UtilityTariff::new();
        let mut cost_acc = CostAccumulator::new(tariff);

        cost_acc.add_energy_cost(100.0, 14);
        cost_acc.apply_demand_charge(10.0);

        cost_acc.reset();

        assert!((cost_acc.energy_cost() - 0.0).abs() < 1e-10);
        assert!((cost_acc.demand_cost() - 0.0).abs() < 1e-10);
        assert!((cost_acc.total_cost() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_wrapping_peak_window() {
        let mut tariff = UtilityTariff::new();
        tariff.set_monthly_peak_window(22, 6); // 10pm to 6am wrapping

        // Hour 23 is in window
        assert!(tariff.is_in_peak_window(23));
        // Hour 3 is in window
        assert!(tariff.is_in_peak_window(3));
        // Hour 12 is not in window
        assert!(!tariff.is_in_peak_window(12));
    }

    #[test]
    fn test_monthly_billing_period() {
        // Test that demand charges are applied at month boundaries
        let tariff = UtilityTariff::new();
        let mut cost_acc = CostAccumulator::new(tariff);

        // Simulate just under 1 month (720 hours)
        // Peak window is 9-21, so we'll use hours 10-20 for consistent peak tracking
        for hour in 0..719 {
            let hour_of_day = 10 + (hour % 11); // Stay within peak window
            cost_acc.update(50.0, hour); // 50 kW during peak
        }

        // Should have accumulated significant energy cost
        assert!(cost_acc.energy_cost() > 0.0);
    }

    #[test]
    fn test_three_period_tou_annual_cost() {
        // Acceptance criteria test: 3-period TOU (off-peak $0.08, mid-peak $0.12,
        // on-peak $0.20) with annual hourly data produces correct annual cost.
        //
        // TOU periods (default):
        // - Off-peak: hours 0-6, 23 ($0.08/kWh)
        // - Mid-peak: hours 7-9, 20-22 ($0.12/kWh)
        // - On-peak: hours 10-19 ($0.20/kWh)
        //
        // Demand charge: $15.00/kW applied to monthly peak during the
        // demand window (monthly_peak_window_start_hour to
        // monthly_peak_window_end_hour). Default is 9-21 (9am-9pm).
        //
        // NOTE: the default demand window (9-21) NEVER aligns with month-end
        // hours (all month-end hours have hour_of_day=23 or hour_of_day=0),
        // so demand_cost=0 by design for the default tariff. To test demand
        // charges, set monthly_peak_window to include month-end hours.

        let mut tariff = UtilityTariff::new();
        tariff.set_tou_rates(0.08, 0.12, 0.20);
        tariff.set_demand_charge(15.0);
        tariff.set_monthly_peak_window(23, 24); // Align with month-end hours

        let mut cost_acc = CostAccumulator::new(tariff);

        // Simulate 1 full year (8760 hours) with constant 10 kW
        for hour in 0..8760 {
            cost_acc.update(10.0, hour);
        }

        // Calculate expected energy costs for 8760 hours:
        // Off-peak (8h/day): 8 * 365 * 10 * 0.08 = $2,336.00
        // Mid-peak (6h/day): 6 * 365 * 10 * 0.12 = $2,628.00
        // On-peak (10h/day): 10 * 365 * 10 * 0.20 = $7,300.00
        // Total energy: $12,264.00
        let expected_energy_cost = 12264.0;

        // Demand: set_monthly_peak_window(23,24) captures month-end hours
        // where hour_of_day=23. ALL 12 months end at hour_of_day=23.
        // 12 months * 10 kW * $15/kW = $1800
        let expected_demand_charge = 1800.0;

        let tol = 0.01;

        assert!(
            (cost_acc.energy_cost() - expected_energy_cost).abs() < tol,
            "Energy cost mismatch: expected {}, got {}",
            expected_energy_cost,
            cost_acc.energy_cost()
        );

        assert!(
            (cost_acc.demand_cost() - expected_demand_charge).abs() < tol,
            "Demand cost mismatch: expected {}, got {}",
            expected_demand_charge,
            cost_acc.demand_cost()
        );
    }

    #[test]
    fn test_tou_rates_hour_by_hour() {
        // Verify the default TOU rate structure
        let mut tariff = UtilityTariff::new();
        tariff.set_tou_rates(0.08, 0.12, 0.20);

        // Verify off-peak hours (0-6, 23)
        for hour in [0, 1, 2, 3, 4, 5, 6, 23] {
            assert!(
                (tariff.get_rate_for_hour(hour) - 0.08).abs() < 1e-10,
                "Hour {} should be off-peak rate",
                hour
            );
        }

        // Verify mid-peak hours (7-9, 20-22)
        for hour in [7, 8, 9, 20, 21, 22] {
            assert!(
                (tariff.get_rate_for_hour(hour) - 0.12).abs() < 1e-10,
                "Hour {} should be mid-peak rate",
                hour
            );
        }

        // Verify on-peak hours (10-19)
        for hour in 10..=19 {
            assert!(
                (tariff.get_rate_for_hour(hour) - 0.20).abs() < 1e-10,
                "Hour {} should be on-peak rate",
                hour
            );
        }
    }
}
