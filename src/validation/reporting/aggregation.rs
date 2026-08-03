//! Aggregation module for monthly/daily reporting from hourly simulation data.
//!
//! This module provides:
//! - Monthly aggregation (compute monthly sums/averages from hourly data)
//! - Daily aggregation (compute daily sums/averages from hourly data)
//! - Custom report filtering by zone, system, plant
//! - SQL output using rusqlite
//!
//! # Example
//!
//! ```rust,ignore
//! use fluxion::validation::reporting::aggregation::{
//!     hourly_to_daily, hourly_to_monthly, CustomReportFilter, SqlReporter
//! };
//!
//! // Convert hourly data to daily
//! let daily = hourly_to_daily(&hourly_data);
//!
//! // Convert hourly data to monthly
//! let monthly = hourly_to_monthly(&hourly_data);
//!
//! // Filter by zone
//! let filter = CustomReportFilter::default().with_zone(0);
//! let filtered = filter.apply(&hourly_data);
//!
//! // Export to SQL
//! let reporter = SqlReporter::new("output.db")?;
//! reporter.insert_hourly_data(&hourly_data)?;
//! ```

use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::validation::diagnostic::HourlyData;

/// Number of hours in a year
#[allow(dead_code)]
const HOURS_PER_YEAR: usize = 8760;
/// Approximate watts per kW conversion
const WATTS_PER_KW: f64 = 1000.0;
/// Seconds per hour for energy conversion (W -> Wh)
const SECONDS_PER_HOUR: f64 = 3600.0;
/// Joules per watt-hour conversion
const JOULES_PER_WATT_HOUR: f64 = 3600.0;

/// Daily aggregated data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyAggregation {
    /// Year (e.g., 2024)
    pub year: u32,
    /// Month (1-12)
    pub month: u32,
    /// Day of month (1-31)
    pub day: u32,
    /// Average outdoor temperature (°C)
    pub avg_outdoor_temp: f64,
    /// Minimum outdoor temperature (°C)
    pub min_outdoor_temp: f64,
    /// Maximum outdoor temperature (°C)
    pub max_outdoor_temp: f64,
    /// Total HVAC heating energy (kWh)
    pub total_heating_kwh: f64,
    /// Total HVAC cooling energy (kWh)
    pub total_cooling_kwh: f64,
    /// Total solar gains (kWh)
    pub total_solar_kwh: f64,
    /// Total internal gains (kWh)
    pub total_internal_kwh: f64,
    /// Total infiltration losses (kWh)
    pub total_infiltration_kwh: f64,
    /// Total envelope conduction (kWh)
    pub total_envelope_kwh: f64,
    /// Average zone temperature (°C)
    pub avg_zone_temp: f64,
    /// Number of hours in this day (typically 24)
    pub hour_count: usize,
}

impl DailyAggregation {
    /// Create a new daily aggregation with zeros.
    pub fn new(year: u32, month: u32, day: u32) -> Self {
        Self {
            year,
            month,
            day,
            avg_outdoor_temp: 0.0,
            min_outdoor_temp: f64::MAX,
            max_outdoor_temp: f64::MIN,
            total_heating_kwh: 0.0,
            total_cooling_kwh: 0.0,
            total_solar_kwh: 0.0,
            total_internal_kwh: 0.0,
            total_infiltration_kwh: 0.0,
            total_envelope_kwh: 0.0,
            avg_zone_temp: 0.0,
            hour_count: 0,
        }
    }

    /// Returns the day of year (1-366).
    pub fn day_of_year(&self) -> u32 {
        let days_in_month = [31u32, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut doy = 0u32;
        for i in 0..(self.month - 1) {
            doy += days_in_month[i as usize];
        }
        doy + self.day
    }
}

/// Monthly aggregated data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyAggregation {
    /// Year (e.g., 2024)
    pub year: u32,
    /// Month (1-12)
    pub month: u32,
    /// Average outdoor temperature (°C)
    pub avg_outdoor_temp: f64,
    /// Minimum outdoor temperature (°C)
    pub min_outdoor_temp: f64,
    /// Maximum outdoor temperature (°C)
    pub max_outdoor_temp: f64,
    /// Total HVAC heating energy (kWh)
    pub total_heating_kwh: f64,
    /// Total HVAC cooling energy (kWh)
    pub total_cooling_kwh: f64,
    /// Total solar gains (kWh)
    pub total_solar_kwh: f64,
    /// Total internal gains (kWh)
    pub total_internal_kwh: f64,
    /// Total infiltration losses (kWh)
    pub total_infiltration_kwh: f64,
    /// Total envelope conduction (kWh)
    pub total_envelope_kwh: f64,
    /// Average zone temperature (°C)
    pub avg_zone_temp: f64,
    /// Number of days in this month
    pub day_count: u32,
    /// Number of hours in this month
    pub hour_count: usize,
}

impl MonthlyAggregation {
    /// Create a new monthly aggregation with zeros.
    pub fn new(year: u32, month: u32) -> Self {
        Self {
            year,
            month,
            avg_outdoor_temp: 0.0,
            min_outdoor_temp: f64::MAX,
            max_outdoor_temp: f64::MIN,
            total_heating_kwh: 0.0,
            total_cooling_kwh: 0.0,
            total_solar_kwh: 0.0,
            total_internal_kwh: 0.0,
            total_infiltration_kwh: 0.0,
            total_envelope_kwh: 0.0,
            avg_zone_temp: 0.0,
            day_count: 0,
            hour_count: 0,
        }
    }

    /// Returns the month name.
    pub fn month_name(&self) -> &'static str {
        match self.month {
            1 => "January",
            2 => "February",
            3 => "March",
            4 => "April",
            5 => "May",
            6 => "June",
            7 => "July",
            8 => "August",
            9 => "September",
            10 => "October",
            11 => "November",
            12 => "December",
            _ => "Unknown",
        }
    }
}

/// Filter criteria for custom reports.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CustomReportFilter {
    /// Filter by zone index (None = all zones)
    pub zone_index: Option<usize>,
    /// Filter by HVAC system ID (None = all systems)
    pub hvac_system_id: Option<String>,
    /// Filter by plant ID (None = all plants)
    pub plant_id: Option<String>,
    /// Start hour (inclusive), None = 0
    pub start_hour: Option<usize>,
    /// End hour (inclusive), None = 8759
    pub end_hour: Option<usize>,
    /// Start month (1-12), None = 1
    pub start_month: Option<u32>,
    /// End month (1-12), None = 12
    pub end_month: Option<u32>,
}

impl CustomReportFilter {
    /// Create a new filter with default values (no filtering).
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by specific zone index.
    pub fn with_zone(mut self, zone_index: usize) -> Self {
        self.zone_index = Some(zone_index);
        self
    }

    /// Filter by HVAC system ID.
    pub fn with_hvac_system(mut self, system_id: &str) -> Self {
        self.hvac_system_id = Some(system_id.to_string());
        self
    }

    /// Filter by plant ID.
    pub fn with_plant(mut self, plant_id: &str) -> Self {
        self.plant_id = Some(plant_id.to_string());
        self
    }

    /// Filter to specific month range.
    pub fn with_month_range(mut self, start: u32, end: u32) -> Self {
        self.start_month = Some(start);
        self.end_month = Some(end);
        self
    }

    /// Filter to specific hour range.
    pub fn with_hour_range(mut self, start: usize, end: usize) -> Self {
        self.start_hour = Some(start);
        self.end_hour = Some(end);
        self
    }

    /// Apply filter to hourly data and return filtered data.
    pub fn apply(&self, hourly_data: &[HourlyData]) -> Vec<HourlyData> {
        hourly_data
            .iter()
            .filter(|h| self.matches(h))
            .cloned()
            .collect()
    }

    /// Check if a single hourly data point matches the filter.
    pub fn matches(&self, h: &HourlyData) -> bool {
        if let Some(zone) = self.zone_index {
            if h.zone_temps.len() <= zone {
                return false;
            }
        }

        if let Some(start) = self.start_hour {
            if h.hour < start {
                return false;
            }
        }

        if let Some(end) = self.end_hour {
            if h.hour > end {
                return false;
            }
        }

        if let Some(start) = self.start_month {
            if h.month < start {
                return false;
            }
        }

        if let Some(end) = self.end_month {
            if h.month > end {
                return false;
            }
        }

        true
    }
}

/// SQL Reporter for exporting simulation data to SQLite.
pub struct SqlReporter {
    conn: Connection,
}

impl SqlReporter {
    /// Create a new SQL reporter with the given database path.
    pub fn new(path: &Path) -> SqlResult<Self> {
        let conn = Connection::open(path)?;

        // Create tables
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hourly_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour INTEGER NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                day INTEGER NOT NULL,
                hour_of_day INTEGER NOT NULL,
                outdoor_temp REAL NOT NULL,
                zone_index INTEGER NOT NULL,
                zone_temp REAL NOT NULL,
                solar_gains REAL NOT NULL,
                hvac_heating REAL NOT NULL,
                hvac_cooling REAL NOT NULL,
                internal_loads REAL NOT NULL,
                infiltration_loss REAL NOT NULL,
                envelope_conduction REAL NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS daily_aggregation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                day INTEGER NOT NULL,
                avg_outdoor_temp REAL NOT NULL,
                min_outdoor_temp REAL NOT NULL,
                max_outdoor_temp REAL NOT NULL,
                total_heating_kwh REAL NOT NULL,
                total_cooling_kwh REAL NOT NULL,
                total_solar_kwh REAL NOT NULL,
                total_internal_kwh REAL NOT NULL,
                total_infiltration_kwh REAL NOT NULL,
                total_envelope_kwh REAL NOT NULL,
                avg_zone_temp REAL NOT NULL,
                hour_count INTEGER NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS monthly_aggregation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                avg_outdoor_temp REAL NOT NULL,
                min_outdoor_temp REAL NOT NULL,
                max_outdoor_temp REAL NOT NULL,
                total_heating_kwh REAL NOT NULL,
                total_cooling_kwh REAL NOT NULL,
                total_solar_kwh REAL NOT NULL,
                total_internal_kwh REAL NOT NULL,
                total_infiltration_kwh REAL NOT NULL,
                total_envelope_kwh REAL NOT NULL,
                avg_zone_temp REAL NOT NULL,
                day_count INTEGER NOT NULL,
                hour_count INTEGER NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hourly_hour ON hourly_data(hour)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hourly_month ON hourly_data(month)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_month ON daily_aggregation(month)",
            [],
        )?;

        Ok(Self { conn })
    }

    /// Insert hourly data into the database.
    pub fn insert_hourly_data(&mut self, hourly_data: &[HourlyData]) -> SqlResult<()> {
        let tx = self.conn.unchecked_transaction()?;

        for h in hourly_data {
            let year = 2024; // Assumed year for the simulation

            // Insert per-zone data
            for zone_idx in 0..h.zone_temps.len() {
                let zone_temp = h.zone_temps.get(zone_idx).copied().unwrap_or(0.0);
                let solar = h.solar_gains.get(zone_idx).copied().unwrap_or(0.0);
                let heating = h.hvac_heating.get(zone_idx).copied().unwrap_or(0.0);
                let cooling = h.hvac_cooling.get(zone_idx).copied().unwrap_or(0.0);
                let internal = h.internal_loads.get(zone_idx).copied().unwrap_or(0.0);
                let infiltration = h.infiltration_loss.get(zone_idx).copied().unwrap_or(0.0);
                let envelope = h.envelope_conduction.get(zone_idx).copied().unwrap_or(0.0);

                self.conn.execute(
                    "INSERT INTO hourly_data (
                        hour, year, month, day, hour_of_day, outdoor_temp,
                        zone_index, zone_temp, solar_gains, hvac_heating,
                        hvac_cooling, internal_loads, infiltration_loss, envelope_conduction
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                    params![
                        h.hour,
                        year,
                        h.month,
                        h.day,
                        h.hour_of_day,
                        h.outdoor_temp,
                        zone_idx,
                        zone_temp,
                        solar,
                        heating,
                        cooling,
                        internal,
                        infiltration,
                        envelope
                    ],
                )?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Insert daily aggregation data into the database.
    pub fn insert_daily_aggregation(&mut self, daily_data: &[DailyAggregation]) -> SqlResult<()> {
        let tx = self.conn.unchecked_transaction()?;

        for d in daily_data {
            self.conn.execute(
                "INSERT INTO daily_aggregation (
                    year, month, day, avg_outdoor_temp, min_outdoor_temp, max_outdoor_temp,
                    total_heating_kwh, total_cooling_kwh, total_solar_kwh, total_internal_kwh,
                    total_infiltration_kwh, total_envelope_kwh, avg_zone_temp, hour_count
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                params![
                    d.year,
                    d.month,
                    d.day,
                    d.avg_outdoor_temp,
                    d.min_outdoor_temp,
                    d.max_outdoor_temp,
                    d.total_heating_kwh,
                    d.total_cooling_kwh,
                    d.total_solar_kwh,
                    d.total_internal_kwh,
                    d.total_infiltration_kwh,
                    d.total_envelope_kwh,
                    d.avg_zone_temp,
                    d.hour_count
                ],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Insert monthly aggregation data into the database.
    pub fn insert_monthly_aggregation(
        &mut self,
        monthly_data: &[MonthlyAggregation],
    ) -> SqlResult<()> {
        let tx = self.conn.unchecked_transaction()?;

        for m in monthly_data {
            self.conn.execute(
                "INSERT INTO monthly_aggregation (
                    year, month, avg_outdoor_temp, min_outdoor_temp, max_outdoor_temp,
                    total_heating_kwh, total_cooling_kwh, total_solar_kwh, total_internal_kwh,
                    total_infiltration_kwh, total_envelope_kwh, avg_zone_temp, day_count, hour_count
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                params![
                    m.year,
                    m.month,
                    m.avg_outdoor_temp,
                    m.min_outdoor_temp,
                    m.max_outdoor_temp,
                    m.total_heating_kwh,
                    m.total_cooling_kwh,
                    m.total_solar_kwh,
                    m.total_internal_kwh,
                    m.total_infiltration_kwh,
                    m.total_envelope_kwh,
                    m.avg_zone_temp,
                    m.day_count,
                    m.hour_count
                ],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Query monthly heating energy sum from database.
    pub fn query_monthly_heating(&self, year: u32, month: u32) -> SqlResult<f64> {
        let mut stmt = self.conn.prepare(
            "SELECT total_heating_kwh FROM monthly_aggregation WHERE year = ?1 AND month = ?2",
        )?;
        let result: f64 = stmt.query_row(params![year, month], |row| row.get(0))?;
        Ok(result)
    }

    /// Query monthly cooling energy sum from database.
    pub fn query_monthly_cooling(&self, year: u32, month: u32) -> SqlResult<f64> {
        let mut stmt = self.conn.prepare(
            "SELECT total_cooling_kwh FROM monthly_aggregation WHERE year = ?1 AND month = ?2",
        )?;
        let result: f64 = stmt.query_row(params![year, month], |row| row.get(0))?;
        Ok(result)
    }

    /// Close the database connection.
    pub fn close(self) {}
}

/// Convert watts to kilowatt-hours.
fn watts_to_kwh(watts: f64) -> f64 {
    watts * SECONDS_PER_HOUR / JOULES_PER_WATT_HOUR / WATTS_PER_KW
}

/// Convert hourly data to daily aggregations.
pub fn hourly_to_daily(hourly_data: &[HourlyData]) -> Vec<DailyAggregation> {
    if hourly_data.is_empty() {
        return Vec::new();
    }

    let mut daily_map: std::collections::HashMap<(u32, u32, u32), DailyAggregation> =
        std::collections::HashMap::new();

    for h in hourly_data {
        let year = 2024; // Assumed year
        let key = (year, h.month, h.day);

        let daily = daily_map
            .entry(key)
            .or_insert_with(|| DailyAggregation::new(year, h.month, h.day));

        // Update outdoor temperature stats
        daily.min_outdoor_temp = daily.min_outdoor_temp.min(h.outdoor_temp);
        daily.max_outdoor_temp = daily.max_outdoor_temp.max(h.outdoor_temp);

        // Accumulate energy values (convert W -> kWh)
        let zone_count = h.zone_temps.len().max(1);

        for zone_idx in 0..zone_count {
            let solar = h.solar_gains.get(zone_idx).copied().unwrap_or(0.0);
            let heating = h.hvac_heating.get(zone_idx).copied().unwrap_or(0.0);
            let cooling = h.hvac_cooling.get(zone_idx).copied().unwrap_or(0.0);
            let internal = h.internal_loads.get(zone_idx).copied().unwrap_or(0.0);
            let infiltration = h.infiltration_loss.get(zone_idx).copied().unwrap_or(0.0);
            let envelope = h.envelope_conduction.get(zone_idx).copied().unwrap_or(0.0);

            daily.total_solar_kwh += watts_to_kwh(solar);
            daily.total_heating_kwh += watts_to_kwh(heating.abs());
            daily.total_cooling_kwh += watts_to_kwh(cooling.abs());
            daily.total_internal_kwh += watts_to_kwh(internal);
            daily.total_infiltration_kwh += watts_to_kwh(infiltration.abs());
            daily.total_envelope_kwh += watts_to_kwh(envelope.abs());
        }

        // Accumulate zone temperatures for averaging
        let avg_zone_temp: f64 = h.zone_temps.iter().sum::<f64>() / zone_count as f64;
        daily.avg_zone_temp += avg_zone_temp;

        daily.hour_count += 1;
    }

    // Compute averages
    let mut result: Vec<DailyAggregation> = daily_map
        .into_values()
        .map(|mut d| {
            if d.hour_count > 0 {
                d.avg_outdoor_temp = (d.min_outdoor_temp + d.max_outdoor_temp) / 2.0;
                d.avg_zone_temp /= d.hour_count as f64;
            }
            d
        })
        .collect();

    result.sort_by_key(|a| (a.year, a.month, a.day));
    result
}

/// Convert hourly data to monthly aggregations.
pub fn hourly_to_monthly(hourly_data: &[HourlyData]) -> Vec<MonthlyAggregation> {
    if hourly_data.is_empty() {
        return Vec::new();
    }

    let mut monthly_map: std::collections::HashMap<(u32, u32), MonthlyAggregation> =
        std::collections::HashMap::new();

    for h in hourly_data {
        let year = 2024; // Assumed year
        let key = (year, h.month);

        let monthly = monthly_map
            .entry(key)
            .or_insert_with(|| MonthlyAggregation::new(year, h.month));

        // Update outdoor temperature stats
        monthly.min_outdoor_temp = monthly.min_outdoor_temp.min(h.outdoor_temp);
        monthly.max_outdoor_temp = monthly.max_outdoor_temp.max(h.outdoor_temp);

        // Accumulate energy values (convert W -> kWh)
        let zone_count = h.zone_temps.len().max(1);
        let mut daily_zone_temps_sum = 0.0;

        for zone_idx in 0..zone_count {
            let solar = h.solar_gains.get(zone_idx).copied().unwrap_or(0.0);
            let heating = h.hvac_heating.get(zone_idx).copied().unwrap_or(0.0);
            let cooling = h.hvac_cooling.get(zone_idx).copied().unwrap_or(0.0);
            let internal = h.internal_loads.get(zone_idx).copied().unwrap_or(0.0);
            let infiltration = h.infiltration_loss.get(zone_idx).copied().unwrap_or(0.0);
            let envelope = h.envelope_conduction.get(zone_idx).copied().unwrap_or(0.0);

            monthly.total_solar_kwh += watts_to_kwh(solar);
            monthly.total_heating_kwh += watts_to_kwh(heating.abs());
            monthly.total_cooling_kwh += watts_to_kwh(cooling.abs());
            monthly.total_internal_kwh += watts_to_kwh(internal);
            monthly.total_infiltration_kwh += watts_to_kwh(infiltration.abs());
            monthly.total_envelope_kwh += watts_to_kwh(envelope.abs());

            let zone_temp = h.zone_temps.get(zone_idx).copied().unwrap_or(0.0);
            daily_zone_temps_sum += zone_temp;
        }

        monthly.avg_zone_temp += daily_zone_temps_sum / zone_count as f64;
        monthly.hour_count += 1;
    }

    // Track days per month
    let mut days_per_month: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::new();
    for h in hourly_data {
        let year = 2024;
        let key = (year, h.month);
        *days_per_month.entry(key).or_insert(0) += 1;
    }

    // Compute averages
    let mut result: Vec<MonthlyAggregation> = monthly_map
        .into_values()
        .map(|mut m| {
            if m.hour_count > 0 {
                m.avg_outdoor_temp = (m.min_outdoor_temp + m.max_outdoor_temp) / 2.0;
                m.avg_zone_temp /= m.hour_count as f64;
            }
            // Days = hours / 24 (rounded up if partial)
            m.day_count = (m.hour_count.div_ceil(24) as u32).max(1);
            if let Some(days) = days_per_month.get(&(m.year, m.month)) {
                m.day_count = (*days / 24).max(1);
            }
            m
        })
        .collect();

    result.sort_by_key(|a| (a.year, a.month));
    result
}

/// Verify monthly aggregation matches hourly sum within tolerance.
///
/// Returns (is_valid, max_error_percent).
pub fn verify_monthly_aggregation(
    hourly_data: &[HourlyData],
    monthly_data: &[MonthlyAggregation],
    tolerance_percent: f64,
) -> (bool, f64) {
    let hourly_monthly = hourly_to_monthly(hourly_data);

    let mut max_error: f64 = 0.0;

    for m in monthly_data {
        if let Some(hourly_m) = hourly_monthly
            .iter()
            .find(|h| h.year == m.year && h.month == m.month)
        {
            let heating_diff = (m.total_heating_kwh - hourly_m.total_heating_kwh).abs();
            let heating_sum = hourly_m.total_heating_kwh.max(0.001);
            let heating_error = (heating_diff / heating_sum) * 100.0;

            let cooling_diff = (m.total_cooling_kwh - hourly_m.total_cooling_kwh).abs();
            let cooling_sum = hourly_m.total_cooling_kwh.max(0.001);
            let cooling_error = (cooling_diff / cooling_sum) * 100.0;

            max_error = max_error.max(heating_error).max(cooling_error);
        }
    }

    (max_error <= tolerance_percent, max_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::diagnostic::HourlyData;

    fn create_test_hourly_data() -> Vec<HourlyData> {
        let mut data = Vec::with_capacity(HOURS_PER_YEAR);

        for hour in 0..HOURS_PER_YEAR {
            let mut h = HourlyData::new(hour, 2);
            h.outdoor_temp = 20.0 + (hour as f64 % 24.0) - 12.0; // 8 to 32°C daily cycle
            h.zone_temps = vec![22.0, 24.0];
            h.solar_gains = vec![1000.0, 1200.0]; // W
            h.hvac_heating = vec![500.0, 600.0]; // W
            h.hvac_cooling = vec![0.0, 0.0]; // W
            h.internal_loads = vec![200.0, 250.0]; // W
            h.infiltration_loss = vec![50.0, 60.0]; // W
            h.envelope_conduction = vec![100.0, 120.0]; // W
            data.push(h);
        }

        data
    }

    #[test]
    fn test_hourly_to_daily() {
        let hourly = create_test_hourly_data();
        let daily = hourly_to_daily(&hourly);

        // Should have 365 days
        assert_eq!(daily.len(), 365);

        // Check first day
        let first = &daily[0];
        assert_eq!(first.month, 1);
        assert_eq!(first.day, 1);
        assert_eq!(first.hour_count, 24);

        // Energy values should be positive
        assert!(first.total_heating_kwh > 0.0);
        assert!(first.total_solar_kwh > 0.0);
    }

    #[test]
    fn test_hourly_to_monthly() {
        let hourly = create_test_hourly_data();
        let monthly = hourly_to_monthly(&hourly);

        // Should have 12 months
        assert_eq!(monthly.len(), 12);

        // Check January
        let jan = &monthly[0];
        assert_eq!(jan.month, 1);
        assert_eq!(jan.hour_count, 744); // 31 days * 24 hours
        assert!(jan.total_heating_kwh > 0.0);
    }

    #[test]
    fn test_verify_monthly_aggregation() {
        let hourly = create_test_hourly_data();
        let monthly = hourly_to_monthly(&hourly);

        let (valid, error) = verify_monthly_aggregation(&hourly, &monthly, 0.1);
        assert!(
            valid,
            "Monthly should match hourly within 0.1%, got {}",
            error
        );
    }

    #[test]
    fn test_custom_filter() {
        let hourly = create_test_hourly_data();

        // Filter to zone 0 only
        let filter = CustomReportFilter::new().with_zone(0);
        let filtered = filter.apply(&hourly);

        // All data should still match zone 0
        assert_eq!(filtered.len(), hourly.len());

        // Filter to January only
        let filter = CustomReportFilter::new().with_month_range(1, 1);
        let filtered = filter.apply(&hourly);

        // Should have 744 hours in January
        assert_eq!(filtered.len(), 744);
    }

    #[test]
    fn test_daily_sum_matches_monthly() {
        let hourly = create_test_hourly_data();
        let daily = hourly_to_daily(&hourly);
        let monthly = hourly_to_monthly(&hourly);

        // Sum of daily heating should roughly equal monthly
        let daily_heating_sum: f64 = daily.iter().map(|d| d.total_heating_kwh).sum();
        let monthly_heating_sum: f64 = monthly.iter().map(|m| m.total_heating_kwh).sum();

        let diff = (daily_heating_sum - monthly_heating_sum).abs();
        let rel_error = diff / monthly_heating_sum.max(0.001);

        assert!(
            rel_error < 0.001,
            "Daily sum should match monthly within 0.1%"
        );
    }
}
