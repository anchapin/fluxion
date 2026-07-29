//! Design Day Selection from TMY3/EPW weather data
//!
//! Identifies extreme heating and cooling design days from hourly weather data
//! following ASHRAE Handbook of Fundamentals Chapter 14 binning methodology.
//!
//! # Design Day Selection
//!
//! - **Heating design day**: Day with lowest average dry-bulb temperature
//! - **Cooling design day**: Day with highest average dry-bulb temperature
//!
//! # Safety Factor
//!
//! Equipment capacity is calculated from peak design-day loads and multiplied
//! by a safety factor (typically 1.1-1.2) per ASHRAE guidelines.

use crate::weather::HourlyWeatherData;

#[derive(Debug, Clone)]
pub struct DesignDaySelector {
    heating_design: Option<DailySummary>,
    cooling_design: Option<DailySummary>,
}

#[derive(Debug, Clone)]
pub struct DailySummary {
    pub month: u32,
    pub day_of_month: u32,
    pub max_temp: f64,
    pub min_temp: f64,
    pub avg_temp: f64,
    pub temp_range: f64,
}

impl DesignDaySelector {
    pub fn new() -> Self {
        Self {
            heating_design: None,
            cooling_design: None,
        }
    }

    pub fn select_from_hourly(&mut self, hourly_data: &[HourlyWeatherData]) -> &Self {
        if hourly_data.is_empty() {
            return self;
        }

        let mut daily_temps: std::collections::HashMap<(u32, u32), Vec<f64>> =
            std::collections::HashMap::new();

        for hour in hourly_data {
            let day = hour.day_of_year();
            let month = ((day / 31) + 1).min(12) as u32;
            let day_of_month = ((day % 31) + 1) as u32;
            daily_temps
                .entry((month, day_of_month))
                .or_default()
                .push(hour.dry_bulb_temp);
        }

        let mut heating_day: Option<DailySummary> = None;
        let mut cooling_day: Option<DailySummary> = None;

        for ((month, day_of_month), temps) in daily_temps {
            if temps.len() < 20 {
                continue;
            }

            let avg_temp = temps.iter().sum::<f64>() / temps.len() as f64;
            let max_temp = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_temp = temps.iter().cloned().fold(f64::INFINITY, f64::min);
            let temp_range = max_temp - min_temp;

            let day_spec = DailySummary {
                month,
                day_of_month,
                max_temp,
                min_temp,
                avg_temp,
                temp_range,
            };

            match &heating_day {
                None => heating_day = Some(day_spec.clone()),
                Some(h) if day_spec.avg_temp < h.avg_temp => {
                    heating_day = Some(day_spec.clone());
                }
                _ => {}
            }

            match &cooling_day {
                None => cooling_day = Some(day_spec.clone()),
                Some(c) if day_spec.avg_temp > c.avg_temp => {
                    cooling_day = Some(day_spec.clone());
                }
                _ => {}
            }
        }

        self.heating_design = heating_day;
        self.cooling_design = cooling_day;
        self
    }

    pub fn heating_design(&self) -> Option<&DailySummary> {
        self.heating_design.as_ref()
    }

    pub fn cooling_design(&self) -> Option<&DailySummary> {
        self.cooling_design.as_ref()
    }
}

impl Default for DesignDaySelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hourly_for_day(month: u32, day: u32, temps: &[f64]) -> Vec<HourlyWeatherData> {
        let cumulative_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let day_index = (month - 1) as usize;
        let day_start = cumulative_days[day_index.min(11)] * 24 + ((day - 1) as usize).min(30) * 24;
        temps
            .iter()
            .enumerate()
            .map(|(i, &t)| HourlyWeatherData::new(t, 0.0, 0.0, 0.0, 2.0, 50.0, day_start + i))
            .collect()
    }

    #[test]
    fn test_select_heating_design_day() {
        let mut selector = DesignDaySelector::new();
        let jan_cold = vec![-25.0; 24];
        let july_warm = vec![35.0; 24];
        let mut hourly = make_hourly_for_day(1, 15, &jan_cold);
        hourly.extend(make_hourly_for_day(7, 15, &july_warm));

        selector.select_from_hourly(&hourly);

        let heating = selector.heating_design().unwrap();
        assert_eq!(heating.month, 1);
        assert!(heating.avg_temp < 0.0);

        let cooling = selector.cooling_design().unwrap();
        assert_eq!(cooling.month, 7);
        assert!(cooling.avg_temp > 20.0);
    }

    #[test]
    fn test_select_from_empty() {
        let mut selector = DesignDaySelector::new();
        selector.select_from_hourly(&[]);
        assert!(selector.heating_design().is_none());
        assert!(selector.cooling_design().is_none());
    }

    #[test]
    fn test_daily_summary_clone() {
        let spec = DailySummary {
            month: 1,
            day_of_month: 15,
            max_temp: -10.0,
            min_temp: -20.0,
            avg_temp: -15.0,
            temp_range: 10.0,
        };
        let cloned = spec.clone();
        assert_eq!(cloned.avg_temp, spec.avg_temp);
    }
}
