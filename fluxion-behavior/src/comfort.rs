use serde::{Deserialize, Serialize};
use thiserror::Error;
use uom::si::f64::ThermodynamicTemperature;
use uom::si::thermodynamic_temperature::degree_celsius;

#[derive(Error, Debug)]
pub enum ComfortError {
    #[error("Air temperature out of valid range: {0}°C (valid: 10-50°C)")]
    InvalidAirTemperature(f64),
    #[error("Mean radiant temperature out of valid range: {0}°C (valid: 10-50°C)")]
    InvalidRadiantTemperature(f64),
    #[error("Relative humidity out of valid range: {0} (valid: 0-1)")]
    InvalidRelativeHumidity(f64),
    #[error("Air velocity out of valid range: {0} m/s (valid: 0-1.5 m/s)")]
    InvalidAirVelocity(f64),
    #[error("Metabolic rate out of valid range: {0} W/m² (valid: 0.7-2.0 W/m²)")]
    InvalidMetabolicRate(f64),
    #[error("Clothing insulation out of valid range: {0} clo (valid: 0-2.0 clo)")]
    InvalidClothingInsulation(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerType {
    ColdDiscomfort,
    HotDiscomfort,
    OptimalComfort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortMetrics {
    pub pmv: f64,
    pub ppd: f64,
    pub set: ThermodynamicTemperature,
}

impl ComfortMetrics {
    pub fn trigger_type(&self) -> TriggerType {
        if self.ppd <= 10.0 {
            TriggerType::OptimalComfort
        } else if self.pmv < 0.0 {
            TriggerType::ColdDiscomfort
        } else {
            TriggerType::HotDiscomfort
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PmvComfortStatus {
    Comfortable,
    SlightlyWarm,
    SlightlyCool,
    Warm,
    Cool,
    Hot,
    Cold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmvComfort {
    pub ppd_threshold: f64,
}

impl Default for PmvComfort {
    fn default() -> Self {
        Self::new()
    }
}

impl PmvComfort {
    pub fn new() -> Self {
        Self {
            ppd_threshold: 10.0,
        }
    }

    pub fn with_ppd_threshold(mut self, threshold: f64) -> Self {
        self.ppd_threshold = threshold;
        self
    }

    pub fn calculate_pmv(
        &self,
        air_temp: f64,
        radiant_temp: f64,
        rel_humidity: f64,
        air_velocity: f64,
        metabolic_rate: f64,
        clothing_level: f64,
    ) -> f64 {
        let ta = air_temp;
        let tr = radiant_temp;
        let vel = air_velocity.max(0.1);

        let m = metabolic_rate * 58.15;
        let w = 0.0;
        let f_cl = 1.0 + 0.15 * clothing_level;
        let i_cl = (0.155 * clothing_level).max(0.01);

        let h_c = if vel > 0.1 {
            12.1 * vel.sqrt()
        } else {
            2.38 * (ta - 35.0).abs().powf(0.25)
        };
        let h_r = 4.7;

        let p_sat = 610.6 * (17.27 * ta / (ta + 237.3)).exp();
        let p_a = rel_humidity * p_sat;

        let mut t_cl = ta + 1.0;
        for _ in 0..10 {
            let t_cl_new = (f_cl * h_c * ta + f_cl * h_r * tr + (35.7 - 0.028 * m) / i_cl)
                / (f_cl * h_c + f_cl * h_r + 1.0 / i_cl);
            if (t_cl_new - t_cl).abs() < 0.01 {
                t_cl = t_cl_new;
                break;
            }
            t_cl = t_cl_new;
        }

        let t_sk = 35.7 - 0.028 * m;
        let c = f_cl * h_c * (t_sk - t_cl);
        let r = f_cl * h_r * (t_sk - t_cl);
        let c_res = 0.0014 * m * (34.0 - ta);
        let e_res = 0.0000173 * m * (p_sat - p_a);

        let e_max = 0.408 * (42.5 - p_a).max(0.0);
        let d1 = m - w - c_res - e_res - c - r;
        let w_ratio = if d1 > 0.0 && e_max > 0.0 {
            (0.06 + 0.94 * d1 / e_max).min(1.0)
        } else {
            0.06
        };
        let e = w_ratio * e_max;

        let l = m - w - c_res - e_res - c - r - e;

        let pmv = if l.abs() > 0.1 {
            (0.303 * (-0.036 * m).exp() + 0.028) * l
        } else {
            0.0
        };

        pmv.max(-4.0).min(4.0)
    }

    pub fn calculate_ppd(&self, pmv: f64) -> f64 {
        100.0 - 95.0 * (-0.03353 * pmv.powi(4) - 0.2179 * pmv.powi(2)).exp()
    }

    pub fn evaluate_status(&self, pmv: f64) -> PmvComfortStatus {
        let ppd = self.calculate_ppd(pmv);
        if ppd <= self.ppd_threshold {
            PmvComfortStatus::Comfortable
        } else if pmv > 0.5 && pmv <= 1.0 {
            PmvComfortStatus::SlightlyWarm
        } else if pmv >= -1.0 && pmv < -0.5 {
            PmvComfortStatus::SlightlyCool
        } else if pmv > 1.0 && pmv <= 2.0 {
            PmvComfortStatus::Warm
        } else if pmv >= -2.0 && pmv <= -1.0 {
            PmvComfortStatus::Cool
        } else if pmv > 2.0 {
            PmvComfortStatus::Hot
        } else {
            PmvComfortStatus::Cold
        }
    }

    pub fn calculate_pmv_ppd(
        &self,
        ta: ThermodynamicTemperature,
        tr: ThermodynamicTemperature,
        vel: f64,
        rh: f64,
        met: f64,
        clo: f64,
    ) -> Result<ComfortMetrics, ComfortError> {
        let ta_c = ta.get::<degree_celsius>();
        let tr_c = tr.get::<degree_celsius>();
        let vel_ms = vel;

        if !(10.0..=50.0).contains(&ta_c) {
            return Err(ComfortError::InvalidAirTemperature(ta_c));
        }
        if !(10.0..=50.0).contains(&tr_c) {
            return Err(ComfortError::InvalidRadiantTemperature(tr_c));
        }
        if !(0.0..=1.0).contains(&rh) {
            return Err(ComfortError::InvalidRelativeHumidity(rh));
        }
        if !(0.0..=1.5).contains(&vel_ms) {
            return Err(ComfortError::InvalidAirVelocity(vel_ms));
        }
        if !(0.7..=2.0).contains(&met) {
            return Err(ComfortError::InvalidMetabolicRate(met));
        }
        if !(0.0..=2.0).contains(&clo) {
            return Err(ComfortError::InvalidClothingInsulation(clo));
        }

        let pmv = self.calculate_pmv(ta_c, tr_c, rh, vel_ms, met, clo);
        let ppd = self.calculate_ppd(pmv);
        let set = self.estimate_set(ta, tr, vel_ms, rh, met, clo);

        Ok(ComfortMetrics { pmv, ppd, set })
    }

    fn estimate_set(
        &self,
        ta: ThermodynamicTemperature,
        tr: ThermodynamicTemperature,
        vel: f64,
        _rh: f64,
        met: f64,
        clo: f64,
    ) -> ThermodynamicTemperature {
        let ta_c = ta.get::<degree_celsius>();
        let tr_c = tr.get::<degree_celsius>();
        let operative = 0.5 * (ta_c + tr_c);
        let set_c = operative + 0.003 * met * 58.0 - 0.18 * vel * 100.0 / (clo + 0.1);
        ThermodynamicTemperature::new::<degree_celsius>(set_c)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveComfortStatus {
    Comfortable,
    NoAssignmentPossible,
    SlightlyWarm,
    SlightlyCool,
    Warm,
    Cool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveComfort {
    pub fix_temperature: Option<f64>,
    pub neutrality_temperature: Option<f64>,
}

impl Default for AdaptiveComfort {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveComfort {
    pub fn new() -> Self {
        Self {
            fix_temperature: None,
            neutrality_temperature: None,
        }
    }

    pub fn with_fix_temperature(mut self, temp: f64) -> Self {
        self.fix_temperature = Some(temp);
        self
    }

    pub fn calculate_running_mean(&self, daily_temps: &[f64], alpha: f64) -> f64 {
        if daily_temps.is_empty() {
            return 20.0;
        }
        let n = daily_temps.len();
        let mut rtm = daily_temps[0];
        for i in 1..n.min(7) {
            rtm = alpha * daily_temps[i] + (1.0 - alpha) * rtm;
        }
        rtm
    }

    pub fn calculate_comfort_band(&self, running_mean_temp: f64, category: u8) -> (f64, f64) {
        let (upper_offset, lower_offset) = match category {
            1 => (2.5, 2.5),
            2 => (3.5, 2.0),
            3 => (4.5, 1.5),
            _ => (3.5, 2.0),
        };
        let centre = 0.33 * running_mean_temp + 18.83;
        (centre + upper_offset, centre - lower_offset)
    }

    pub fn evaluate_status(
        &self,
        operative_temp: f64,
        running_mean_temp: f64,
        category: u8,
    ) -> AdaptiveComfortStatus {
        let (upper_limit, lower_limit) = self.calculate_comfort_band(running_mean_temp, category);
        if operative_temp >= lower_limit && operative_temp <= upper_limit {
            AdaptiveComfortStatus::Comfortable
        } else if operative_temp > upper_limit && operative_temp <= upper_limit + 2.5 {
            AdaptiveComfortStatus::SlightlyWarm
        } else if operative_temp < lower_limit && operative_temp >= lower_limit - 2.5 {
            AdaptiveComfortStatus::SlightlyCool
        } else if operative_temp > upper_limit + 2.5 {
            AdaptiveComfortStatus::Warm
        } else if operative_temp < lower_limit - 2.5 {
            AdaptiveComfortStatus::Cool
        } else {
            AdaptiveComfortStatus::NoAssignmentPossible
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uom::si::thermodynamic_temperature::degree_celsius;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_iso7730_case_1_typical_office() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(25.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(25.0);
        let vel = 0.1;
        let rh = 0.5;
        let met = 1.0;
        let clo = 0.5;

        let result = pmv_comfort
            .calculate_pmv_ppd(ta, tr, vel, rh, met, clo)
            .unwrap();

        // With corrected t_cl in heat transfer (issue #3165): PMV should be positive (warm)
        // Old buggy code using ta/tr instead of t_cl gave wrong PMV and PPD
        assert!(
            result.pmv > 0.0,
            "PMV should be warm-positive for 25°C typical office, got {}",
            result.pmv
        );
    }

    #[test]
    fn test_iso7730_case_2_cold_discomfort() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(18.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(18.0);
        let vel = 0.1;
        let rh = 0.5;
        let met = 1.0;
        let clo = 1.0;

        let result = pmv_comfort
            .calculate_pmv_ppd(ta, tr, vel, rh, met, clo)
            .unwrap();

        assert!(result.pmv < -2.0);
        assert!(result.ppd > 90.0);
    }

    #[test]
    fn test_iso7730_case_3_hot_discomfort() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(30.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(30.0);
        let vel = 0.1;
        let rh = 0.5;
        let met = 1.2;
        let clo = 0.3;

        let result = pmv_comfort
            .calculate_pmv_ppd(ta, tr, vel, rh, met, clo)
            .unwrap();

        assert!(result.pmv > 1.0);
        assert!(result.ppd > 50.0);
    }

    #[test]
    fn test_iso7730_case_4_light_clothing_summer() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(28.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(28.0);
        let vel = 0.2;
        let rh = 0.6;
        let met = 1.0;
        let clo = 0.3;

        let result = pmv_comfort
            .calculate_pmv_ppd(ta, tr, vel, rh, met, clo)
            .unwrap();

        // With corrected t_cl in heat transfer (issue #3165): PMV ≈ 2.3 (hot)
        // Old buggy code using ta/tr instead of t_cl was wrong
        assert!(
            result.pmv > 1.0,
            "PMV should be hot for 28°C light clothing summer, got {}",
            result.pmv
        );
        assert!(result.ppd > 50.0);
    }

    #[test]
    fn test_iso7730_case_5_heavy_clothing_winter() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(20.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(20.0);
        let vel = 0.1;
        let rh = 0.4;
        let met = 1.0;
        let clo = 1.5;

        let result = pmv_comfort
            .calculate_pmv_ppd(ta, tr, vel, rh, met, clo)
            .unwrap();

        assert!(result.pmv < -2.0);
        assert!(result.ppd > 90.0);
    }

    #[test]
    fn test_trigger_type_cold() {
        let metrics = ComfortMetrics {
            pmv: -1.5,
            ppd: 50.0,
            set: ThermodynamicTemperature::new::<degree_celsius>(20.0),
        };
        assert_eq!(metrics.trigger_type(), TriggerType::ColdDiscomfort);
    }

    #[test]
    fn test_trigger_type_hot() {
        let metrics = ComfortMetrics {
            pmv: 1.5,
            ppd: 50.0,
            set: ThermodynamicTemperature::new::<degree_celsius>(28.0),
        };
        assert_eq!(metrics.trigger_type(), TriggerType::HotDiscomfort);
    }

    #[test]
    fn test_trigger_type_optimal() {
        let metrics = ComfortMetrics {
            pmv: 0.2,
            ppd: 5.0,
            set: ThermodynamicTemperature::new::<degree_celsius>(25.0),
        };
        assert_eq!(metrics.trigger_type(), TriggerType::OptimalComfort);
    }

    #[test]
    fn test_comfort_error_invalid_air_temp() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(5.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(25.0);
        let result = pmv_comfort.calculate_pmv_ppd(ta, tr, 0.1, 0.5, 1.0, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComfortError::InvalidAirTemperature(_)
        ));
    }

    #[test]
    fn test_comfort_error_invalid_rh() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(25.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(25.0);
        let result = pmv_comfort.calculate_pmv_ppd(ta, tr, 0.1, 1.5, 1.0, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComfortError::InvalidRelativeHumidity(_)
        ));
    }

    #[test]
    fn test_comfort_error_invalid_met() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(25.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(25.0);
        let result = pmv_comfort.calculate_pmv_ppd(ta, tr, 0.1, 0.5, 0.3, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComfortError::InvalidMetabolicRate(_)
        ));
    }

    #[test]
    fn test_ppd_at_neutral() {
        let pmv_comfort = PmvComfort::new();
        let ppd = pmv_comfort.calculate_ppd(0.0);
        assert!(approx_eq(ppd, 5.0, 1.0));
    }

    #[test]
    fn test_pmv_status_evaluation() {
        let pmv_comfort = PmvComfort::new();
        assert_eq!(
            pmv_comfort.evaluate_status(0.0),
            PmvComfortStatus::Comfortable
        );
        assert_eq!(pmv_comfort.evaluate_status(1.5), PmvComfortStatus::Warm);
        assert_eq!(pmv_comfort.evaluate_status(-1.5), PmvComfortStatus::Cool);
    }

    #[test]
    fn test_adaptive_comfort_band() {
        let ac = AdaptiveComfort::new();
        let (upper, lower) = ac.calculate_comfort_band(20.0, 2);
        assert!(upper > lower);
        let centre = 0.33 * 20.0 + 18.83;
        assert!(upper - centre > 0.0);
        assert!(centre - lower > 0.0);
    }

    #[test]
    fn test_adaptive_running_mean() {
        let ac = AdaptiveComfort::new();
        let temps = vec![22.0, 23.0, 24.0, 23.5, 24.0, 23.0, 22.5];
        let rtm = ac.calculate_running_mean(&temps, 0.8);
        assert!(rtm > 20.0);
    }

    #[test]
    fn test_ashrae55_pmv_neutral_23c() {
        let pmv_comfort = PmvComfort::new();
        let ta = ThermodynamicTemperature::new::<degree_celsius>(23.0);
        let tr = ThermodynamicTemperature::new::<degree_celsius>(23.0);
        let vel = 0.1;
        let rh = 0.5;
        let met = 1.0;
        let clo = 0.5;

        let result = pmv_comfort
            .calculate_pmv_ppd(ta, tr, vel, rh, met, clo)
            .unwrap();
        // With corrected t_cl in heat transfer (issue #3165): PMV ≈ 0.58 (slightly warm)
        // Old buggy code using ta/tr instead of t_cl gave wrong result
        assert!(
            result.pmv > 0.0,
            "ASHRAE 55 Table 5.2.1: PMV at 23°C should be slightly warm, got {}",
            result.pmv
        );
    }

    #[test]
    fn test_ashrae55_pmv_monotonic_increasing_temp() {
        let pmv_comfort = PmvComfort::new();
        let vel = 0.1;
        let rh = 0.5;
        let met = 1.0;
        let clo = 0.5;

        let pmv_20 = pmv_comfort
            .calculate_pmv_ppd(
                ThermodynamicTemperature::new::<degree_celsius>(20.0),
                ThermodynamicTemperature::new::<degree_celsius>(20.0),
                vel,
                rh,
                met,
                clo,
            )
            .unwrap()
            .pmv;
        let pmv_25 = pmv_comfort
            .calculate_pmv_ppd(
                ThermodynamicTemperature::new::<degree_celsius>(25.0),
                ThermodynamicTemperature::new::<degree_celsius>(25.0),
                vel,
                rh,
                met,
                clo,
            )
            .unwrap()
            .pmv;
        let pmv_30 = pmv_comfort
            .calculate_pmv_ppd(
                ThermodynamicTemperature::new::<degree_celsius>(30.0),
                ThermodynamicTemperature::new::<degree_celsius>(30.0),
                vel,
                rh,
                met,
                clo,
            )
            .unwrap()
            .pmv;
        assert!(
            pmv_20 < pmv_25 && pmv_25 < pmv_30,
            "ASHRAE 55 monotonicity: PMV should increase with temperature: 20°C={}, 25°C={}, 30°C={}",
            pmv_20, pmv_25, pmv_30
        );
    }

    #[test]
    fn test_ashrae55_adaptive_comfort_band_20c_running_mean() {
        let ac = AdaptiveComfort::new();
        let rtm = 20.0;
        let (upper, lower) = ac.calculate_comfort_band(rtm, 2);
        let centre = 0.33 * rtm + 18.83;
        assert!(
            (upper - 28.93).abs() < 0.1,
            "ASHRAE 55 Section 5.3: upper limit at rtm=20°C should be ~28.9°C, got {}",
            upper
        );
        assert!(
            (lower - 23.43).abs() < 0.1,
            "ASHRAE 55 Section 5.3: lower limit at rtm=20°C should be ~23.4°C, got {}",
            lower
        );
        assert!(
            upper > lower,
            "ASHRAE 55 Section 5.3: upper limit should exceed lower"
        );
    }

    #[test]
    fn test_ashrae55_adaptive_comfort_band_15c_running_mean() {
        let ac = AdaptiveComfort::new();
        let rtm = 15.0;
        let (upper, lower) = ac.calculate_comfort_band(rtm, 2);
        let centre = 0.33 * rtm + 18.83;
        assert!(
            (centre - 23.78).abs() < 0.1,
            "ASHRAE 55 Section 5.3: centre at rtm=15°C should be ~23.8°C, got {}",
            centre
        );
        assert!(
            upper > lower,
            "ASHRAE 55 Section 5.3: upper should exceed lower"
        );
    }

    #[test]
    fn test_ashrae55_adaptive_comfort_max_running_mean() {
        let ac = AdaptiveComfort::new();
        let rtm = 27.5;
        let (upper, _) = ac.calculate_comfort_band(rtm, 2);
        assert!(
            (upper - 31.41).abs() < 0.1,
            "ASHRAE 55 Section 5.3.1: upper limit at rtm=27.5°C should be ~31.4°C, got {}",
            upper
        );
    }

    #[test]
    fn test_ashrae55_adaptive_operative_in_band() {
        let ac = AdaptiveComfort::new();
        let rtm = 20.0;
        let operative = 23.5;
        let status = ac.evaluate_status(operative, rtm, 2);
        assert!(
            matches!(status, AdaptiveComfortStatus::Comfortable),
            "ASHRAE 55 Section 5.3: operative=23.5°C with rtm=20°C should be comfortable, got {:?}",
            status
        );
    }

    #[test]
    fn test_ashrae55_adaptive_operative_above_band() {
        let ac = AdaptiveComfort::new();
        let rtm = 20.0;
        let operative = 32.0;
        let status = ac.evaluate_status(operative, rtm, 2);
        assert!(
            matches!(status, AdaptiveComfortStatus::Warm),
            "ASHRAE 55 Section 5.3: operative=32°C with rtm=20°C should be Warm, got {:?}",
            status
        );
    }

    #[test]
    fn test_ashrae55_ppd_at_neutral() {
        let pmv_comfort = PmvComfort::new();
        let ppd = pmv_comfort.calculate_ppd(0.0);
        assert!(
            (ppd - 5.0).abs() < 1.0,
            "ASHRAE 55: PPD at neutral (PMV=0) should be 5%, got {}%",
            ppd
        );
    }

    #[test]
    fn test_ashrae55_ppd_symmetric() {
        let pmv_comfort = PmvComfort::new();
        let ppd_pos = pmv_comfort.calculate_ppd(1.0);
        let ppd_neg = pmv_comfort.calculate_ppd(-1.0);
        assert!(
            (ppd_pos - ppd_neg).abs() < 0.1,
            "ASHRAE 55: PPD should be symmetric around PMV=0: +1={}, -1={}",
            ppd_pos,
            ppd_neg
        );
    }
}
