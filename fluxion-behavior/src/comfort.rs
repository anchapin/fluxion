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

    // The max/min chain is kept instead of `clamp`: for a NaN PMV it yields
    // -4.0 (f64::max/min ignore NaN), whereas `clamp` would propagate NaN —
    // swapping it would change edge-case behavior.
    #[allow(clippy::manual_clamp)]
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
        let m = metabolic_rate * 58.15;
        let w = 0.0;

        // Round 99 (ISO 7730:2005 residuals sweep): the clothing model
        // moves to the ISO 7730:2005 Annex D reference-program forms,
        // cross-checked against EnergyPlus ThermalComfort.cc
        // `CalcFangerPMV` ("based on ANSI/ASHRAE Standard 55-2020 and
        // ISO 7730:2005"). Removed non-ISO constructs: f_cl = 1.0 +
        // 0.15·clo (ASHRAE-Handbook linearization), the t_cl solve
        // linearized at h_r = 4.7 W/m²K, the h_c selection keyed on
        // `vel > 0.1` at |ta − 35|, and the 0.1 m/s velocity floor.
        let i_cl = 0.155 * clothing_level;
        let f_cl = if i_cl < 0.078 {
            1.0 + 1.29 * i_cl
        } else {
            1.05 + 0.645 * i_cl
        };
        let clo_insul = i_cl * f_cl; // ISO: I_cl·f_cl

        // t_cl: implicit clothing-surface balance with the radiative
        // quartic inside, fixed-point iteration in the P1..P4 scaled
        // forms (ISO Annex D program; E+ parameters MaxIter 150,
        // StopIterCrit 0.00015), h_c re-evaluated each pass as
        // max(12.1·√v, 2.38·|t_cl − ta|^0.25). Absolute-offset convention
        // +273.0 per the ISO Annex D program, uniform with the round-97
        // quartic below (E+ converts at 273.15 — recorded choice).
        let p2 = clo_insul * 3.96;
        let p3 = clo_insul * 100.0;
        let p1 = clo_insul * (ta + 273.0);
        let t_tr = (tr + 273.0) / 100.0;
        let p4 = 308.7 - 0.028 * m + p2 * (t_tr * t_tr * t_tr * t_tr);
        let abs_air = ta + 273.0;
        let mut xn = (abs_air + (35.5 - ta) / (3.5 * i_cl + 0.1)) / 100.0;
        let hc_for = 12.1 * air_velocity.sqrt();
        let mut hc = hc_for;
        let mut xf = xn;
        let mut it = 0;

        let p_sat = 610.6 * (17.27 * ta / (ta + 237.3)).exp();
        let p_a = rel_humidity * p_sat;

        while ((xn - xf).abs() > 0.00015 || it == 0) && it < 150 {
            xf = (xf + xn) / 2.0;
            let hc_nat = 2.38 * (100.0 * xf - abs_air).abs().powf(0.25);
            hc = hc_for.max(hc_nat);
            xn = (p4 + p1 * hc - p2 * (xf * xf * xf * xf)) / (100.0 + p3 * hc);
            it += 1;
        }
        let t_cl = 100.0 * xn - 273.0;
        let h_c = hc;

        // Dry c/r terms (round 97; ISO 7730:2005): convection from the
        // clothing surface `f_cl·h_c·(t_cl − t_a)` plus the radiative
        // quartic `3.96e-8·f_cl·[(t_cl+273)⁴ − (t_r+273)⁴]` — the intent of
        // the #3165 correction. Pre-fix both terms used (t_sk − t_cl), the
        // across-clothing difference, against the film coefficients — not
        // an ISO term (the across-clothing flux is the conduction
        // K = (t_sk − t_cl)/I_cl, already embedded in the t_cl solution).
        let c = f_cl * h_c * (t_cl - ta);
        let r = 3.96e-8 * f_cl * ((t_cl + 273.0).powi(4) - (tr + 273.0).powi(4));
        let c_res = 0.0014 * m * (34.0 - ta);
        // Latent respiration (round 97; ISO 7730:2005 HL3): 1.7e-5·M·(5867 − p_a)
        // W/m² with p_a in Pa; 5867 Pa = saturated water-vapour pressure of the
        // exhaled air. Pre-fix: 1.73e-5·M·(p_sat − p_a) — Fanger-family
        // coefficient against the ambient saturation pressure, roughly half
        // the loss at room temperatures (1.59 vs 4.23 W/m² at 25 °C / 50 % rh).
        let e_res = 0.000017 * m * (5867.0 - p_a);

        // Evaporative heat loss from the skin (round 98, Issue #4598): ISO
        // 7730:2005's FIXED split — E_dif + E_rsw — replaces the
        // load-proportional wettedness regulator
        // (`w·e_max`, w = clamp(0.06 + 0.94·d1/e_max, 0.06, 1.0)) this copy
        // carried since the round-96 units fix:
        //
        //   HL1 — diffusion through the skin, always present, unclamped as
        //         in the ISO Basic program and EnergyPlus
        //         (EvapHeatLossDiff = 0.00305·(5733 − 6.99·H − P_a)):
        //         E_dif = 3.05e-3·(5733 − 6.99·M − p_a) W/m², M in W/m²,
        //         p_a in Pa (work w = 0 here, so M = MW). The same bracket
        //         arrived in round 96 as the e_max capacity (itself
        //         replacing the mmHg-form `0.408·(42.5 − p_a)` that zeroed
        //         at any normal humidity).
        //   Sweating — E_rsw = 0.42·max(M − 58.15, 0) W/m², active only
        //         above the comfort metabolic state 58.15 W/m² (1 met;
        //         EnergyPlus guards the same way: E_rsw = 0.42·(H − 58.2)
        //         for H > 58.2, else 0).
        //
        // Premise (verified 2026-10-25 against the EnergyPlus Engineering
        // Reference Fanger section and the ISO 7730:2005 equation set): the
        // Fanger/ISO PMV carries NO wettedness feedback — skin wettedness
        // is a two-node-model (Pierce/KSU) construct. The regulator floored
        // the cold side at 6 % of capacity (~0.8 W/m² where E_dif ≈ 13.1
        // W/m² at 18 °C / clo 1.0 — reads warm) and saturated at e_max on
        // the warm side, never crediting sweating above 1 met. Beyond the
        // ISO validity domain (hot-humid extremes) the unclamped E_dif
        // bracket goes negative; ISO's reference program computes it
        // unclamped and the PMV ±4 clamp absorbs that regime.
        let e_dif = 3.05e-3 * (5733.0 - 6.99 * m - p_a);
        let e_rsw = 0.42 * (m - 58.15).max(0.0);
        let e = e_dif + e_rsw;

        let l = m - w - c_res - e_res - c - r - e;

        // Round 99: PMV computed unconditionally — ISO 7730:2005 and the
        // E+ reference carry no zero arm near neutral (the removed
        // |L| <= 0.1 gate quantized the neutral region). The max/min
        // chain keeps the NaN -> -4.0 edge semantics documented above.
        ((0.303 * (-0.036 * m).exp() + 0.028) * l)
            .max(-4.0)
            .min(4.0)
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
        } else if (-1.0..-0.5).contains(&pmv) {
            PmvComfortStatus::SlightlyCool
        } else if pmv > 1.0 && pmv <= 2.0 {
            PmvComfortStatus::Warm
        } else if (-2.0..=-1.0).contains(&pmv) {
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
        for &temp in daily_temps.iter().take(n.min(7)).skip(1) {
            rtm = alpha * temp + (1.0 - alpha) * rtm;
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

        // Round 97: with ISO's clothing-surface dry terms and the 5867 Pa
        // respiration form, 25 °C / clo 0.5 / 50 % rh is essentially
        // neutral (PMV ≈ -0.023); the pre-round-97 `pmv > 0` expectation
        // encoded the (t_sk − t_cl) deviation.
        // Round 98 (Issue #4598, fixed ISO 7730 evaporative split): 25 °C
        // sits below the balance, where the regulator scaled the diffusive
        // loss below capacity; the fixed E_dif restores it.
        // -0.023355613 -> -0.389260211 (round 98). Round 99 (ISO clothing
        // model: piecewise f_cl, implicit-quartic t_cl, in-iteration
        // h_c): -0.389260211 -> -0.401646093.
        assert!(
            (result.pmv - (-0.401646093)).abs() < 1e-9,
            "PMV for 25°C typical office should sit at the corrected pin, got {}",
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

        // Round 97 corrected pins: slightly cool, not the pre-fix cold
        // reading. Round 98 (Issue #4598) then replaced the recorded
        // wettedness-regulator residual itself with ISO's fixed split —
        // see the re-pinned values below.
        // Round 98 (Issue #4598) fixed-split values: -0.612278191 ->
        // -1.416959230. Round 99 (ISO clothing model): -1.416959230 ->
        // -1.407963263.
        assert!((result.pmv - (-1.407963263)).abs() < 1e-9);
        assert!((result.ppd - 45.936682908).abs() < 1e-9);
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

        // Round 98 (Issue #4598): at met 1.2 the ISO sweating term
        // 0.42·(69.78 − 58.15) = 4.885 W/m² engages: 1.734916701 ->
        // 1.478118082. Round 99 (ISO clothing model): 1.478118082 ->
        // 1.428325186.
        assert!((result.pmv - 1.428325186).abs() < 1e-9);
        assert!((result.ppd - 47.026996319).abs() < 1e-9);
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

        // Round 97 corrected pins: warm-leaning, not the pre-fix hot
        // reading. Round 99 (ISO clothing model): 0.195674534 ->
        // 0.143714899.
        assert!((result.pmv - 0.143714899).abs() < 1e-9);
        assert!((result.ppd - 5.427939622).abs() < 1e-9);
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

        // Round 98 (Issue #4598, fixed split): -0.009075675 ->
        // -0.151261250. Round 99 (ISO clothing model): -0.151261250 ->
        // -0.187107459.
        assert!((result.pmv - (-0.187107459)).abs() < 1e-9);
        assert!((result.ppd - 5.725825125).abs() < 1e-9);
    }

    #[test]
    fn evaporative_capacity_uses_iso7730_pa_form_units() {
        // Round 96 units fix (same defect as src/sim/thermal_model/comfort.rs):
        // the skin-diffusion capacity was `0.408·(42.5 − p_a)` — a mmHg-form
        // constant (NIST SP 491: E_dif = 0.408·[P_s − RH·P_a] W/m², mmHg;
        // 42.5 mmHg = 5666 Pa) — fed p_a in Pa, so the term was zero at any
        // normal humidity. Corrected to ISO 7730:2005 Annex D HL1,
        // `3.05e-3·(5733 − 6.99·M − p_a)` W/m², p_a in Pa. Reference values
        // computed independently in Python replicating this function's exact
        // operation order (round-96 worklog; re-pinned round 97 — dry c/r
        // terms to ISO 7730's clothing-surface form, then respiration to
        // the 5867 Pa exhaled-air form, worklog r97_comfort_refs.py);
        // ±1e-9 is cross-language libm
        // ulp headroom, not a physics tolerance.
        let pmv_comfort = PmvComfort::new();
        let dry = pmv_comfort.calculate_pmv(30.0, 30.0, 0.0005, 0.1, 1.0, 0.5);
        let humid = pmv_comfort.calculate_pmv(30.0, 30.0, 0.5, 0.1, 1.0, 0.5);
        let very_humid = pmv_comfort.calculate_pmv(30.0, 30.0, 0.9, 0.1, 1.0, 0.5);
        assert!(
            (dry - 0.918966774).abs() < 1e-9,
            "dry-air 30 C PMV: got {dry}"
        );
        assert!(
            (humid - 1.478136931).abs() < 1e-9,
            "50 % rh 30 C PMV: got {humid}"
        );
        assert!(
            (very_humid - 1.925920840).abs() < 1e-9,
            "90 % rh 30 C PMV: got {very_humid}"
        );
        assert!(
            dry < humid && humid < very_humid,
            "PMV must rise with humidity once the evaporative term engages"
        );
    }

    #[test]
    fn respiration_uses_exhaled_air_saturation_pressure_iso7730() {
        // Round 97 (ISO 7730:2005 HL3): E_re = 1.7e-5·M·(5867 − p_a) W/m²
        // with p_a in Pa; 5867 Pa is the saturated water-vapour pressure of
        // the EXHALED air (the ISO 7730 Basic program carries the same
        // constant; EnergyPlus prints 0.000017·M·(5867 − VapPress)). Pre-fix
        // this term was 1.73e-5·M·(p_sat(ta) − p_a) — the Fanger-family
        // coefficient paired with the ambient-air saturation pressure, which
        // roughly halved the loss at room temperatures (1.59 vs 4.23 W/m²
        // at 25 °C / 50 % rh). Scenario at met 1.5 doubles the respiration
        // flux so the bracket difference is ~2x the met=1 effect:
        // PMV 1.448437078 (pre-r97) -> 1.297428735 (r97). Round 98
        // (Issue #4598) moves it again, to 0.795372740: at met 1.5 the
        // fixed split's sweating term 0.42·(87.23 − 58.15) = 12.21 W/m²
        // engages (the regulator saturated at e_max instead). Reference
        // values computed independently in Python replicating this
        // function's exact operation order (r97/r98/r99_comfort_refs.py);
        // ±1e-9 is cross-language libm ulp headroom, not a physics
        // tolerance. Round 99 (ISO clothing model): 0.795372740 ->
        // 0.778813182.
        let pmv_comfort = PmvComfort::new();
        let pmv = pmv_comfort.calculate_pmv(26.0, 26.0, 0.5, 0.1, 1.5, 0.5);
        assert!(
            (pmv - 0.778813182).abs() < 1e-9,
            "respiration 5867 Pa form at 26 C / met 1.5: got {pmv}"
        );
        // Fix direction: more respiration loss must read cooler, and
        // moister air (higher p_a narrows the bracket) must warm.
        assert!(pmv < 1.448437078, "fix direction: got {pmv}");
        let humid = pmv_comfort.calculate_pmv(26.0, 26.0, 0.9, 0.1, 1.5, 0.5);
        assert!(
            (humid - 1.029305956).abs() < 1e-9 && humid > pmv,
            "humidity must warm: {humid} vs {pmv}"
        );
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
        // Round 96 moved PMV at 23 °C +0.58 -> -0.0120 (ISO 7730 Pa-form
        // evaporative capacity engaged at 50 % rh; the pre-fix "slightly
        // warm" +0.58 was compensation from the zero-at-normal-humidity
        // evaporative term). Round 97 moves it again, to -0.424073441 (dry
        // c/r terms to ISO's clothing-surface t_cl form, then respiration
        // to the 5867 Pa exhaled-air form); this function's neutral point
        // now sits at ~26.0 °C. Round 98 (Issue #4598, fixed evaporative
        // split) moves it to -1.158944320: at 23 °C the d1-proportional
        // regulator held the diffusive loss below ISO's fixed E_dif.
        // Pinned exactly (Python replication, r97/r98/r99_comfort_refs.py;
        // ±1e-9 libm headroom). Round 99 (ISO clothing model):
        // -1.158944320 -> -1.133427674.
        assert!(
            (result.pmv - (-1.133427674)).abs() < 1e-9,
            "ASHRAE 55 PMV at 23 °C (round-97/98/99 comfort fixes): got {}",
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
        let _centre = 0.33 * rtm + 18.83;
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

    #[test]
    fn evaporative_split_is_fixed_iso7730_not_load_regulated() {
        // Round 98 (Issue #4598): the wettedness regulator
        // (w = clamp(0.06 + 0.94·d1/e_max, 0.06, 1.0); e = w·e_max) is
        // replaced by ISO 7730:2005's fixed split E = E_dif + E_rsw — the
        // diffusion term is always present (unclamped, as in the ISO Basic
        // program and EnergyPlus's EvapHeatLossDiff) and sweating engages
        // only above 58.15 W/m² (1 met; EnergyPlus guards the same way:
        // E_rsw = 0.42·(H − 58.2) for H > 58.2, else 0). The regulator
        // floored this 14 °C / clo 1.0 scenario near 6 % of capacity
        // (PMV -1.623601745 pre-fix); the fixed diffusion restores the
        // loss (PMV -2.471865351). Round 99 (ISO clothing model):
        // -2.471865351 -> -2.496170958. Reference values from Python
        // replication of this function's exact operation order
        // (r98/r99_comfort_refs.py); ±1e-9 is cross-language libm ulp
        // headroom, not a physics tolerance.
        let pmv_comfort = PmvComfort::new();
        let pmv = pmv_comfort.calculate_pmv(14.0, 14.0, 0.5, 0.1, 1.0, 1.0);
        assert!(
            (pmv - (-2.496170958)).abs() < 1e-9,
            "fixed ISO evaporative split at 14 C / clo 1.0: got {pmv}"
        );
    }
}
