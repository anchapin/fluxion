use std::f64::consts::E;

const C1: f64 = -5800.2206;
const C2: f64 = 1.3914993;
const C3: f64 = -0.048640239;
const C4: f64 = 0.000041764768;
const C5: f64 = -0.000000014452093;
const C6: f64 = 0.0;
const C7: f64 = 6.5459673;

const R_A: f64 = 287.055;
const M_W_RATIO: f64 = 0.62198;
const CP_A: f64 = 1.006;
const CP_V: f64 = 1.86;
const H_FG: f64 = 2501.0;

#[derive(Debug, Clone, PartialEq)]
pub struct PsychrometricState {
    pub t_db: f64,
    pub rh: f64,
    pub pressure: f64,
    pub p_ws: f64,
    pub p_w: f64,
    pub humidity_ratio: f64,
    pub enthalpy: f64,
    pub dew_point: f64,
    pub wet_bulb: f64,
    pub specific_volume: f64,
    pub degree_of_saturation: f64,
}

pub fn calculate(t_db: f64, rh: f64, pressure: f64) -> PsychrometricState {
    assert!((0.0..=100.0).contains(&rh), "RH must be 0-100%");
    assert!(pressure > 0.0, "Pressure must be positive");

    let rh_frac = rh / 100.0;
    let t_k = t_db + 273.15;

    let p_ws = saturation_pressure(t_k);
    let p_w = rh_frac * p_ws;

    let w = M_W_RATIO * p_w / (pressure - p_w);
    let w_s = M_W_RATIO * p_ws / (pressure - p_ws);

    let h = CP_A * t_db + w * (H_FG + CP_V * t_db);
    let v = R_A * t_k * (1.0 + 1.6078 * w) / pressure;
    let t_dp = dew_point_from_vapor_pressure(p_w);
    let t_wb = wet_bulb_temperature(t_db, w, pressure);
    let mu = w / w_s;

    PsychrometricState {
        t_db,
        rh,
        pressure,
        p_ws,
        p_w,
        humidity_ratio: w,
        enthalpy: h,
        dew_point: t_dp,
        wet_bulb: t_wb,
        specific_volume: v,
        degree_of_saturation: mu,
    }
}

fn saturation_pressure(t_k: f64) -> f64 {
    let ln_pws = C1 / t_k
        + C2
        + C3 * t_k
        + C4 * t_k * t_k
        + C5 * t_k.powi(3)
        + C6 * t_k.powi(4)
        + C7 * t_k.ln();
    E.powf(ln_pws)
}

fn dew_point_from_vapor_pressure(p_w: f64) -> f64 {
    if p_w <= 0.0 {
        return f64::NAN;
    }
    let alpha = (p_w / 610.94).ln();
    243.04 * alpha / (17.625 - alpha)
}

fn wet_bulb_temperature(t_db: f64, w: f64, pressure: f64) -> f64 {
    let mut t_wb = t_db;
    let h_actual = CP_A * t_db + w * (H_FG + CP_V * t_db);
    let t_dp = dew_point_from_vapor_pressure(w * pressure / (M_W_RATIO + w));

    for _ in 0..200 {
        let t_wb_k = t_wb + 273.15;
        let p_ws_wb = saturation_pressure(t_wb_k);
        let w_s_wb = M_W_RATIO * p_ws_wb / (pressure - p_ws_wb);
        let h_wb = CP_A * t_wb + w_s_wb * (H_FG + CP_V * t_wb);
        let residual = h_actual - h_wb;

        if residual.abs() < 1e-8 {
            return t_wb;
        }

        let cp_eff = CP_A
            + w_s_wb * CP_V
            + M_W_RATIO * (H_FG + CP_V * t_wb) * p_ws_wb * C7 / (t_wb_k * pressure)
            - M_W_RATIO * p_ws_wb * (H_FG + CP_V * t_wb) / (pressure - p_ws_wb).max(1.0);

        if cp_eff.abs() < 1e-15 {
            break;
        }
        let step = residual / cp_eff;
        let clamped_step = step.max(-10.0).min(10.0);
        t_wb -= clamped_step;
        let lower = if t_dp.is_finite() { t_dp - 1.0 } else { -60.0 };
        t_wb = t_wb.clamp(lower, t_db + 1.0);
    }
    t_wb
}

#[cfg(test)]
mod tests {
    use super::*;

    const T_DB: f64 = 25.0;
    const RH: f64 = 50.0;
    const P: f64 = 101325.0;
    const TOL: f64 = 0.02;

    fn ref_state() -> PsychrometricState {
        calculate(T_DB, RH, P)
    }

    #[test]
    fn test_saturation_pressure() {
        let state = ref_state();
        let expected_pws = 3168.5;
        assert!(
            (state.p_ws - expected_pws).abs() < 1.0,
            "P_ws: got {}, expected ~{}",
            state.p_ws,
            expected_pws
        );
    }

    #[test]
    fn test_vapor_pressure() {
        let state = ref_state();
        let expected_pw = 1584.3;
        assert!(
            (state.p_w - expected_pw).abs() < 1.0,
            "P_w: got {}, expected ~{}",
            state.p_w,
            expected_pw
        );
    }

    #[test]
    fn test_humidity_ratio() {
        let state = ref_state();
        let expected_w = 0.00988;
        assert!(
            (state.humidity_ratio - expected_w).abs() < 0.0001,
            "W: got {:.6}, expected ~{}",
            state.humidity_ratio,
            expected_w
        );
    }

    #[test]
    fn test_enthalpy() {
        let state = ref_state();
        let expected_h = 50.32;
        assert!(
            (state.enthalpy - expected_h).abs() < TOL,
            "h: got {:.4}, expected ~{}",
            state.enthalpy,
            expected_h
        );
    }

    #[test]
    fn test_dew_point() {
        let state = ref_state();
        let expected_dp = 13.89;
        assert!(
            (state.dew_point - expected_dp).abs() < 0.1,
            "T_dp: got {:.4}, expected ~{}",
            state.dew_point,
            expected_dp
        );
    }

    #[test]
    fn test_specific_volume() {
        let state = ref_state();
        let expected_v = 0.858;
        assert!(
            (state.specific_volume - expected_v).abs() < 0.005,
            "v: got {:.6}, expected ~{}",
            state.specific_volume,
            expected_v
        );
    }

    #[test]
    fn test_wet_bulb() {
        let state = ref_state();
        let expected_wb = 17.95;
        assert!(
            (state.wet_bulb - expected_wb).abs() < 0.2,
            "T_wb: got {:.4}, expected ~{}",
            state.wet_bulb,
            expected_wb
        );
    }

    #[test]
    fn test_degree_of_saturation() {
        let state = ref_state();
        assert!(
            (state.degree_of_saturation - 0.50).abs() < 0.01,
            "mu: got {:.6}, expected ~0.50",
            state.degree_of_saturation
        );
    }

    #[test]
    fn test_saturated_air() {
        let state = calculate(20.0, 100.0, P);
        assert!(
            (state.dew_point - 20.0).abs() < 0.05,
            "At 100% RH, dew point should equal dry-bulb: got {}",
            state.dew_point
        );
        assert!(
            (state.wet_bulb - 20.0).abs() < 0.05,
            "At 100% RH, wet-bulb should equal dry-bulb: got {}",
            state.wet_bulb
        );
    }

    #[test]
    fn test_dry_air() {
        let state = calculate(30.0, 0.0, P);
        assert!(
            state.humidity_ratio < 1e-10,
            "At 0% RH, humidity ratio should be ~0: got {}",
            state.humidity_ratio
        );
        let expected_h = CP_A * 30.0;
        assert!(
            (state.enthalpy - expected_h).abs() < 0.01,
            "h: got {:.4}, expected ~{}",
            state.enthalpy,
            expected_h
        );
    }

    #[test]
    fn test_consistency_dew_point_roundtrip() {
        let state = ref_state();
        let p_w_recovered = 610.94 * (17.625 * state.dew_point / (243.04 + state.dew_point)).exp();
        assert!(
            (p_w_recovered - state.p_w).abs() < 1.0,
            "Roundtrip vapor pressure: got {:.2}, expected {:.2}",
            p_w_recovered,
            state.p_w
        );
    }

    #[test]
    fn test_enthalpy_from_components() {
        let state = ref_state();
        let h_sensible = CP_A * T_DB;
        let h_latent = state.humidity_ratio * H_FG;
        let h_vapor_sensible = state.humidity_ratio * CP_V * T_DB;
        let h_total = h_sensible + h_latent + h_vapor_sensible;
        assert!(
            (state.enthalpy - h_total).abs() < 0.01,
            "Enthalpy decomposition: got {:.4}, summed {:.4}",
            state.enthalpy,
            h_total
        );
    }

    #[test]
    #[should_panic(expected = "RH must be 0-100%")]
    fn test_invalid_rh_high() {
        calculate(25.0, 101.0, P);
    }

    #[test]
    #[should_panic(expected = "RH must be 0-100%")]
    fn test_invalid_rh_negative() {
        calculate(25.0, -5.0, P);
    }

    #[test]
    #[should_panic(expected = "Pressure must be positive")]
    fn test_invalid_pressure() {
        calculate(25.0, 50.0, 0.0);
    }
}
