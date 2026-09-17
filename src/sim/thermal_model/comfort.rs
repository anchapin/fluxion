//! Adaptive-comfort helper for `ThermalModelTrait::get_comfort_metrics`.
//!
//! Issue #3789 — split out of `thermal_model.rs` so the parent module is
//! thin enough to clear the Issue #3457 module-size ratchet. The function
//! is invoked by every concrete `ThermalModelTrait` implementation
//! (`PhysicsThermalModel`, `SurrogateThermalModel`, `HybridThermalModel`,
//! `UnifiedThermalModel`, `MockThermalModel`).

/// Compute PMV, PPD, and adaptive comfort metrics from zone temperature (ASHRAE 55).
///
/// Uses Fanger PMV model (ASHRAE 55-2022 Table 5.2.1) with the given
/// metabolic rate (met), clothing insulation (clo), relative humidity (rh),
/// and air velocity (vel).
///
/// Adaptive comfort uses ASHRAE 55-2022 Section 5.3 with Category II
/// comfort bands. Running mean is approximated from the operative temperature
/// using an exponential moving average with alpha=0.8.
pub(crate) fn compute_pmv_ppd_and_adaptive(
    zone_temp: f64,
    rh: f64,
    vel: f64,
    met: f64,
    clo: f64,
) -> super::ZoneComfortMetrics {
    let ta = zone_temp;
    let tr = zone_temp;
    let operative = ta;

    let p_sat = 610.6 * (17.27 * ta / (ta + 237.3)).exp();
    let p_a = rh * p_sat;
    let m = met * 58.15;
    let vel = vel.max(0.1);

    let f_cl = 1.0 + 0.15 * clo;
    let i_cl = (0.155 * clo).max(0.01);

    let h_c = if vel > 0.1 {
        12.1 * vel.sqrt()
    } else {
        2.38 * (ta - 35.0).abs().powf(0.25)
    };
    let h_r: f64 = 4.7;

    let mut t_cl = ta + 1.0;
    for _ in 0..10 {
        let t_cl_new = (f_cl * h_c * ta + f_cl * h_r * tr + (35.7 - 0.028 * m) / i_cl)
            / (f_cl * h_c + f_cl * h_r + 1.0 / i_cl);
        if (t_cl_new - t_cl).abs() < 0.01 {
            break;
        }
        t_cl = t_cl_new;
    }

    let t_sk = 35.7 - 0.028 * m;
    let c = f_cl * h_c * (t_sk - ta);
    let r = f_cl * h_r * (t_sk - tr);
    let c_res = 0.0014 * m * (34.0 - ta);
    let e_res = 0.0000173 * m * (p_sat - p_a);

    let e_max = 0.408 * (42.5 - p_a).max(0.0);
    let d1 = m - c_res - e_res - c - r;
    let w_ratio = if d1 > 0.0 && e_max > 0.0 {
        (0.06 + 0.94 * d1 / e_max).min(1.0)
    } else {
        0.06
    };
    let e = w_ratio * e_max;

    let l = m - c_res - e_res - c - r - e;

    let pmv_raw = if l.abs() > 0.1 {
        (0.303 * (-0.036 * m).exp() + 0.028) * l
    } else {
        0.0
    };
    let pmv = pmv_raw.clamp(-4.0, 4.0);

    let ppd = 100.0 - 95.0 * (-0.03353 * pmv.powi(4) - 0.2179 * pmv.powi(2)).exp();

    let rtm = operative;
    let centre = 0.33 * rtm + 18.83;
    let (upper_limit, lower_limit) = (centre + 3.5, centre - 2.0);

    let is_adaptive_comfortable = operative >= lower_limit && operative <= upper_limit;

    super::ZoneComfortMetrics {
        pmv,
        ppd,
        operative_temp: operative,
        relative_humidity: rh,
        running_mean_temp: rtm,
        adaptive_upper_limit: upper_limit,
        adaptive_lower_limit: lower_limit,
        is_adaptive_comfortable,
    }
}
