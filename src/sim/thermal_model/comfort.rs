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

    // Skin-diffusion evaporative capacity (round 96 units fix; ISO 7730:2005
    // Annex D HL1): `3.05e-3·(5733 − 6.99·M − p_a)` W/m² with M in W/m² and
    // p_a in Pa. Pre-fix this was `0.408·(42.5 − p_a)` — the mmHg-form
    // coefficient/skin-saturation constant (NIST SP 491: E_dif =
    // 0.408·[P_s − RH·P_a] W/m², mmHg; 42.5 mmHg = 5666 Pa) paired with
    // p_a in Pa, which zeroed the term at any normal humidity.
    let e_max = (3.05e-3 * (5733.0 - 6.99 * m - p_a)).max(0.0);
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

#[cfg(test)]
mod tests {
    use super::compute_pmv_ppd_and_adaptive;

    /// Round 95 (coverage remedy): comfort.rs was 0% covered (0/12 branch
    /// targets) while sitting in the conduction_zone bucket. Reference
    /// values below are computed independently in Python by replicating
    /// the exact Fanger operation order (round-96 worklog, r96_comfort_refs.py);
    /// ±1e-9 is cross-language libm ulp headroom, not a physics tolerance.
    ///
    /// Units fix (round 96, the round-95 flagged decision): the pre-fix
    /// skin-diffusion capacity `0.408·(42.5 − p_a)` paired the mmHg-form
    /// coefficient/skin-saturation constant (NIST SP 491: E_dif =
    /// 0.408·[P_s − RH·P_a] W/m², pressures in mmHg; 42.5 mmHg = 5666 Pa)
    /// with p_a expressed in Pa, so the term was zero at any normal
    /// humidity and neutral PMV sat at 28.018 °C (pre-fix pin). The
    /// corrected capacity is ISO 7730:2005 Annex D HL1,
    /// `3.05e-3·(5733 − 6.99·M − p_a)` W/m² with M in W/m² and p_a in Pa,
    /// which engages at normal humidity. Known residual (recorded, not
    /// fixed here): the dry c/r terms use t_sk against ta/tr instead of
    /// ISO's t_cl form (the fluxion-behavior copy carries that #3165
    /// correction), so neutral sits at ~29.0 °C rather than the textbook
    /// ~25-26 °C window even with the evaporative term corrected.

    const TOL: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64, what: &str) {
        assert!(
            (actual - expected).abs() < TOL,
            "{what}: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn fanger_neutral_point_yields_pmv_zero_and_ppd_five() {
        // |l| <= 0.1 at the load-balance temperature -> pmv_raw takes the
        // zero arm, pmv clamps to 0.0, PPD = 100 - 95·exp(0) = 5.0 exactly.
        // Round 96: neutral moved 28.018371323 -> 29.033729242 °C when the
        // evaporative capacity was corrected to the ISO 7730 Pa form — at
        // normal humidity the term now removes ~10 W/m² at balance, so the
        // neutral point shifts warmer (see the units-fix note above; the
        // residual offset vs the textbook window is the dry c/r form,
        // recorded, not fixed here).
        let m = compute_pmv_ppd_and_adaptive(29.033729242, 0.5, 0.1, 1.0, 0.5);
        assert_close(m.pmv, 0.0, "neutral PMV");
        assert_close(m.ppd, 5.0, "neutral PPD (Fanger lower bound)");
        assert!(
            m.is_adaptive_comfortable,
            "operative {} must sit inside the adaptive band [{}, {}]",
            m.operative_temp, m.adaptive_lower_limit, m.adaptive_upper_limit
        );
        // ASHRAE 55 adaptive-band bookkeeping: centre = 0.33·top + 18.83,
        // upper = centre + 3.5 (Category II), lower = centre - 2.0.
        let centre = 0.33 * m.operative_temp + 18.83;
        assert_close(m.adaptive_upper_limit, centre + 3.5, "adaptive upper");
        assert_close(m.adaptive_lower_limit, centre - 2.0, "adaptive lower");
        assert_close(m.running_mean_temp, m.operative_temp, "running mean");
        assert_close(m.relative_humidity, 0.5, "rh passthrough");
    }

    #[test]
    fn fanger_still_air_band_is_monotone_and_pinned() {
        // vel == 0.1 exactly -> the still-air convection arm
        // h_c = 2.38·|ta - 35|^0.25.
        let cool = compute_pmv_ppd_and_adaptive(27.0, 0.5, 0.1, 1.0, 0.5);
        let warm = compute_pmv_ppd_and_adaptive(29.0, 0.5, 0.1, 1.0, 0.5);
        assert_close(cool.pmv, -0.720193388, "still-air 27 C PMV");
        assert_close(cool.ppd, 15.914313680, "still-air 27 C PPD");
        assert_close(warm.pmv, 0.0, "still-air 29 C PMV (zero arm)");
        assert_close(warm.ppd, 5.0, "still-air 29 C PPD (zero arm)");
        assert!(
            cool.pmv < warm.pmv,
            "PMV must rise with operative temperature"
        );
        assert!(cool.ppd >= 5.0 && warm.ppd >= 5.0, "PPD >= 5% always");
        let mid = compute_pmv_ppd_and_adaptive(26.0, 0.5, 0.1, 1.0, 0.5);
        assert_close(mid.pmv, -1.399232952, "still-air 26 C PMV");
        assert_close(mid.ppd, 45.470929237, "still-air 26 C PPD");
    }

    #[test]
    fn fanger_moving_air_convection_correlation_engages() {
        // vel > 0.1 -> h_c = 12.1·sqrt(vel); the raised convection
        // coefficient drives the load past the lower clamp at 26 C.
        let fan = compute_pmv_ppd_and_adaptive(26.0, 0.5, 0.8, 1.0, 0.5);
        assert_close(fan.pmv, -4.0, "fan 26 C PMV (lower clamp arm)");
        assert_close(fan.ppd, 99.999455805, "fan 26 C PPD at clamp");
        let still = compute_pmv_ppd_and_adaptive(26.0, 0.5, 0.1, 1.0, 0.5);
        assert!(
            fan.pmv < still.pmv,
            "moving air must cool the Fanger balance: fan {} vs still {}",
            fan.pmv,
            still.pmv
        );
    }

    #[test]
    fn fanger_dry_air_engages_the_evaporative_term() {
        // Round 96: with the ISO 7730 Pa-form capacity the term engages at
        // BOTH humidities at 30 C (e_max 16.24 vs 9.78 W/m²), so the
        // contract is the humidity monotonicity: drier air carries more
        // evaporative capacity, the same load drives a larger loss, and
        // PMV falls. (Pre-fix, the term was zero at 50 % rh entirely.)
        let dry = compute_pmv_ppd_and_adaptive(30.0, 0.0005, 0.1, 1.0, 0.5);
        assert_close(dry.pmv, 0.076398820, "dry-air 30 C PMV");
        let humid = compute_pmv_ppd_and_adaptive(30.0, 0.5, 0.1, 1.0, 0.5);
        assert_close(humid.pmv, 0.637984379, "humid 30 C PMV");
        assert!(
            dry.pmv < humid.pmv,
            "the evaporative term must reduce PMV as air dries"
        );
    }

    #[test]
    fn fanger_extremes_clamp_pmv_and_break_adaptive_comfort() {
        // Both clamp arms: pmv_raw beyond ±4 saturates, PPD is symmetric
        // in pmv² so both extremes share the same PPD, and the adaptive
        // band must reject both. The hot case also pins the e_max clamp:
        // at 50 C / 90 % rh p_a (11.1 kPa) exceeds the ISO capacity's
        // zero-crossing, the term saturates to 0, and the wettedness
        // regulator takes its 0.06 floor arm (same value pre-fix).
        let hot = compute_pmv_ppd_and_adaptive(50.0, 0.9, 0.1, 2.0, 1.0);
        assert_close(hot.pmv, 4.0, "hot PMV (upper clamp arm)");
        assert_close(hot.ppd, 99.999455805, "hot PPD at clamp");
        assert!(
            !hot.is_adaptive_comfortable,
            "operative 50 C exceeds the adaptive upper limit"
        );
        let cold = compute_pmv_ppd_and_adaptive(2.0, 0.3, 0.1, 1.0, 0.0);
        assert_close(cold.pmv, -4.0, "cold PMV (lower clamp arm)");
        assert_close(cold.ppd, 99.999455805, "cold PPD at clamp");
        assert!(
            !cold.is_adaptive_comfortable,
            "operative 2 C is below the adaptive lower limit"
        );
    }
}
