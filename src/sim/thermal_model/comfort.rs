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

    // Dry c/r terms (round 97; ISO 7730:2005): convection from the CLOTHING
    // SURFACE `f_cl·h_c·(t_cl − t_a)` plus the radiative quartic
    // `3.96e-8·f_cl·[(t_cl+273)⁴ − (t_r+273)⁴]`, both driven by the t_cl
    // solved above from the linearized clothing-surface balance. Pre-fix
    // both terms used the skin temperature t_sk against ta/tr, overstating
    // the dry loss at balance and holding neutral ~3 °C warm after the
    // round-96 evaporative units fix.
    let c = f_cl * h_c * (t_cl - ta);
    let r = 3.96e-8 * f_cl * ((t_cl + 273.0).powi(4) - (tr + 273.0).powi(4));
    let c_res = 0.0014 * m * (34.0 - ta);
    // Latent respiration (round 97; ISO 7730:2005 HL3): 1.7e-5·M·(5867 − p_a)
    // W/m² with p_a in Pa. 5867 Pa is the saturated water-vapour pressure of
    // the EXHALED air (the ISO 7730 Basic program carries the same constant;
    // EnergyPlus prints 0.000017·M·(5867 − VapPress)). Pre-fix this was
    // 1.73e-5·M·(p_sat − p_a) — the Fanger-family coefficient paired with the
    // ambient-air saturation pressure instead of the exhaled-air constant,
    // which roughly halved the term at room temperatures (1.59 vs 4.23 W/m²
    // at 25 °C / 50 % rh).
    let e_res = 0.000017 * m * (5867.0 - p_a);

    // Evaporative heat loss from the skin (round 98, Issue #4598): ISO
    // 7730:2005's FIXED split — E_dif + E_rsw — replaces the wettedness
    // regulator (`w·e_max`, w = clamp(0.06 + 0.94·d1/e_max, 0.06, 1.0)) this
    // copy carried since the round-96 units fix:
    //
    //   HL1 — diffusion through the skin, always present, unclamped as in
    //         the ISO Basic program and EnergyPlus
    //         (EvapHeatLossDiff = 0.00305·(5733 − 6.99·H − P_a)):
    //         E_dif = 3.05e-3·(5733 − 6.99·M − p_a) W/m², M in W/m², p_a in
    //         Pa (work w = 0 here, so M = MW). This same bracket arrived in
    //         round 96 as the e_max capacity (itself replacing the mmHg-form
    //         `0.408·(42.5 − p_a)` that zeroed at any normal humidity).
    //   Sweating — E_rsw = 0.42·max(M − 58.15, 0) W/m², active only above
    //         the comfort metabolic state 58.15 W/m² (1 met; EnergyPlus
    //         guards the same way: E_rsw = 0.42·(H − 58.2) for H > 58.2,
    //         else 0).
    //
    // Premise (verified 2026-10-25 against the EnergyPlus Engineering
    // Reference Fanger section and the ISO 7730:2005 equation set): the
    // Fanger/ISO PMV carries NO wettedness feedback — skin wettedness is a
    // two-node-model (Pierce/KSU) construct. The regulator floored the cold
    // side at 6 % of capacity (~0.8 W/m² where E_dif ≈ 13.1 W/m² at
    // 18 °C / clo 1.0 — reads warm) and saturated at e_max on the warm side,
    // never crediting sweating above 1 met. Beyond the ISO validity domain
    // (hot-humid extremes) the unclamped E_dif bracket goes negative;
    // ISO's reference program computes it unclamped and the PMV ±4 clamp
    // absorbs that regime.
    let e_dif = 3.05e-3 * (5733.0 - 6.99 * m - p_a);
    let e_rsw = 0.42 * (m - 58.15).max(0.0);
    let e = e_dif + e_rsw;

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
    /// which engages at normal humidity. Round 97 (this fix): the dry c/r
    /// terms moved from t_sk against ta/tr to ISO 7730:2005's
    /// clothing-surface form — `f_cl·h_c·(t_cl − t_a)` and the radiative
    /// quartic `3.96e-8·f_cl·[(t_cl+273)⁴ − (t_r+273)⁴]` — the residual
    /// flagged above; neutral came down 29.033729242 -> 25.580475167 °C
    /// (dry terms), then to 26.009076839 °C (respiration to the 5867 Pa
    /// exhaled-air form), inside the textbook window.
    /// Round 98 (2026-10-25, Issue #4598): the residual above is fixed —
    /// the evaporative model is now ISO 7730:2005's fixed HL1 diffusion +
    /// sweating split (E_dif always present, unclamped; E_rsw =
    /// 0.42·max(M − 58.15, 0)), replacing the wettedness regulator
    /// (`w·e_max`, w from 0.06 to 1) that floored the cold side at 6 % of
    /// capacity (~12.3 W/m² of missing diffusion at 18 °C) and never
    /// credited sweating above 1 met.

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
        // Round 96 moved neutral 28.018371323 -> 29.033729242 °C (evaporative
        // capacity to the ISO 7730 Pa form). Round 97 stage 1 moves it to
        // 25.580475167 °C as the dry c/r terms move to ISO's
        // clothing-surface t_cl form (bisection of the exact operation
        // order, r97_comfort_refs.py). Stage 2 (respiration to the 5867 Pa
        // exhaled-air form) lands it at 26.009076839 °C — the pinned value.
        // Round 98 (#4598 fixed evaporative split) leaves it bit-identical:
        // at met 1.0 near neutral the regulator already sat at full
        // diffusive capacity, so the fixed E_dif crosses the same balance.
        let m = compute_pmv_ppd_and_adaptive(26.009076839, 0.5, 0.1, 1.0, 0.5);
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
        assert_close(cool.pmv, 0.382578260, "still-air 27 C PMV");
        assert_close(cool.ppd, 8.048119849, "still-air 27 C PPD");
        assert_close(warm.pmv, 1.154399049, "still-air 29 C PMV");
        assert_close(warm.ppd, 33.049831077, "still-air 29 C PPD");
        assert!(
            cool.pmv < warm.pmv,
            "PMV must rise with operative temperature"
        );
        assert!(cool.ppd >= 5.0 && warm.ppd >= 5.0, "PPD >= 5% always");
        let mid = compute_pmv_ppd_and_adaptive(26.0, 0.5, 0.1, 1.0, 0.5);
        assert_close(
            mid.pmv,
            0.0,
            "still-air 26 C PMV (zero arm at the corrected neutral)",
        );
        assert_close(mid.ppd, 5.0, "still-air 26 C PPD (Fanger lower bound)");
    }

    #[test]
    fn fanger_moving_air_convection_correlation_engages() {
        // vel > 0.1 -> h_c = 12.1·sqrt(vel); the raised convection cools
        // the balance below the still-air value at 26 C. (Pre-round-97 this
        // clamped at the -4 arm; the corrected dry terms leave it mid-band.
        // The clamp arms are pinned in the extremes test.)
        let fan = compute_pmv_ppd_and_adaptive(26.0, 0.5, 0.8, 1.0, 0.5);
        // Round 98 (Issue #4598, fixed ISO 7730 evaporative split): the
        // raised convection puts this scenario below the balance, where
        // the wettedness regulator floored the diffusive loss; the fixed
        // E_dif restores it. -0.298312723 -> -0.981489399.
        assert_close(fan.pmv, -0.981489399, "fan 26 C PMV");
        assert_close(fan.ppd, 25.346776419, "fan 26 C PPD");
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
        assert_close(dry.pmv, 0.979980337, "dry-air 30 C PMV");
        let humid = compute_pmv_ppd_and_adaptive(30.0, 0.5, 0.1, 1.0, 0.5);
        assert_close(humid.pmv, 1.539150494, "humid 30 C PMV");
        assert!(
            dry.pmv < humid.pmv,
            "the evaporative term must reduce PMV as air dries"
        );
    }

    #[test]
    fn fanger_extremes_clamp_pmv_and_break_adaptive_comfort() {
        // Both clamp arms: pmv_raw beyond ±4 saturates, PPD is symmetric
        // in pmv² so both extremes share the same PPD, and the adaptive
        // band must reject both. Round 98 (Issue #4598, fixed split): at
        // 50 C / 90 % rh p_a (11.1 kPa) lies far outside the ISO validity
        // domain, the unclamped E_dif bracket goes negative (as in the ISO
        // Basic program / EnergyPlus EvapHeatLossDiff) and met-2 sweating
        // adds 24.42 W/m² — L stays ≈ +198 W/m² and the +4 clamp arm
        // holds. On the cold arm the fixed E_dif (16.0 W/m² vs the old
        // regulator's 0.06 floor) deepens L to ≈ −243; the −4 clamp arm
        // holds.
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

    #[test]
    fn fanger_iso7730_fixed_evaporative_split_cold_side_and_sweating() {
        // Round 98 (Issue #4598): the evaporative model is ISO 7730:2005's
        // FIXED split — E_dif (always present) + 0.42·max(M − 58.15, 0)
        // (sweating above 1 met) — replacing the wettedness regulator
        // (w = clamp(0.06 + 0.94·d1/e_max, 0.06, 1.0); e = w·e_max). The
        // regulator floored the cold side at 6 % of capacity: at
        // 18 °C / clo 1.0 / 50 % rh it carried e ≈ 0.786 W/m² where ISO's
        // fixed E_dif is ≈ 13.099 W/m² (+12.31 W/m² of restored diffusive
        // loss — the issue's reading-warm defect). Sweating case: at
        // met 1.2 the ISO sweating term credits 0.42·(69.78 − 58.15) =
        // 4.885 W/m² the regulator never credited (it saturated at e_max).
        // Reference values from Python replication of the exact operation
        // order (r98_comfort_refs.py); ±1e-9 is cross-language libm ulp
        // headroom, not a physics tolerance.
        let cold = compute_pmv_ppd_and_adaptive(18.0, 0.5, 0.1, 1.0, 1.0);
        assert_close(cold.pmv, -1.416959230, "fixed split 18 C PMV");
        assert_close(cold.ppd, 46.417718088, "fixed split 18 C PPD");
        let sweat = compute_pmv_ppd_and_adaptive(30.0, 0.5, 0.1, 1.2, 0.5);
        assert_close(
            sweat.pmv,
            1.651473783,
            "fixed split met-1.2 sweating 30 C PMV",
        );
        assert_close(
            sweat.ppd,
            59.139238541,
            "fixed split met-1.2 sweating 30 C PPD",
        );
    }
}
