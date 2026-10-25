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

    // Round 99 (ISO 7730:2005 residuals sweep): the clothing model moves
    // to the ISO 7730:2005 Annex D reference-program forms, cross-checked
    // against EnergyPlus ThermalComfort.cc `CalcFangerPMV` ("based on
    // ANSI/ASHRAE Standard 55-2020 and ISO 7730:2005"). Pre-fix constructs
    // that are NOT in ISO, all removed here: f_cl = 1.0 + 0.15·clo (the
    // ASHRAE-Handbook linearization), a t_cl solve linearized at
    // h_r = 4.7 W/m²K, an h_c selection keyed on `vel > 0.1` at
    // |ta − 35|, and the 0.1 m/s velocity floor.
    let i_cl = 0.155 * clo;
    let f_cl = if i_cl < 0.078 {
        1.0 + 1.29 * i_cl
    } else {
        1.05 + 0.645 * i_cl
    };
    let clo_insul = i_cl * f_cl; // ISO: I_cl·f_cl

    // t_cl: implicit clothing-surface balance with the radiative quartic
    // inside, fixed-point iteration in the P1..P4 scaled forms (ISO Annex
    // D program; E+ parameters MaxIter 150, StopIterCrit 0.00015), with
    // h_c re-evaluated each pass as max(12.1·√v, 2.38·|t_cl − ta|^0.25).
    // Absolute-offset convention +273.0 per the ISO Annex D program,
    // uniform with the round-97 quartic below (E+ converts at 273.15 —
    // recorded choice, ~0.2 W/m² class effect on R).
    let p2 = clo_insul * 3.96;
    let p3 = clo_insul * 100.0;
    let p1 = clo_insul * (ta + 273.0);
    let t_tr = (tr + 273.0) / 100.0;
    let p4 = 308.7 - 0.028 * m + p2 * (t_tr * t_tr * t_tr * t_tr);
    let abs_air = ta + 273.0;
    let mut xn = (abs_air + (35.5 - ta) / (3.5 * i_cl + 0.1)) / 100.0;
    let hc_for = 12.1 * vel.sqrt();
    let mut hc = hc_for;
    let mut xf = xn;
    let mut it = 0;
    while ((xn - xf).abs() > 0.00015 || it == 0) && it < 150 {
        xf = (xf + xn) / 2.0;
        let hc_nat = 2.38 * (100.0 * xf - abs_air).abs().powf(0.25);
        hc = hc_for.max(hc_nat);
        xn = (p4 + p1 * hc - p2 * (xf * xf * xf * xf)) / (100.0 + p3 * hc);
        it += 1;
    }
    let t_cl = 100.0 * xn - 273.0;
    let h_c = hc;

    // Dry c/r terms (round 97; ISO 7730:2005): convection from the CLOTHING
    // SURFACE `f_cl·h_c·(t_cl − t_a)` plus the radiative quartic
    // `3.96e-8·f_cl·[(t_cl+273)⁴ − (t_r+273)⁴]`, both driven by the t_cl
    // solved above from ISO's implicit clothing-surface balance (round 99;
    // pre-r99 linearized at h_r = 4.7). Pre-fix
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

    // Round 99: PMV computed unconditionally — ISO 7730:2005 and the E+
    // reference carry no zero arm near neutral (the removed |L| <= 0.1
    // gate quantized the neutral region). The ±4 output clamp is a domain
    // guard, kept.
    let pmv = ((0.303 * (-0.036 * m).exp() + 0.028) * l).clamp(-4.0, 4.0);

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
        // Round 99 (ISO 7730:2005 residuals sweep): the t_cl solve moved
        // from the linearized h_r = 4.7 rational form to ISO's implicit
        // clothing-surface balance (radiative quartic inside, fixed-point
        // iteration, P1..P4 forms per the ISO Annex D / EnergyPlus
        // CalcFangerPMV reference), f_cl to the piecewise 1.0+1.29·i_cl /
        // 1.05+0.645·i_cl form, and h_c to max(12.1·√v, 2.38·|t_cl−ta|^0.25)
        // re-evaluated in-iteration. The |L| <= 0.1 zero arm is gone (not
        // an ISO construct), so neutral is now the true balance crossing:
        // 26.009076839 -> 26.085298390 °C (bisection of the exact
        // operation order, r99_comfort_refs.py). At the pinned neutral
        // L ≈ 0 to machine precision, PMV ≈ 0, PPD = 5.0.
        // History: 28.018371323 (pre-r96) -> 29.033729242 (r96) ->
        // 25.580475167 (r97 dry terms) -> 26.009076839 (r97 respiration,
        // held by r98) -> 26.085298390 (r99 clothing model).
        let m = compute_pmv_ppd_and_adaptive(26.085298390, 0.5, 0.1, 1.0, 0.5);
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
        // vel == 0.1 exactly: h_c = max(12.1·√0.1, 2.38·|t_cl − ta|^0.25);
        // at 27-29 °C the forced arm 3.83 W/m²K wins (pre-round-99 the
        // still-air arm 2.38·|ta − 35|^0.25 was taken unconditionally).
        let cool = compute_pmv_ppd_and_adaptive(27.0, 0.5, 0.1, 1.0, 0.5);
        let warm = compute_pmv_ppd_and_adaptive(29.0, 0.5, 0.1, 1.0, 0.5);
        assert_close(cool.pmv, 0.341105653, "still-air 27 C PMV");
        assert_close(cool.ppd, 7.420324732, "still-air 27 C PPD");
        assert_close(warm.pmv, 1.096578683, "still-air 29 C PMV");
        assert_close(warm.ppd, 30.357703742, "still-air 29 C PPD");
        assert!(
            cool.pmv < warm.pmv,
            "PMV must rise with operative temperature"
        );
        assert!(cool.ppd >= 5.0 && warm.ppd >= 5.0, "PPD >= 5% always");
        let mid = compute_pmv_ppd_and_adaptive(26.0, 0.5, 0.1, 1.0, 0.5);
        // Round 99: with the zero arm gone, 26 °C is simply 0.085 °C below
        // the corrected neutral — slightly cool, continuously.
        assert_close(mid.pmv, -0.031687346, "still-air 26 C PMV");
        assert_close(mid.ppd, 5.020786059, "still-air 26 C PPD");
    }

    #[test]
    fn fanger_moving_air_convection_correlation_engages() {
        // vel > 0.1 -> h_c = 12.1·sqrt(vel); the raised convection cools
        // the balance below the still-air value at 26 C. (Pre-round-97 this
        // clamped at the -4 arm; the corrected dry terms leave it mid-band.
        // The clamp arms are pinned in the extremes test.)
        let fan = compute_pmv_ppd_and_adaptive(26.0, 0.5, 0.8, 1.0, 0.5);
        // Round 98: fixed ISO 7730 evaporative split
        // (-0.298312723 -> -0.981489399). Round 99 (ISO clothing model):
        // h_c = max(12.1·√0.8, 2.38·|t_cl − ta|^0.25) = 10.82 W/m²K
        // (forced arm wins; pre-r99 the same scenario read h_c = 10.82
        // only via the vel > 0.1 branch, and the t_cl solve was
        // linearized) — the stronger in-iteration convection deepens the
        // cool reading: -0.981489399 -> -1.072602286.
        assert_close(fan.pmv, -1.072602286, "fan 26 C PMV");
        assert_close(fan.ppd, 29.274362617, "fan 26 C PPD");
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
        assert_close(dry.pmv, 0.918966774, "dry-air 30 C PMV");
        let humid = compute_pmv_ppd_and_adaptive(30.0, 0.5, 0.1, 1.0, 0.5);
        assert_close(humid.pmv, 1.478136931, "humid 30 C PMV");
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
        assert_close(cold.pmv, -1.407963263, "fixed split 18 C PMV");
        assert_close(cold.ppd, 45.936682908, "fixed split 18 C PPD");
        let sweat = compute_pmv_ppd_and_adaptive(30.0, 0.5, 0.1, 1.2, 0.5);
        assert_close(
            sweat.pmv,
            1.606211963,
            "fixed split met-1.2 sweating 30 C PMV",
        );
        assert_close(
            sweat.ppd,
            56.683297632,
            "fixed split met-1.2 sweating 30 C PPD",
        );
    }
}
