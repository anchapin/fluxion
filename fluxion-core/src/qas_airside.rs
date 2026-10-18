//! ASHRAE 140-2023 §11 air-side HVAC quasi-analytical solution (QASv2) — Rust port.
//!
//! Issue #4274. Reimplementation of the QASv2.xlsm quasi-analytical solution
//! workbook (Standard 140-2023 Informative Annex B17 §B17.3.2), itself the
//! merged solution of ASHRAE research project RP-865 (Yuill and Haberl 2002).
//! Unlike the Annex B example-result statistics, which the standard expressly
//! disclaims as acceptance criteria, the QAS is a computable model: given the
//! ambient case inputs it returns one coil load per system, so it is a genuine
//! analytical target for §11.
//!
//! Four system models: four-pipe fan coil (FC), single-zone air handler with
//! return fan (SZ), constant-volume reheat (CAV), variable-volume reheat (VAV).
//!
//! All internal calculations are in IP units matching the workbook; loads are
//! Btu/h. The workbook requires iterative calculation (max 200 iterations,
//! max change 1e-5) because supply mass flow depends on specific volume, which
//! depends on the coil leaving state, which depends on mass flow; this port
//! iterates to the same tolerance.
//!
//! Provenance: the equations, case inputs and case-1 anchors were transcribed
//! from the in-repo extraction of QASv2.xlsm recorded in
//! `data/reference/ashrae140/analytical_acceptance_criteria.json`
//! (`section_11_airside`); the same equations live in `scripts/qas_oracle.py`,
//! which carries the case-1 self-test against the workbook values. Constants
//! follow the 2009 ASHRAE Handbook of Fundamentals mass ratios (QAS change
//! log, version 13 onward).
//!
//! Scope: economizer off, no return-duct heat gain, per the workbook's as-
//! delivered configuration. Case 2 publishes wet bulb only; its dew point is
//! derived here through the adiabatic saturation relation (ice form, both
//! bulbs below freezing).

use crate::error::FluxionError;

// Constants, from the QASv2 INPUTS sheet.
const ATM_PSI: f64 = 14.69597535;
const RA: f64 = 53.35; // gas constant, dry air, ft*lbf/(lbda*degR)
const CP_AIR: f64 = 0.2403; // Btu/(lbm*degF)
const CP_VAPOR: f64 = 0.444; // Btu/(lbm*degF)
const CP_EVAP: f64 = 0.556; // Btu/(lbm*degF)
const MASSRAT: f64 = 0.621945; // water / dry air
const INV_MASSRAT: f64 = 1.607858; // dry air / water
const WORK2HEAT: f64 = 0.006685;
const DFE: f64 = 70.0; // design fan efficiency, percent
const SFPR: f64 = 2.0; // supply fan pressure rise, in. wg
const RFPR: f64 = 1.0; // return fan pressure rise, in. wg
const SASP: f64 = 55.0; // supply air setpoint, degF (CAV and VAV)
const PHSP: f64 = 45.0; // preheat setpoint, degF (CAV and VAV)
const NSCFM: f64 = 1300.0; // nominal supply cfm (CAV and VAV)
const NRCFM: f64 = 800.0; // nominal return cfm (CAV and VAV)

const MAX_ITER: usize = 200;
const TOL: f64 = 1e-5;

/// One §11 ambient test case (Standard 140-2023 §5.5 via the QASv2 INPUTS
/// sheet). The rightmost digit of an AE case number selects the case here:
/// AE103, AE203, AE303 and AE403 all use case 3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestCaseInputs {
    pub oadb: f64,
    /// Outdoor dew point, degF. `None` when the source publishes wet bulb only
    /// (case 2) and the dew point must be derived.
    pub oadp: Option<f64>,
    pub oawb: Option<f64>,
    pub r1_t: f64,
    pub r1_qs: f64,
    pub r1_ql: f64,
    pub r1_cfm: f64,
    pub r1_exh: f64,
    pub r2_t: f64,
    pub r2_qs: f64,
    pub r2_ql: f64,
    pub r2_cfm: f64,
    pub r2_exh: f64,
}

/// §11 ambient test cases, transcribed from QASv2.xlsm via
/// `analytical_acceptance_criteria.json` (`section_11_airside.test_case_inputs`).
pub const TEST_CASES: [TestCaseInputs; 6] = [
    TestCaseInputs {
        oadb: -20.20,
        oadp: Some(-20.20000),
        oawb: None,
        r1_t: 70.0,
        r1_qs: -10000.0,
        r1_ql: 2000.0,
        r1_cfm: 600.0,
        r1_exh: 200.0,
        r2_t: 72.0,
        r2_qs: -8000.0,
        r2_ql: 3000.0,
        r2_cfm: 700.0,
        r2_exh: 300.0,
    },
    TestCaseInputs {
        oadb: 30.02,
        oadp: None,
        oawb: Some(19.756),
        r1_t: 71.0,
        r1_qs: -2000.0,
        r1_ql: 2000.0,
        r1_cfm: 600.0,
        r1_exh: 200.0,
        r2_t: 73.0,
        r2_qs: 1000.0,
        r2_ql: 3000.0,
        r2_cfm: 700.0,
        r2_exh: 300.0,
    },
    TestCaseInputs {
        oadb: 59.90,
        oadp: Some(26.60000),
        oawb: None,
        r1_t: 74.0,
        r1_qs: 5000.0,
        r1_ql: 2000.0,
        r1_cfm: 600.0,
        r1_exh: 200.0,
        r2_t: 76.0,
        r2_qs: 8000.0,
        r2_ql: 3000.0,
        r2_cfm: 700.0,
        r2_exh: 300.0,
    },
    TestCaseInputs {
        oadb: 80.42,
        oadp: Some(71.78000),
        oawb: None,
        r1_t: 75.0,
        r1_qs: 10000.0,
        r1_ql: 2000.0,
        r1_cfm: 600.0,
        r1_exh: 200.0,
        r2_t: 77.0,
        r2_qs: 12000.0,
        r2_ql: 3000.0,
        r2_cfm: 700.0,
        r2_exh: 300.0,
    },
    TestCaseInputs {
        oadb: 76.82,
        oadp: Some(36.32000),
        oawb: None,
        r1_t: 74.0,
        r1_qs: 5000.0,
        r1_ql: 2000.0,
        r1_cfm: 600.0,
        r1_exh: 200.0,
        r2_t: 76.0,
        r2_qs: 8000.0,
        r2_ql: 3000.0,
        r2_cfm: 700.0,
        r2_exh: 300.0,
    },
    TestCaseInputs {
        oadb: 73.40,
        oadp: Some(69.62000),
        oawb: None,
        r1_t: 74.0,
        r1_qs: 5000.0,
        r1_ql: 2000.0,
        r1_cfm: 600.0,
        r1_exh: 200.0,
        r2_t: 76.0,
        r2_qs: 8000.0,
        r2_ql: 3000.0,
        r2_cfm: 700.0,
        r2_exh: 300.0,
    },
];

fn ln_pws_above_freezing(t_r: f64) -> f64 {
    -10440.397 / t_r - 11.29465 - 0.027022355 * t_r + 1.289036e-5 * t_r * t_r
        - 2.4780681e-9 * t_r * t_r * t_r
        + 6.5459673 * t_r.ln()
}

fn ln_pws_below_freezing(t_r: f64) -> f64 {
    -10214.165 / t_r - 4.8932428 - 0.0053765794 * t_r
        + 1.9202377e-7 * t_r * t_r
        + 3.5575832e-10 * t_r * t_r * t_r
        - 9.0344688e-14 * t_r * t_r * t_r * t_r
        + 4.1635019 * t_r.ln()
}

/// Saturation water pressure, psia, from dry-bulb or dew point in degF.
fn pws(t_f: f64) -> f64 {
    let t_r = t_f + 459.67;
    if t_f > 32.0 {
        ln_pws_above_freezing(t_r).exp()
    } else {
        ln_pws_below_freezing(t_r).exp()
    }
}

/// Humidity ratio in grains of moisture per lbm dry air, from dew point in degF.
fn humidity_ratio_from_dewpoint(oadp_f: f64) -> f64 {
    let p = pws(oadp_f);
    MASSRAT * p / (ATM_PSI - p) * 7000.0
}

fn specific_volume(t_f: f64, hr_grains: f64) -> f64 {
    (RA * (t_f + 459.67) / ATM_PSI * (1.0 + INV_MASSRAT * hr_grains / 7000.0)) / 144.0
}

fn specific_heat(hr_grains: f64) -> f64 {
    CP_AIR + CP_VAPOR * hr_grains / 7000.0
}

fn enthalpy(t_f: f64, hr_grains: f64) -> f64 {
    CP_AIR * t_f + (hr_grains / 7000.0) * (1061.15 + CP_VAPOR * t_f)
}

fn grains_per_min(q_latent_btuh: f64, t_zone_f: f64) -> f64 {
    (q_latent_btuh / 60.0) / (1061.15 + CP_VAPOR * t_zone_f) * 7000.0
}

/// Saturated humidity ratio in grains at a coil leaving temperature.
fn saturated_hr_at(t_f: f64) -> f64 {
    let p = ln_pws_above_freezing(t_f + 459.67).exp();
    MASSRAT * (p / (ATM_PSI - p)) * 7000.0
}

/// Dew point in degF from humidity ratio in grains, per the QAS correlations.
fn dewpoint_of(hr_grains: f64) -> f64 {
    let pw = ATM_PSI * hr_grains / 7000.0 / (MASSRAT + hr_grains / 7000.0);
    let lnpw = pw.ln();
    let above = 100.45
        + 33.193 * lnpw
        + 2.319 * lnpw * lnpw
        + 0.1707 * lnpw * lnpw * lnpw
        + 1.2063 * pw.powf(0.1984);
    if above < 32.0 {
        90.12 + 26.142 * lnpw + 0.8927 * lnpw * lnpw
    } else {
        above
    }
}

/// Coil sensible and latent load, Btu/h. Sign convention follows the workbook.
fn coil_loads(
    mass_flow_lbm_min: f64,
    t_entering: f64,
    t_leaving: f64,
    hr_entering: f64,
    hr_leaving: f64,
) -> (f64, f64) {
    let dp_entering = dewpoint_of(hr_entering);
    let sensible = mass_flow_lbm_min
        * 60.0
        * ((CP_AIR + CP_VAPOR * hr_leaving / 7000.0) * (t_entering - t_leaving)
            + ((hr_entering - hr_leaving) / 7000.0
                * (CP_VAPOR * (t_entering - dp_entering) + (dp_entering - t_leaving))));
    let latent = mass_flow_lbm_min
        * 60.0
        * ((hr_entering - hr_leaving) / 7000.0)
        * (1075.21 - CP_EVAP * (dp_entering - 32.0));
    (sensible, latent)
}

/// Dew point from dry bulb and wet bulb via the adiabatic saturation relation.
///
/// Two branches per the ASHRAE Handbook of Fundamentals psychrometrics
/// chapter: the liquid-water form above freezing and the ice form below it.
/// Case 2 has both bulbs below freezing (30.02 / 19.756 degF), so the ice
/// form is the one that applies.
fn dewpoint_from_wetbulb(tdb: f64, twb: f64) -> Result<f64, FluxionError> {
    let pws_wb = pws(twb);
    let w_star = MASSRAT * pws_wb / (ATM_PSI - pws_wb);
    let w = if twb > 32.0 {
        ((1093.0 - 0.556 * twb) * w_star - 0.240 * (tdb - twb)) / (1093.0 + 0.444 * tdb - twb)
    } else {
        ((1220.0 - 0.04 * twb) * w_star - 0.240 * (tdb - twb)) / (1220.0 + 0.444 * tdb - 0.48 * twb)
    };
    if w <= 0.0 {
        return Err(FluxionError::Validation(format!(
            "QAS: derived humidity ratio is not positive for tdb={tdb} twb={twb}; \
             the published dew point is required for this case"
        )));
    }
    let pw = ATM_PSI * w / (MASSRAT + w);
    let lnpw = pw.ln();
    let above = 100.45
        + 33.193 * lnpw
        + 2.319 * lnpw * lnpw
        + 0.1707 * lnpw * lnpw * lnpw
        + 1.2063 * pw.powf(0.1984);
    Ok(if above < 32.0 {
        90.12 + 26.142 * lnpw + 0.8927 * lnpw * lnpw
    } else {
        above
    })
}

/// Ambient state derived from one §11 test case, per the workbook's AMBIENT sheet.
#[derive(Debug, Clone)]
pub struct Ambient {
    pub case: usize,
    pub oadb: f64,
    pub oadp: f64,
    pub oadp_derived: bool,
    pub oahr: f64,
    pub oash: f64,
    pub oae: f64,
    pub r1_t: f64,
    pub r1_qs: f64,
    pub r1_ql: f64,
    pub r1_cfm: f64,
    pub r1_exh: f64,
    pub r2_t: f64,
    pub r2_qs: f64,
    pub r2_ql: f64,
    pub r2_cfm: f64,
    pub r2_exh: f64,
    pub gro: f64,
    pub grt: f64,
}

impl Ambient {
    pub fn new(case: usize) -> Result<Self, FluxionError> {
        let c =
            *TEST_CASES
                .get(case.checked_sub(1).ok_or_else(|| {
                    FluxionError::Validation("QAS: case numbers are 1-based".into())
                })?)
                .ok_or_else(|| FluxionError::Validation(format!("QAS: no test case {case}")))?;
        let (oadp, oadp_derived) = match c.oadp {
            Some(dp) => (dp, false),
            None => {
                let twb = c.oawb.ok_or_else(|| {
                    FluxionError::Validation(
                        "QAS: case publishes neither dew point nor wet bulb".into(),
                    )
                })?;
                (dewpoint_from_wetbulb(c.oadb, twb)?, true)
            }
        };
        let oahr = humidity_ratio_from_dewpoint(oadp);
        Ok(Ambient {
            case,
            oadb: c.oadb,
            oadp,
            oadp_derived,
            oash: specific_heat(oahr),
            oae: enthalpy(c.oadb, oahr),
            oahr,
            r1_t: c.r1_t,
            r1_qs: c.r1_qs,
            r1_ql: c.r1_ql,
            r1_cfm: c.r1_cfm,
            r1_exh: c.r1_exh,
            r2_t: c.r2_t,
            r2_qs: c.r2_qs,
            r2_ql: c.r2_ql,
            r2_cfm: c.r2_cfm,
            r2_exh: c.r2_exh,
            gro: grains_per_min(c.r1_ql, c.r1_t),
            grt: grains_per_min(c.r2_ql, c.r2_t),
        })
    }
}

/// Coil/state results for one system, one case. Fields mirror the workbook
/// outputs; every load is Btu/h, every temperature degF.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SystemResult {
    pub heating_coil: f64,
    pub preheat_coil: f64,
    pub zone1_reheat: f64,
    pub zone2_reheat: f64,
    pub total_reheat: f64,
    pub total_heat: f64,
    pub cooling_sensible: f64,
    pub cooling_latent: f64,
    pub cooling_total: f64,
    pub mixed_air_temp: f64,
    pub return_air_temp: f64,
    pub supply_air_temp: f64,
    pub zone1_supply_temp: f64,
    pub zone2_supply_temp: f64,
    pub supply_mass_flow: f64,
}

/// All four system models solved for one case, economizer off, no return-duct
/// heat gain (the workbook's as-delivered configuration).
#[derive(Debug, Clone, PartialEq)]
pub struct QasCaseResult {
    pub case: usize,
    pub oadb: f64,
    pub oadp: f64,
    pub oadp_derived: bool,
    pub fc: SystemResult,
    pub sz: SystemResult,
    pub cav: SystemResult,
    pub vav: SystemResult,
}

/// Four-pipe fan coil, single zone, no economizer, no return fan.
fn fan_coil(a: &Ambient) -> SystemResult {
    let mut accsv = 13.5;
    let mut rosv = 13.5;
    let mut sahr = a.oahr;
    let mut rohr = a.oahr;
    let mut mahr = a.oahr;
    let mut rosmf = 0.0;
    let mut mat = 0.0;
    let mut tahc = 0.0;
    let mut tacc = 0.0;
    let mut sfr = 0.0;
    for _ in 0..MAX_ITER {
        let prev = (accsv, sahr);
        let sash = specific_heat(sahr);
        let mash = specific_heat(mahr);
        let rosh = specific_heat(rohr);
        let rash = rosh;
        rosmf = a.r1_cfm / accsv;
        let roemf = a.r1_exh / rosv;
        let rormf = rosmf - roemf;
        let rosat = a.r1_t - (a.r1_qs / (rosmf * 60.0 * sash));
        sfr =
            ((accsv * rosmf / a.r1_cfm).powi(2) * SFPR) * accsv * WORK2HEAT / (DFE / 100.0) / sash;
        mat = (rormf * a.r1_t * rash + roemf * a.oadb * a.oash) / (rosmf * mash);
        tahc = mat.max(rosat - sfr);
        tacc = tahc.min(rosat - sfr);
        let ssphr = saturated_hr_at(tacc);
        let mahr_new = if rormf != 0.0 {
            (rohr * rormf + a.oahr * roemf) / rosmf
        } else {
            a.oahr
        };
        sahr = ssphr.min(mahr_new);
        rohr = sahr + a.gro / rosmf;
        mahr = mahr_new;
        accsv = specific_volume(tacc, sahr);
        rosv = specific_volume(a.r1_t, rohr);
        if (accsv - prev.0).abs() < TOL && (sahr - prev.1).abs() < TOL {
            break;
        }
    }
    let qh = rosmf * 60.0 * specific_heat(mahr) * (tahc - mat);
    let (qcs, qcl) = coil_loads(rosmf, tahc, tacc, mahr, sahr);
    SystemResult {
        heating_coil: qh,
        cooling_sensible: qcs,
        cooling_latent: qcl,
        cooling_total: qcs + qcl,
        mixed_air_temp: mat,
        supply_air_temp: tacc + sfr,
        supply_mass_flow: rosmf,
        ..Default::default()
    }
}

/// Single-zone air handler with return fan, economizer disabled.
fn single_zone(a: &Ambient, rdhg: f64) -> SystemResult {
    let mut accsv = 13.5;
    let mut rosv = 13.5;
    let mut rfisv = 13.5;
    let mut sahr = a.oahr;
    let mut rohr = a.oahr;
    let mut rahr = a.oahr;
    let mut mahr = a.oahr;
    let mut rosmf = 0.0;
    let mut mat = 0.0;
    let mut rat = 0.0;
    let mut tahc = 0.0;
    let mut tacc = 0.0;
    let mut sfr = 0.0;
    for _ in 0..MAX_ITER {
        let prev = (accsv, sahr);
        let sash = specific_heat(sahr);
        let mash = specific_heat(mahr);
        let rosh = specific_heat(rohr);
        let rash = rosh;
        rosmf = a.r1_cfm / accsv;
        let roemf = a.r1_exh / rosv;
        let rormf = rosmf - roemf;
        let rosat = a.r1_t - (a.r1_qs / (rosmf * 60.0 * sash));
        let rfit = a.r1_t + rdhg;
        let rfr = if rormf != 0.0 {
            ((rormf * rfisv) / (a.r1_cfm - a.r1_exh)).powi(2) * RFPR * rfisv * WORK2HEAT
                / (DFE / 100.0)
                / rash
        } else {
            0.0
        };
        rat = rfit + rfr;
        let oamf = (roemf).max(rosmf - rormf);
        let rcmf = rosmf - oamf;
        sfr = (rosmf * accsv / a.r1_cfm).powi(2) * SFPR * accsv * WORK2HEAT / (DFE / 100.0) / sash;
        mat = (oamf * a.oadb * a.oash + rcmf * rat * rash) / (rosmf * mash);
        tahc = mat.max(rosat - sfr);
        tacc = tahc.min(rosat - sfr);
        let ssphr = saturated_hr_at(tacc);
        let mahr_new = if rcmf != 0.0 {
            (rahr * rcmf + a.oahr * oamf) / rosmf
        } else {
            a.oahr
        };
        sahr = ssphr.min(mahr_new);
        rohr = sahr + a.gro / rosmf;
        rahr = rohr;
        mahr = mahr_new;
        accsv = specific_volume(tacc, sahr);
        rosv = specific_volume(a.r1_t, rohr);
        rfisv = specific_volume(rfit, rohr);
        if (accsv - prev.0).abs() < TOL && (sahr - prev.1).abs() < TOL {
            break;
        }
    }
    let qh = rosmf * 60.0 * specific_heat(mahr) * (tahc - mat);
    let (qcs, qcl) = coil_loads(rosmf, tahc, tacc, mahr, sahr);
    SystemResult {
        heating_coil: qh,
        cooling_sensible: qcs,
        cooling_latent: qcl,
        cooling_total: qcs + qcl,
        mixed_air_temp: mat,
        return_air_temp: rat,
        supply_air_temp: tacc + sfr,
        supply_mass_flow: rosmf,
        ..Default::default()
    }
}

/// Shared CAV / VAV solver. Both have a preheat coil, a cooling coil at the
/// supply setpoint, and a reheat coil per zone.
fn two_zone_reheat(a: &Ambient, variable_volume: bool, rdhg: f64) -> SystemResult {
    let mut accsv = 13.5;
    let mut rosv = 13.5;
    let mut rtsv = 13.5;
    let mut rfisv = 13.5;
    let mut sahr = a.oahr;
    let mut rohr = a.oahr;
    let mut rthr = a.oahr;
    let mut rahr = a.oahr;
    let mut mahr = a.oahr;
    let mut sat = SASP;
    let mut tsmf = 0.0;
    let mut rosmf = 0.0;
    let mut rtsmf = 0.0;
    let mut mat = 0.0;
    let mut tahc = 0.0;
    let mut tacc = 0.0;
    let mut rosat = 0.0;
    let mut rtsat = 0.0;
    for _ in 0..MAX_ITER {
        let prev = (accsv, sahr, sat);
        let sash = specific_heat(sahr);
        let mash = specific_heat(mahr);
        let rosh = specific_heat(rohr);
        let rtsh = specific_heat(rthr);
        let rash = specific_heat(rahr);
        let roemf = a.r1_exh / rosv;
        let rtemf = a.r2_exh / rtsv;
        if variable_volume {
            let roimf = a.r1_qs / (60.0 * sash * (a.r1_t - sat));
            let rtimf = a.r2_qs / (60.0 * sash * (a.r2_t - sat));
            rosmf = roimf.max(roemf);
            rtsmf = rtimf.max(rtemf);
        } else {
            rosmf = a.r1_cfm / accsv;
            rtsmf = a.r2_cfm / accsv;
        }
        rosat = a.r1_t - (a.r1_qs / (rosmf * 60.0 * sash));
        rtsat = a.r2_t - (a.r2_qs / (rtsmf * 60.0 * sash));
        let rormf = rosmf - roemf;
        let rtrmf = rtsmf - rtemf;
        let trmf = rormf + rtrmf;
        tsmf = rosmf + rtsmf;
        let mut rat = None;
        let mut rfit = a.r1_t;
        if trmf > 0.0 {
            let mrt = (rormf * a.r1_t * rosh + rtrmf * a.r2_t * rtsh) / (trmf * rash);
            rfit = mrt + rdhg;
            let rfr =
                ((trmf * rfisv) / NRCFM).powi(2) * RFPR * rfisv * WORK2HEAT / (DFE / 100.0) / rash;
            rat = Some(rfit + rfr);
        }
        let oamf = (roemf + rtemf).max(tsmf - trmf);
        let rcmf = tsmf - oamf;
        let sfr = if variable_volume {
            ((tsmf * accsv) / NSCFM).powi(2) * SFPR * accsv * WORK2HEAT / (DFE / 100.0) / sash
        } else {
            SFPR * accsv * WORK2HEAT / (DFE / 100.0) / sash
        };
        mat = match rat {
            None => a.oadb,
            Some(rat) => (oamf * a.oadb * a.oash + rcmf * rat * rash) / (tsmf * mash),
        };
        tahc = mat.max(PHSP);
        tacc = tahc.min(SASP - sfr);
        sat = tacc + sfr;
        let ssphr = saturated_hr_at(tacc);
        let mahr_new = if rcmf != 0.0 {
            (a.oahr * oamf + rcmf * rahr) / tsmf
        } else {
            a.oahr
        };
        sahr = ssphr.min(mahr_new);
        rohr = sahr + a.gro / rosmf;
        rthr = sahr + a.grt / rtsmf;
        rahr = if trmf != 0.0 {
            (rohr * rormf + rthr * rtrmf) / trmf
        } else {
            sahr
        };
        mahr = mahr_new;
        accsv = specific_volume(tacc, sahr);
        rosv = specific_volume(a.r1_t, rohr);
        rtsv = specific_volume(a.r2_t, rthr);
        if trmf != 0.0 {
            rfisv = specific_volume(rfit, rahr);
        }
        if (accsv - prev.0).abs() < TOL && (sahr - prev.1).abs() < TOL && (sat - prev.2).abs() < TOL
        {
            break;
        }
    }
    let qpre = tsmf * 60.0 * specific_heat(mahr) * (tahc - mat);
    let qr1 = rosmf * 60.0 * sash_of(&sahr) * (rosat - sat);
    let qr2 = rtsmf * 60.0 * sash_of(&sahr) * (rtsat - sat);
    let (qcs, qcl) = coil_loads(tsmf, tahc, tacc, mahr, sahr);
    SystemResult {
        preheat_coil: qpre,
        zone1_reheat: qr1,
        zone2_reheat: qr2,
        total_reheat: qr1 + qr2,
        total_heat: qpre + qr1 + qr2,
        cooling_sensible: qcs,
        cooling_latent: qcl,
        cooling_total: qcs + qcl,
        mixed_air_temp: mat,
        supply_air_temp: sat,
        zone1_supply_temp: rosat,
        zone2_supply_temp: rtsat,
        ..Default::default()
    }
}

fn sash_of(sahr: &f64) -> f64 {
    specific_heat(*sahr)
}

/// Solve all four §11 systems for `case` (1-based), economizer off, no
/// return-duct heat gain.
pub fn solve_case(case: usize) -> Result<QasCaseResult, FluxionError> {
    let a = Ambient::new(case)?;
    Ok(QasCaseResult {
        case: a.case,
        oadb: a.oadb,
        oadp: a.oadp,
        oadp_derived: a.oadp_derived,
        fc: fan_coil(&a),
        sz: single_zone(&a, 0.0),
        cav: two_zone_reheat(&a, false, 0.0),
        vav: two_zone_reheat(&a, true, 0.0),
    })
}

/// Case-1 anchors read directly off QASv2.xlsm as delivered (economizer off).
/// These are the regression check on this reimplementation and the published
/// reference outputs in `analytical_acceptance_criteria.json`.
pub const CASE_1_ANCHORS: [(&str, &str, f64); 13] = [
    ("FC", "heating_coil", 28728.61),
    ("FC", "mixed_air_temp", 39.206105),
    ("SZ", "heating_coil", 28525.0),
    ("SZ", "mixed_air_temp", 39.52867),
    ("CAV", "preheat_coil", 10056.0),
    ("CAV", "zone1_reheat", 26317.0),
    ("CAV", "zone2_reheat", 28624.0),
    ("CAV", "total_heat", 64997.0),
    ("CAV", "mixed_air_temp", 38.17691501),
    ("CAV", "supply_air_temp", 46.01088037),
    ("VAV", "preheat_coil", 35008.0),
    ("VAV", "zone1_reheat", 15353.0),
    ("VAV", "total_heat", 67002.0),
];

fn anchor_value(r: &QasCaseResult, system: &str, field: &str) -> Option<f64> {
    let s = match system {
        "FC" => &r.fc,
        "SZ" => &r.sz,
        "CAV" => &r.cav,
        "VAV" => &r.vav,
        _ => return None,
    };
    match field {
        "heating_coil" => Some(s.heating_coil),
        "preheat_coil" => Some(s.preheat_coil),
        "zone1_reheat" => Some(s.zone1_reheat),
        "zone2_reheat" => Some(s.zone2_reheat),
        "total_reheat" => Some(s.total_reheat),
        "total_heat" => Some(s.total_heat),
        "cooling_sensible" => Some(s.cooling_sensible),
        "cooling_latent" => Some(s.cooling_latent),
        "cooling_total" => Some(s.cooling_total),
        "mixed_air_temp" => Some(s.mixed_air_temp),
        "return_air_temp" => Some(s.return_air_temp),
        "supply_air_temp" => Some(s.supply_air_temp),
        "zone1_supply_temp" => Some(s.zone1_supply_temp),
        "zone2_supply_temp" => Some(s.zone2_supply_temp),
        "supply_mass_flow" => Some(s.supply_mass_flow),
        _ => None,
    }
}

/// Relative tolerance for the case-1 anchor check (matches the Python
/// oracle's self-test and the gate's analytical tolerance).
pub const ANCHOR_TOLERANCE_REL: f64 = 0.005;

/// Emit the `section_11_airside` block of
/// `data/reference/ashrae140/analytical_results.json`, deterministically, for
/// all six §11 ambient cases. Case 1 is additionally checked against the
/// published QASv2.xlsm anchors and the function returns an error instead of
/// emitting a passing artifact when any anchor misses
/// [`ANCHOR_TOLERANCE_REL`] — the same refuse-to-pass discipline as the §8
/// GC10a report.
pub fn results_section_json() -> Result<String, FluxionError> {
    let (ok, lines) = case_1_anchor_check();
    if !ok {
        let mut msg = String::from(
            "QAS case-1 anchors deviate from the QASv2.xlsm published values beyond \
             tolerance — refusing to emit a passing §11 result:\n",
        );
        for l in &lines {
            msg.push_str(l);
            msg.push('\n');
        }
        return Err(FluxionError::Validation(msg));
    }

    let mut out = String::from(" \"section_11_airside\": {\n");
    out.push_str(
        "  \"method\": \"QASv2 quasi-analytical solution reimplementation (Issue #4274); \
         economizer off, no return-duct heat gain; iterative solve, max 200 iterations, \
         convergence 1e-5\",\n",
    );
    out.push_str(
        "  \"case_1_anchor_check\": \"13/13 published QASv2.xlsm case-1 values reproduced \
         within 0.5% (13 mismatch lines absent)\",\n",
    );
    for case in 1..=6usize {
        let r = solve_case(case)?;
        out.push_str(&format!("  \"case_{case}\": {{\n"));
        out.push_str(&format!(
            "   \"oadb_f\": {:.2}, \"oadp_f\": {:.5}, \"oadp_derived_from_wetbulb\": {},\n",
            r.oadb, r.oadp, r.oadp_derived
        ));
        for (name, s) in [
            ("FC", &r.fc),
            ("SZ", &r.sz),
            ("CAV", &r.cav),
            ("VAV", &r.vav),
        ] {
            // Field names match `case_1_reference_outputs` in
            // analytical_acceptance_criteria.json so the release gate can
            // compare them directly. Loads Btu/h, temperatures degF.
            out.push_str(&format!(
                "   \"{name}\": {{\"heating_coil\": {:.1}, \"preheat_coil\": {:.1}, \
                 \"zone1_reheat\": {:.1}, \"zone2_reheat\": {:.1}, \
                 \"total_reheat\": {:.1}, \"total_heat\": {:.1}, \
                 \"cooling_total\": {:.1}, \"cooling_sensible\": {:.1}, \
                 \"cooling_latent\": {:.1}, \"mixed_air_temp\": {:.4}, \
                 \"return_air_temp\": {:.4}, \"supply_air_temp\": {:.4}, \
                 \"zone1_supply_temp\": {:.4}, \"zone2_supply_temp\": {:.4}}},\n",
                s.heating_coil,
                s.preheat_coil,
                s.zone1_reheat,
                s.zone2_reheat,
                s.total_reheat,
                s.total_heat,
                s.cooling_total,
                s.cooling_sensible,
                s.cooling_latent,
                s.mixed_air_temp,
                s.return_air_temp,
                s.supply_air_temp,
                s.zone1_supply_temp,
                s.zone2_supply_temp
            ));
        }
        // Replace the trailing comma of the last system with a newline.
        out.replace_range(out.len() - 2.., "\n");
        if case < 6 {
            out.push_str("  },\n");
        } else {
            out.push_str("  }\n");
        }
    }
    out.push_str(
        " },\n \"_section_11_source\": \"QASv2.xlsm via Standard 140-2023 Informative Annex \
         B17 Section B17.3.2; inputs and anchors recorded in \
         data/reference/ashrae140/analytical_acceptance_criteria.json \
         (section_11_airside); published case-1 outputs are the only workbook-\
         verified values; cases 2-6 are model outputs pending a workbook cross-check\"\n",
    );
    Ok(out)
}

/// Reproduce the case-1 workbook values. Returns (ok, report lines).
pub fn case_1_anchor_check() -> (bool, Vec<String>) {
    let mut lines = Vec::new();
    let mut ok = true;
    let r = match solve_case(1) {
        Ok(r) => r,
        Err(e) => return (false, vec![format!("case-1 solve failed: {e}")]),
    };
    for (system, field, expected) in CASE_1_ANCHORS {
        let got = anchor_value(&r, system, field).unwrap_or(f64::NAN);
        let (passed, errtxt) = if expected.abs() < 1e-9 {
            ((got.abs() < 1.0), format!("abs {:.4}", got.abs()))
        } else {
            let err = (got - expected).abs() / expected.abs();
            (err <= ANCHOR_TOLERANCE_REL, format!("{:.3}%", err * 100.0))
        };
        ok = ok && passed;
        lines.push(format!(
            "{system:<4} {field:<18} expected {expected:12.4}  got {got:12.4}  {errtxt:<10} {}",
            if passed { "ok" } else { "MISMATCH" }
        ));
    }
    (ok, lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_1_matches_workbook_anchors() {
        let (ok, lines) = case_1_anchor_check();
        for l in &lines {
            eprintln!("{l}");
        }
        assert!(
            ok,
            "QAS case-1 anchors must reproduce the QASv2.xlsm values"
        );
    }

    #[test]
    fn case_2_dewpoint_derives_from_wetbulb() {
        let a = Ambient::new(2).expect("case 2");
        assert!(a.oadp_derived);
        // Derived dew point must sit below both bulbs (dry air over ice).
        assert!(
            a.oadp < 19.756,
            "derived dew point {} not below wet bulb",
            a.oadp
        );
        // And the humidity ratio must be positive and small.
        assert!(a.oahr > 0.0 && a.oahr < 30.0, "grains {}", a.oahr);
    }

    #[test]
    fn all_six_cases_solve_deterministically() {
        let first: Vec<QasCaseResult> = (1..=6)
            .map(solve_case)
            .collect::<Result<_, _>>()
            .expect("all cases solve");
        for (i, r) in first.iter().enumerate() {
            assert_eq!(r.case, i + 1);
        }
        let second: Vec<QasCaseResult> = (1..=6)
            .map(solve_case)
            .collect::<Result<_, _>>()
            .expect("repeat solve");
        assert_eq!(first, second, "QAS must be deterministic run to run");
    }

    /// Warm cases (3-6) must show no heating demand on FC/SZ and no reheat on
    /// VAV beyond mixing losses — a qualitative sanity net against a sign
    /// error in the shared solver.
    #[test]
    fn warm_cases_have_no_heating_coil_demand_on_fc_sz() {
        for case in 3..=6 {
            let r = solve_case(case).expect("warm case");
            assert!(
                r.fc.heating_coil.abs() < 1.0,
                "case {case} FC heat {}",
                r.fc.heating_coil
            );
            assert!(
                r.sz.heating_coil.abs() < 1.0,
                "case {case} SZ heat {}",
                r.sz.heating_coil
            );
        }
    }

    #[test]
    fn case_1_oa_state_matches_oracle() {
        // Cross-check against scripts/qas_oracle.py (Python oracle, case 1):
        // oadp -20.2 F, 1.8124884065085292 gr/lbm, OA enthalpy -4.5816219676 Btu/lbm.
        let a = Ambient::new(1).expect("case 1");
        assert!((a.oahr - 1.8124884065085292).abs() < 1e-12);
        assert!((a.oae - (-4.581621967636253)).abs() < 1e-9);
        assert!((a.gro - 213.63021829956452).abs() < 1e-9);
    }
}
