#!/usr/bin/env python3
"""Quasi-analytical solution oracle for ASHRAE 140-2023 Section 11 (Air-Side HVAC).

Reimplementation of QASv2.xlsm, the merged quasi-analytical solution from ASHRAE
Technical Research Project 865 (Yuill and Haberl 2002), referenced by Standard 140
Informative Annex B17 Section B17.3.2.

Why this exists: unlike the Annex B example-result statistics, which the standard
explicitly disclaims as acceptance criteria, the QAS is a computable model. Given the
ambient case inputs it returns one coil load per system, so it is a genuine analytical
target for Section 11.

Four system models: four-pipe fan coil (FC), single-zone air handler (SZ), constant
air volume with zone reheat (CAV), variable air volume with zone reheat (VAV).

All internal calculations are in IP units, matching the workbook. Loads are Btu/h.
The workbook requires iterative calculation (max 200 iterations, max change 1e-5)
because supply mass flow depends on specific volume, which depends on the coil
leaving state, which depends on mass flow. This module iterates to the same tolerance.
"""
import math

# Constants, from the QASv2 INPUTS sheet. Mass ratios per the 2009 ASHRAE Handbook of
# Fundamentals (QAS change log, version 13 onward).
ATM_PSI = 14.69597535
RA = 53.35                  # gas constant, dry air, ft*lbf/(lbda*degR)
CP_AIR = 0.2403             # Btu/(lbm*degF)
CP_VAPOR = 0.444            # Btu/(lbm*degF)
CP_EVAP = 0.556             # Btu/(lbm*degF)
MASSRAT = 0.621945          # water / dry air
INV_MASSRAT = 1.607858      # dry air / water
WORK2HEAT = 0.006685
DFE = 70.0                  # design fan efficiency, percent
SFPR = 2.0                  # supply fan pressure rise, in. wg
RFPR = 1.0                  # return fan pressure rise, in. wg
SASP = 55.0                 # supply air setpoint, degF (CAV and VAV)
PHSP = 45.0                 # preheat setpoint, degF (CAV and VAV)
NSCFM = 1300.0              # nominal supply cfm (CAV and VAV)
NRCFM = 800.0               # nominal return cfm (CAV and VAV)

MAX_ITER = 200
TOL = 1e-5

# Section 11 ambient test cases. The rightmost digit of an AE case number selects the
# case here: AE103, AE203, AE303 and AE403 all use case 3.
TEST_CASES = {
    1: dict(oadb=-20.20, oadp=-20.20000, r1_t=70.0, r1_qs=-10000, r1_ql=2000,
            r1_cfm=600, r1_exh=200, r2_t=72.0, r2_qs=-8000, r2_ql=3000,
            r2_cfm=700, r2_exh=300),
    2: dict(oadb=30.02, oadp=None, oawb=19.756, r1_t=71.0, r1_qs=-2000, r1_ql=2000,
            r1_cfm=600, r1_exh=200, r2_t=73.0, r2_qs=1000, r2_ql=3000,
            r2_cfm=700, r2_exh=300),
    3: dict(oadb=59.90, oadp=26.60000, r1_t=74.0, r1_qs=5000, r1_ql=2000,
            r1_cfm=600, r1_exh=200, r2_t=76.0, r2_qs=8000, r2_ql=3000,
            r2_cfm=700, r2_exh=300),
    4: dict(oadb=80.42, oadp=71.78000, r1_t=75.0, r1_qs=10000, r1_ql=2000,
            r1_cfm=600, r1_exh=200, r2_t=77.0, r2_qs=12000, r2_ql=3000,
            r2_cfm=700, r2_exh=300),
    5: dict(oadb=76.82, oadp=36.32000, r1_t=74.0, r1_qs=5000, r1_ql=2000,
            r1_cfm=600, r1_exh=200, r2_t=76.0, r2_qs=8000, r2_ql=3000,
            r2_cfm=700, r2_exh=300),
    6: dict(oadb=73.40, oadp=69.62000, r1_t=74.0, r1_qs=5000, r1_ql=2000,
            r1_cfm=600, r1_exh=200, r2_t=76.0, r2_qs=8000, r2_ql=3000,
            r2_cfm=700, r2_exh=300),
}


def ln_pws_above_freezing(t_r):
    return (-10440.397 / t_r - 11.29465 - 0.027022355 * t_r
            + 1.289036e-5 * t_r ** 2 - 2.4780681e-9 * t_r ** 3
            + 6.5459673 * math.log(t_r))


def ln_pws_below_freezing(t_r):
    return (-10214.165 / t_r - 4.8932428 - 0.0053765794 * t_r
            + 1.9202377e-7 * t_r ** 2 + 3.5575832e-10 * t_r ** 3
            - 9.0344688e-14 * t_r ** 4 + 4.1635019 * math.log(t_r))


def pws(t_f):
    """Saturation water pressure, psia, from dry-bulb or dew point in degF."""
    t_r = t_f + 459.67
    return math.exp(ln_pws_above_freezing(t_r) if t_f > 32
                    else ln_pws_below_freezing(t_r))


def humidity_ratio_from_dewpoint(oadp_f):
    """Grains of moisture per lbm dry air, from dew point in degF."""
    p = pws(oadp_f)
    return MASSRAT * p / (ATM_PSI - p) * 7000.0


def specific_volume(t_f, hr_grains):
    return ((RA * (t_f + 459.67) / ATM_PSI) * (1 + INV_MASSRAT * hr_grains / 7000.0)) / 144.0


def specific_heat(hr_grains):
    return CP_AIR + CP_VAPOR * hr_grains / 7000.0


def enthalpy(t_f, hr_grains):
    return CP_AIR * t_f + (hr_grains / 7000.0) * (1061.15 + CP_VAPOR * t_f)


def grains_per_min(q_latent_btuh, t_zone_f):
    return (q_latent_btuh / 60.0) / (1061.15 + CP_VAPOR * t_zone_f) * 7000.0


def saturated_hr_at(t_f):
    """Saturated humidity ratio in grains at a coil leaving temperature."""
    p = math.exp(ln_pws_above_freezing(t_f + 459.67))
    return MASSRAT * (p / (ATM_PSI - p)) * 7000.0


def dewpoint_of(hr_grains):
    """Dew point in degF from humidity ratio in grains, per the QAS correlations."""
    pw = ATM_PSI * hr_grains / 7000.0 / (MASSRAT + hr_grains / 7000.0)
    lnpw = math.log(pw)
    above = (100.45 + 33.193 * lnpw + 2.319 * lnpw ** 2 + 0.1707 * lnpw ** 3
             + 1.2063 * pw ** 0.1984)
    if above < 32.0:
        return 90.12 + 26.142 * lnpw + 0.8927 * lnpw ** 2
    return above


def coil_loads(mass_flow_lbm_min, t_entering, t_leaving, hr_entering, hr_leaving):
    """Cooling coil sensible and latent, Btu/h. Sign convention follows the workbook."""
    dp_entering = dewpoint_of(hr_entering)
    sensible = (mass_flow_lbm_min * 60.0) * (
        (CP_AIR + CP_VAPOR * hr_leaving / 7000.0) * (t_entering - t_leaving)
        + ((hr_entering - hr_leaving) / 7000.0
           * (CP_VAPOR * (t_entering - dp_entering) + (dp_entering - t_leaving))))
    latent = (mass_flow_lbm_min * 60.0) * ((hr_entering - hr_leaving) / 7000.0) * (
        1075.21 - CP_EVAP * (dp_entering - 32.0))
    return sensible, latent


class Ambient:
    def __init__(self, case):
        c = TEST_CASES[case]
        self.case = case
        self.oadb = c["oadb"]
        if c.get("oadp") is None:
            # Case 2 publishes wet bulb only. Solve dew point from the wet bulb by
            # matching the adiabatic saturation humidity ratio.
            self.oadp = _dewpoint_from_wetbulb(c["oadb"], c["oawb"])
            self.oadp_derived = True
        else:
            self.oadp = c["oadp"]
            self.oadp_derived = False
        self.oahr = humidity_ratio_from_dewpoint(self.oadp)
        self.oasv = specific_volume(self.oadb, self.oahr)
        self.oash = specific_heat(self.oahr)
        self.oae = enthalpy(self.oadb, self.oahr)
        self.r1_t, self.r1_qs, self.r1_ql = c["r1_t"], c["r1_qs"], c["r1_ql"]
        self.r1_cfm, self.r1_exh = c["r1_cfm"], c["r1_exh"]
        self.r2_t, self.r2_qs, self.r2_ql = c["r2_t"], c["r2_qs"], c["r2_ql"]
        self.r2_cfm, self.r2_exh = c["r2_cfm"], c["r2_exh"]
        self.gro = grains_per_min(self.r1_ql, self.r1_t)
        self.grt = grains_per_min(self.r2_ql, self.r2_t)


def _dewpoint_from_wetbulb(tdb, twb):
    """Dew point from dry bulb and wet bulb, via the adiabatic saturation relation.

    Two branches, per the ASHRAE Handbook of Fundamentals psychrometrics chapter: the
    liquid-water form above freezing and the ice form below it. Case 2 has both bulbs
    below freezing (30.02 / 19.756 degF), so the ice form is the one that applies.
    """
    pws_wb = pws(twb)
    w_star = MASSRAT * pws_wb / (ATM_PSI - pws_wb)
    if twb > 32.0:
        w = ((1093.0 - 0.556 * twb) * w_star - 0.240 * (tdb - twb)) / (
            1093.0 + 0.444 * tdb - twb)
    else:
        w = ((1220.0 - 0.04 * twb) * w_star - 0.240 * (tdb - twb)) / (
            1220.0 + 0.444 * tdb - 0.48 * twb)
    if w <= 0:
        raise ValueError(
            "derived humidity ratio is not positive for tdb=%s twb=%s; the published "
            "dew point is required for this case" % (tdb, twb))
    pw = ATM_PSI * w / (MASSRAT + w)
    lnpw = math.log(pw)
    above = (100.45 + 33.193 * lnpw + 2.319 * lnpw ** 2 + 0.1707 * lnpw ** 3
             + 1.2063 * pw ** 0.1984)
    if above < 32.0:
        return 90.12 + 26.142 * lnpw + 0.8927 * lnpw ** 2
    return above


# --------------------------------------------------------------------------- #
# System models. Each iterates to convergence because supply mass flow depends
# on specific volume, which depends on the coil leaving state.
# --------------------------------------------------------------------------- #

def fan_coil(a):
    """Four-pipe fan coil, single zone, no economizer, no return fan."""
    accsv = rosv = 13.5
    sahr = rohr = mahr = a.oahr
    for _ in range(MAX_ITER):
        prev = (accsv, sahr)
        sash, mash, rosh = specific_heat(sahr), specific_heat(mahr), specific_heat(rohr)
        rash = rosh
        rosmf = a.r1_cfm / accsv
        roemf = a.r1_exh / rosv
        rormf = rosmf - roemf
        rosat = a.r1_t - (a.r1_qs / (rosmf * 60.0 * sash))
        sfr = ((accsv * rosmf / a.r1_cfm) ** 2 * SFPR) * accsv * WORK2HEAT / (DFE / 100.0) / sash
        mat = (rormf * a.r1_t * rash + roemf * a.oadb * a.oash) / (rosmf * mash)
        tahc = mat if mat > (rosat - sfr) else (rosat - sfr)
        tacc = tahc if tahc < (rosat - sfr) else (rosat - sfr)
        ssphr = saturated_hr_at(tacc)
        mahr_new = (rohr * rormf + a.oahr * roemf) / rosmf if rormf else a.oahr
        sahr = min(ssphr, mahr_new)
        rohr = sahr + a.gro / rosmf
        mahr = mahr_new
        accsv = specific_volume(tacc, sahr)
        rosv = specific_volume(a.r1_t, rohr)
        if abs(accsv - prev[0]) < TOL and abs(sahr - prev[1]) < TOL:
            break
    qh = (rosmf * 60.0) * mash * (tahc - mat)
    qcs, qcl = coil_loads(rosmf, tahc, tacc, mahr, sahr)
    return dict(system="FC", heating_coil=qh, cooling_sensible=qcs, cooling_latent=qcl,
                cooling_total=qcs + qcl, mixed_air_temp=mat, supply_air_temp=tacc + sfr,
                supply_mass_flow=rosmf)


def single_zone(a, rdhg=0.0):
    """Single-zone air handler with return fan, economizer disabled."""
    accsv = rosv = rfisv = 13.5
    sahr = rohr = rahr = mahr = a.oahr
    for _ in range(MAX_ITER):
        prev = (accsv, sahr)
        sash, mash, rosh = specific_heat(sahr), specific_heat(mahr), specific_heat(rohr)
        rash = rosh
        rosmf = a.r1_cfm / accsv
        roemf = a.r1_exh / rosv
        rormf = rosmf - roemf
        rosat = a.r1_t - (a.r1_qs / (rosmf * 60.0 * sash))
        rfit = a.r1_t + rdhg
        rfr = (((rormf * rfisv) / (a.r1_cfm - a.r1_exh)) ** 2 * RFPR
               * rfisv * WORK2HEAT / (DFE / 100.0) / rash) if rormf else 0.0
        rat = rfit + rfr
        oamf = max(roemf, rosmf - rormf)
        rcmf = rosmf - oamf
        sfr = (rosmf * accsv / a.r1_cfm) ** 2 * SFPR * accsv * WORK2HEAT / (DFE / 100.0) / sash
        mat = (oamf * a.oadb * a.oash + rcmf * rat * rash) / (rosmf * mash)
        tahc = mat if mat > (rosat - sfr) else (rosat - sfr)
        tacc = tahc if tahc < (rosat - sfr) else (rosat - sfr)
        ssphr = saturated_hr_at(tacc)
        mahr_new = (rahr * rcmf + a.oahr * oamf) / rosmf if rcmf else a.oahr
        sahr = min(ssphr, mahr_new)
        rohr = sahr + a.gro / rosmf
        rahr = rohr
        mahr = mahr_new
        accsv = specific_volume(tacc, sahr)
        rosv = specific_volume(a.r1_t, rohr)
        rfisv = specific_volume(rfit, rohr)
        if abs(accsv - prev[0]) < TOL and abs(sahr - prev[1]) < TOL:
            break
    qh = (rosmf * 60.0) * mash * (tahc - mat)
    qcs, qcl = coil_loads(rosmf, tahc, tacc, mahr, sahr)
    return dict(system="SZ", heating_coil=qh, cooling_sensible=qcs, cooling_latent=qcl,
                cooling_total=qcs + qcl, mixed_air_temp=mat, return_air_temp=rat,
                supply_air_temp=tacc + sfr, supply_mass_flow=rosmf)


def _two_zone_reheat(a, variable_volume, rdhg=0.0):
    """Shared CAV / VAV solver. Both have a preheat coil, a cooling coil at the
    supply setpoint, and a reheat coil per zone."""
    accsv = 13.5
    rosv = rtsv = rfisv = 13.5
    sahr = rohr = rthr = rahr = mahr = a.oahr
    sat = SASP
    for _ in range(MAX_ITER):
        prev = (accsv, sahr, sat)
        sash = specific_heat(sahr)
        mash = specific_heat(mahr)
        rosh, rtsh, rash = specific_heat(rohr), specific_heat(rthr), specific_heat(rahr)
        roemf = a.r1_exh / rosv
        rtemf = a.r2_exh / rtsv
        if variable_volume:
            roimf = a.r1_qs / (60.0 * sash * (a.r1_t - sat))
            rtimf = a.r2_qs / (60.0 * sash * (a.r2_t - sat))
            rosmf = max(roimf, roemf)
            rtsmf = max(rtimf, rtemf)
        else:
            rosmf = a.r1_cfm / accsv
            rtsmf = a.r2_cfm / accsv
        rosat = a.r1_t - (a.r1_qs / (rosmf * 60.0 * sash))
        rtsat = a.r2_t - (a.r2_qs / (rtsmf * 60.0 * sash))
        rormf = rosmf - roemf
        rtrmf = rtsmf - rtemf
        trmf = rormf + rtrmf
        tsmf = rosmf + rtsmf
        if trmf > 0:
            mrt = (rormf * a.r1_t * rosh + rtrmf * a.r2_t * rtsh) / (trmf * rash)
            rfit = mrt + rdhg
            rfr = (((trmf * rfisv) / NRCFM) ** 2 * RFPR) * rfisv * WORK2HEAT / (DFE / 100.0) / rash
            rat = rfit + rfr
        else:
            mrt = rfit = rat = None
        oamf = max(roemf + rtemf, tsmf - trmf)
        rcmf = tsmf - oamf
        if variable_volume:
            sfr = (((tsmf * accsv) / NSCFM) ** 2 * SFPR) * accsv * WORK2HEAT / (DFE / 100.0) / sash
        else:
            sfr = SFPR * accsv * WORK2HEAT / (DFE / 100.0) / sash
        if rat is None:
            mat = a.oadb
        else:
            mat = (oamf * a.oadb * a.oash + rcmf * rat * rash) / (tsmf * mash)
        tahc = mat if mat > PHSP else PHSP
        tacc = tahc if tahc < (SASP - sfr) else (SASP - sfr)
        sat = tacc + sfr
        ssphr = saturated_hr_at(tacc)
        mahr_new = (oamf * a.oahr + rcmf * rahr) / tsmf if rcmf else a.oahr
        sahr = min(ssphr, mahr_new)
        rohr = sahr + a.gro / rosmf
        rthr = sahr + a.grt / rtsmf
        rahr = ((rohr * rormf + rthr * rtrmf) / trmf) if trmf else sahr
        mahr = mahr_new
        accsv = specific_volume(tacc, sahr)
        rosv = specific_volume(a.r1_t, rohr)
        rtsv = specific_volume(a.r2_t, rthr)
        if trmf:
            rfisv = specific_volume(rfit, rahr)
        if (abs(accsv - prev[0]) < TOL and abs(sahr - prev[1]) < TOL
                and abs(sat - prev[2]) < TOL):
            break
    qpre = (tsmf * 60.0) * mash * (tahc - mat)
    qr1 = (rosmf * 60.0) * sash * (rosat - sat)
    qr2 = (rtsmf * 60.0) * sash * (rtsat - sat)
    qcs, qcl = coil_loads(tsmf, tahc, tacc, mahr, sahr)
    return dict(system="VAV" if variable_volume else "CAV",
                preheat_coil=qpre, zone1_reheat=qr1, zone2_reheat=qr2,
                total_reheat=qr1 + qr2, total_heat=qpre + qr1 + qr2,
                cooling_sensible=qcs, cooling_latent=qcl, cooling_total=qcs + qcl,
                mixed_air_temp=mat, supply_air_temp=sat,
                zone1_supply_temp=rosat, zone2_supply_temp=rtsat)


def cav_reheat(a, rdhg=0.0):
    return _two_zone_reheat(a, variable_volume=False, rdhg=rdhg)


def vav_reheat(a, rdhg=0.0):
    return _two_zone_reheat(a, variable_volume=True, rdhg=rdhg)


def solve(case, rdhg=0.0):
    a = Ambient(case)
    return {"case": case, "oadb": a.oadb, "oadp": a.oadp,
            "oadp_derived": a.oadp_derived,
            "FC": fan_coil(a), "SZ": single_zone(a, rdhg),
            "CAV": cav_reheat(a, rdhg), "VAV": vav_reheat(a, rdhg)}


# Anchors read directly off QASv2.xlsm as delivered, which is set to case 1 with the
# economizer off. These are the regression check on this reimplementation.
CASE_1_ANCHORS = {
    "FC":  {"heating_coil": 28728.61, "cooling_total": 0.0, "mixed_air_temp": 39.206105},
    "SZ":  {"heating_coil": 28525.0,  "cooling_total": 0.0, "mixed_air_temp": 39.52867},
    "CAV": {"preheat_coil": 10056.0, "zone1_reheat": 26317.0, "zone2_reheat": 28624.0,
            "total_heat": 64997.0, "mixed_air_temp": 38.17691501,
            "supply_air_temp": 46.01088037},
    "VAV": {"preheat_coil": 35008.0, "zone1_reheat": 15353.0, "zone2_reheat": 16642.0,
            "total_heat": 67002.0, "supply_air_temp": 45.134299,
            "zone1_supply_temp": 116.4560, "zone2_supply_temp": 96.8703},
}


def self_test(rel_tol=0.005):
    """Reproduce the case-1 workbook values. Returns (ok, report lines)."""
    r = solve(1)
    lines, ok = [], True
    for system, anchors in CASE_1_ANCHORS.items():
        for field, expected in anchors.items():
            got = r[system][field]
            if abs(expected) < 1e-9:
                passed = abs(got) < 1.0
                err = abs(got)
                errtxt = "abs %.4f" % err
            else:
                err = abs(got - expected) / abs(expected)
                passed = err <= rel_tol
                errtxt = "%.3f%%" % (err * 100)
            ok = ok and passed
            lines.append("%-4s %-18s expected %12.4f  got %12.4f  %-10s %s"
                         % (system, field, expected, got, errtxt,
                            "ok" if passed else "MISMATCH"))
    return ok, lines


if __name__ == "__main__":
    import sys
    ok, lines = self_test()
    for line in lines:
        print(line)
    print("\nself test:", "PASS" if ok else "FAIL")
    if ok:
        print("\nSection 11 QAS targets, Btu/h (economizer off, no return duct heat gain)\n")
        hdr = ("case", "FC heat", "SZ heat", "CAV total heat", "VAV total heat",
               "CAV cool tot", "VAV cool tot")
        print("%-5s %12s %12s %15s %15s %14s %14s" % hdr)
        for c in sorted(TEST_CASES):
            r = solve(c)
            print("%-5s %12.1f %12.1f %15.1f %15.1f %14.1f %14.1f"
                  % (c, r["FC"]["heating_coil"], r["SZ"]["heating_coil"],
                     r["CAV"]["total_heat"], r["VAV"]["total_heat"],
                     r["CAV"]["cooling_total"], r["VAV"]["cooling_total"]))
    sys.exit(0 if ok else 1)
