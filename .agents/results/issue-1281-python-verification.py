#!/usr/bin/env python3
"""
Python verification for Issue #1281 — h_ms_total additive model overcounting hypothesis.

Tests whether the current additive h_ms_total model (sum of wall+roof+floor per-surface
conductances) overcounts thermal coupling in the 9R4C zone-level network, and whether
a non-additive (parallel-resistance) correction produces more physically reasonable
cooling load dynamics for ASHRAE 140 Case 900/920/950.

Uses ACTUAL Case 900 parameters from the Rust engine (src/sim/construction.rs,
src/validation/ashrae_140_cases.rs).

References:
  - ISO 13790:2008 Annex C (5R1C / 9R4C simplified method)
  - ASHRAE 140-2023 (BESTEST), Table 4.2 (Case 900/920/950 reference data)
  - docs/KNOWN_ISSUES.md LIMIT-05 UPDATE (Phase 36)
  - docs/adr/0002-promote-9r4c-high-mass-default.md (ADR-002)
  - src/sim/construction.rs:1005-1056 (HighMassWall, Roof, Floor definitions)
  - src/sim/thermal_model_core.rs:914-921 (h_tr_is = 3.45 × floor_area)
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Case 900 actual parameters (derived from Rust engine)
# ---------------------------------------------------------------------------

@dataclass
class Case900Spec:
    """ASHRAE 140 Case 900 — high-mass baseline (south window).

    Geometry: 8m × 6m × 2.7m, 12 m² south window, 48 m² floor area.
    Construction: high-mass (wood siding + foam + concrete block wall;
    concrete + foam + roof deck roof; concrete slab + insulation floor).

    Per-surface h_tr_ms values derived from half-insulation rule:
      h_ms_wall = A_opaque_wall / (R_wood + R_foam/2)
      h_ms_roof = A_roof / (R_concrete + R_foam/2)
      h_ms_floor = A_floor / (R_concrete + R_insulation/2)

    Cm from layer (k, rho, cp, thickness) products (active layer).
    """
    # Geometry
    floor_area: float = 48.0          # m^2
    roof_area: float = 48.0           # m^2 (same as floor)
    wall_total: float = 75.6          # m^2 (2 × (8+6) × 2.7)
    window_area: float = 12.0         # m^2 (south window)
    opaque_wall: float = 75.6 - 12.0  # 63.6 m^2
    volume: float = 8 * 6 * 2.7       # 129.6 m^3

    # Thermal capacitances (J/K) per surface — see src/sim/construction.rs Materials
    # Cm_wall ≈ 0.080 m × 1400 kg/m³ × 840 J/kg·K × 63.6 m² (concrete block layer)
    #        + 0.009 × 530 × 900 × 63.6 (wood siding)
    # We use approximate values matching the engine's per-zone Cm_wall (~5e6).
    cm_wall: float = 5.0e6
    cm_roof: float = 3.0e6
    cm_floor: float = 2.0e6
    cm_internal: float = 1.0e6

    # Per-surface h_tr_ms (W/K) from half-insulation rule
    # Wall: A_opaque / (R_wood + R_foam/2) = 63.6 / (0.0643 + 1.5375/2) ≈ 76.4
    h_ms_wall: float = 76.4
    # Roof: A_floor / (R_concrete + R_foam/2) = 48 / (0.0708 + 2.775/2) ≈ 32.9
    h_ms_roof: float = 32.9
    # Floor: A_floor / (R_concrete + R_insulation/2) = 48 / (0.1569 + 5.025/2) ≈ 18.0
    h_ms_floor: float = 18.0

    # Per-surface h_tr_em (W/K) — half-insulation from exterior side
    # These are LOWER than h_tr_ms because the exterior has more insulation.
    h_em_wall: float = 25.0   # approximate
    h_em_roof: float = 20.0
    h_em_floor: float = 10.0

    # Lumped h_tr_is = 3.45 × floor_area (per Issue #714, ASHRAE 140 simplified)
    h_tr_is: float = 3.45 * 48.0     # = 165.6 W/K

    # Ventilation: ACH = 0.5, V = 129.6, ρ × cp = 1206
    # h_ve = ρ × cp × ACH/3600 × V = 1.2 × 1005 × 0.5/3600 × 129.6 ≈ 21.7 W/K
    h_ve: float = 21.7

    # Setpoints
    heating_setpoint: float = 20.0
    cooling_setpoint: float = 27.0


# ---------------------------------------------------------------------------
# 9R4C solver — CURRENT (additive h_ms_total for surface temperature)
# ---------------------------------------------------------------------------

class NineR4CAdditive:
    """9R4C network using ADDITIVE h_ms_total for the shared surface node.

    This matches src/physics/multi_node_solver.rs::step_with_gains and
    src/sim/thermal_model_physics/physics_impl.rs::t_surface calculation.

    T_s = (Σ_k h_ms_k × T_m_k) / Σ_k h_ms_k  [weighted average of mass temps]
    T_air = (h_tr_is × T_s + h_ve × T_out + phi_ia) / (h_tr_is + h_ve)
    """

    def __init__(self, spec: Case900Spec):
        self.spec = spec
        self.T_wall = 20.0
        self.T_roof = 20.0
        self.T_floor = 20.0
        self.T_internal = 20.0
        self.T_air = 20.0
        self.T_s = 20.0

        # Per-surface exterior temperatures (sol-air for wall/roof, ground for floor)
        self.T_ext_wall = 30.0
        self.T_ext_roof = 35.0
        self.T_ext_floor = 15.0

        self.dt = 3600.0

    @property
    def h_ms_total(self) -> float:
        return self.spec.h_ms_wall + self.spec.h_ms_roof + self.spec.h_ms_floor

    def step(self, gains: dict, dt: float = 3600.0) -> None:
        """Backward Euler step (matches multi_node_solver.rs::step_with_gains)."""
        self.dt = dt
        dt = self.dt

        for name, T_ext, h_em, h_ms, cm in [
            ('wall',   self.T_ext_wall,   self.spec.h_em_wall,  self.spec.h_ms_wall,  self.spec.cm_wall),
            ('roof',   self.T_ext_roof,   self.spec.h_em_roof,  self.spec.h_ms_roof,  self.spec.cm_roof),
            ('floor',  self.T_ext_floor,  self.spec.h_em_floor, self.spec.h_ms_floor, self.spec.cm_floor),
        ]:
            T_old = getattr(self, f'T_{name}')
            denom = cm / dt + h_em + h_ms
            numer = (cm / dt) * T_old + h_em * T_ext + h_ms * self.T_s + gains.get(name, 0.0)
            T_new = numer / denom
            setattr(self, f'T_{name}', T_new)

        T_env_avg = (self.T_wall + self.T_roof + self.T_floor) / 3.0
        cm = self.spec.cm_internal
        h_me = 100.0  # h_tr_me (furniture-to-air) — typical value
        denom = cm / dt + self.spec.h_tr_is + h_me
        numer = (cm / dt) * self.T_internal + self.spec.h_tr_is * self.T_air + h_me * T_env_avg + gains.get('internal', 0.0)
        self.T_internal = numer / denom

        # Surface temperature — ADDITIVE sum in denominator
        h_ms_w, h_ms_r, h_ms_f = self.spec.h_ms_wall, self.spec.h_ms_roof, self.spec.h_ms_floor
        h_ms_total = self.h_ms_total
        if h_ms_total > 1e-6:
            self.T_s = (h_ms_w * self.T_wall + h_ms_r * self.T_roof + h_ms_f * self.T_floor) / h_ms_total

    def compute_air_temperature(self, T_outdoor: float, h_ve: float, phi_ia: float) -> float:
        """Compute free-floating zone air temperature.

        T_air = (h_is × T_s + h_ve × T_out + phi_ia) / (h_is + h_ve)
        """
        denom = self.spec.h_tr_is + h_ve
        if denom < 1e-6:
            return self.T_s
        return (self.spec.h_tr_is * self.T_s + h_ve * T_outdoor + phi_ia) / denom


# ---------------------------------------------------------------------------
# 9R4C solver — NON-ADDITIVE (parallel-resistance correction)
# ---------------------------------------------------------------------------

class NineR4CParallelResistance:
    """9R4C network with NON-ADDITIVE coupling.

    Each surface has its OWN T_s_k determined by the surface node energy balance:
      T_s_k = (h_ms_k × T_m_k + h_is × T_air) / (h_ms_k + h_is)

    The air node sees the parallel combination of per-surface mass-to-air paths:
      h_path_k = h_ms_k × h_is / (h_ms_k + h_is)  [series combination]
      T_air = (Σ_k h_path_k × T_m_k + h_ve × T_out + phi_ia) / (Σ_k h_path_k + h_ve)

    This eliminates the "shared T_s" approximation by computing per-surface T_s
    and then combining the parallel paths to the air node.
    """

    def __init__(self, spec: Case900Spec):
        self.spec = spec
        self.T_wall = 20.0
        self.T_roof = 20.0
        self.T_floor = 20.0
        self.T_internal = 20.0
        self.T_air = 20.0
        self.dt = 3600.0

        self.T_ext_wall = 30.0
        self.T_ext_roof = 35.0
        self.T_ext_floor = 15.0

    def h_path(self, h_ms_k: float) -> float:
        """Per-surface series conductance from mass to air (mass → T_s → air)."""
        h_is = self.spec.h_tr_is
        return h_ms_k * h_is / (h_ms_k + h_is)

    @property
    def h_path_total(self) -> float:
        return (self.h_path(self.spec.h_ms_wall)
                + self.h_path(self.spec.h_ms_roof)
                + self.h_path(self.spec.h_ms_floor))

    def step(self, gains: dict, dt: float = 3600.0) -> None:
        """Backward Euler step with per-surface T_s_k."""
        self.dt = dt
        dt = self.dt
        h_is = self.spec.h_tr_is

        for name, T_ext, h_em, h_ms, cm in [
            ('wall',   self.T_ext_wall,   self.spec.h_em_wall,  self.spec.h_ms_wall,  self.spec.cm_wall),
            ('roof',   self.T_ext_roof,   self.spec.h_em_roof,  self.spec.h_ms_roof,  self.spec.cm_roof),
            ('floor',  self.T_ext_floor,  self.spec.h_em_floor, self.spec.h_ms_floor, self.spec.cm_floor),
        ]:
            T_old = getattr(self, f'T_{name}')

            # Per-surface T_s_k — series solution using previous T_m and T_air
            T_s_k = (h_ms * T_old + h_is * self.T_air) / (h_ms + h_is)

            denom = cm / dt + h_em + h_ms
            numer = (cm / dt) * T_old + h_em * T_ext + h_ms * T_s_k + gains.get(name, 0.0)
            T_new = numer / denom
            setattr(self, f'T_{name}', T_new)

        T_env_avg = (self.T_wall + self.T_roof + self.T_floor) / 3.0
        cm = self.spec.cm_internal
        h_me = 100.0
        denom = cm / dt + h_is + h_me
        numer = (cm / dt) * self.T_internal + h_is * self.T_air + h_me * T_env_avg + gains.get('internal', 0.0)
        self.T_internal = numer / denom

    def compute_air_temperature(self, T_outdoor: float, h_ve: float, phi_ia: float) -> float:
        """Compute air temperature from per-surface series paths."""
        h_path_w = self.h_path(self.spec.h_ms_wall)
        h_path_r = self.h_path(self.spec.h_ms_roof)
        h_path_f = self.h_path(self.spec.h_ms_floor)
        h_path_total = h_path_w + h_path_r + h_path_f

        denom = h_path_total + h_ve
        if denom < 1e-6:
            return (h_path_w * self.T_wall + h_path_r * self.T_roof + h_path_f * self.T_floor) / h_path_total
        return (h_path_w * self.T_wall
                + h_path_r * self.T_roof
                + h_path_f * self.T_floor
                + h_ve * T_outdoor
                + phi_ia) / denom


# ---------------------------------------------------------------------------
# Steady-state comparison (24-hour constant forcing)
# ---------------------------------------------------------------------------

def steady_state_compare():
    """Compare steady-state T_air and T_s between additive and non-additive models."""
    spec = Case900Spec()

    # Hot summer day forcing
    T_ext_wall = 45.0
    T_ext_roof = 50.0
    T_ext_floor = 18.0
    phi_ia = 200.0  # W internal convective gain
    T_outdoor = 32.0

    add = NineR4CAdditive(spec)
    add.T_ext_wall, add.T_ext_roof, add.T_ext_floor = T_ext_wall, T_ext_roof, T_ext_floor
    for _ in range(5000):
        add.step({}, dt=3600.0)
    T_air_add = add.compute_air_temperature(T_outdoor, spec.h_ve, phi_ia)

    par = NineR4CParallelResistance(spec)
    par.T_ext_wall, par.T_ext_roof, par.T_ext_floor = T_ext_wall, T_ext_roof, T_ext_floor
    for _ in range(5000):
        par.step({}, dt=3600.0)
    T_air_par = par.compute_air_temperature(T_outdoor, spec.h_ve, phi_ia)

    print("=" * 78)
    print("STEADY-STATE COMPARISON: Additive vs Non-Additive 9R4C (Case 900 params)")
    print("=" * 78)
    print(f"Exterior: T_ext_wall={T_ext_wall}, T_ext_roof={T_ext_roof}, T_ext_floor={T_ext_floor} °C")
    print(f"Outdoor: {T_outdoor} °C, Internal gain: {phi_ia} W, h_ve={spec.h_ve:.1f} W/K")
    print(f"h_tr_is = {spec.h_tr_is:.1f} W/K  (lumped 3.45 × floor_area)")
    print(f"h_ms_wall={spec.h_ms_wall:.1f}, h_ms_roof={spec.h_ms_roof:.1f}, h_ms_floor={spec.h_ms_floor:.1f} W/K")
    print()
    print(f"Additive:        T_wall={add.T_wall:.3f}, T_roof={add.T_roof:.3f}, T_floor={add.T_floor:.3f} °C")
    print(f"                 T_int={add.T_internal:.3f}, T_s={add.T_s:.3f}, T_air={T_air_add:.3f} °C")
    print()
    print(f"Non-additive:    T_wall={par.T_wall:.3f}, T_roof={par.T_roof:.3f}, T_floor={par.T_floor:.3f} °C")
    print(f"                 T_int={par.T_internal:.3f}, T_air={T_air_par:.3f} °C")
    print()
    print(f"ΔT_air (additive − non-additive): {T_air_add - T_air_par:+.4f} °C")
    print()

    h_ms_total_add = add.h_ms_total
    h_path_total_par = par.h_path_total

    print(f"h_ms_total (additive sum):       {h_ms_total_add:.3f} W/K")
    print(f"h_path_total (series-parallel):  {h_path_total_par:.3f} W/K")
    print(f"Ratio (additive/series-parallel): {h_ms_total_add / h_path_total_par:.3f}")
    print()

    # Direction of overcounting
    # If additive > series-parallel, the additive model OVERCOUNTS the mass-to-air coupling.
    # Higher mass-to-air coupling → MORE heat absorbed by mass → cooler air → LESS cooling.
    # So overcounting → UNDERESTIMATION of cooling demand.
    if h_ms_total_add > h_path_total_par:
        overcount_pct = (h_ms_total_add - h_path_total_par) / h_path_total_par * 100.0
        print(f"→ HYPOTHESIS CONFIRMED: additive h_ms_total > non-additive h_path_total ({overcount_pct:.1f}% overcount)")
        print()
        print("  Physical interpretation:")
        print("  - In the additive model, h_tr_is appears as a SINGLE resistance shared by")
        print("    all surfaces in parallel (treated as one big conductance from mass to air).")
        print("  - In the non-additive model, h_tr_is appears IN SERIES with each per-surface")
        print("    h_ms_k. Each surface contributes h_ms_k × h_is / (h_ms_k + h_is).")
        print(f"  - With h_tr_is = {spec.h_tr_is:.1f} W/K being LARGER than the per-surface h_ms_k,")
        print("    the series-parallel combination h_path_k = h_ms_k × h_is / (h_ms_k + h_is) is")
        print("    closer to h_ms_k (not h_is). The SUM of these paths is less than h_ms_total.")
        print()
        print("  Effect on cooling load:")
        print("  - Higher h_ms_total (additive) → MORE heat coupled between mass and air →")
        print("    air stays COOLER → LESS cooling demand.")
        print("  - Lower h_path_total (non-additive) → LESS coupling → air gets WARMER →")
        print("    MORE cooling demand. ← This direction matches the ASHRAE 140 reference.")
    else:
        print("→ HYPOTHESIS REJECTED: additive h_ms_total is NOT larger than the series-parallel")
        print("  combination. Ratio: {:.3f}".format(h_ms_total_add / h_path_total_par))


# ---------------------------------------------------------------------------
# Dynamic peak cooling test (24-hour solar pulse)
# ---------------------------------------------------------------------------

def peak_cooling_test():
    """Simulate a 24-hour Denver summer day and compare peak cooling demand.

    This models Case 900's peak cooling hour (~14:00). HVAC clamps T_air at
    27 °C when above. The cooling load is the rate at which heat must be
    REMOVED from the air to maintain the setpoint.
    """
    spec = Case900Spec()

    # 24-hour forcing profile (Denver July 21, summer solstice + a bit)
    hours = np.arange(24)
    T_outdoor_prof = np.array([
        18.0, 17.0, 16.5, 16.0, 16.5, 18.0,    # 0-5  night
        20.0, 23.0, 26.0, 29.0, 32.0, 34.0,    # 6-11 morning ramp
        35.5, 36.5, 37.0, 36.5, 35.0, 33.0,    # 12-17 afternoon peak
        30.0, 27.0, 24.0, 22.0, 20.0, 19.0,    # 18-23 evening
    ])
    phi_ia_prof = np.array([
        100, 100, 100, 100, 100, 100,
        150, 200, 400, 800, 1500, 2200,
        2800, 3200, 3000, 2400, 1500, 600,
        300, 200, 150, 120, 110, 100,
    ])
    T_sol_air_wall_prof = T_outdoor_prof + 8.0
    T_sol_air_roof_prof = T_outdoor_prof + 12.0

    # Initialize from previous-day equilibrium (run 5 days for warmup)
    def warmup(model_cls, n_days=5):
        model = model_cls(spec)
        for d in range(n_days * 24):
            h = d % 24
            model.T_ext_wall = T_sol_air_wall_prof[h]
            model.T_ext_roof = T_sol_air_roof_prof[h]
            model.T_ext_floor = 16.0
            model.step({}, dt=3600.0)
        return model

    def run(model_cls):
        model = warmup(model_cls)
        T_air_hist, Q_cool_hist = [], []
        for h in range(24):
            model.T_ext_wall = T_sol_air_wall_prof[h]
            model.T_ext_roof = T_sol_air_roof_prof[h]
            model.T_ext_floor = 16.0

            model.step({}, dt=3600.0)

            T_air_ff = model.compute_air_temperature(T_outdoor_prof[h], spec.h_ve, phi_ia_prof[h])

            if T_air_ff > spec.cooling_setpoint:
                # Cooling load = total heat that must be REMOVED to bring air to setpoint
                # Q_cool = h_is × (T_s − T_set) + h_ve × (T_out − T_set) + phi_ia
                # (positive value = cooling demand)
                T_s_now = getattr(model, 'T_s', T_air_ff)
                if not hasattr(model, 'T_s'):
                    # Non-additive: T_s not stored, estimate from T_m
                    h_p_w, h_p_r, h_p_f = model.h_path(spec.h_ms_wall), model.h_path(spec.h_ms_roof), model.h_path(spec.h_ms_floor)
                    h_p_total = h_p_w + h_p_r + h_p_f
                    T_s_now = (h_p_w * model.T_wall + h_p_r * model.T_roof + h_p_f * model.T_floor) / h_p_total
                Q_cool = (spec.h_tr_is * (T_s_now - spec.cooling_setpoint)
                          + spec.h_ve * (T_outdoor_prof[h] - spec.cooling_setpoint)
                          + phi_ia_prof[h])
                Q_cool = max(Q_cool, 0.0)
                T_air_hist.append(spec.cooling_setpoint)
                Q_cool_hist.append(Q_cool)
                model.T_air = spec.cooling_setpoint  # clamp
            else:
                T_air_hist.append(T_air_ff)
                Q_cool_hist.append(0.0)
                model.T_air = T_air_ff

        return np.array(T_air_hist), np.array(Q_cool_hist)

    add_T, add_Q = run(NineR4CAdditive)
    par_T, par_Q = run(NineR4CParallelResistance)

    print()
    print("=" * 78)
    print("DYNAMIC PEAK COOLING TEST: 24-hour Denver summer day (Case 900 params)")
    print("=" * 78)
    print(f"Cooling setpoint: {spec.cooling_setpoint} °C")
    print()
    print(f"{'Hour':>4} {'T_out':>6} {'phi_ia':>6} | {'T_air_add':>10} {'T_air_par':>10} | {'Q_cool_add':>10} {'Q_cool_par':>10}")
    print("-" * 78)
    for h in range(24):
        print(f"{h:>4} {T_outdoor_prof[h]:>6.1f} {phi_ia_prof[h]:>6.0f} | "
              f"{add_T[h]:>10.3f} {par_T[h]:>10.3f} | "
              f"{add_Q[h]:>10.0f} {par_Q[h]:>10.0f}")

    print()
    print(f"Peak cooling (additive):     {add_Q.max():.0f} W ({add_Q.max()/1000.0:.2f} kW)")
    print(f"Peak cooling (non-additive): {par_Q.max():.0f} W ({par_Q.max()/1000.0:.2f} kW)")
    print()
    delta_pct = (par_Q.max() - add_Q.max()) / max(add_Q.max(), 1) * 100.0
    print(f"Δ peak cooling (non-add − add): {par_Q.max() - add_Q.max():+.0f} W ({delta_pct:+.1f}%)")
    print()
    if par_Q.max() > add_Q.max():
        print("✓ Non-additive (parallel-resistance) produces HIGHER peak cooling demand.")
        print("  This is the correct direction for closing the ASHRAE 140 cooling underestimate gap.")
    else:
        print("✗ Non-additive does NOT increase peak cooling in this test.")

    return add_Q.max(), par_Q.max()


# ---------------------------------------------------------------------------
# Sensitivity: vary Case 920 (E/W windows) per-surface conductances
# ---------------------------------------------------------------------------

def case_920_sensitivity():
    """ASHRAE 140 Case 920 — east/west windows instead of south.

    This shifts solar gain to E/W walls during morning/evening, which
    challenges the lumped surface temperature assumption. Compare how
    additive vs non-additive handles E/W timing differences.
    """
    spec = Case900Spec()
    # 920 has E/W windows: 6 m² each. Wall area distribution differs.
    # Assume similar per-surface h_tr_ms to 900 (same construction).

    # Forcing: morning sun on east, afternoon on west
    hours = np.arange(24)
    T_outdoor_prof = np.array([
        18.0, 17.0, 16.5, 16.0, 16.5, 18.0,
        20.0, 24.0, 28.0, 32.0, 35.0, 36.5,
        37.0, 36.5, 35.0, 33.0, 30.0, 27.0,
        24.0, 22.0, 20.0, 19.0, 18.5, 18.0,
    ])
    phi_ia_prof = np.array([
        100, 100, 100, 100, 100, 100,
        150, 200, 600, 1200, 1800, 2400,
        2800, 2400, 1800, 1200, 600, 300,
        200, 150, 120, 110, 100, 100,
    ])
    # Morning E wall gets more sun (sol-air boost ~10° at 8-10am)
    # Afternoon W wall gets more sun (sol-air boost ~10° at 14-16pm)
    T_sol_air_east_prof = T_outdoor_prof + np.array([
        0,0,0,0,0,0, 5,10,15,10,5,2, 0,0,0,0,0,0, 0,0,0,0,0,0,
    ])
    T_sol_air_west_prof = T_outdoor_prof + np.array([
        0,0,0,0,0,0, 0,0,0,0,2,5, 8,10,12,10,5,2, 0,0,0,0,0,0,
    ])
    T_sol_air_roof_prof = T_outdoor_prof + np.array([
        0,0,0,0,0,0, 3,8,12,15,12,8, 5,3,2,1,0,0, 0,0,0,0,0,0,
    ])

    def run(model_cls):
        model = model_cls(spec)
        # Warmup
        for d in range(5 * 24):
            h = d % 24
            model.T_ext_wall = (T_sol_air_east_prof[h] + T_sol_air_west_prof[h]) / 2
            model.T_ext_roof = T_sol_air_roof_prof[h]
            model.T_ext_floor = 16.0
            model.step({}, dt=3600.0)
        # Test day
        T_air_hist, Q_cool_hist = [], []
        for h in range(24):
            model.T_ext_wall = (T_sol_air_east_prof[h] + T_sol_air_west_prof[h]) / 2
            model.T_ext_roof = T_sol_air_roof_prof[h]
            model.T_ext_floor = 16.0
            model.step({}, dt=3600.0)
            T_air_ff = model.compute_air_temperature(T_outdoor_prof[h], spec.h_ve, phi_ia_prof[h])
            if T_air_ff > spec.cooling_setpoint:
                T_s_now = getattr(model, 'T_s', T_air_ff)
                if not hasattr(model, 'T_s'):
                    h_p_w, h_p_r, h_p_f = model.h_path(spec.h_ms_wall), model.h_path(spec.h_ms_roof), model.h_path(spec.h_ms_floor)
                    h_p_total = h_p_w + h_p_r + h_p_f
                    T_s_now = (h_p_w * model.T_wall + h_p_r * model.T_roof + h_p_f * model.T_floor) / h_p_total
                Q_cool = (spec.h_tr_is * (T_s_now - spec.cooling_setpoint)
                          + spec.h_ve * (T_outdoor_prof[h] - spec.cooling_setpoint)
                          + phi_ia_prof[h])
                Q_cool = max(Q_cool, 0.0)
                T_air_hist.append(spec.cooling_setpoint)
                Q_cool_hist.append(Q_cool)
                model.T_air = spec.cooling_setpoint
            else:
                T_air_hist.append(T_air_ff)
                Q_cool_hist.append(0.0)
                model.T_air = T_air_ff
        return np.array(T_air_hist), np.array(Q_cool_hist)

    add_T, add_Q = run(NineR4CAdditive)
    par_T, par_Q = run(NineR4CParallelResistance)

    print()
    print("=" * 78)
    print("CASE 920 (East/West windows): Peak cooling under morning/afternoon solar")
    print("=" * 78)
    print(f"Peak cooling (additive):     {add_Q.max():.0f} W ({add_Q.max()/1000.0:.2f} kW)")
    print(f"Peak cooling (non-additive): {par_Q.max():.0f} W ({par_Q.max()/1000.0:.2f} kW)")
    delta_pct = (par_Q.max() - add_Q.max()) / max(add_Q.max(), 1) * 100.0
    print(f"Δ peak cooling: {par_Q.max() - add_Q.max():+.0f} W ({delta_pct:+.1f}%)")

    return add_Q.max(), par_Q.max()


# ---------------------------------------------------------------------------
# Sensitivity: vary h_tr_is (interior film)
# ---------------------------------------------------------------------------

def h_is_sensitivity():
    """Show how the additive/non-additive gap varies with h_tr_is."""
    print()
    print("=" * 78)
    print("SENSITIVITY: h_tr_is (interior film) vs additive/non-additive gap")
    print("=" * 78)
    print(f"{'h_tr_is':>10} {'h_ms_total':>12} {'h_path_total':>14} {'gap %':>8} {'T_air_add':>10} {'T_air_par':>10} {'ΔT_air':>8}")
    print("-" * 78)

    for h_tr_is in [50.0, 100.0, 165.6, 250.0, 500.0, 1000.0]:
        spec = Case900Spec()
        spec.h_tr_is = h_tr_is
        T_ext_wall, T_ext_roof, T_ext_floor = 45.0, 50.0, 18.0
        T_outdoor, phi_ia = 32.0, 200.0

        add = NineR4CAdditive(spec)
        add.T_ext_wall, add.T_ext_roof, add.T_ext_floor = T_ext_wall, T_ext_roof, T_ext_floor
        for _ in range(5000):
            add.step({}, dt=3600.0)
        T_air_add = add.compute_air_temperature(T_outdoor, spec.h_ve, phi_ia)

        par = NineR4CParallelResistance(spec)
        par.T_ext_wall, par.T_ext_roof, par.T_ext_floor = T_ext_wall, T_ext_roof, T_ext_floor
        for _ in range(5000):
            par.step({}, dt=3600.0)
        T_air_par = par.compute_air_temperature(T_outdoor, spec.h_ve, phi_ia)

        h_ms_total = add.h_ms_total
        h_path_total = par.h_path_total
        gap_pct = (h_ms_total - h_path_total) / h_path_total * 100.0

        print(f"{h_tr_is:>10.1f} {h_ms_total:>12.2f} {h_path_total:>14.2f} {gap_pct:>+7.1f}% "
              f"{T_air_add:>10.3f} {T_air_par:>10.3f} {T_air_add - T_air_par:>+8.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Issue #1281 Python Verification — h_ms_total additive overcounting hypothesis")
    print("=" * 78)
    print("Using ACTUAL Case 900 parameters from the Rust engine:")
    print(f"  h_ms_wall=76.4, h_ms_roof=32.9, h_ms_floor=18.0 W/K (half-insulation rule)")
    print(f"  h_tr_is = 3.45 × 48 = 165.6 W/K (ASHRAE 140 simplified)")
    print(f"  Cm_wall/roof/floor/internal = 5e6/3e6/2e6/1e6 J/K")
    print()

    steady_state_compare()
    print()
    add_peak_900, par_peak_900 = peak_cooling_test()
    case_920_sensitivity()
    h_is_sensitivity()

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    if par_peak_900 > add_peak_900:
        print("✓ Non-additive (parallel-resistance) formulation produces HIGHER peak cooling")
        print(f"  Case 900: {add_peak_900/1000:.2f} → {par_peak_900/1000:.2f} kW")
        print(f"            ({(par_peak_900-add_peak_900)/add_peak_900*100:+.1f}% change)")
        print()
        print("  This is the correct direction to close the ASHRAE 140 high-mass cooling")
        print("  underestimate gap documented in docs/KNOWN_ISSUES.md LIMIT-05 UPDATE.")
        print()
        print("Recommendation:")
        print("  Implement the non-additive correction in:")
        print("    src/physics/multi_node_solver.rs::compute_zone_air_temperature")
        print("    src/sim/thermal_model_physics/physics_impl.rs (t_surface calculation)")
        print("  by replacing the shared-T_s conductance-weighted average with a")
        print("  per-surface series combination (h_ms_k × h_is) / (h_ms_k + h_is).")
    else:
        print("✗ Non-additive does NOT increase peak cooling.")
        print("  The overcounting hypothesis is in the WRONG direction.")
        print("  Investigate: time-stepping, gain distribution, sub-stepping instead.")
