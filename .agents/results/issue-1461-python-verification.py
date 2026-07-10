"""Numerical verification for the ThermalManifold data structure (issue #1461).

Verifies that the matrix-form gauge transport T_new = T + dt*(M*T + A) is
exactly the simultaneous forward-Euler step of the simplified 5R1C ODE that
`ThermalManifold::from_5r1c_parameters` encodes:

    C_air  · dT_air/dt = (T_mass - T_air)/R_eq + Q_internal
    C_mass · dT_mass/dt = (T_air - T_mass)/R_eq + Q_solar

with parameters and timestep matching `test_from_5r1c_matches_legacy_ode` in
`src/physics/geometry_tensor.rs`.
"""
import numpy as np

R_EQ = 0.10
C_AIR = 10_000.0
C_MASS = 50_000.0
T_AIR_0 = 20.0
T_MASS_0 = 20.0
Q_INT = 200.0
Q_SOLAR = 800.0
DT = 60.0
STEPS = 50


def main() -> None:
    g_eq = 1.0 / R_EQ
    M = np.zeros((4, 4))
    # Air row (idx 0):
    M[0, 0] = -g_eq / C_AIR
    M[0, 1] = +g_eq / C_AIR
    # Mass row (idx 1):
    M[1, 0] = +g_eq / C_MASS
    M[1, 1] = -g_eq / C_MASS
    # Roof / floor slots (idx 2, 3) stay at 0 — inert.

    # Source vector — internal gains go to the air slot, solar to the mass slot.
    A = np.zeros(4)
    A[0] = Q_INT / C_AIR
    A[1] = Q_SOLAR / C_MASS

    # Lock-step Euler (legacy, simultaneous) ↔ matrix form (geometric).
    legacy_air = T_AIR_0
    legacy_mass = T_MASS_0
    T = np.array([T_AIR_0, T_MASS_0, 0.0, 0.0])

    max_drift = 0.0
    for k in range(STEPS):
        air_rate = ((legacy_mass - legacy_air) / R_EQ + Q_INT) / C_AIR
        mass_rate = ((legacy_air - legacy_mass) / R_EQ + Q_SOLAR) / C_MASS
        legacy_air += DT * air_rate
        legacy_mass += DT * mass_rate

        T = T + DT * (M @ T + A)
        drift = max(abs(T[0] - legacy_air), abs(T[1] - legacy_mass))
        max_drift = max(max_drift, drift)

    print(f"After {STEPS} steps:")
    print(f"  T_air:  legacy={legacy_air:.10f}, matrix={T[0]:.10f}")
    print(f"  T_mass: legacy={legacy_mass:.10f}, matrix={T[1]:.10f}")
    print(f"  Max |legacy − matrix|: {max_drift:.3e}")
    print(f"  T_roof:  {T[2]:.10e}  (must be 0)")
    print(f"  T_floor: {T[3]:.10e}  (must be 0)")
    assert max_drift < 1e-9, "matrix-form drift exceeds floating-point tolerance"
    assert abs(T[2]) < 1e-12 and abs(T[3]) < 1e-12, "inert slots must stay at 0"
    print("OK — matrix form tracks legacy 5R1C to floating-point precision.")


if __name__ == "__main__":
    main()
