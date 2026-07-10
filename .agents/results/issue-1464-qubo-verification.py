"""QUBO mapping verification — Issue #1464.

Companion script to `src/quantum/qubo_mapping.rs`. Verifies the encoding math
that the Rust module implements:

    Q[(i,k), (j,l)] = metric[i,j] * 2^k * 2^l / scale^2

so that for any binary `x`:

    x^T Q x = T_recon^T · metric · T_recon      (+ optional gauge bias)

where T_recon = (Σ_k 2^k x[(i,k)]) / scale.

Run from repo root: `python3 .agents/results/issue-1464-qubo-verification.py`.
"""

import numpy as np


def build_qubo(metric, K, scale_factor):
    N = 4 * K
    Q = np.zeros((N, N))
    for i in range(4):
        for j in range(4):
            for k in range(K):
                for l in range(K):
                    Q[i * K + k, j * K + l] += metric[i, j] * (2**k) * (2**l) / (scale_factor ** 2)
    return (Q + Q.T) / 2.0


def add_gauge_bias(Q, gauge, K, scale_factor, coeff=1.0):
    for i in range(4):
        for k in range(K):
            Q[i * K + k, i * K + k] -= coeff * gauge[i] * (2**k) / scale_factor
    return Q


def encode_temps(scalar, K, scale_factor):
    bits = np.zeros((4, K), dtype=int)
    for i in range(4):
        val = int(round(scalar[i] * scale_factor))
        val = max(0, min((1 << K) - 1, val))
        for k in range(K):
            bits[i, k] = (val >> k) & 1
    return bits


def decode_temps(bits, K, scale_factor):
    T = np.zeros(4)
    for i in range(4):
        val = sum(int(bits[i, k]) * 2**k for k in range(K))
        T[i] = val / scale_factor
    return T


def qubo_to_ising(Q):
    """Exact QUBO → Ising conversion (symmetric Q)."""
    N = Q.shape[0]
    ones = np.ones(N)
    h = 0.5 * (Q @ ones)
    J = 0.25 * Q.copy()
    np.fill_diagonal(J, 0.0)
    c = 0.25 * np.trace(Q) + 0.25 * (ones @ Q @ ones)
    return h, J, c


def qubo_energy(Q, x):
    return x @ Q @ x


def run_scenarios():
    # 5R1C scene
    r_eq, c_air, c_mass = 0.1, 1000.0, 5000.0
    g_eq = 1.0 / r_eq
    metric_5r1c = np.zeros((4, 4))
    metric_5r1c[0, 0] = -g_eq / c_air
    metric_5r1c[0, 1] = g_eq / c_air
    metric_5r1c[1, 0] = g_eq / c_mass
    metric_5r1c[1, 1] = -g_eq / c_mass

    # 9R4C scene
    g_w, g_r, g_f = 50.0, 30.0, 20.0
    c_a, c_w, c_r2, c_f2 = 1000.0, 5000.0, 3000.0, 8000.0
    metric_9r4c = np.zeros((4, 4))
    metric_9r4c[0, 0] = -(g_w + g_r + g_f) / c_a
    metric_9r4c[0, 1] = g_w / c_a
    metric_9r4c[0, 2] = g_r / c_a
    metric_9r4c[0, 3] = g_f / c_a
    metric_9r4c[1, 0] = g_w / c_w
    metric_9r4c[1, 1] = -g_w / c_w
    metric_9r4c[2, 0] = g_r / c_r2
    metric_9r4c[2, 2] = -g_r / c_r2
    metric_9r4c[3, 0] = g_f / c_f2
    metric_9r4c[3, 3] = -g_f / c_f2

    scenarios = [
        ("5R1C cold zone",     metric_5r1c, np.array([10.0,  8.0, 0.0, 0.0]), np.array([ 500.0,  100.0,    0.0,    0.0])),
        ("5R1C warm zone",     metric_5r1c, np.array([24.0, 26.0, 0.0, 0.0]), np.array([-200.0,  300.0,    0.0,    0.0])),
        ("9R4C mid-temp",      metric_9r4c, np.array([22.0, 20.0, 23.0, 18.0]), np.array([ 100.0,  200.0,   50.0,   30.0])),
        ("Flat manifold",      np.eye(4),   np.array([20.0, 20.0, 20.0, 20.0]), np.zeros(4)),
    ]

    K = 8
    SCALE_MAX = 50.0
    s = (2**K - 1) / SCALE_MAX

    print(f"{'Scenario':<22} {'T_max_err (°C)':<16} {'E_qubo':<14} {'E_recon':<14} {'match':<6}")
    print("-" * 72)
    for name, M, T, g in scenarios:
        Q = build_qubo(M, K, s)
        Q = add_gauge_bias(Q, g, K, s)
        bits = encode_temps(T, K, s)
        T_recon = decode_temps(bits, K, s)
        x = bits.flatten()
        E_qubo = qubo_energy(Q, x)
        E_recon = T_recon @ M @ T_recon - g @ T_recon
        max_err = float(np.max(np.abs(T - T_recon)))
        match = bool(np.isclose(E_qubo, E_recon, rtol=1e-10, atol=1e-9))
        print(f"{name:<22} {max_err:<16.6f} {E_qubo:<14.6f} {E_recon:<14.6f} {match}")

    # Ising conversion sanity check (16 random trials)
    print("\n=== Ising conversion (16 random QUBO solutions) ===")
    M = metric_5r1c
    T = np.array([21.0, 22.0, 0.0, 0.0])
    Q = build_qubo(M, K, s)
    Q = add_gauge_bias(Q, np.array([100.0, 200.0, 50.0, 30.0]), K, s)
    h, J, c = qubo_to_ising(Q)
    rng = np.random.default_rng(42)
    all_match = True
    for trial in range(16):
        x = rng.integers(0, 2, size=4 * K)
        s_ising = 2 * x - 1
        E_qubo = qubo_energy(Q, x)
        E_ising = s_ising @ J @ s_ising + h @ s_ising + c
        ok = bool(np.isclose(E_qubo, E_ising, rtol=1e-10, atol=1e-9))
        all_match &= ok
        print(f"  Trial {trial:2d}: QUBO={E_qubo:>12.6f}  Ising={E_ising:>12.6f}  match={ok}")
    print(f"\nAll Ising trials match: {all_match}")
    return all_match


if __name__ == "__main__":
    ok = run_scenarios()
    raise SystemExit(0 if ok else 1)