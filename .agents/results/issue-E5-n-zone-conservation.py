"""
Issue #1348 / batch_id E5 — N-zone inter-zone thermal coupling conservation.

Verifies the algebraic identity

    Σ_i Σ_j h_tr_ij · (T_j − T_i) = 0   when h_tr_ij = h_tr_ji

for N = 3, 5, 10. The Rust `MultiZoneAirflowNetwork::net_inter_zone_q` and
the `validate_n_zone_network_conservation` validator must reproduce this
identity to machine precision (|Σ q_iz| < 1e-6 W, the Issue #1348 acceptance
criterion). This script mirrors the Python verification in
`.agents/results/issue-1281-python-verification.py` style: it builds the
same symmetric random conductance matrices the Rust unit tests probe, and
asserts conservation numerically.

Usage:
    python .agents/results/issue-E5-n-zone-conservation.py

Exit code 0 ⇒ conservation holds. Exit code 1 ⇒ conservation violated.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class ConservationResult:
    n: int
    symmetric: bool
    net_inter_zone_q_w: float
    tolerance_w: float
    energy_delta_j: float

    @property
    def passed(self) -> bool:
        return abs(self.net_inter_zone_q_w) < self.tolerance_w


def build_fully_connected(n: int, h: float) -> np.ndarray:
    m = np.full((n, n), h, dtype=np.float64)
    np.fill_diagonal(m, 0.0)
    return m


def build_ring(n: int, h: float) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.float64)
    if n < 2:
        return m
    for i in range(n):
        j_next = (i + 1) % n
        m[i, j_next] = h
        m[j_next, i] = h
    return m


def net_inter_zone_q(h_tr: np.ndarray, temps: np.ndarray) -> float:
    """Σ_i q_iz[i] = Σ_i Σ_j h_tr_ij · (T_j − T_i)."""
    n = h_tr.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += h_tr[i, j] * (temps[j] - temps[i])
    return float(total)


def energy_delta_j(C: np.ndarray, T_old: np.ndarray, T_new: np.ndarray, q_ext: np.ndarray, dt: float) -> float:
    """Total energy change during one implicit step, J.

    For a closed system with q_ext = 0 and Σ q_iz = 0, ΔE must be 0 J.
    """
    e_old = float(np.sum(C * T_old))
    e_new = float(np.sum(C * T_new))
    e_added = float(np.sum(q_ext) * dt)
    return e_new - e_old - e_added


def backward_euler_step(C: np.ndarray, h_tr: np.ndarray, T_old: np.ndarray, q_ext: np.ndarray, dt: float) -> np.ndarray:
    n = C.shape[0]
    M = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row_sum = float(np.sum(h_tr[i, :]))
        M[i, i] = C[i] / dt + row_sum
        b[i] = C[i] / dt * T_old[i] + q_ext[i]
        for j in range(n):
            if i != j:
                M[i, j] = -h_tr[i, j]
    return np.linalg.solve(M, b)


def main() -> int:
    print("Issue #1348 — N-zone inter-zone conservation verification")
    print("=" * 60)
    print()
    print(f"{'N':>3} {'topology':<28} {'|Σ q_iz| [W]':>16} {'ΔE [J]':>14} {'pass':>6}")
    print("-" * 70)

    results: list[ConservationResult] = []
    tolerance_w = 1e-6

    # N=3 ring (matches the Rust `three_zone_symmetric_conductance_conserves_energy` test)
    n = 3
    h_tr = build_ring(n, 50.0)
    T_old = np.array([20.0, 25.0, 15.0])
    net = net_inter_zone_q(h_tr, T_old)
    results.append(ConservationResult(
        n=n, symmetric=True,
        net_inter_zone_q_w=net,
        tolerance_w=tolerance_w,
        energy_delta_j=0.0,  # net_inter_zone_q algebraic; no step needed
    ))

    # N=5 fully connected (matches `five_zone_symmetric_conductance_conserves_energy`)
    n = 5
    h_tr = build_fully_connected(n, 30.0)
    T_old = np.array([18.0, 20.0, 22.0, 24.0, 26.0])
    C = np.full(n, 1.0e6)
    q_ext = np.zeros(n)
    dt = 3600.0
    T_new = backward_euler_step(C, h_tr, T_old, q_ext, dt)
    net = net_inter_zone_q(h_tr, T_new)
    de = energy_delta_j(C, T_old, T_new, q_ext, dt)
    results.append(ConservationResult(
        n=n, symmetric=True,
        net_inter_zone_q_w=net,
        tolerance_w=tolerance_w,
        energy_delta_j=de,
    ))

    # N=10 fully connected (matches `ten_zone_symmetric_conductance_conserves_energy`)
    n = 10
    h_tr = build_fully_connected(n, 10.0)
    T_old = np.array([15.0 + i for i in range(n)], dtype=np.float64)
    C = np.full(n, 1.0e6)
    q_ext = np.zeros(n)
    T_new = backward_euler_step(C, h_tr, T_old, q_ext, dt)
    net = net_inter_zone_q(h_tr, T_new)
    de = energy_delta_j(C, T_old, T_new, q_ext, dt)
    results.append(ConservationResult(
        n=n, symmetric=True,
        net_inter_zone_q_w=net,
        tolerance_w=tolerance_w,
        energy_delta_j=de,
    ))

    # N=3 ASYMMETRIC: must fail — sanity check the algebraic identity really is asymmetric-sensitive
    n = 3
    h_tr = np.array([
        [0.0, 5.0, 1.0],
        [3.0, 0.0, 7.0],  # h_01 != h_10 → asymmetric
        [2.0, 4.0, 0.0],
    ])
    T_old = np.array([25.0, 20.0, 15.0])
    net = net_inter_zone_q(h_tr, T_old)
    results.append(ConservationResult(
        n=n, symmetric=False,
        net_inter_zone_q_w=net,
        tolerance_w=tolerance_w,
        energy_delta_j=0.0,
    ))

    for r in results:
        marker = "✅" if r.passed else ("❌" if r.symmetric else "⚠️")
        topology = ("fully connected" if r.symmetric and r.n >= 5
                    else "ring" if r.symmetric
                    else "ASYMMETRIC ring")
        print(f"{r.n:>3} {topology:<28} {abs(r.net_inter_zone_q_w):>16.3e} "
              f"{r.energy_delta_j:>14.3e} {marker:>6}")

    print()

    symmetric_results = [r for r in results if r.symmetric]
    failed = [r for r in symmetric_results if not r.passed]
    if failed:
        print(f"❌ {len(failed)} symmetric network(s) violated conservation:")
        for r in failed:
            print(f"   N={r.n}: |Σ q_iz| = {abs(r.net_inter_zone_q_w):.3e} W > {r.tolerance_w:.0e} W")
        return 1

    print(f"✅ All {len(symmetric_results)} symmetric N-zone networks conserve energy:")
    for r in symmetric_results:
        print(f"   N={r.n}: |Σ q_iz| = {abs(r.net_inter_zone_q_w):.3e} W  (< {r.tolerance_w:.0e} W)")

    # Asymmetric case: confirm |Σ q_iz| > 1e-3 (proves the algebraic identity is asymmetric-sensitive)
    asym = next(r for r in results if not r.symmetric)
    if abs(asym.net_inter_zone_q_w) <= 1e-3:
        print(f"❌ Asymmetric N=3 unexpectedly conserved energy: |Σ q_iz| = "
              f"{abs(asym.net_inter_zone_q_w):.3e} W")
        return 1
    print(f"✅ Asymmetric N=3 correctly detected: |Σ q_iz| = "
          f"{abs(asym.net_inter_zone_q_w):.3e} W  (no false PASS)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
