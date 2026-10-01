# LIMIT-32 — investigation history

Narrative history for **LIMIT-32**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-32` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-32: CTF steady-state flux for low-mass walls has the wrong sign

**Category:** LIMIT (known open physics defect, tracked upstream)

**Status:** Resolved (fix merged, PR #4062 — single-node dual-boundary B-coupling)

**Evidence (Issue #4062, found by `tests/all_tests/conduction_1052rp_analytical.rs`):**
under constant forcing (20 °C zone / 28 °C sol-air) the `CTFSolverWrapper`
steady-state flux for an 80 mm EPS wall now converges to **+3.688 W/m²** (was −5.53 W/m²), matching the physical value U_filmed·ΔT = +3.69 W/m². Heavy, medium, and multi-layer walls unchanged. The EPS wall now reproduces the analytical ZOH reference to 5+ significant figures.

**Root cause (`src/physics/state_space_ctf/mod.rs::build_state_space_matrices`):**
for a single-layer wall whose discretization yields exactly ONE state node
(EPS 80 mm at dt = 3600 s: Fo = 0.64, dxn = 0.0907 m > L = 0.08 m →
n = ceil(L/dxn) = 1), the node is simultaneously the exterior and interior
boundary. The exterior-boundary branch in the if/else chain fires first and sets
`b_mat[i][0]` (exterior surface coupling) but leaves `b_mat[i][1] = 0`
(interior surface coupling never set). The interior temperature then enters
only through the D direct term, doubling the Y-column DC gain
(−2U instead of −U) and inverting the steady-state flux sign.
Fix: when `is_exterior_boundary && is_interior_boundary`, also set
`b_mat[i][1] = k * dxtmp_boundary` inside the exterior branch. The A
diagonal stays −2k·dxtmp_b (each surface counts as one k·dxtmp_b neighbor).
DC gain becomes exactly [[U, −U], [U, −U]]. The old `abs().max(0)`
clamping in `ctf_coefficients.rs` is dead reference code (superseded by
state_space_ctf); the LIMIT-32 diagnosis pre-dates the state-space migration.

**Guardrails:** `ctf_stays_within_linear_envelope` and
`ctf_1052rp_ss_validations` pin the corrected behavior; the proptest in
`debug_new_expm_tests.rs` now checks both X and Y DC gains to 1% across
10,000 randomised wall configurations. The evolution test
`golden_summary_matches_all_walls` confirms production == seed kernel.
