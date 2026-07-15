# cargo-mutants Peak RSS Verification — Issue #1619

## Measurement

**Command**: `/usr/bin/time -v cargo mutants --config .cargo/mutants.toml -p fluxion --list`

**Peak RSS**: 23,402,804 KB ≈ **22.3 GB**

**Target**: < 4 GB (architectural target from issue #1619)

**Gap**: 18.3 GB above target (~5.6x the target)

## System Environment

- OS: Linux Mint 22.3 (Ubuntu 24.04 base)
- Rust toolchain: rustc 1.95.0, cargo 1.95.0
- Runner: Local workstation (8 cores, 32 GB RAM)

## Outcome

- [x] cargo mutants --list completes without OOM (exit status 0)
- [x] Peak RSS measured and recorded
- [ ] Peak RSS < 4 GB — **NOT MET** (measured 22.3 GB)

## Analysis

Peak RSS (~22.3 GB) is **5.6x above** the <4 GB target and **1.4x above** the
<16 GB ceiling noted in `.cargo/mutants.toml`. The gap is significant and
consistent with the pre-existing note in `.cargo/mutants.toml`:

> "peak RSS may still approach the 32 GB limit"

The top contributors (per exclude_globs comments in `.cargo/mutants.toml`):
- `src/physics/state_space_ctf.rs` (4120 lines)
- `src/physics/multi_node_solver.rs` (1744 lines)
- `src/physics/ctf_coefficients.rs` (1776 lines)
- `src/physics/fd_discretization.rs` (739 lines)
- `src/physics/fd_solver.rs` (896 lines)
- `src/physics/ctf_solver.rs` (761 lines)

## Out-of-Scope Actions (per issue #1619)

Per issue #1619 scope, the following are **out of scope** for this measurement task:
- Moving modules to fluxion-core
- Changing exclude_globs
- Modifying ort feature-gating

## Next Steps

To meet the <4 GB target, a future effort would need to:
1. Move additional heavy modules to fluxion-core (breaking the fluxion-core cycle-breaking rule)
2. Expand exclude_globs to cover more files
3. Further ort feature-gating (currently enabled via --features ort)

---

*Generated: 2026-07-15*
