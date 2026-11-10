# CONTRACT — fluxion

Canonical source for every line: this repo at `develop` a0ca7fde (2026-11-10), or Alex's standing
directives (operator record). A worker reads this instead of
re-deriving ground rules per brief. Citations are checked by
the plan-coder harness drift check;
a line whose source disappears is stale and must be repaired.

## Never regress

- Energy balance must hold: conduction + convection + radiation +
  solar + HVAC sum to zero for any zone. (RULES.md §Must-Always 1)
- Never hardcode or tune outputs/constants to pass ASHRAE 140; fix
  the root cause. Known failing gates are structural LIMIT-* gaps,
  not permission to relax baselines. (RULES.md §Must-Never 2;
  AGENTS.md intro)
- Never raise the strict-energy-gate baseline to hide Case 600/900
  regressions. (AGENTS.md, strict ±15% gate note)
- Simulation output values changing is not, by itself, a
  regression: re-pin guards/baselines to current values and
  proceed. (Standing directive, Alex, 2026-10-25)
- `SCORECARD.md` is generated — never hand-edit it; regenerate with
  `python3 scripts/generate_scorecard.py`. (AGENTS.md)
- Adding/removing tests obliges: regenerate
  `tests/test_inventory.json`, refresh AGENTS.md count citations,
  bump grown ratchets in `scripts/check_test_inventory_drift.py`
  with a dated comment. (AGENTS.md §Issue #3442)
- `fluxion-core` is a dependency-light leaf: it must not import
  root `sim`, `physics`, `ai`, or `validation` modules; keep
  `src/sim/assembly.rs` and `src/sim/multi_node_thermal.rs` as
  re-export shims. (AGENTS.md)

## Definition of done here

- `cargo fmt` and `cargo clippy` clean; unit tests included for
  new functionality. (RULES.md §Must-Always 3)
- Physics-related change: full validation suite run; any change:
  at minimum the affected cases before/after. (RULES.md,
  Safety Boundaries: Validation)
- Generated files (scorecard, test inventory, topology diagrams)
  regenerated from a clean state in the same PR, never mid-rebase.
  (AGENTS.md, topology + inventory notes)

## Verify with

```bash
cargo nextest run --workspace --exclude fluxion-tauri --test-threads=2 --no-fail-fast  # canonical CI form (AGENTS.md)
cargo test --workspace --exclude fluxion-tauri --no-fail-fast   # local-debug equivalent
cargo clippy --lib -- -D warnings                               # RULES.md pre-commit form
python3 scripts/check_test_inventory_drift.py                   # when tests moved
```

## Hard stops — flag, don't decide

- Physics results moved AND the physics intent is unclear: stop
  and flag for Alex. (Standing directives, 2026-09-29 + 2026-10-25)
- Workflow, branch-protection, or release changes: Alex's call.
  Gate changes are edits to `.github/ci-gates.yaml` (Gate
  Registry, Issue #4254) — never edit `.github/workflows/*.yml`
  to change a gate. (Standing directive, 2026-09-29;
  .github/ci-gates.yaml header)
- Large refactors: Alex's call. (Standing directive, 2026-09-29)

## Scope conventions

- Base branch is `develop`; PRs target `develop`, never `main`;
  never push directly to `main` or `develop`. (RULES.md §4,
  §Must-Never 4)
- Bare `cargo test` covers the ROOT crate only — always use the
  `--workspace` form (or `-p <package>` deliberately). (AGENTS.md,
  Issue #3587)
- `fluxion-fluid/` is the acausal HVAC crate (not
  `fluxion-core/src/fluid/`); run the MCP package with
  `cargo run -p fluxion-mcp`. (AGENTS.md)
