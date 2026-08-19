# deny.toml `[bans]` duplicate-version budget tracking (Issue #2994)

- **Owner**: Issue #2994 — long-term plan to reduce the duplicate-version count toward zero and ultimately flip `deny.toml [bans] multiple-versions = "deny"`.
- **Live state**: 45 duplicate-version diagnostics (matches `# duplicates_baseline: 45` in deny.toml and `total_duplicates` in the JSON baseline; updated 2026-08-17).
- **Gate**: `.github/workflows/security.yml` `deny` job fails when the live count exceeds the baseline. The Python re-implementation (`scripts/check_deny_duplicate_budget.py`) was removed 2026-08-19 as orphan — see `.agents/results/result-pm.md`; the inline bash counting step in `security.yml` is the only live gate.
- **Cluster inventory (machine-readable)**: `tests/reference_data/deny_budget_baseline.json` (schema_version=1, 10 clusters spanning 45 crates).
- **Reduction roadmap**: 45 → 30 → 15 → 0 (v1.4-M1, M2, M3). Each milestone lowers both the JSON baseline and the deny.toml comment in the same PR.

## How the gate works

`deny.toml [bans]` is configured with `multiple-versions = "warn"` and a `# duplicates_baseline: 45` comment (cargo-deny 0.20.2 rejects unknown keys, so the budget lives as a comment that the CI step parses). `.github/workflows/security.yml`'s `deny` job runs `cargo deny -f json check bans`, counts `"code":"duplicate"` diagnostics, and fails when the count exceeds the baseline. The Python re-implementation (`scripts/check_deny_duplicate_budget.py`, removed 2026-08-19 as orphan — see `.agents/results/result-pm.md`) added a cross-check against the JSON baseline so a future reduction PR could lower the budget atomically in both places; that cross-check is no longer available, and a future reduction PR must lower both fields in the same PR by hand.

## Cluster inventory

Ten logical clusters spanning 45 distinct crates (each cluster entry in the JSON baseline maps to one or more `cargo deny` diagnostics):

| Cluster | Crates | Diagnostic count | Apparent source | Reduction strategy |
|---|---|---:|---|---|
| `windows-sys-family` | `windows-sys`, `windows-targets`, six `windows_*` arch crates | 9 | transitive via faer 0.24, directories 5, rustls 0.23, anstream/colored/tempfile | bump the entire family to a single major (likely 0.61) |
| `nalgebra-family` | `nalgebra`, `ndarray`, `simba`, `wide`, `equator`, `equator-macro` | 6 | statrs pulls nalgebra 0.33; workspace is on 0.35 | bump statrs (or drop it) and align the simba/equator chain |
| `rand-family` | `rand`, `rand_chacha`, `rand_core`, `rand_distr`, `getrandom` | 5 | rustls et al. on rand 0.8; reqwest path on rand 0.9 | resolves once rustls/reqwest align on rand 0.9 |
| `thiserror-family` | `thiserror`, `thiserror-impl` | 2 | redox_users/sysctl on 1.x; workspace-direct on 2.x | bump the thiserror 1.x dependency chain |
| `directories-family` | `directories`, `dirs-sys` | 2 | fluxion-direct on 5; plotters via font-kit on 6 | resolves once plotters/font-kit drop 5 |
| `single-version-pairs` | 17 two-version clusters (block-buffer, core-foundation, cpufeatures, crypto-common, digest, foldhash, foreign-types, foreign-types-shared, libloading, redox_users, rustc-hash, rustls-webpki, safe_arch, sha2, smallvec, syn, winnow) | 17 | various transitive sources | tackle per crate, lowest-effort first |
| `uom` | `uom` | 1 | fluxion on 0.35; fluxion-fluid on 0.38 | resolves once fluxion bumps to 0.38 |
| `r-efi` | `r-efi` | 1 | rustls pulls both 5.x and 6.x depending on patch release | resolves once rustls pin updates |
| `bitflags` | `bitflags` | 1 | legacy 1.3.2 + current 2.11.0 | identify the lone 1.3.2 transitive; `cargo update` |
| `hashbrown` | `hashbrown` | 1 | reqwest on 0.15; criterion plot on 0.16 | resolves once criterion plot bumps to 0.16 |

The full per-crate versions + lock-line pointers live in `tests/reference_data/deny_budget_baseline.json`.

## Reduction roadmap

| Milestone | Target | Target clusters | Notes |
|---|---:|---|---|
| **v1.4-M1** | 30 | `bitflags`, `cpufeatures`, `hashbrown`, `directories-family`, `uom`, `foldhash` | Lowest-effort cluster wins. Each is a single dependency bump or workspace pin. |
| **v1.4-M2** | 15 | `r-efi`, `rand-family`, `winnow`, `thiserror-family`, `syn` | Touches rustls pin + pyo3-style transitive chain bumps. |
| **v1.4-M3** | 0  | `windows-sys-family`, `nalgebra-family`, residual pairs | Then flip `deny.toml [bans] multiple-versions = "deny"` and migrate any unavoidable entries to `skip` with `reason`. |

Each milestone decrements both `deny.toml [bans] duplicates_baseline` (comment) and `tests/reference_data/deny_budget_baseline.json` (`duplicates_baseline`, `total_duplicates`) in the same PR; the JSON file is the source of truth the script reads, the deny.toml comment is what `.github/workflows/security.yml` parses.

## Operator workflow

```bash
# 1. Reproduce the live count (inline bash in security.yml does this in CI).
#    Locally, run the same cargo-deny invocation the workflow runs:
cargo deny -f json check bans 2>&1 \
    | grep -c '"code":"duplicate"' || true
# → 45 (matches the deny.toml comment + JSON baseline)

# 2. To machine-readable JSON: see the inline counting step in
#    .github/workflows/security.yml `deny` job (the only live gate as of
#    2026-08-19 — the Python re-implementation was removed as orphan).

# 3. To reduce a cluster (example: bitflags):
#    - identify the 1.3.2 transitive (rg "bitflags = \"1\"" --type toml),
#    - bump or `cargo update -p bitflags --precise 2.11.0`,
#    - rerun step 1; live count drops to 44,
#    - edit tests/reference_data/deny_budget_baseline.json: drop the "bitflags" entry from `single-version-pairs.crates`,
#      decrement `duplicates_baseline` and `total_duplicates` to 44,
#    - edit deny.toml: change `# duplicates_baseline: 45` to `# duplicates_baseline: 44`,
#    - commit both files in one PR.

# 4. The CI gate from #2933 catches regressions automatically (the security.yml job
#    runs every PR).
```

## References

- **Issue #2994** — this tracking dashboard (work to reduce 45 → 0).
- **Issue #2933** — the CI gate (budget enforcement + regression detection).
- **Issue #2699** — original cargo-deny introduction (closed).
- **`deny.toml [bans]`** — `multiple-versions = "warn"`, `# duplicates_baseline: 45`.
- **`.github/workflows/security.yml` `deny` job** — runs the inline counting step every PR.
- **`tests/reference_data/deny_budget_baseline.json`** — machine-readable cluster inventory (schema_version=1).
- *Removed 2026-08-19 (see `.agents/results/result-pm.md`): `scripts/check_deny_duplicate_budget.py` (Python re-implementation with JSON baseline) and `scripts/ci/test_check_deny_duplicate_budget.py` (pytest regression harness, 6 cases).*