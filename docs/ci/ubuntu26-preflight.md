# Ubuntu 26 Runner Migration Preflight — Issue #3960

Preflight audit for the GitHub `ubuntu-latest` → Ubuntu 26.04 migration (rolling out Oct 19 – Nov 19, 2026, not a single cutover). Inventory at `develop` 73ed427: 53 workflows, all 53 reference `ubuntu-latest` (216 refs), zero pin an explicit version. This report classifies the risk-sensitive lanes (β-soak, physics/ASHRAE, perf, TLS-fail-closed, apt-install surfaces), assesses the SHA-pinned setup actions, documents the act/local divergence, and records the adopted policy: **pin `ubuntu-24.04` everywhere** (decided 2026-09-26) — the mechanical sweep (51 files, 208 refs) is implemented in this branch alongside the audit.

## 1. Inventory (measured 2026-09-26 on `develop` @ `73ed427`)

- **53** workflow files in `.github/workflows/`, **all 53** reference `ubuntu-latest`, **216** total refs.
- The issue's 2026-09-24 count (52 workflows / 426 refs) is superseded: PR #4090's duplicate-YAML-key consolidation removed ~200 duplicated `runs-on:` lines, and one workflow was added since.
- Top consumers:

| Workflow | `ubuntu-latest` refs | Lane / role |
|---|---|---|
| `performance_dashboard.yml` | 28 | nightly perf dashboards |
| `ashrae_validation.yml` | 27 | Lane 2 ASHRAE 140 validation |
| `rust-tests.yml` | 15 | Lane 1 per-PR physics gauntlet |
| `ci-gates.yml` | 12 | phase-1 required-check gate |
| `wasm-build.yml` / `python-bindings.yml` | 11 / 11 | bindings builds |
| `determinism_check.yml` | 7 | determinism gate |
| `ci.yml` | 6 | CI coordinator |
| `security.yml` / `fast_math_check.yml` | 5 / 5 | security scans / fast math |

- **Zero** workflows pin an explicit version (`ubuntu-24.04` etc.). The only non-`-latest` variant is `loom-stress.yml`'s `ubuntu-latest-8-cores`.
- 16 workflows use the `vars.FLUXION_LINUX_RUNNER` expression form (self-hosted on trusted push, `ubuntu-latest` fallback on PRs); the fallback literal is what the flip touches.
- Note: `perf_lane.yml` (PR #4094, unmerged at audit time) already pins `ubuntu-24.04` by design.

## 2. Risk classification

### 2a. β-soak — `nightly-ashrae-140-gauge.yml:57`
Single `runs-on: ubuntu-latest`. The 0/30-night counter (Issue #3286, §LIMIT-21) would change OS mid-soak. Mitigating context: per ADR-0017 the flip authority moved from β-soak to the #3986 teacher validation suite, and the streak is pinned at 0/30 (all red) — so a flip corrupts the record-keeping, not a near-complete production streak. Still needs the flip-day annotation (draft in §6).

### 2b. Physics / ASHRAE lanes
- `ashrae_validation.yml` (27 refs, mixed plain + `FLUXION_LINUX_RUNNER` expression form), `rust-tests.yml` (15 refs, heavy apt-install — see §2e).
- Companion gates on the vars-expression form: `.github/workflows/physics-pr.yml` (job: "ASHRAE 140 Strict Energy Gate (Issue #1333)"), `h_tr_em_regression_gate.yml`, plus `ashrae_140_validation.yml`, `fast_math_check.yml`, `determinism_check.yml`.
- Determinism gate is environment-sensitive by definition — absolute-output comparisons across an OS flip are exactly what it exists to catch.

### 2c. Perf lanes
- `performance_dashboard.yml` (28 plain `ubuntu-latest` refs) — nightly absolute-throughput dashboards; the flip changes the throughput context.
- The contention-sensitive TIMING pair (#3952/#3957, kept in-suite with retries) and the new `perf_lane.yml` (pinned 24.04 already) live here.

### 2d. TLS-fail-closed surfaces
`rumqttc-upstream.yml`, `security.yml` cover the `fluxion-twin` MQTT and REST proxy-validation paths. glibc/OpenSSL/CA-bundle jumps on Ubuntu 26 are the exposure; these tests are fail-closed by design, so breakage is loud (good), but the failure mode could be mistaken for a code regression (bad).

### 2e. apt-install build deps (unpinned — highest breakage vector)
- `ashrae_validation.yml`: 20 apt-get refs; `rust-tests.yml`: 14; `fast_math_check.yml`: 4; `ci-gates.yml`: 4; `ashrae_benchmark_harness.yml`: 4.
- These install distro packages by name on the floating image; package renames/removals on 26.04 are the classic migration breaker.

## 3. Setup-action check

All actions are SHA-pinned (verified via `check_workflow_pin.py`):
- OS-agnostic, safe: `actions/checkout` (123 refs), `actions/upload-artifact` (49), `nick-fields/retry` (41), `actions/cache` (35), `actions/github-script` (8).
- Toolchain/environment — SHA-pinned but OS-image-adjacent, **needs live validation** on `ubuntu-26.04`: `actions/setup-python` (16), `Swatinem/rust-cache` (24), `docker/*` (9), `actions/setup-node` (3), `aws-actions/configure-aws-credentials` (3), `awalsh128/cache-apt-pkgs-action` (2), `baptiste0928/cargo-install` (2), `taiki-e/install-action` (1), `dawidd6/action-download-artifact` (2).
- `dtolnay/rust-toolchain` is SHA-pinned but **3 different SHAs are in use** across workflows — minor inconsistency worth unifying in the pin PR; the toolchain itself is image-independent (rustup), and GitHub's own 26.04 image ships the same Rust 1.98.1 as 24.04, so MSRV 1.98.0 is unaffected either way.
- Known 26.04 stack deltas (from the official migration issue actions/runner-images#14748): Node 24, Python 3.14, kernel 7.0, Java 17 default. Workflows assuming Node 20-era or Python 3.12-era preinstalled behavior should be re-verified; nothing in the fluxion tree was found hardcoding those versions.

## 4. `.actrc` / local-validation divergence

- `.actrc` pins `catthehacker/ubuntu:act-22.04` — two LTS behind CI's 24.04 today, three behind after the flip. The pin is deliberate (Docker Hub rate limits, preinstalled rustup/cargo/node/python) and local `act` runs are shape-checking, not environment parity — but the fidelity gap is widening and should be re-acknowledged, not silently inherited.
- `bc` gap (still present): `scripts/disk-space-gate.sh` (pre-push) and `scripts/annual_ashrae_revalidation.sh` shell out to `bc`, which is not guaranteed on the act image or minimal dev machines. Unrelated to the Ubuntu 26 flip (GH images ship `bc`), but it is the same class of undeclared-dependency risk the migration exposes.

## 5. Recommendation: pin everywhere (decided 2026-09-26)

**Pin `ubuntu-24.04` explicitly in ALL workflows in one mechanical workflow-only PR** (qualifies for the `required_checks_workflow_only` lane) — 51 files, 208 refs. Alex chose pin-everywhere over pin-10-and-ride: the rollout is gradual (Oct 19 – Nov 19), so riding workflows would get non-deterministic OS assignment per job for a month, making "is this the OS or my change?" expensive to debug; the sed is trivially reviewable; and uniformity means December's unpin is a single sweep. The "early 26.04 signal" canary argument loses to a deliberate December validation window over accidental October breakage.

Deliberately NOT renamed: the self-hosted custom labels `ubuntu-latest-4-cores` / `ubuntu-latest-8-cores` (used by `ripr-preflight.yml`, `loom-stress.yml`, `mutation-nightly.yml`) — these match Alex's self-hosted Hetzner runner labels, not GitHub-hosted images, so renaming them in workflow files would break runner matching. Relabeling the self-hosted runners themselves is a separate ops task.

Mechanical rule: `ubuntu-latest` → `ubuntu-24.04` everywhere except the `-4-cores`/`-8-cores` self-hosted labels; in the `vars.FLUXION_LINUX_RUNNER` expression form, only the fallback literal changes (guard shape unchanged); matrix values, `fromJSON` matrices, artifact-name derivations, and descriptive comments/echoes updated in lock-step so nothing references a stale name.

### `check_runner_routing_policy.py` implications
None blocking. The gate constrains the guard conjunction around `vars.FLUXION_LINUX_RUNNER` (invariants 1–3), not the fallback literal — swapping the literal preserves the trust boundary. Run the gate after the sweep regardless.

### Pinning is deferral, not avoidance
Set a re-validation deadline (~Dec 2026): run the pinned lanes once against `ubuntu-26.04` via `workflow_dispatch` on a scratch branch, then unpin. Ubuntu 24.04 images have a finite support tail; the pin buys a controlled cutover, not a permanent exemption.

## 6. β-soak flip-day annotation (draft for `docs/KNOWN_ISSUES.md` §LIMIT-21)

> **LIMIT-21 UPDATE (flip day, Oct/Nov 2026):** GitHub's rolling `ubuntu-latest` → Ubuntu 26.04 migration (Oct 19 – Nov 19, 2026) moved the β-soak runner (`nightly-ashrae-140-gauge.yml`) to the new image on <date>. Per ADR-0017 the production flip authority sits with the #3986 teacher validation suite; the β-soak continues as the nightly authority on the `gauge-solver` prototype arm. Counter state at flip: <x>/30 consecutive green (0/30 as of 2026-09-26). Any streak progress after this date is measured on Ubuntu 26.04 and is not directly comparable to pre-flip runs.

## 7. Decision (recorded 2026-09-26)

**Pin everywhere.** The mechanical sweep is implemented in this branch alongside the audit. Left for follow-up (not folded in): the `dtolnay/rust-toolchain` 3-SHA unification (§3) and the self-hosted runner relabeling (§5).
