# Node.js 20 Deprecation Warning — Investigation Report

**Issue**: #2179
**Status**: Not actionable in repository code
**Date**: 2026-07-28

## Summary

The issue claims `.github/actions/setup-rust-env/` uses Node.js 20 explicitly, causing the deprecation warning:

> Node 20 is being deprecated. This workflow is running with Node 24 by default.

**This claim is incorrect.**

## Investigation Findings

### 1. `setup-rust-env` contains NO Node.js references

The action at `.github/actions/setup-rust-env/action.yml` uses only:
- `dtolnay/rust-toolchain@1.94.0` (Rust toolchain installer)
- `mozilla-actions/sccache-action@v0.0.9` (compiler caching)
- `actions/cache@v4` (GitHub Actions cache)

It has no `actions/setup-node`, no `node` version strings, and no Node.js configuration whatsoever.

### 2. The deprecation warning comes from GitHub infrastructure

The warning originates from:
- **GitHub Actions runner infrastructure** — runners now default to Node 24 and warn when running JavaScript actions built for Node 20
- **Third-party JavaScript actions** used in CI workflows:
  - `taiki-e/install-action@v2` (installs cargo-llvm-cov in `code-coverage.yml:53`)
  - `codecov/codecov-action@v4` (uploads coverage in `code-coverage.yml:112`)

### 3. This is NOT fixable in repository code

- GitHub controls the runner's Node.js version default
- Third-party action maintainers must rebuild their actions with Node 24
- There is no configuration in any `.github/actions/` or `.github/workflows/` file that can change this

## Code Coverage Gate Status

The `Code Coverage Gate (Issue #1932)` job **does NOT fail due to this warning**. The warning is informational and does not cause job failure.

If the Code Coverage Gate is failing, the root cause is elsewhere — check the actual job logs for the specific failure reason.

## Recommendation

Close Issue #2179 as **not actionable** — the warning comes from GitHub infrastructure and third-party actions outside this repository's control.
