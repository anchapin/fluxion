# Security

This document covers security policy and accepted-risk advisories for the
Fluxion dependency graph.

## Reporting vulnerabilities

Email security@fluxion.org (PGP key on request) for any suspected security
issue. Do **not** open a public GitHub issue for unreported vulnerabilities.

## Accepted-risk advisories

The advisories below are suppressed in `.cargo/audit.toml` with a documented
justification. Each entry includes the rationale and the conditions under
which it should be revisited.

### `RUSTSEC-2026-0192` — `ttf-parser` (unmaintained)

| Field | Value |
| --- | --- |
| Crate | `ttf-parser` v0.20.0 |
| Type | unmaintained (no security vulnerability) |
| Parents | `plotters` v0.3.7 (default `ttf` feature) → `font-kit` v0.14.x → `ttf-parser` |
| Fluxion use site | `src/validation/report.rs` (chart caption rendering via `("sans-serif", 50).into_font()`) |
| Issue | [#1458](https://github.com/anchapin/fluxion/issues/1458) |
| First seen | 2026-06-29 (advisory publication) |

#### Why ignored

- **Not a vulnerability.** Rustsec type is `unmaintained`, not a CVE. There is
  no patched version (latest published release, v0.25.1 from Nov 2024, is
  also unmaintained and carries the same advisory). [Advisory
  link](https://rustsec.org/advisories/RUSTSEC-2026-0192.html).
- **Depth-2 transitive.** Fluxion does not import `ttf-parser` directly. The
  crate enters via `plotters`'s default `ttf` feature, which pulls
  `font-kit` → `ttf-parser`. Confirmed via
  `cargo tree -i ttf-parser`:
  ```
  ttf-parser v0.20.0
  └── plotters v0.3.7
      └── fluxion v1.0.0
  ```
- **No mainstream successor.** The advisory recommends `skrifa` (Google
  Fonts "oxidize" / fontations). `skrifa` is **not** adopted by `plotters`
  0.3.7; switching requires either waiting for an upstream `plotters`
  release or wiring it ourselves.

#### Why not removed

There is no `ttf-parser` version that is currently maintained, so no version
bump can clear the warning. Two removal paths were considered and deferred:

1. **Switch `plotters` to its `ab_glyph` font feature.**
   `plotters` already supports a `ttf`-free font path (`ab_glyph` +
   `register_font`). However, `ab_glyph` does not auto-load system fonts, so
   the existing `("sans-serif", 50).into_font()` call sites in
   `src/validation/report.rs` would require bundling a TTF/OTF font file
   (with license review) and adding a `register_font` call. This exceeds the
   S/<=4h low-risk scope of issue #1458.
2. **Replace `plotters` entirely.** Out of scope: `plotters` is used
   only for two placeholder chart generators that are currently inert
   (not invoked from any test or production code path). Larger refactor
   belongs in a dedicated cleanup issue.

#### Revisit when

- `ttf-parser` ships a release after June 2026 (advisory entry updated to
  "patched" with a non-zero version range), **or**
- `plotters` 0.3.x or 0.4 ships a release that drops or replaces its
  `font-kit`/`ttf-parser` font backend, **or**
- Fluxion removes the inactive chart-placeholder code in
  `src/validation/report.rs` and the `ttf`/`bitmap_encoder` `plotters`
  features can be safely dropped from the root `Cargo.toml`, removing the
  `ttf-parser` entry from all three lockfiles.

Tracking issue: [#1458](https://github.com/anchapin/fluxion/issues/1458).

---

### `RUSTSEC-2024-0436` — `paste` (unmaintained)

| Field | Value |
| --- | --- |
| Crate | `paste` |
| Type | unmaintained (no security vulnerability) |
| Parents | `nalgebra`, `statrs`, `faer` (transitive) |
| Resolution | No replacement released. `paste` `1.x` is widely used and
  baked into the Rust numerical ecosystem. |

### `RUSTSEC-2026-0177` — PyO3 0.22 `Sync` bound (PR-B)

| Field | Value |
| --- | --- |
| Crate | `pyo3` 0.22.6 |
| Type | memory-safety (PyCFunction::new_closure missing `Sync` bound) |
| Affects | pyo3 0.15.0..<0.29.0 |
| Fixed in | pyo3 ≥ 0.29.0 |
| Fluxion use site | `Cargo.toml:205` pins `pyo3 = "0.22"`; used by `src/python/` and `src/bin/fluxion-py/` |
| Tracking | [Issue #2553](https://github.com/anchapin/fluxion/issues/2553) (PR-B) |
| Time-bound | **2026-12-31** — if migration has not landed by this date, REMOVE this ignore from `.cargo/audit.toml` so cargo audit surfaces the failure and forces the work to be prioritized |

#### Why ignored

- **Actively known vulnerability.** PyO3 0.22.6 is missing the `Sync` bound
  on `PyCFunction::new_closure` (GHSA-chgr-c6px-7xpp). Patched in
  pyo3 ≥ 0.29.0.
- **Migration is non-trivial.** pyo3 0.22 → 0.29 spans 7 minor versions and
  includes breaking API changes:
  - `Bound<'py, T>` migration (replaces the old `&PyAny` / `&PyTuple` API)
  - `IntoPy` / `FromPyObject` rewrites
  - GIL-refs removal (the `Python::with_gil` / `PyGil::acquire` API is gone)
  - `abi3-py310` → `abi3-py312` (or higher) abi bump
  - `pyo3-macros` and `pyo3-build-config` version alignment
  - `src/python/` and `src/bin/fluxion-py/` must be rewritten against the new API
- **Out of scope** for issue #2553's "remove stale ignores" mandate.

#### Why not removed

Removing this ignore would make `cargo audit` exit non-zero on every PR
until the PyO3 migration lands, blocking all development. The
**time-bound** in `.cargo/audit.toml` (target 2026-12-31) is the
explicit deadline: after that date, the ignore must be removed and the
PyO3 migration becomes a release blocker.

#### Revisit when

- pyo3 ≥ 0.29.0 migration is complete and `Cargo.toml` pins the new version, **or**
- 2026-12-31 arrives without migration — at which point REMOVE the ignore.

### `RUSTSEC-2026-0098`, `RUSTSEC-2026-0099` — remediated transitively in #2553 (PR-A)

Both rustls-webpki name-constraint advisories (URI names, wildcard DNS) are
patched in `rustls-webpki >= 0.103.12`. Fluxion's `Cargo.lock` resolves
`rustls-webpki` to `0.103.13` (transitively via `rustls 0.23.37` →
`hyper-rustls 0.27.7` → `reqwest 0.12.28`). The previous ignore entries
were stale suppressions; they were removed in PR-A so that `cargo audit`
now verifies the fix on every PR.

---

## Audit configuration

The file `.cargo/audit.toml` is the canonical `cargo audit` configuration
(per the [rustsec example](https://github.com/rustsec/rustsec/blob/main/cargo-audit/audit.toml.example)).
An older `audit.toml` at the repository root is a historical artifact from
before the convention was adopted and is **not** read by `cargo audit` 0.22+.

To reproduce locally:

```bash
cargo audit           # unmaintained warnings are informational, no failure
cargo audit -D unmaintained  # strictest gate (used in CI if/when enforced)
```

The CI workflow at `.github/workflows/security.yml` runs `cargo audit` on every
pull request and weekly on the main branch.
