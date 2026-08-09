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

### `RUSTSEC-2026-0177`, `RUSTSEC-2026-0098`, `RUSTSEC-2026-0099`

Status as of issue [#2553](https://github.com/anchapin/fluxion/issues/2553):

| Advisory | Crate | Status | Reason | Tracking |
| --- | --- | --- | --- | --- |
| `RUSTSEC-2026-0177` | `pyo3` (GHSA-chgr-c6px-7xpp) | **OPEN — CI failure** | Affects `pyo3` 0.15.0..<0.29.0; we pin `pyo3 = "0.22"` (Cargo.lock: `pyo3 0.22.6`). Missing `Sync` bound on `PyCFunction::new_closure` closures (thread-safety bug). | [#2553](https://github.com/anchapin/fluxion/issues/2553) |
| `RUSTSEC-2026-0098` | `rustls-webpki` (GHSA-965h-392x-2mh5) | **REMEDIATED** (transitive) | Patched in `rustls-webpki` >= 0.103.12. Cargo.lock resolves to 0.103.13 (`rustls 0.23.37` → `hyper-rustls 0.27.7` → `reqwest 0.12.28`; also via `rumqttc` pinned to the `fix/rustsec-2026-webpki` git branch in `fluxion-twin`). | [#2553](https://github.com/anchapin/fluxion/issues/2553) |
| `RUSTSEC-2026-0099` | `rustls-webpki` (GHSA-xgp8-3hg3-c2mh) | **REMEDIATED** (transitive) | Same patch as RUSTSEC-2026-0098 (`rustls-webpki >= 0.103.12`). Already fixed in our lockfile. | [#2553](https://github.com/anchapin/fluxion/issues/2553) |

#### Why the three entries were removed from `.cargo/audit.toml`

The previous ignore list bundled the three advisories together, but they have
fundamentally different remediation states. Splitting them surfaced the real
gap (0177) while letting the cargo-audit CI gate verify the other two on every
PR.

#### Upgrade plan for `RUSTSEC-2026-0177`

`pyo3` 0.22 → 0.29 is a 7-minor-version jump with several breaking API
changes (`PyAny::iter` / `IntoPy` / `gil-refs` removed in 0.23, `Bound<'py, T>`
migration through 0.23–0.26, abi3 abi bumps, `pyo3-macros` /
`pyo3-build-config` version alignment). The Python bindings in `src/python/`
and the `fluxion-py` binary entry point must be rewritten against the new
API. This is out of scope for the audit-suppression cleanup and is tracked
in [#2553](https://github.com/anchapin/fluxion/issues/2553) as a follow-up
PR.

Required when the upgrade lands:

1. Bump `pyo3 = "0.22"` → `pyo3 = "0.29"` (Cargo.toml:205).
2. Bump `pyo3-build-config = "0.22"` → `pyo3-build-config = "0.29"`
   (Cargo.toml:277).
3. Migrate `src/python/` and `src/bin/fluxion-py/` to the `Bound<'py, T>` API
   and the new `IntoPy` / `FromPyObject` traits.
4. Re-run `cargo audit` to confirm `RUSTSEC-2026-0177` is cleared.

Until that PR lands, the security.yml CI job will fail on this advisory,
which is the intended outcome of #2553.

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
