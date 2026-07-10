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

`PyO3` and `reqwest` advisories marked as accepted-risk pending upstream
bumps that introduce breaking API changes (see `.cargo/audit.toml` for
per-line rationale).

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
