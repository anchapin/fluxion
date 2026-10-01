# CI-03 — investigation history

Narrative history for **CI-03**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `CI-03` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### CI-03: `ort` pinned to a release candidate (issue #2691) — no stable 2.0 on crates.io

- **Affected:** Dependency hygiene / release-gates; both root `fluxion` crate and
  `fluxion-behavior` sibling.
- **Status:** 🔄 Intentional — tracked until `ort` 2.0 stable ships.
- **Details:** The `ort` crate (ONNX Runtime Rust bindings, pykeio/ort) has no stable
  2.0 release on crates.io. As of this writing the latest version is `2.0.0-rc.13`
  (verified via `cargo search ort`); no `2.0.0` (non-prerelease) exists. The v1.3
  milestone therefore pins the newest RC for freshness. The earlier `2.0.0-rc.10`
  pin was bumped to `2.0.0-rc.13`, and — critically — `fluxion-behavior`'s `ort`
  feature was moved OUT of `default` (now `default = []`), so the release-candidate
  ONNX runtime is no longer pulled into every consumer of that sibling. The root
  crate already gated `ort` behind a non-default `ort`/`onnx` feature (issue #1294),
  so default builds never compiled `ort`.
- **Resolution:** When `ort` 2.0 stable (a version without `-rc`/`-alpha`/`-beta`)
  is published, bump the pin in `Cargo.toml` and `fluxion-behavior/Cargo.toml`,
  update the comments, and drop this CI-03 entry. Until then the RC pin is
  intentional and the non-default feature ensures it is opt-in everywhere.
