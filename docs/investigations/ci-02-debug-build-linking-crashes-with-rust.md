# CI-02 — investigation history

Narrative history for **CI-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `CI-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### CI-02: Debug build linking crashes with rust-lld segfault (issue #2297)

- **Affected:** Local debug builds (`cargo build`, `cargo test`, `cargo clippy`) on
  disk-space-constrained systems.
- **Status:** 🔄 Known limitation — release builds (`--release`) work correctly.
- **Details:** Debug builds of `fluxion-rest` (and other large targets) crash during
  linking with SIGSEGV in rust-lld. This is an environmental issue — the linker
  runs out of memory or disk I/O bandwidth during the debug build's heavy compilation
  unit count. Observed during earth tube integration work (PR #2280). Not reproducible
  in CI (GitHub-hosted runners have more disk space and memory). The fix is to use
  release builds for local development, or ensure sufficient free disk space (~100 GB+
  recommended).
- **Workaround:** Use `cargo build --release` or `cargo test --release` for local
  development. CI uses release builds by default and is unaffected.
