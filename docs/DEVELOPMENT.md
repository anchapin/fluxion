# Fluxion Development Guide

Developer-facing build environment notes for Fluxion: the memory-safe Cargo
defaults enforced in `.cargo/config.toml` (low-memory linker selection and
debug-info splitting on Linux, Issue #3765), the recommended build-job
concurrency limits that keep local and automated builds from exhausting
system RAM, and the opt-in tooling around them. Platform profiles and CI
invocations are unchanged; this page only explains the defaults every
`cargo build` / `cargo test` picks up from this repository.

*Last Updated: 2026-09-14*

## Memory-safe build defaults (Issue #3765)

On high-core hosts, parallel debug links can exhaust RAM: GNU `ld` streams
full DWARF debug info and Cargo runs up to `nproc` jobs at once. The
repository ships two conservative, Linux-only defaults in
`.cargo/config.toml`:

1. **Low-memory linker** — `[target.'cfg(target_os = "linux")']` sets
   `linker = ".cargo/linker-wrapper.sh"`. The wrapper prefers `mold`, then
   `lld`, and falls back to the system `cc` driver when neither is
   installed, so machines and CI images without them keep building exactly
   as before. Install either one to opt in:

   ```bash
   sudo apt install mold      # Debian/Ubuntu (provides /usr/bin/ld.mold)
   # or
   sudo apt install lld       # provides /usr/bin/ld.lld
   ```

   No configuration change is needed once installed; the next build picks
   it up automatically.

2. **Debug-info splitting** — Linux dev/test builds run with
   `-C split-debuginfo=unpacked`, keeping DWARF data in `.dwo` files beside
   the binary instead of streaming gigabytes of debug data through the
   linker. The `test` and `bench` profiles inherit the `dev` setting; the
   `ci` profile builds with `debug = 0`, so the flag is a no-op there.
   Release profiles are untouched (`Cargo.toml` pins
   `split-debuginfo = "packed"` for `[profile.release]`).

Caveats:

- A dev/test binary run from outside `target/` may show backtraces without
  line numbers, because its debug info lives in sibling `.dwo` files. Run
  it from the build directory (or debug with `rust-gdb`/`rust-lldb`) when
  you need source-level traces.
- If you export `RUSTFLAGS` yourself, Cargo ignores the config `rustflags`
  entry entirely (environment wins). Append
  `-C split-debuginfo=unpacked` manually in that case.
- `mold` requires gcc ≥ 8 / clang with `-fuse-ld=` support; the wrapper
  checks for `ld.mold`/`ld.lld` on `PATH` (the names the compiler drivers
  look up), never failing the build when they are absent.

macOS and Windows are unaffected: the `[target]` table is gated on
`cfg(target_os = "linux")`.

## Concurrency (jobs) recommendations

Repo-level `.cargo/config.toml` deliberately does **not** pin `[build] jobs`
— the right number depends on your cores and RAM, and a repo default would
override every developer's hardware as well as CI sizing. Constrain it at
the user level (`~/.cargo/config.toml`) instead:

```toml
[build]
# Limit parallel compilation/linking jobs to prevent RAM/swap exhaustion
jobs = 4
```

or per invocation:

```bash
cargo build -j 4
cargo test -j 4
```

Rules of thumb: allow roughly 1–2 GB RAM per parallel rustc job; on a
12-core host with 16 GB RAM, `jobs = 4` is a safe ceiling (the value used
by the maintainers). CI is already constrained elsewhere (`cargo nextest`
runs with `--test-threads=2`; see `docs/ci/nextest-rollout.md`), and
automated loops should pass an explicit `-j` so a fast machine cannot
oversubscribe itself. Before large builds, run
`./scripts/disk-space-check.sh` (10 GB minimum) as noted in `AGENTS.md`.
