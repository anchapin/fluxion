# Local CI Validation via `act` — Issue #3577

Run a curated subset of GitHub Actions workflows locally with [`act`](https://nektosact.com/) before pushing, so workflow-shape failures (wrong action refs, missing env, typos in step names) are caught on the operator's machine instead of burning a slot in the GH-hosted runner queue. Configured by `.actrc` (catthehacker/ubuntu:act-22.04 image, `linux/amd64`, default branch `develop`) and driven by `scripts/ci-local.sh` (default suite: `scorecard-drift`, `docs-hygiene`, `architecture_drift`, `scripts-tests`). See `docs/CONTRIBUTING.md` for the broader workflow guide and `AGENTS.md` §"Commands That Are Easy to Guess Wrong" for the one-line pre-push invocation.

## When to run this

- **Before every `git push`** that touches anything in `scripts/`, `.github/workflows/`, `.actrc`, or `docs/`. Run `./scripts/disk-space-check.sh && ./scripts/ci-local.sh` — the disk-space gate (10 GB minimum) precedes the `act` suite so a half-finished run cannot exhaust the filesystem.
- **Authoring or editing a workflow YAML.** Workflow files parse against the GH Actions schema in subtle ways (matrix syntax, expression quoting, `${{ secrets.* }}` resolution); the local `act` run is the cheapest way to surface those errors.
- **Debugging a CI failure** that doesn't reproduce on a clean clone. Replay the failing job locally to iterate without round-tripping through the GH queue.

Skip it for Rust-matrix jobs — those run on native toolchains and cannot be replicated inside the act Docker container; use `cargo test --workspace` (or `cargo nextest run --workspace`) directly.

## Quick start

```bash
# 1. Install act (one-time)
#    macOS:   brew install act
#    Linux:   https://nektosact.com/installation/

# 2. Pre-push pair: disk gate then curated `act` suite (~3m total)
./scripts/disk-space-check.sh && ./scripts/ci-local.sh

# 3. Run a single workflow from the curated set
./scripts/ci-local.sh docs             # only docs-hygiene
./scripts/ci-local.sh scorecard        # only scorecard-drift
./scripts/ci-local.sh architecture     # only architecture_drift
./scripts/ci-local.sh scripts          # only scripts-tests

# 4. Run a specific workflow by file path (escape hatch)
./scripts/ci-local.sh .github/workflows/foo.yml

# 5. Show the curated alias list and timings
./scripts/ci-local.sh --help
```

## What `.actrc` pins

`.actrc` at the repo root configures `act` so contributors cannot hit "missing toolchain" CI-shape bugs:

| Setting | Value | Why |
|---|---|---|
| `-P ubuntu-latest=catthehacker/ubuntu:act-22.04` | Image pinning | Ships rustup + cargo + node + python + common build tools; `:act-22.04` lives on Docker Hub where anonymous rate limits are more lenient than `:full-latest` on ghcr.io. |
| `--container-architecture linux/amd64` | Platform | Avoids ARM emulation on Apple Silicon running x86 Docker. Override to `linux/arm64` if you are on an ARM-native host. |
| `--defaultbranch develop` | Branch | Matches the GH default branch so `pull_request` events resolve correctly. |
| `--verbose` | Logging | Loud by default; pass `-q` (or set `act` flags yourself) to silence. |

## What the curated suite covers (and what it skips)

The default suite is four Python-only workflows chosen for low RAM cost and short wall time:

| Alias | Workflow file | Approx. runtime | What it gates |
|---|---|---|---|
| `scorecard` | `scorecard-drift.yml` | ~30s | `python3 scripts/generate_scorecard.py` + drift detection (Issue #3128) |
| `docs` | `docs-hygiene.yml` | ~1m | `python3 scripts/check_docs_summaries.py`, `check_doc_inventory_fresh.py`, `check_root_hygiene.py` |
| `architecture` | `architecture_drift.yml` | ~30s | `python3 scripts/check_architecture_drift.py` (boundary drift between `ARCHITECTURE.md` and the codebase) |
| `scripts` | `scripts-tests.yml` | ~1m | `pytest` harness for the `scripts/` Python utilities |

What does NOT run under `scripts/ci-local.sh` and why:

- **Self-hosted-only jobs** (`[self-hosted,*]` labels) — act cannot resolve them on a developer machine.
- **Rust matrix jobs** — `cargo test` / `cargo nextest run --workspace` on the host is faster than inside the act container and produces byte-identical results.
- **Jobs with `timeout-minutes > 10`** — host RAM/time constraints; long-running validation suites should run on the GH-hosted runner.
- **macOS / Windows runners** — act's Linux Docker cannot emulate them; jobs targeting those runners will SKIP locally (matching GH behavior with caveats).

## Caveats and false positives

- **Detached worktree checkout.** `act` runs the workflow's `checkout` action against the worktree path it was launched from, which may differ from the GH-runner's checkout root. Expect an occasional `path not found` warning on absolute-path steps; if it does not break the actual check, treat it as a false positive and rely on the GH queue for the authoritative run.
- **macOS / Windows jobs SKIP.** As noted above; this matches GH behavior roughly. Any SKIP in local `act` output does not constitute a CI failure.
- **Container network.** Some actions hit external services (e.g. `actions/cache` against a remote backend). On a corporate network or behind a VPN, the local run may hang at a network step; pass `--no-skip-checkout` carefully and consider increasing Docker's network resources.

## Troubleshooting

- **`ERR: 'act' not in PATH`** — install via `brew install act` (macOS) or follow <https://nektosact.com/installation/> (Linux). The script intentionally exits 2 (no fallback) so you notice.
- **`ERR: Docker daemon not reachable`** — start Docker Desktop (or `systemctl start docker` on Linux). act shells out to `docker info` as a hard precondition.
- **`unknown workflow '$arg'`** — pass one of `fmt`, `scorecard`, `docs`, `architecture`, `scripts`, or a path to a `.yml` file under `.github/workflows/`. The curated aliases are listed by `./scripts/ci-local.sh --help`.
- **A specific job fails locally but passes on GH** — capture the full `act` output (`-v` or `--verbose` is already on by default from `.actrc`) and open an issue or paste it in the PR thread. Treat `act` as a debugging aid, not the source of truth — the GH-hosted runner is authoritative.

## Source files

- `.actrc` — image / architecture / default-branch pinning
- `scripts/ci-local.sh` — curated suite driver
- `scripts/disk-space-check.sh` — 10 GB minimum gate that precedes `ci-local.sh`
- `AGENTS.md` §"Commands That Are Easy to Guess Wrong" — one-line pre-push pair
- `docs/CONTRIBUTING.md` §"Running CI locally with `act`" — broader context and example invocations

## Related

- Issue #3577 — this doc was created to resolve it
- PR #3568 — initial landing of `.actrc` + `scripts/ci-local.sh`
- `.github/BRANCH_PROTECTION.md` — required status checks; local validation does not bypass them, it shortens the iteration loop before they fire
