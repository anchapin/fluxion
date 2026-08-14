# Mojo toolchain setup (issue #2979)

Install guide for the [Mojo programming language](https://mojolang.org/) and
the [Modular CLI (`max`)](https://docs.modular.com/cli/) on developer
machines. Unblocks evaluation work in #2938, #2937, and #2940.

Installation is **optional**: Fluxion remains Rust-only per `ARCHITECTURE.md`
and `RULES.md`. Mojo prototypes are evaluation-only; the Rust path is the
source of truth for production code paths.

Verify install status any time with `bash scripts/check_mojo_toolchain.sh`
(advisory gate, exits 0 even when the toolchain is missing — never blocks
CI). The Modular install steps below were captured from
https://mojolang.org/install/ (v1.0.0) on 2026-08-14; re-check upstream
if installation fails.

## System requirements (Mojo v1.0.0)

Source: <https://mojolang.org/docs/requirements/>

- **OS**: Linux (glibc ≥ 2.34, e.g. Ubuntu 22.04 LTS+), macOS Sequoia (15+ on
  Apple silicon), or Windows via WSL with a compatible Ubuntu.
- **CPU**: x86-64-v3 (Haswell-class, ~2013+) on Linux; ARM64 Neoverse N1+ on
  Linux; Apple silicon on macOS.
- **RAM**: ≥ 8 GiB for Mojo development (more for MAX inference / serving).
- **C compiler** (`cc`, `gcc`, or `clang`) on Linux — used as a linker.
- **Xcode 16+ or Command Line Tools 16+** on macOS.
- **GPU**: optional (NVIDIA ≥ driver 580, AMD, Apple silicon).
- Network access to `https://pixi.sh`, `https://astral.sh`, and
  `https://conda.modular.com/max/`.

Fluxion's `deny.toml` does NOT need to be amended for Mojo — Mojo is an
**evaluation-time** dependency, not a crate or workspace member. No Cargo
feature is required to gate Mojo work; the toolchain is purely a developer
workstation concern.

## Install paths

Modular distributes Mojo as a Python/Conda package. Pick **one** of the three
installers below. Pixi is recommended (most reliable); `uv` is the lightest
single-user install.

### Path A — `pixi` (recommended)

[pixi](https://pixi.sh/latest/) is a Conda-compatible package manager
maintained by Prefix.dev; Modular publishes Mojo to the
`https://conda.modular.com/max/` channel.

```sh
# 1. Install pixi if absent.
curl -fsSL https://pixi.sh/install.sh | sh

# 2. Initialise a Mojo project (creates ./life/ with pixi.toml).
mkdir -p ~/mojo-projects && cd ~/mojo-projects
pixi init mojo-hello \
  -c https://conda.modular.com/max/ -c conda-forge \
  && cd mojo-hello

# 3. Add the mojo package.
pixi add mojo

# 4. Verify (this also warms the JIT cache).
pixi run mojo --version
#   -> mojo 25.x.y (or similar; format may shift)
```

To invoke `mojo` outside the project directory, enter the project's shell:

```sh
pixi shell          # enter
mojo --version      # works inside the shell
exit                # leave the shell
```

### Path B — `uv` (lightweight)

[`uv`](https://docs.astral.sh/uv/) is an extremely fast Python package
manager that can install Mojo into a project-scoped virtualenv.

```sh
# 1. Install uv if absent.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Mojo globally (user-level), OR create a project.
#    a) Global user install:
uv pip install --system mojo

#    b) Project-scoped (preferred — matches get-started tutorial):
uv init mojo-hello && cd mojo-hello
uv add mojo
uv run mojo --version
```

### Path C — Legacy `modular` CLI (`curl get.modular.com | sh`)

The issue body cites `curl https://get.modular.com | sh` followed by
`modular install mojo`. That installer still exists at
<https://get.modular.com> and is appropriate on bare Linux/macOS hosts where
you want a system-wide install outside a project directory. It is **not**
required by the Mojo language itself — Pixi/uv cover all current Mojo use
cases — but it remains the canonical "single binary on PATH" install:

```sh
# Linux (Debian/Ubuntu) or macOS. Requires sudo.
curl -fsSL https://get.modular.com | sh -

# Then either `modular install mojo` (binary in ~/.modular/bin) or
# `modular install max` (full MAX framework + CLI).
modular install mojo
mojo --version
```

This path downloads a Debian/RPM/Brew package and adds `~/.modular/bin` to
the user's PATH for subsequent shells.

### Windows

Mojo does not run natively on Windows. Use [WSL2 with Ubuntu
22.04+](https://learn.microsoft.com/en-us/windows/wsl/install) and follow
**Path A** (`pixi`) inside the WSL distro.

## Verifying the install

After any path:

```sh
mojo --version   # MUST print a non-empty version string
max --version    # MUST print a non-empty version string (MAX framework)
```

Both commands must succeed in a fresh shell (PATH includes the install
location — `~/.modular/bin` for Path C, the pixi/uv project venv for
Paths A/B).

You can also run the project-local advisory gate:

```sh
bash scripts/check_mojo_toolchain.sh
```

Expected output (with Mojo installed):

```text
PASS  mojo found: /home/.../mojo
PASS  mojo version: mojo 25.x.y
PASS  max found: /home/.../max
PASS  max version: max 25.x.y
```

Expected output (without Mojo — typical on CI runners and most
contributor laptops):

```text
WARN  mojo not found on PATH (Mojo SDK not installed)
WARN  max not found on PATH (MAX framework CLI not installed)
INFO  Install guide: docs/agents/mojo-setup.md
```

The gate always exits 0; it never blocks CI.

## Troubleshooting

- **`mojo: command not found` after install.** Open a fresh shell so
  `~/.modular/bin` (Path C) or the project venv (Paths A/B) is on PATH. For
  permanent PATH addition in `~/.bashrc` / `~/.zshrc`, see your shell docs.
- **`pixi` install fails behind a corporate proxy.** Set
  `HTTP_PROXY` / `HTTPS_PROXY` (lower-case also works) before running
  `curl -fsSL https://pixi.sh/install.sh | sh`.
- **`uv pip install mojo` says "no such package".** Modular's Mojo PyPI
  distribution is published as `mojo`; if pip fails to find it, ensure
  your `uv` is ≥ 0.4 (older versions do not resolve Modular's index).
  Re-run `uv self update`.
- **`mojo --version` shows an old build.** Run `pixi update mojo` (Path A)
  or `uv sync --upgrade` (Path B) inside the project directory.
- **GPU not detected.** Optional — Mojo runs on CPU. If a GPU is desired,
  verify the driver meets Modular's
  [GPU compatibility matrix](https://mojolang.org/docs/requirements/#gpu-compatibility)
  (NVIDIA driver ≥ 580 on Linux).
- **glibc older than 2.34.** Modular does not test distributions older
  than Ubuntu 22.04. Run Mojo in a container with `ubuntu:22.04` as the
  base image, or upgrade the host.

## Why is this not in `Cargo.toml`?

Mojo is an **evaluation language**, not a workspace dependency. Adding
Mojo to Cargo.toml would:
- trigger `cargo deny` supply-chain scans for a language we don't ship,
- require `cargo audit` to whitelist the Mojo PyPI index,
- bloat the published crate size (`.cargoignore` currently strips docs/,
  models/, etc. to stay <10 MB — see `AGENTS.md` Toolchain Quirks).

The Mojo prototypes referenced by #2938 live in their own evaluation
directory (see #2938 for repo layout). When/if Mojo is ever promoted to
a production dependency, this guide will be retired and the install will
move to `scripts/install_mojo.sh` with a CI gate.

## Related

- Issue #2979 (this document) — installation tracking.
- Issue #2938 — Vectorized 5R1C / Perez Sky Model Mojo prototype.
- Issue #2937 — Mojo & MAX framework evaluation for AI surrogate kernels.
- Issue #2940 — Umbrella Mojo roadmap epic.
- `scripts/check_mojo_toolchain.sh` — advisory detect gate.
- <https://mojolang.org/install/> — upstream install reference.
- <https://docs.modular.com/cli/> — `max` CLI reference.

