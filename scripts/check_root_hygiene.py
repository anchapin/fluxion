#!/usr/bin/env python3
"""
Root hygiene check for Fluxion.

Enforces ``AGENTS.md`` §Repository Hygiene: only allow-listed files may live
at the repo root. Transient artifacts (case dumps, scratch scripts, log
archives, committed binaries, scratch directories) must be moved to ``tmp/``,
``docs/archive/``, ``docs/investigations/``, or deleted.

Originally ``.md``-only (see ``#2466`` and the git history of
``check_root_md_policy.py``). Widened to cover the transient file types that
repeatedly slipped past the ``.md``-only gate:

    - scratch file extensions: ``.txt .csv .rs .py .sh .json .zip``
    - no-extension blobs / committed binaries (e.g. ELF files)
    - known scratch directory names (``fixes/ results/ reports/ ...``)

Legit root files fall into clear buckets and are allow-listed so the gate
does not false-positive on real config:

    - dotfiles (``.gitignore``, ``.rustfmt.toml``, ...) — skipped wholesale
    - ``.md`` allow-list (``README.md``, ``ARCHITECTURE.md``, ...) — unchanged
    - build/config extensions that are NEVER blocked: ``.toml``, ``.yaml``/
      ``.yml``, ``.lock``, ``.pyi``, ``.skill``, ``.onnx``
    - no-extension allow-list: ``LICENSE``, ``Dockerfile``, ``Makefile``
    - per-extension exception allow-list for legit files that use a normally
      blocked extension (e.g. ``requirements-dev.txt``)

A thin backward-compat alias lives at ``scripts/check_root_md_policy.py`` so
``.github/workflows/docs-hygiene.yml`` and ``.pre-commit-config.yaml`` keep
working unchanged.

Usage::

    python3 scripts/check_root_hygiene.py             # run the check
    python3 scripts/check_root_hygiene.py --self-test # deterministic self-test

Exit codes:
    0 — root is clean
    1 — one or more transient/blocked artifacts at root
    2 — script error
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# `.md` policy (unchanged from the original check_root_md_policy.py / #2466)
# Keep in sync with AGENTS.md §Repository Hygiene.
# ---------------------------------------------------------------------------
ROOT_MD_ALLOWLIST: frozenset[str] = frozenset(
    {
        "README.md",
        "ARCHITECTURE.md",
        "CODEBASE_MAP.md",
        "CONTRIBUTING.md",
        "RULES.md",
        "CHANGELOG.md",
        "AGENTS.md",
        "SCORECARD.md",
    }
)

# Auto-generated per-session file that should never be committed.
# Warns but never fails. Belt-and-braces protection lives in `.gitignore`.
ROOT_MD_WARNLIST: frozenset[str] = frozenset({"CLAUDE.md"})

# ---------------------------------------------------------------------------
# Extended hygiene policy (widened gate)
# ---------------------------------------------------------------------------

# Scratch file extensions that should never appear at the repo root.
# Legit exceptions are listed in ROOT_BLOCKED_EXT_ALLOWLIST below.
ROOT_BLOCKED_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".csv", ".rs", ".py", ".sh", ".json", ".zip"}
)

# Legit files that use a normally-blocked extension. Add here if a genuine
# root config/dev file ever needs one of these extensions (e.g. `setup.py`).
ROOT_BLOCKED_EXT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "requirements-dev.txt",
        "requirements.txt",
    }
)

# Build/config extensions that are NEVER blocked at root. Listed explicitly
# so the gate's intent is self-documenting; not in ROOT_BLOCKED_EXTENSIONS.
ROOT_NEVER_BLOCKED_EXTENSIONS: frozenset[str] = frozenset(
    {".toml", ".yaml", ".yml", ".lock", ".pyi", ".skill", ".onnx"}
)

# No-extension files permitted at root. Anything else without an extension
# is treated as a suspect blob (catches committed ELF/Mach-O/PE binaries,
# gh-CLI response dumps, etc.).
ROOT_NO_EXT_ALLOWLIST: frozenset[str] = frozenset({"LICENSE", "Dockerfile", "Makefile"})

# Directory names that are always scratch at root. This is a denylist (not
# an allowlist) so the many legit root dirs (src, docs, tests, scripts,
# crates, ...) are unaffected. Add names here as new scratch dirs appear.
ROOT_BLOCKED_DIRECTORIES: frozenset[str] = frozenset(
    {
        "fixes",
        "results",
        "reports",
        "bem-engineer-workspace",
        "bem-engineer",
        "scratch",
        "output",
        "outputs",
        "artifacts",
    }
)

# Dotfile/dotdir entries at repo root that are LEGIT root config and
# therefore skip the gitignore/tracked cross-check introduced in #2954.
# These are the dot-prefixed counterparts of the `.toml/.yaml/.lock`
# allow-list: real tracked config that Git itself needs to see.
#
# Anything dot-prefixed that is NOT in this allow-list must either be
# matched by `.gitignore` (via `git check-ignore`) OR be already tracked
# in git (legacy runtime dirs that were committed before their
# gitignore line existed). Anything else is an unmanaged dotfile/dotdir
# at root and causes the gate to FAIL.
ROOT_DOTFILE_ALLOWLIST: frozenset[str] = frozenset(
    {
        ".agents",                 # orchestration runtime dir (issues, results, skills)
        ".cargo",                  # cargo config (audit.toml, config.toml, mutants.toml)
        ".cargoignore",            # cargo publish ignore
        ".dockerignore",           # docker ignore
        ".editorconfig",           # editor config
        ".env.example",            # env example (NOTE: `.env` itself is gitignored)
        ".git",                    # git's own dir (always present)
        ".github",                 # GitHub config dir
        ".gitattributes",          # git attributes
        ".githooks",               # local git hooks (tracked for CI parity)
        ".gitignore",              # git ignore
        ".npmignore",              # npm ignore
        ".planning",               # project planning artifacts (AGENTS.md allows)
        ".pre-commit-config.yaml", # pre-commit config
        ".rustfmt.toml",           # rustfmt config
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_gitignored(path: Path) -> bool:
    """Return True if `path` is matched by a `.gitignore` rule.

    Uses `git check-ignore` so the answer matches what Git itself considers
    ignored. Falls back to False if Git is unavailable or the path is not in
    a Git repo.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return False
    return result.returncode == 0


def is_tracked(path: Path) -> bool:
    """Return True if `path` is tracked by git.

    Used as a fallback for the dotfile-root hygiene check introduced in
    #2954: if a dotfile/dotdir at root is already tracked in git (e.g. an
    agent-runtime dir that was committed before the gitignore line was
    added), we accept it rather than demanding an immediate ``git rm``. The
    ``.gitignore`` line is the real defense — it prevents NEW files from
    being committed. Falls back to False if Git is unavailable.
    """
    import subprocess

    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(rel)],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def is_dotfile_root_compliant(path: Path) -> bool:
    """Return True if `path` (a dotfile/dotdir at repo root) is allowed.

    Allowed when ANY of:

      * ``path.name`` is in :data:`ROOT_DOTFILE_ALLOWLIST` — legit root
        config that Git itself needs to see (``.git/``, ``.cargo/``,
        ``.gitignore``, ...).
      * ``path`` is matched by ``.gitignore`` (``git check-ignore``) —
        the gate verifies the gitignore line is in place for new dirs.
      * ``path`` is already tracked in git (``git ls-files --error-unmatch``)
        — handles legacy runtime dirs that were committed before their
        gitignore line existed.

    Otherwise the path is an *unmanaged* dotfile/dotdir at root and the
    gate fails. See issue #2954.
    """
    if path.name in ROOT_DOTFILE_ALLOWLIST:
        return True
    if is_gitignored(path):
        return True
    if is_tracked(path):
        return True
    return False


def _is_dotfile(name: str) -> bool:
    return name.startswith(".") and name not in {"."}


def _has_extension(name: str) -> bool:
    return "." in name.lstrip(".")


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------
class RootScan:
    """Categorized scan of the repo root. ``violations`` drives FAIL."""

    def __init__(self) -> None:
        self.md_allow: list[Path] = []
        self.md_warn: list[Path] = []
        self.md_transient: list[Path] = []
        self.blocked_ext: list[Path] = []
        self.no_ext_blocked: list[Path] = []
        self.blocked_dirs: list[Path] = []
        # Issue #2954: dotfile/dotdir at repo root that is neither
        # allow-listed, gitignored, nor tracked.
        self.dotfile_unmanaged: list[Path] = []

    @property
    def violations(self) -> list[Path]:
        """All findings that should cause the gate to FAIL."""
        return (
            self.md_transient
            + self.blocked_ext
            + self.no_ext_blocked
            + self.blocked_dirs
            + self.dotfile_unmanaged
        )


def scan_root(repo_root: Path) -> RootScan:
    """Scan `repo_root` (non-recursive) and categorize every entry."""
    scan = RootScan()

    for path in sorted(repo_root.iterdir()):
        name = path.name

        # Dotfiles / dotdirs (.gitignore, .rustfmt.toml, .editorconfig,
        # .git/, .cargo/, ...): cross-checked against the dotfile allow-list,
        # `.gitignore`, and git tracking. Issue #2954 widened the gate so a
        # dotfile-prefixed runtime directory is no longer structurally
        # invisible. Anything not allow-listed, not gitignored, and not
        # already tracked is flagged as ``dotfile_unmanaged``.
        # NOTE: this check must run BEFORE the ``path.is_dir()`` short-circuit
        # below, otherwise a dotdir like ``.mytool/`` would be skipped
        # wholesale (it is a directory but not a known scratch dir).
        if _is_dotfile(name):
            if not is_dotfile_root_compliant(path):
                scan.dotfile_unmanaged.append(path)
            continue

        # Directories: check against the scratch denylist.
        if path.is_dir():
            if name in ROOT_BLOCKED_DIRECTORIES:
                scan.blocked_dirs.append(path)
            continue

        # `.md` files: original #2466 policy (allow-list / warn-list).
        if path.suffix == ".md":
            if name in ROOT_MD_ALLOWLIST:
                scan.md_allow.append(path)
            elif name in ROOT_MD_WARNLIST:
                # CLAUDE.md: warn if present and NOT gitignored, else compliant.
                if is_gitignored(path):
                    scan.md_allow.append(path)
                else:
                    scan.md_warn.append(path)
            else:
                scan.md_transient.append(path)
            continue

        # No extension: allow-list only (LICENSE, Dockerfile, Makefile).
        if not _has_extension(name):
            if name in ROOT_NO_EXT_ALLOWLIST:
                continue
            scan.no_ext_blocked.append(path)
            continue

        # Blocked scratch extensions: allow-list exceptions pass.
        if path.suffix in ROOT_BLOCKED_EXTENSIONS:
            if name in ROOT_BLOCKED_EXT_ALLOWLIST:
                continue
            scan.blocked_ext.append(path)
            continue

        # Everything else (never-blocked extensions like .toml/.yaml/.lock/
        # .pyi/.skill/.onnx, plus any unlisted extension): permitted.
        continue

    return scan


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_remediation() -> None:
    print()
    print("Remediation (per AGENTS.md §Repository Hygiene):")
    print("  1. Move the file/dir to tmp/, docs/archive/, docs/investigations/,")
    print("     or delete it if it is pure scratch.")
    print("  2. If it is a case analysis: docs/investigations/")
    print("  3. If it is a session summary / batch log: docs/archive/planning/")
    print("  4. If it is a legit root config file that uses a normally-blocked")
    print("     extension, add it to ROOT_BLOCKED_EXT_ALLOWLIST or")
    print("     ROOT_NO_EXT_ALLOWLIST in scripts/check_root_hygiene.py.")
    print("  5. If it is a dotfile/dotdir (issue #2954): add it to `.gitignore`")
    print("     (preferred) so the gate accepts it on every checkout, or add")
    print("     it to ROOT_DOTFILE_ALLOWLIST in scripts/check_root_hygiene.py")
    print("     if it is legit root config that must stay tracked.")
    print("  6. Update any links in other docs to the new path.")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return _self_test()

    print("=== Fluxion Root Hygiene Check ===")
    print(f"Repo: {REPO_ROOT}")
    print(f".md allow-list: {sorted(ROOT_MD_ALLOWLIST)}")
    print(f".md warn-list (non-blocking): {sorted(ROOT_MD_WARNLIST)}")
    print(f"Blocked extensions: {sorted(ROOT_BLOCKED_EXTENSIONS)}")
    print(f"Blocked directories: {sorted(ROOT_BLOCKED_DIRECTORIES)}")
    print(f"Dotfile allow-list: {sorted(ROOT_DOTFILE_ALLOWLIST)}")
    print()

    scan = scan_root(REPO_ROOT)

    for p in scan.md_allow:
        pass  # compliant — silent
    for p in scan.md_warn:
        print(f"    WARN: {p.name} present but NOT git-ignored")
    for p in scan.md_transient:
        print(f"    FAIL (.md not allow-listed): {p.name}")
    for p in scan.blocked_ext:
        print(f"    FAIL (scratch extension {p.suffix}): {p.name}")
    for p in scan.no_ext_blocked:
        print(f"    FAIL (no-extension blob): {p.name}")
    for p in scan.blocked_dirs:
        print(f"    FAIL (scratch directory): {p.name}/")
    for p in scan.dotfile_unmanaged:
        kind = "dir" if p.is_dir() else "file"
        print(
            f"    FAIL (unmanaged dotfile/dotdir, not allow-listed/gitignored/tracked): {p.name} ({kind})"
        )

    total_files = len(
        [p for p in REPO_ROOT.iterdir() if p.is_file() and not _is_dotfile(p.name)]
    )
    total_dotfiles = len([p for p in REPO_ROOT.iterdir() if _is_dotfile(p.name)])
    print()
    print(
        f"Scanned {total_files} non-dotfile(s) + {total_dotfiles} dotfile(s) at repo root: "
        f"{len(scan.md_allow)} allow-listed, {len(scan.md_warn)} warned, "
        f"{len(scan.violations)} violation(s)."
    )

    if scan.md_warn:
        print()
        print(
            f"WARN: {len(scan.md_warn)} warn-listed `.md` file(s) present and "
            f"NOT git-ignored (CLAUDE.md is auto-generated per AGENTS.md and "
            f"should be added to `.gitignore`):"
        )
        for p in scan.md_warn:
            print(f"  - {p.name}")

    if not scan.violations:
        print()
        print("PASS: No transient artifacts at repo root.")
        return 0

    print()
    print(f"FAIL: {len(scan.violations)} transient artifact(s) at repo root:")
    for p in scan.violations:
        kind = "dir" if p.is_dir() else "file"
        print(f"  - {p.name} ({kind})")
    _print_remediation()
    return 1


# ---------------------------------------------------------------------------
# Self-test (deterministic; no repo state required)
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Build a mock repo root in a tmpdir and assert the scanner classifies
    every planted file correctly. Returns 0 on success, 2 on failure."""
    print("=== check_root_hygiene.py self-test ===")
    failures: list[str] = []

    def _touch(
        root: Path, name: str, *, is_dir: bool = False, binary: bool = False
    ) -> Path:
        p = root / name
        if is_dir:
            p.mkdir()
        elif binary:
            p.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56)
        else:
            p.write_text("x")
        return p

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # --- legit (must NOT be flagged) ---
        legit = [
            ("README.md", False),  # .md allow-list
            ("ARCHITECTURE.md", False),
            ("LICENSE", False),  # no-ext allow-list
            ("Dockerfile", False),
            ("Cargo.toml", False),  # never-blocked ext
            ("deny.toml", False),
            ("release_gates.yaml", False),
            ("codecov.yml", False),
            ("Cargo.lock", False),
            ("uv.lock", False),
            ("fluxion.pyi", False),
            ("bem-engineer.skill", False),
            ("tests_tmp_dummy.onnx", False),
            ("requirements-dev.txt", False),  # blocked-ext allow-list
            ("requirements.txt", False),
            (".gitignore", False),  # dotfile allow-list (#2954)
            (".rustfmt.toml", False),
            (".env.example", False),
            (".git", True),  # dotdir allow-list (#2954)
            (".cargo", True),
            (".githooks", True),
            (".github", True),
            (".planning", True),
            (".agents", True),
            ("src", True),  # legit dir (not in denylist)
            ("docs", True),
            ("crates", True),
        ]
        for name, is_dir in legit:
            _touch(root, name, is_dir=is_dir)

        # --- violations (MUST be flagged) ---
        expected_violations: dict[str, str] = {
            "scratch.md": "md_transient",
            "CASE_600.md": "md_transient",
            "dump.txt": "blocked_ext",
            "out.csv": "blocked_ext",
            "orphan.rs": "blocked_ext",
            "helper.py": "blocked_ext",
            "run.sh": "blocked_ext",
            "data.json": "blocked_ext",
            "logs.zip": "blocked_ext",
            "test_cases": "no_ext_blocked",  # no-extension binary blob
            "pull_request": "no_ext_blocked",
            "fixes": "blocked_dirs",
            "results": "blocked_dirs",
            "reports": "blocked_dirs",
            "bem-engineer-workspace": "blocked_dirs",
            "bem-engineer": "blocked_dirs",
            ".mytool": "dotfile_unmanaged",  # #2954: unmanaged dotdir
        }
        for name in expected_violations:
            is_dir = expected_violations[name] in {"blocked_dirs", "dotfile_unmanaged"}
            binary = (
                expected_violations[name] == "no_ext_blocked" and name == "test_cases"
            )
            _touch(root, name, is_dir=is_dir, binary=binary)

        # CLAUDE.md warn-list is non-blocking; plant one, gitignored() returns
        # False in the tmpdir so it should land in md_warn (not a violation).
        _touch(root, "CLAUDE.md")

        # Monkey-patch REPO_ROOT + is_gitignored + is_tracked so the tmpdir
        # scan is hermetic. The dotfile cross-check (#2954) consults all
        # three signals, so we have to neutralize every one of them.
        import check_root_hygiene as mod

        orig_root = mod.REPO_ROOT
        mod.REPO_ROOT = root
        mod.is_gitignored = lambda p: False  # type: ignore[assignment]
        mod.is_tracked = lambda p: False  # type: ignore[assignment]
        try:
            scan = mod.scan_root(root)
        finally:
            mod.REPO_ROOT = orig_root

        # Assert every legit file is absent from violations.
        violation_names = {p.name for p in scan.violations}
        for name, _ in legit:
            if name in violation_names:
                failures.append(f"legit '{name}' was flagged as a violation")

        # Assert every expected violation is present and in the right bucket.
        for name, bucket in expected_violations.items():
            bucket_names = {p.name for p in getattr(scan, bucket)}
            if name not in bucket_names:
                failures.append(
                    f"expected '{name}' in scan.{bucket}, got {sorted(bucket_names)}"
                )

        # Assert CLAUDE.md went to warn (non-blocking), not violations.
        if "CLAUDE.md" not in {p.name for p in scan.md_warn}:
            failures.append("CLAUDE.md should be in md_warn (non-blocking)")

        # Assert no false positives among never-blocked extensions.
        if scan.violations and any(
            p.name in {n for n, _ in legit} for p in scan.violations
        ):
            failures.append("a legit never-blocked file was flagged")

    if failures:
        print("FAIL: self-test assertions did not hold:")
        for f in failures:
            print(f"  - {f}")
        return 2

    print(
        f"PASS: {len(legit)} legit items unflagged, "
        f"{len(expected_violations)} violations caught, CLAUDE.md warned."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
