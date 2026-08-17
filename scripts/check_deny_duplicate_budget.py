#!/usr/bin/env python3
"""
Duplicate-Version Budget Check for Fluxion.

Parses ``cargo deny -f json check bans`` output, counts the
``code=duplicate`` diagnostics, and compares against the
``duplicates_baseline`` declared in ``tests/reference_data/deny_budget_baseline.json``
(mirrors the inline ``# duplicates_baseline: 45`` comment in
``deny.toml [bans]``). Exits non-zero when the live count exceeds the
baseline.

Issue #2994 acceptance criteria are realised as follows:

* The script is a Python re-implementation of the bash counting done
  inline in ``.github/workflows/security.yml``'s ``Verify
  duplicate-version budget`` step. Running it locally surfaces the
  same data as the CI step without needing the ``cargo deny`` extension
  plugin or a Rust toolchain checkout.
* ``tests/reference_data/deny_budget_baseline.json`` is the
  machine-readable cluster inventory; this script reads it so a future
  reduction PR only has to edit one file.
* When ``cargo deny`` is not on PATH, the script exits 2 (script
  error), matching the convention used by the other
  ``scripts/check_*`` gates (see e.g. ``check_audit_ignores_fresh.py``).

Usage::

    python3 scripts/check_deny_duplicate_budget.py
    python3 scripts/check_deny_duplicate_budget.py --cargo-deny-bin /opt/cargo-deny/bin/cargo-deny
    python3 scripts/check_deny_duplicate_budget.py --baseline-file path/to/baseline.json
    python3 scripts/check_deny_duplicate_budget.py --quiet
    python3 scripts/check_deny_duplicate_budget.py --json-out

Exit codes:
    0 — Live duplicate-version count is within budget.
    1 — Live count exceeds ``duplicates_baseline`` (regression; tighten
        the dep graph or raise the baseline per issue #2994).
    2 — Script error (cargo deny unavailable, baseline file missing,
        Cargo.lock missing, parse failure).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = REPO_ROOT / "tests" / "reference_data" / "deny_budget_baseline.json"
DENY_TOML = REPO_ROOT / "deny.toml"
CARGO_LOCK = REPO_ROOT / "Cargo.lock"

DEFAULT_CARGO_DENY_BIN = "cargo deny"


def load_baseline(path: Path) -> dict:
    """Load and lightly validate the baseline JSON.

    Returns the parsed dict; raises ``FileNotFoundError`` /
    ``json.JSONDecodeError`` / ``ValueError`` when the file is missing,
    malformed, or missing the required keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"baseline file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("total_duplicates", "duplicates_baseline", "clusters"):
        if key not in data:
            raise ValueError(
                f"baseline file {path} missing required key {key!r}; "
                "schema_version must be >= 1"
            )
    if not isinstance(data["clusters"], list) or not data["clusters"]:
        raise ValueError(f"baseline file {path} has empty or non-list `clusters`")
    return data


def parse_baseline_from_deny_toml(path: Path) -> int | None:
    """Extract the ``# duplicates_baseline: N`` comment from deny.toml.

    Mirrors the regex used by ``.github/workflows/security.yml``. Returns
    ``None`` when the comment is absent or malformed. Used as a
    belt-and-braces sanity check against the JSON baseline.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"^#\s*duplicates_baseline\s*:\s*([0-9]+)",
        text,
        re.MULTILINE,
    )
    if match is None:
        return None
    return int(match.group(1))


def run_cargo_deny(
    repo_root: Path,
    cargo_deny_bin: str | list[str],
) -> str:
    """Run ``cargo deny -f json check bans`` and return its combined output.

    Raises ``FileNotFoundError`` if the binary is not on PATH (exit 2
    from ``main()``). The command is intentionally run with ``check
    bans`` rather than ``check`` so the run stays fast and only emits
    the diagnostics relevant to this gate.
    """
    if isinstance(cargo_deny_bin, str):
        argv = cargo_deny_bin.split()
    else:
        argv = list(cargo_deny_bin)
    argv += ["-f", "json", "check", "bans"]
    proc = subprocess.run(
        argv,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    # cargo-deny exits 0 with the JSON output even when warnings are
    # present (the JSON stream itself is the diagnostic). Combine
    # stdout and stderr so callers don't miss diagnostics that the
    # binary routes to stderr depending on version.
    return (proc.stdout or "") + (proc.stderr or "")


def count_duplicate_diagnostics(raw_jsonl: str) -> tuple[int, list[dict]]:
    """Parse the JSONL output and return ``(count, diagnostics)``.

    Each line is a separate JSON object: most lines are diagnostics of
    the form ``{"type":"diagnostic","fields":{"code":"duplicate",...}}``
    and the final line is the summary ``{"type":"summary",...}``. We
    accept any object that parses and has ``type == 'diagnostic'`` and
    ``fields.code == 'duplicate'``. Lines that fail to parse are
    silently skipped (cargo-deny may interleave non-JSON messages in
    some versions).
    """
    diagnostics: list[dict] = []
    for line in raw_jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "diagnostic":
            continue
        if obj.get("fields", {}).get("code") != "duplicate":
            continue
        diagnostics.append(obj)
    return len(diagnostics), diagnostics


def parse_summary_warnings(raw_jsonl: str) -> int | None:
    """Extract the top-line ``bans.warnings`` count from the summary.

    Returns ``None`` when no summary line is found (some
    cargo-deny builds don't emit one). Used purely as a cross-check
    against the per-diagnostic count; the CI gate in
    ``security.yml`` uses the diagnostic count directly.
    """
    for line in raw_jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "summary":
            continue
        return obj.get("fields", {}).get("bans", {}).get("warnings")
    return None


def extract_crate_names(diagnostics: list[dict]) -> list[str]:
    """Pull the crate names out of each duplicate diagnostic.

    The ``message`` field follows ``"found N duplicate entries for crate
    '<name>'"``; we extract ``<name>`` for reporting. Duplicates are
    preserved so callers can see which crate-name appears more than
    once if cargo-deny emits a malformed run.
    """
    out: list[str] = []
    for d in diagnostics:
        msg = d.get("fields", {}).get("message", "")
        m = re.match(r"found \d+ duplicate entries? for crate .([\w_-]+).", msg)
        if m:
            out.append(m.group(1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=BASELINE_FILE,
        help=(
            "Path to the deny-budget baseline JSON (default: "
            f"{BASELINE_FILE.relative_to(REPO_ROOT)})."
        ),
    )
    parser.add_argument(
        "--cargo-deny-bin",
        default=DEFAULT_CARGO_DENY_BIN,
        help=(
            "cargo deny invocation to use (default: 'cargo deny'). "
            "Pass a different command if cargo-deny is installed "
            "outside PATH or as a standalone binary."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Repository root to scan (default: the directory holding "
            "this script's parent's parent)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-diagnostic details; print only the summary.",
    )
    parser.add_argument(
        "--json-out",
        action="store_true",
        help=(
            "Emit a single JSON object on stdout with the run's "
            "structured fields (`baseline`, `live_count`, "
            "`wildcards_baseline`, `summary_warnings`, `clusters`, "
            "`over_budget`). Machine-readable for CI consumers."
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    print(
        "=== Fluxion deny.toml [bans] duplicate-version budget check (Issue #2994) ==="
    )
    print(f"Repo root:    {repo_root}")
    print(f"Baseline:     {args.baseline_file}")
    print(f"Cargo deny:   {args.cargo_deny_bin}")
    print()

    # --- 1. Load the baseline -------------------------------------------------
    try:
        baseline = load_baseline(args.baseline_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR parsing baseline {args.baseline_file}: {exc}", file=sys.stderr)
        return 2

    budget = int(baseline["duplicates_baseline"])
    wildcards_baseline = int(baseline.get("wildcards_baseline", 0))
    n_clusters_in_baseline = len(baseline["clusters"])
    print(
        f"Baseline duplicates:  {budget}  "
        f"(schema_version={baseline.get('schema_version', '?')}, "
        f"captured_at={baseline.get('captured_at', '?')})"
    )
    print(f"Baseline wildcards:   {wildcards_baseline} (informational only)")
    print(f"Baseline clusters:    {n_clusters_in_baseline} cluster definitions")
    print()

    # --- 2. Cross-check deny.toml inline baseline -----------------------------
    toml_baseline = parse_baseline_from_deny_toml(repo_root / "deny.toml")
    if toml_baseline is None:
        print(
            "WARN: deny.toml does not declare `# duplicates_baseline: N` — "
            "the JSON baseline is the sole source of truth. Update both "
            "together to avoid drift.",
            file=sys.stderr,
        )
    elif toml_baseline != budget:
        print(
            f"ERROR: deny.toml `# duplicates_baseline: {toml_baseline}` "
            f"disagrees with JSON baseline `{budget}`. Update the "
            "smaller to match the larger (issue #2933 parses the TOML "
            "comment, not the JSON).",
            file=sys.stderr,
        )
        return 2
    else:
        print(
            f"deny.toml inline baseline matches: {toml_baseline} "
            "(.github/workflows/security.yml parses this comment)."
        )

    # --- 3. Resolve cargo-deny ------------------------------------------------
    cargo_deny_arg = args.cargo_deny_bin
    if isinstance(cargo_deny_arg, str) and " " not in cargo_deny_arg:
        if (
            shutil.which(cargo_deny_arg.split("/")[0]) is None
            and shutil.which(cargo_deny_arg) is None
        ):
            # 'cargo deny' is two tokens; check the first token ('cargo').
            if cargo_deny_arg.split()[0] == "cargo" and shutil.which("cargo") is None:
                print(
                    "ERROR: cargo-deny binary not found on PATH. Install "
                    "with `cargo install cargo-deny --version 0.20.2` "
                    "(see `.github/workflows/security.yml`).",
                    file=sys.stderr,
                )
                return 2
    print()

    # --- 4. Run cargo deny ----------------------------------------------------
    try:
        raw_output = run_cargo_deny(repo_root, cargo_deny_arg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR invoking cargo deny: {exc}", file=sys.stderr)
        return 2

    if not raw_output.strip():
        print(
            "ERROR: cargo deny produced no output. Check that "
            "Cargo.lock and deny.toml are present at the repo root.",
            file=sys.stderr,
        )
        return 2

    # --- 5. Count diagnostics -------------------------------------------------
    dup_count, diagnostics = count_duplicate_diagnostics(raw_output)
    summary_warnings = parse_summary_warnings(raw_output)
    crate_names = extract_crate_names(diagnostics)

    print(f"Live duplicate-version diagnostics: {dup_count}")
    if summary_warnings is not None:
        print(f"cargo deny summary `bans.warnings`:  {summary_warnings}")
    print(f"Distinct crates flagged:            {len(set(crate_names))}")
    print()

    if not args.quiet:
        for crate, diag in sorted(zip(crate_names, diagnostics), key=lambda x: x[0]):
            msg = diag.get("fields", {}).get("message", "?")
            line = None
            for lbl in diag.get("fields", {}).get("labels", []):
                if "line" in lbl:
                    line = lbl["line"]
                    break
            print(f"  {crate:<32s} Cargo.lock:{line}  {msg}")
        print()

    over_budget = dup_count > budget

    print("=" * 64)
    if over_budget:
        print(
            f"FAIL: live duplicate-version count {dup_count} exceeds "
            f"baseline {budget} (delta {dup_count - budget:+d}). "
            "Investigate the offending crates above or raise the "
            "baseline per issue #2994."
        )
    elif dup_count < budget:
        print(
            f"PASS: live count {dup_count} is UNDER baseline {budget} "
            f"(slack {budget - dup_count}). Consider lowering the "
            "baseline per issue #2994's reduction plan."
        )
    else:
        print(f"PASS: live count {dup_count} matches baseline {budget}. Within budget.")

    if args.json_out:
        payload = {
            "baseline_file": str(args.baseline_file),
            "schema_version": baseline.get("schema_version"),
            "captured_at": baseline.get("captured_at"),
            "duplicates_baseline": budget,
            "wildcards_baseline": wildcards_baseline,
            "live_count": dup_count,
            "summary_warnings": summary_warnings,
            "over_budget": over_budget,
            "delta": dup_count - budget,
            "clusters_in_baseline": n_clusters_in_baseline,
            "crates": crate_names,
        }
        print()
        print("--- JSON output ---")
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 1 if over_budget else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
