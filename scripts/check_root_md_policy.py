#!/usr/bin/env python3
"""
Backward-compat alias for ``check_root_hygiene.py``.

The root-hygiene gate was widened beyond ``.md``-only (see git history and
``AGENTS.md`` §Repository Hygiene). The real logic now lives in
``scripts/check_root_hygiene.py``. This shim keeps the historical filename
alive so existing references — ``.github/workflows/docs-hygiene.yml`` and
``.pre-commit-config.yaml`` — keep working without churn. All argv is
forwarded to the new script; the exit code is propagated.

Usage and exit codes are identical to ``check_root_hygiene.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_NEW_SCRIPT = Path(__file__).resolve().parent / "check_root_hygiene.py"


def main() -> int:
    result = subprocess.run(
        [sys.executable, str(_NEW_SCRIPT), *sys.argv[1:]],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # pragma: no cover - delegate surfaces real errors
        print(f"ERROR: alias failed to delegate: {e}", file=sys.stderr)
        sys.exit(2)
