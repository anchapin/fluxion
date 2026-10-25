"""Module entry point — ``python -m fluxion <command>``.

Mirrors the ``fluxion`` console script declared in ``pyproject.toml``
(``[project.scripts] fluxion = "fluxion.cli:main"``) so the CLI is runnable
straight from a source checkout, without installing the package first::

    python -m fluxion apply-measures --model base.json --measures measures/
"""

from __future__ import annotations

import sys

from fluxion.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
