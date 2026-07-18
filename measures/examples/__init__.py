"""measures.examples — example Fluxion measures (Issue #1814).

Each module in this subdirectory defines one or more concrete
:class:`fluxion.FluxionMeasure` subclasses. They are discovered by
``fluxion apply-measures --measures measures/examples/`` and applied to a
loaded base model.

This ``__init__.py`` is intentionally empty — measure discovery uses
``importlib.util`` to load each ``*.py`` file directly, so module-level
imports of sibling measures would be redundant (and would break the
"discover individual files" path used by the CLI).
"""
