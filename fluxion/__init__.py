"""Fluxion — Neuro-Symbolic Building Energy Modeling engine.

The Rust core lives in ``src/lib.rs`` and is exposed to Python via PyO3.
This package re-exports the native classes at the top level so callers can
write ``fluxion.Model(...)`` directly.

Python-side additions (Issue #1814) live in submodules:

- :mod:`fluxion.measures` — :class:`FluxionMeasure` base class + AOT
  runner. See ``docs/measures.md`` for the design rationale.
- :mod:`fluxion.cli` — ``fluxion apply-measures`` entrypoint.

Importing the Rust extension is wrapped in a try/except so that ``import
fluxion`` works even on a slim install (e.g. documentation builds) where
the ``.abi3.so`` is unavailable.

Solver selection (Issue #3282)
------------------------------

``MultiZoneThermalModel.from_case_spec`` accepts optional ``zone_solver``
(``"gauge"`` default | ``"5r1c"`` | ``"9r4c"``) and ``conduction_solver``
(``"default"`` default | ``"ctf"`` | ``"fd"``) keyword arguments. The
experimental ``"6r2c"`` / ``"8r3c"`` zone solvers raise ``ValueError``
unless the ``FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1`` environment variable
is set — and stay unavailable even then until the experimental cargo
feature ships (issue #3291).
"""

from __future__ import annotations

try:
    from fluxion.fluxion import *  # noqa: F401,F403  (re-export native classes)
    from fluxion.fluxion import __doc__ as _native_doc  # noqa: F401
except ImportError as _exc:  # pragma: no cover — slim install path
    _NATIVE_IMPORT_ERROR = _exc
else:
    _NATIVE_IMPORT_ERROR = None

# Re-export the measures submodule so callers can `from fluxion import
# FluxionMeasure`. This is a soft import — measures can be used without
# the native module (e.g. in unit tests that mock fluxion.Model).
from fluxion.measures import (  # noqa: E402
    FluxionMeasure,
    apply_measures,
    digest_of_json_payload,
    discover_measures,
    dict_to_model,
    load_model,
    make_applied_delta,
    model_to_dict,
    save_model,
)

__all__ = [
    "FluxionMeasure",
    "apply_measures",
    "digest_of_json_payload",
    "discover_measures",
    "dict_to_model",
    "load_model",
    "make_applied_delta",
    "model_to_dict",
    "save_model",
]


def __getattr__(name: str):
    """Lazy attribute access for native classes.

    ``fluxion.Model`` etc. live in the compiled extension module. Importing
    the extension at module load time would force every Python process (even
    those only using :class:`FluxionMeasure` for static analysis) to load
    the Rust shared library. We delegate attribute access so the extension is
    loaded only when actually needed.
    """
    if _NATIVE_IMPORT_ERROR is not None:
        raise ImportError(
            f"fluxion native extension is unavailable ({_NATIVE_IMPORT_ERROR}). "
            "Install with `maturin develop` or `pip install fluxion`."
        ) from _NATIVE_IMPORT_ERROR
    from fluxion import fluxion as _native

    try:
        return getattr(_native, name)
    except AttributeError as e:
        raise AttributeError(f"module 'fluxion' has no attribute {name!r}") from e


# Static type stubs (PEP 561): see ``fluxion.pyi`` at the repo root for the
# generated typing surface.
