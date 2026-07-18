"""
FluxionMeasure base class + Ahead-of-Time (AOT) measure runner (Issue #1814).

This module is the Python analogue of OpenStudio's `OpenStudio::Measure::ModelMeasure`,
adapted to Fluxion's Neuro-Symbolic architecture. Crucially, **measures are
pre-processing steps** — they mutate a model once and serialize it for the Rust
runtime to consume. They MUST NOT run inside the timestepping loop.

Why "AOT only"?
---------------
Rust's `rayon` parallel iterator drives the timestepping loop with a thread
pool. If a Python measure were to run inside that loop, the GIL would be
contended against every rayon worker, serializing all parallel work and
cancelling the speedup we get from `par_iter`. To preserve Fluxion's
high-throughput BatchOracle path (>=10k configs/sec on 8 cores), measures are
*forbidden* from running inside any per-timestep callback.

The runtime guard below emits a `RuntimeWarning` if a measure detects it's
running on a thread whose name matches the rayon worker naming pattern, or if
the env var `FLUXION_INSIDE_TIMESTEPPING=1` is set. The warning is loud
(emitted every apply) but does not raise — measures are sometimes invoked
deliberately from worker contexts (e.g. parallel calibration); the warning is
informational and CI-friendly.

Module layout
-------------
- :class:`FluxionMeasure` — abstract base class. Subclasses override
  :meth:`apply` and (optionally) :meth:`arguments`.
- :func:`load_model_json` / :func:`save_model_json` — minimal AOT
  serialization helpers. The full model schema is still being stabilized
  in ARCHITECTURE.md; for now we round-trip via the snapshot API.
- :func:`discover_measures` — walk a directory and import every
  :class:`FluxionMeasure` subclass declared in any ``*.py`` file.
- :func:`apply_measures` — load a base model, run each measure in sequence,
  serialize the result.

See :mod:`fluxion.cli` for the ``fluxion apply-measures`` entrypoint.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import pathlib
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator


# =============================================================================
# Runtime context detection
# =============================================================================


# Names that `rayon` and most Rust thread-pool crates use by default. The list
# is intentionally conservative — false positives are OK (just a warning),
# false negatives are bad (would silently break the rayon throughput gate).
_RAYON_THREAD_NAME_PATTERNS: tuple[str, ...] = (
    "rayon",
    "rayon-",
    "tokio-runtime-worker",
    "tokio-",
    "tokio_reactor",
)


def _inside_timestepping() -> bool:
    """Return True if the current thread looks like a Rust worker thread.

    The check is heuristic: it matches the current thread name against the
    common rayon/tokio patterns. We also honour an explicit escape hatch via
    the ``FLUXION_INSIDE_TIMESTESTEPPING`` env var, which downstream tooling
    can set when invoking a Python callback from inside a simulation loop.
    """
    if os.environ.get("FLUXION_INSIDE_TIMESTEPPING") == "1":
        return True
    try:
        name = threading.current_thread().name
    except Exception:  # pragma: no cover — defensive
        return False
    lowered = name.lower()
    return any(pattern in lowered for pattern in _RAYON_THREAD_NAME_PATTERNS)


def _warn_if_inside_timestepping(measure_name: str) -> None:
    """Emit a runtime warning if the measure is running on a worker thread."""
    if _inside_timestepping():
        warnings.warn(
            (
                f"FluxionMeasure '{measure_name}' is running on a thread that "
                "looks like a Rust rayon/tokio worker. Measures are AOT "
                "pre-processors and MUST NOT run inside the timestepping loop — "
                "doing so will serialize the parallel BatchOracle path. "
                "Move this logic to `fluxion apply-measures` (run once before "
                "simulation) or to a Rust trait implementation."
            ),
            RuntimeWarning,
            stacklevel=3,
        )


# =============================================================================
# FluxionMeasure base class
# =============================================================================


class _FluxionMeasureMeta(type):
    """Metaclass that wraps :meth:`apply` with the timestepping guard.

    Every call to a subclass's ``apply(...)`` first invokes
    :func:`_warn_if_inside_timestepping` so the warning fires regardless of
    whether the caller uses :func:`apply_measures` or calls ``apply``
    directly. This is the only piece of behaviour enforced automatically by
    the base class; everything else is opt-in via :meth:`arguments` /
    :meth:`parse_arguments`.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        # Skip the wrap for the base class itself — its apply() raises
        # NotImplementedError, and we want the warning *only* to fire on
        # real subclasses (which override apply).
        if bases == ():
            return cls
        original_apply = namespace.get("apply")
        if original_apply is None:
            return cls

        def guarded_apply(self, model, arguments):
            _warn_if_inside_timestepping(name)
            return original_apply(self, model, arguments)

        # Preserve the original signature metadata where possible.
        try:
            guarded_apply.__wrapped__ = original_apply  # type: ignore[attr-defined]
            guarded_apply.__name__ = original_apply.__name__
            guarded_apply.__qualname__ = original_apply.__qualname__
            guarded_apply.__doc__ = original_apply.__doc__
        except (AttributeError, TypeError):
            pass
        # Bind onto the subclass.
        cls.apply = guarded_apply
        return cls


class FluxionMeasure(metaclass=_FluxionMeasureMeta):
    """Abstract base class for Fluxion AOT measures.

    A measure mutates a building model **once** before the Rust runtime
    consumes it. The class mirrors OpenStudio's ``ModelMeasure`` API:

    - :meth:`arguments` returns an argument spec (list of dicts). The format
      is deliberately permissive — Fluxion is still stabilizing its CLI
      argument vocabulary, and the exact keys accepted will evolve alongside
      the model schema in ``ARCHITECTURE.md``.
    - :meth:`apply` mutates the supplied ``model`` in place using the parsed
      ``arguments`` dict. The model is the PyO3 ``fluxion.Model`` instance.

    Subclassing rules
    -----------------
    - Concrete measures MUST override :meth:`apply`.
    - :meth:`arguments` MAY be overridden; the default returns an empty list,
      which means "no user-tunable parameters". The CLI passes an empty
      dict for the args when the measure exposes no arguments.
    - Subclasses are discovered by name — see :func:`discover_measures`.
      Every non-abstract subclass with a ``__name__ != "FluxionMeasure"``
      will be picked up.

    Threading contract
    ------------------
    ``apply`` is invoked by :func:`apply_measures` from the main thread, on
    a freshly-loaded or freshly-mutated ``Model``. The metaclass
    :class:`_FluxionMeasureMeta` automatically wraps every subclass
    ``apply`` with :func:`_warn_if_inside_timestepping` — the warning
    fires whether the user calls :func:`apply_measures` or invokes
    ``measure.apply(model, args)`` directly.
    """

    #: Class-level metadata. Subclasses may override.
    name: str = ""
    description: str = ""

    def __init__(self) -> None:
        if not self.name:
            # Default to the class name, stripped of "Measure" suffix for
            # nicer CLI output.
            self.name = type(self).__name__
            if self.name.endswith("Measure") and self.name != "FluxionMeasure":
                self.name = self.name[: -len("Measure")]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def arguments(self) -> list[dict[str, Any]]:
        """Return the argument specification for this measure.

        Mirrors OpenStudio's ``arguments()`` method, which returns a list of
        argument descriptors. Each entry is a dict with at minimum:

        - ``name`` (str) — the argument key, used as the dict key in
          :meth:`apply`.
        - ``type`` (str) — one of ``"string"``, ``"double"``, ``"integer"``,
          ``"bool"``, ``"choice"``.
        - ``default`` (optional) — default value if the user omits it.

        Optional keys: ``"description"``, ``"required"`` (bool),
        ``"choices"`` (list, for ``type == "choice"``), ``"min"`` / ``"max"``
        for numeric types.

        The default implementation returns an empty list, meaning "no
        arguments". Concrete measures with user-tunable parameters should
        override.
        """
        return []

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        """Mutate ``model`` in place using ``arguments``.

        Subclasses MUST override. The base implementation raises
        ``NotImplementedError``.

        Parameters
        ----------
        model : fluxion.Model
            The PyO3 model instance to mutate. The measure may freely call
            :meth:`fluxion.Model.set_surfaces`, ``set_hvac_system``, etc.
            Note that the model uses a snapshot/owned-value pattern: surface
            lists must be re-applied via :meth:`set_surfaces` to persist.
        arguments : dict
            Parsed user arguments. Keys come from :meth:`arguments`. Missing
            keys fall back to the declared defaults.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.apply() must be overridden by the subclass"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def parse_arguments(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        """Merge user-supplied args with the declared spec defaults.

        Unknown keys are passed through (with a warning) so that measures
        which declare a permissive spec still accept arbitrary JSON.
        """
        spec = self.arguments()
        declared: dict[str, Any] = {}
        for entry in spec:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            declared[entry["name"]] = entry.get("default", _SENTINEL)
        merged: dict[str, Any] = {k: v for k, v in declared.items() if v is not _SENTINEL}
        if raw:
            for k, v in raw.items():
                if k not in declared and declared:
                    warnings.warn(
                        f"FluxionMeasure '{self.name}' received unknown argument "
                        f"'{k}' (not declared in arguments())",
                        UserWarning,
                        stacklevel=2,
                    )
                merged[k] = v
        return merged


_SENTINEL: Any = object()


# =============================================================================
# Model serialization (AOT file format)
# =============================================================================


# Bump when the on-disk format changes in an incompatible way.
SCHEMA_VERSION = "1.0.0"


def model_to_dict(model: Any) -> dict[str, Any]:
    """Serialize a ``fluxion.Model`` to a JSON-compatible dict.

    Uses the snapshot API (``model.zones()``, ``model.surfaces()``,
    ``model.hvac_system()``) so the output is consistent with the
    PyO3 ownership contract described in ``docs/bindings.md``.

    The schema is intentionally minimal and versioned — see
    :data:`SCHEMA_VERSION`. The Rust runtime today does not yet consume this
    format directly; the serialized file is meant for round-tripping through
    :func:`dict_to_model` and for CI smoke tests that verify a measure chain
    is idempotent.
    """
    zones = [
        {
            "index": z.index,
            "temperature": z.temperature,
            "area": z.area,
            "heating_setpoint": z.heating_setpoint,
            "cooling_setpoint": z.cooling_setpoint,
            "hvac_enabled": z.hvac_enabled,
            "surfaces": [_surface_to_dict(s) for s in z.surfaces],
        }
        for z in model.zones()
    ]
    flat_surfaces = [_surface_to_dict(s) for s in model.surfaces()]
    hvac = model.hvac_system()
    return {
        "schema_version": SCHEMA_VERSION,
        "num_zones": model.num_zones(),
        "temperatures": list(model.get_temperatures()),
        "zones": zones,
        "surfaces": flat_surfaces,
        "hvac_system": {
            "heating_capacity": hvac.heating_capacity,
            "cooling_capacity": hvac.cooling_capacity,
            "cop_heating": hvac.cop_heating,
            "cop_cooling": hvac.cop_cooling,
            "stages": hvac.stages,
            "min_outdoor_temp": hvac.min_outdoor_temp,
            "max_outdoor_temp": hvac.max_outdoor_temp,
            "vav_enabled": hvac.vav_enabled,
            "economizer_enabled": hvac.economizer_enabled,
            "supply_air_temp": hvac.supply_air_temp,
        },
    }


def dict_to_model(payload: dict[str, Any]) -> Any:
    """Reconstruct a ``fluxion.Model`` from a :func:`model_to_dict` payload."""
    import fluxion  # local import — measures may run without fluxion at import time

    model = fluxion.Model(num_zones=int(payload.get("num_zones", 1)))
    if "temperatures" in payload:
        model.set_temperatures([float(t) for t in payload["temperatures"]])
    if "surfaces" in payload:
        flat = [_dict_to_surface(s) for s in payload["surfaces"]]
        model.set_surfaces(flat)
    if "hvac_system" in payload:
        model.set_hvac_system(_dict_to_hvac(payload["hvac_system"]))
    return model


def _enum_name(value: Any) -> str:
    """Extract the variant name from a PyO3 enum (which lacks ``.name``)."""
    rep = repr(value)
    # repr() is e.g. ``Orientation.South``; split on the last dot.
    return rep.rsplit(".", 1)[-1]


def _surface_to_dict(s: Any) -> dict[str, Any]:
    # The PyO3 PyShadingDevice type does not expose its fields as Python
    # attributes (the Rust struct has no PyMethods for shading_type /
    # overhang_depth / fin_width / mounting_height). Instead, reconstruct
    # the shading list from the shorthand overhang/fin fields on PySurface.
    shading_devices: list[dict[str, Any]] = []
    if s.overhang_depth is not None and s.overhang_height is not None:
        stype = "Fins" if s.fin_width else "Overhang"
        if s.overhang_depth is not None and s.fin_width:
            stype = "OverhangAndFins"
        shading_devices.append(
            {
                "shading_type": stype,
                "overhang_depth": s.overhang_depth,
                "fin_width": s.fin_width if s.fin_width is not None else 0.0,
                "mounting_height": s.overhang_height,
            }
        )
    elif s.overhang_depth is not None:
        shading_devices.append(
            {
                "shading_type": "Overhang",
                "overhang_depth": s.overhang_depth,
                "fin_width": 0.0,
                "mounting_height": s.overhang_height or 0.0,
            }
        )
    elif s.fin_width is not None:
        shading_devices.append(
            {
                "shading_type": "Fins",
                "overhang_depth": 0.0,
                "fin_width": s.fin_width,
                "mounting_height": 0.0,
            }
        )

    return {
        "area": s.area,
        "u_value": s.u_value,
        "orientation": _enum_name(s.orientation),
        "window_area": s.window_area,
        "overhang_depth": s.overhang_depth,
        "overhang_height": s.overhang_height,
        "fin_width": s.fin_width,
        "shading_devices": shading_devices,
    }


def _dict_to_surface(d: dict[str, Any]) -> Any:
    import fluxion

    orient_name = d.get("orientation", "South")
    orientation = getattr(fluxion.Orientation, orient_name, fluxion.Orientation.South)
    s = fluxion.Surface(
        area=float(d.get("area", 0.0)),
        u_value=float(d.get("u_value", 0.0)),
        orientation=orientation,
        window_area=float(d.get("window_area", 0.0)),
    )
    s.overhang_depth = d.get("overhang_depth")
    s.overhang_height = d.get("overhang_height")
    s.fin_width = d.get("fin_width")
    # ShadingDevice has no Python __init__ — use the static factory methods
    # matching each shading type. append_shading() still accepts the
    # constructed device.
    for dev in d.get("shading_devices", []):
        st_name = dev.get("shading_type", "None")
        if st_name == "Overhang":
            device = fluxion.ShadingDevice.overhang(
                depth=float(dev.get("overhang_depth", 0.0)),
                height=float(dev.get("mounting_height", 0.0)),
            )
        elif st_name == "Fins":
            device = fluxion.ShadingDevice.fins(width=float(dev.get("fin_width", 0.0)))
        elif st_name == "OverhangAndFins":
            device = fluxion.ShadingDevice.overhang_and_fins(
                overhang_depth=float(dev.get("overhang_depth", 0.0)),
                fin_width=float(dev.get("fin_width", 0.0)),
                height=float(dev.get("mounting_height", 0.0)),
            )
        else:
            # "None" or unknown — skip; the shorthand fields already capture
            # any overhang/fin depth if present.
            continue
        s.append_shading(device)
    return s


def _dict_to_hvac(d: dict[str, Any]) -> Any:
    import fluxion

    return fluxion.HVACSystem(
        heating_capacity=float(d.get("heating_capacity", 10000.0)),
        cooling_capacity=float(d.get("cooling_capacity", 8000.0)),
        cop_heating=float(d.get("cop_heating", 3.0)),
        cop_cooling=float(d.get("cop_cooling", 3.2)),
        stages=int(d.get("stages", 1)),
        min_outdoor_temp=float(d.get("min_outdoor_temp", -10.0)),
        max_outdoor_temp=float(d.get("max_outdoor_temp", 40.0)),
        vav_enabled=bool(d.get("vav_enabled", False)),
        economizer_enabled=bool(d.get("economizer_enabled", False)),
        supply_air_temp=float(d.get("supply_air_temp", 13.0)),
    )


def save_model(model: Any, path: str | os.PathLike[str]) -> None:
    """Write a model to disk as JSON (or msgpack if ``msgpack`` is installed).

    The output format is selected by file extension: ``.json`` -> JSON,
    ``.msgpack`` -> msgpack (if available; falls back to JSON if msgpack is
    not importable). All other extensions default to JSON for safety.
    """
    payload = model_to_dict(model)
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".msgpack":
        try:
            import msgpack  # type: ignore[import-not-found]

            target.write_bytes(msgpack.packb(payload, use_bin_type=True))
            return
        except ImportError:
            warnings.warn(
                "msgpack not installed — falling back to JSON for output",
                RuntimeWarning,
                stacklevel=2,
            )
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_model(path: str | os.PathLike[str]) -> Any:
    """Load a model previously written by :func:`save_model`."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".msgpack":
        try:
            import msgpack  # type: ignore[import-not-found]

            payload = msgpack.unpackb(source.read_bytes(), raw=False)
        except ImportError as e:
            raise RuntimeError(
                "Cannot load .msgpack model: msgpack package not installed"
            ) from e
    else:
        payload = json.loads(source.read_text())
    return dict_to_model(payload)


# =============================================================================
# Measure discovery
# =============================================================================


def _iter_measure_classes(module: Any) -> Iterator[type[FluxionMeasure]]:
    """Yield every concrete FluxionMeasure subclass defined in ``module``.

    A class is treated as "abstract" (and therefore skipped) if:

    - ``inspect.isabstract()`` returns ``True`` (catches classes using
      :class:`abc.ABCMeta` + :func:`abc.abstractmethod` directly on
      ``apply``), OR
    - :meth:`FluxionMeasure.apply` has not been overridden *and* the
      subclass explicitly marks itself abstract via :class:`abc.ABCMeta`.
      The :class:`_FluxionMeasureMeta` wraps subclass ``apply`` methods
      for the runtime guard, which obscures the ``@abstractmethod``
      decorator from ``inspect``. To handle that case we look through the
      wrapper chain (``__wrapped__``) for an abstract ``apply``.
    """
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is FluxionMeasure:
            continue
        if not issubclass(obj, FluxionMeasure):
            continue
        if inspect.isabstract(obj):
            continue
        # Fallback: detect abstract through the metaclass wrapper.
        if _apply_is_abstract(obj):
            continue
        if obj.__module__ != module.__name__:
            continue
        yield obj


def _apply_is_abstract(cls: type[FluxionMeasure]) -> bool:
    """Return True if ``cls.apply`` is an ``@abstractmethod`` (after unwrap)."""
    attr = cls.__dict__.get("apply")
    # Walk the __wrapped__ chain installed by the metaclass guard.
    while attr is not None and hasattr(attr, "__wrapped__"):
        attr = attr.__wrapped__
    return bool(
        attr is not None
        and hasattr(attr, "__isabstractmethod__")
        and getattr(attr, "__isabstractmethod__", False)
    )


def discover_measures(measure_dir: str | os.PathLike[str]) -> list[type[FluxionMeasure]]:
    """Discover every concrete :class:`FluxionMeasure` subclass in ``measure_dir``.

    Walks ``measure_dir`` recursively, imports every ``*.py`` file (skipping
    files that start with ``_`` so authors can keep helpers out of the
    discovery scan), and returns the union of concrete subclasses.

    The directory is added to ``sys.path`` for the duration of the import.
    Returns an empty list if the directory does not exist or contains no
    measures.
    """
    base = Path(measure_dir)
    if not base.exists():
        return []
    base_str = str(base.resolve())
    added = False
    if base_str not in sys.path:
        sys.path.insert(0, base_str)
        added = True
    try:
        results: list[type[FluxionMeasure]] = []
        for path in sorted(base.rglob("*.py")):
            if path.name.startswith("_"):
                continue
            mod_name = "_fluxion_measure_" + path.stem + "_" + str(hash(str(path)) & 0xFFFF)
            spec = importlib.util.spec_from_file_location(mod_name, str(path))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                warnings.warn(
                    f"Failed to import measure file {path}: {e}",
                    ImportWarning,
                    stacklevel=2,
                )
                continue
            for cls in _iter_measure_classes(module):
                results.append(cls)
        # Stable order by class name for deterministic CLI output.
        results.sort(key=lambda c: c.__name__)
        return results
    finally:
        if added:
            try:
                sys.path.remove(base_str)
            except ValueError:  # pragma: no cover
                pass


# =============================================================================
# Top-level runner
# =============================================================================


def apply_measures(
    model: Any,
    measures: Iterable[type[FluxionMeasure] | FluxionMeasure],
    measure_args: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Apply each measure in sequence to ``model``, mutating it in place.

    Returns the list of measure names that were applied, in order. Each
    measure is instantiated (if a class was passed) and invoked once.

    Parameters
    ----------
    model : fluxion.Model
        The model to mutate. May be a freshly-constructed ``Model`` or one
        loaded from disk via :func:`load_model`.
    measures : iterable of classes or instances
        Order matters — measures run sequentially. Each entry is either a
        :class:`FluxionMeasure` subclass (instantiated here) or an already-
        constructed instance.
    measure_args : dict, optional
        Mapping of ``measure.name`` -> parsed arguments dict. Measures not
        present in the mapping receive an empty dict (which is then merged
        with declared defaults via :meth:`FluxionMeasure.parse_arguments`).
    """
    measure_args = measure_args or {}
    applied: list[str] = []
    for entry in measures:
        instance = entry() if inspect.isclass(entry) else entry
        if not isinstance(instance, FluxionMeasure):
            raise TypeError(
                f"apply_measures() expected FluxionMeasure classes/instances, "
                f"got {type(entry).__name__}"
            )
        # Refuse to run on a worker thread (defensive).
        _warn_if_inside_timestepping(instance.name or instance.__class__.__name__)
        raw = measure_args.get(instance.name, {}) or {}
        args = instance.parse_arguments(raw)
        instance.apply(model, args)
        applied.append(instance.name or instance.__class__.__name__)
    return applied


__all__ = [
    "FluxionMeasure",
    "SCHEMA_VERSION",
    "apply_measures",
    "discover_measures",
    "dict_to_model",
    "load_model",
    "model_to_dict",
    "save_model",
]
