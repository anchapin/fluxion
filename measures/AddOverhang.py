"""AddOverhang — standard-library Fluxion measure (Issue #3995).

Adds a horizontal **overhang** (a projecting shading device) to the façades
of a model. This is the canonical *shading* parametric study in building
energy modeling — "how much does a 1 m south overhang save?" — and the
direct analogue of the OpenStudio ``AddOverhang`` measure.

This measure promotes the teaching example
``measures/examples/add_overhang.py`` (class ``AddSouthOverhang``) to a
first-class standard-library measure: same snapshot round-trip, generalised
from *south only* to *any façade orientation*.

Definition
----------
For every eligible surface the measure attaches an overhang device with

* ``depth``  — the horizontal projection of the overhang, in metres;
* ``height`` — the mounting height above the bottom of the surface (the
  window head for a wall with glazing), in metres.

The overhang shades the surface below its projection, cutting the summer
solar gain while leaving the winter gain (low sun angles) largely intact.
No opaque-envelope property is changed: the surface area, U-value and
window area are untouched — only the shading topology grows.

Bounds
------
Both dimensions are checked in :func:`validate_dimension` **before** the
model is touched, so an invalid request never leaves the model
half-mutated:

* ``MIN_OVERHANG_DEPTH_M`` (0.0 m) / ``MAX_OVERHANG_DEPTH_M`` (5.0 m) —
  the bounds of the teaching example. A 5 m projection is already a
  generous canopy; beyond that the request is a unit error.
* ``MIN_OVERHANG_HEIGHT_M`` (0.0 m) / ``MAX_OVERHANG_HEIGHT_M`` (10.0 m) —
  the mounting height above the surface's base. 10 m clears the tallest
  storey of a typical building.

The bounds live in this measure — the Rust engine has no overhang constant
to borrow.

Orientation filter
------------------
The ``orientation`` argument selects which façades receive the overhang.
The accepted spellings are the cardinal names of the native
``fluxion.Orientation`` enum (``north``, ``east``, ``south``, ``west``),
case-insensitively, plus the enum values themselves (e.g.
``fluxion.Orientation.South``).

The native ``Orientation`` enum has **no** "all exterior" variant — it only
carries the four cardinal orientations plus ``Up`` (roof), ``Down`` (floor)
and ``Horizontal``. The measure therefore accepts the ``all`` spelling as
the "all exterior façade" convenience: it matches every **cardinal
vertical** façade (north, east, south and west). Roofs and floors are never
eligible — a horizontal overhang on a roof or floor has no meaning.

The surface snapshot does not expose an exterior / interior flag (the
native ``Surface`` carries only ``area``, ``u_value``, ``orientation``,
``window_area`` and its shading devices — see ``fluxion.pyi``), so the
orientation is the only geometric filter available on today's bindings.

``only_with_windows`` (default ``False``) further restricts the measure to
surfaces that already carry glazing (``window_area > 0``), which is what
makes an overhang meaningful: shading an opaque wall blocks nothing.

Persistence
-----------
The measure follows the snapshot / owned-value contract from
``docs/bindings.md``: the surfaces are read **once** via
``model.surfaces()``, the overhang is attached to the *snapshots*, and the
mutated list is pushed back through ``model.set_surfaces(...)``. Mutating
a snapshot does not touch the model on its own — the push-back is what
makes the change stick.

If the model exposes no ``set_surfaces`` setter, the change is **advisory**
(the same pattern as ``SetHVACEfficiency``'s advisory COP fields and
``ReplaceHVACWithVAV``): the measure logs a warning and the overhang is
visible on the surface snapshot only.

Loud failure
------------
If no surface is eligible — no façade of the requested orientation, or
none of them carries glazing while ``only_with_windows`` is set — the
measure raises :class:`NoEligibleSurfaceError`. Reporting "0 overhangs
added" as a success would put a shading delta on the provenance chain that
belongs to a configuration the model never had.

Provenance (Issue #1816)
------------------------
When run through :func:`fluxion.measures.apply_measures` with an
``applied_deltas`` accumulator, a ``python_measure`` entry is appended
automatically. This measure itself performs no provenance bookkeeping — the
runner owns the chain.

Run with::

    fluxion apply-measures \\
        --model base.json \\
        --measures measures/ \\
        --measure-args args.json \\
        --output model.shaded.json

Where ``args.json`` is::

    {
        "AddOverhang": {
            "depth": 1.0,
            "height": 2.5,
            "orientation": "south",
            "only_with_windows": true
        }
    }

Or from Python::

    from fluxion import Model
    from fluxion.measures import apply_measures
    from measures.AddOverhang import AddOverhang

    model = Model(num_zones=2)
    apply_measures(model, [AddOverhang],
                   {"AddOverhang": {"depth": 1.0, "height": 2.5}})
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: Default overhang projection (m). Same default as the teaching example.
DEFAULT_OVERHANG_DEPTH_M = 1.0

#: Smallest overhang projection (m) the measure accepts.
MIN_OVERHANG_DEPTH_M = 0.0

#: Largest overhang projection (m) the measure accepts.
MAX_OVERHANG_DEPTH_M = 5.0

#: Default mounting height above the surface base (m).
DEFAULT_OVERHANG_HEIGHT_M = 2.5

#: Smallest mounting height (m) the measure accepts.
MIN_OVERHANG_HEIGHT_M = 0.0

#: Largest mounting height (m) the measure accepts.
MAX_OVERHANG_HEIGHT_M = 10.0

#: Default façade orientation.
DEFAULT_ORIENTATION = "south"

#: The "all exterior façade" spelling — every cardinal vertical façade.
ORIENTATION_ALL = "all"

#: The cardinal (vertical façade) orientations of the native
#: ``fluxion.Orientation`` enum. ``Up`` (roof), ``Down`` (floor) and
#: ``Horizontal`` are deliberately absent: a horizontal overhang on a roof
#: or a floor has no meaning.
VERTICAL_ORIENTATIONS: tuple[str, ...] = ("north", "east", "south", "west")

#: The accepted ``orientation`` argument values.
ORIENTATION_CHOICES: tuple[str, ...] = VERTICAL_ORIENTATIONS + (ORIENTATION_ALL,)

#: Spellings that map onto :data:`ORIENTATION_ALL`.
_ALL_ALIASES: frozenset[str] = frozenset(
    {
        "all",
        "all exterior",
        "all_exterior",
        "all exterior orientations",
        "exterior",
        "any",
    }
)


class NoEligibleSurfaceError(ValueError):
    """Raised when no surface is eligible for the requested overhang.

    A shading study applied to a model whose façades do not match the
    request would report a change that never happened: the provenance
    chain would carry a shading delta while the model kept its unshaded
    surfaces. The measure fails loudly instead.
    """


def validate_dimension(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Return ``value`` as a validated overhang dimension (metres).

    The dimension must be a finite number within ``[minimum, maximum]``.
    Anything else raises ``ValueError``.

    Pure function — unit-testable without the native bindings.
    """
    try:
        dimension = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Overhang {name} must be a number, got {value!r}"
        ) from exc

    if dimension != dimension:  # NaN
        raise ValueError(f"Overhang {name} must not be NaN")
    if dimension in (float("inf"), float("-inf")):
        raise ValueError(f"Overhang {name} must be finite")
    if dimension < float(minimum):
        raise ValueError(
            f"Overhang {name} must be >= {minimum} m, got {dimension}"
        )
    if dimension > float(maximum):
        raise ValueError(
            f"Overhang {name} must be <= {maximum} m, got {dimension}"
        )
    return dimension


def validate_depth(value: Any) -> float:
    """Return ``value`` as a validated overhang depth (m).

    Pure function — unit-testable without the native bindings.
    """
    return validate_dimension(
        value, "depth", MIN_OVERHANG_DEPTH_M, MAX_OVERHANG_DEPTH_M
    )


def validate_height(value: Any) -> float:
    """Return ``value`` as a validated overhang mounting height (m).

    Pure function — unit-testable without the native bindings.
    """
    return validate_dimension(
        value, "height", MIN_OVERHANG_HEIGHT_M, MAX_OVERHANG_HEIGHT_M
    )


def orientation_name(value: Any) -> str:
    """Return the lowercase orientation name carried by ``value``.

    Accepts the native ``fluxion.Orientation`` enum values (whose
    ``repr`` is ``"Orientation.South"`` and whose ``.name`` is
    ``"South"``), plain strings, and any other object whose ``repr``
    ends in a variant name.

    Pure function — unit-testable without the native bindings.
    """
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.strip().lower()
    if isinstance(value, str):
        return value.strip().lower()
    text = repr(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().strip("'\"").lower()


def normalize_orientation(value: Any) -> str:
    """Return the canonical orientation key for ``value``.

    The canonical keys are :data:`ORIENTATION_CHOICES` — the four
    cardinal façades plus ``"all"`` (every cardinal vertical façade).

    Raises :class:`ValueError` for an unknown spelling, listing the
    accepted ones.

    Pure function — unit-testable without the native bindings.
    """
    if value is None:
        return DEFAULT_ORIENTATION

    raw = value if isinstance(value, str) else orientation_name(value)
    key = raw.strip().lower()
    if key in _ALL_ALIASES:
        return ORIENTATION_ALL
    if key in ORIENTATION_CHOICES:
        return key
    raise ValueError(
        f"Unknown overhang orientation {value!r}. Accepted values: "
        + ", ".join(ORIENTATION_CHOICES)
    )


def matches_orientation(surface_orientation: Any, wanted: str) -> bool:
    """True when ``surface_orientation`` matches the wanted façade.

    ``wanted`` is a canonical orientation key (see
    :func:`normalize_orientation`). The ``"all"`` key matches every
    cardinal vertical façade.

    Pure function — unit-testable without the native bindings.
    """
    name = orientation_name(surface_orientation)
    if wanted == ORIENTATION_ALL:
        return name in VERTICAL_ORIENTATIONS
    return name == wanted


def window_area_of(surface: Any) -> float:
    """Return the glazed area (m²) carried by ``surface``.

    Surfaces without the attribute (or with a non-numeric one) are
    treated as unglazed.

    Pure function — unit-testable without the native bindings.
    """
    value = getattr(surface, "window_area", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def is_eligible(
    surface: Any,
    orientation: str,
    only_with_windows: bool = False,
) -> bool:
    """True when ``surface`` may receive the requested overhang.

    A surface is eligible when it faces the requested orientation and,
    when ``only_with_windows`` is set, already carries glazing.

    Pure function — unit-testable without the native bindings.
    """
    if not matches_orientation(getattr(surface, "orientation", None), orientation):
        return False
    if only_with_windows and window_area_of(surface) <= 0.0:
        return False
    return True


def select_eligible(
    surfaces: Any,
    orientation: str,
    only_with_windows: bool = False,
) -> list[Any]:
    """Return the eligible subset of ``surfaces``.

    Pure function — unit-testable without the native bindings.
    """
    return [
        surface
        for surface in surfaces
        if is_eligible(surface, orientation, only_with_windows)
    ]


def read_surfaces(model: Any) -> list[Any]:
    """Return the model's surface snapshots.

    Mirrors the native ``Model.surfaces()`` accessor (a method returning
    an owned snapshot list). A model that exposes the surfaces as a plain
    attribute is supported too.

    Raises :class:`NoEligibleSurfaceError` when the model exposes no
    usable surfaces accessor.
    """
    accessor = getattr(model, "surfaces", None)
    if accessor is None:
        raise NoEligibleSurfaceError(
            "The model exposes no 'surfaces' accessor — cannot attach an "
            "overhang to a model without surfaces."
        )
    if callable(accessor):
        try:
            surfaces = accessor()
        except (AttributeError, TypeError, NotImplementedError) as exc:
            raise NoEligibleSurfaceError(
                "The model's 'surfaces' accessor could not be called — "
                "cannot attach an overhang."
            ) from exc
    else:
        surfaces = accessor
    return list(surfaces)


def persist_surfaces(model: Any, surfaces: list[Any]) -> str:
    """Push the mutated surface snapshots back into ``model``.

    Returns the persistence path that was taken:

    * ``"set_surfaces"`` — the setter exists and the mutated snapshot
      list was written back into the model.
    * ``"advisory"`` — no setter exists, so the overhang is visible on
      the surface snapshot only.
    """
    setter = getattr(model, "set_surfaces", None)
    if callable(setter):
        setter(surfaces)
        return "set_surfaces"
    return "advisory"


class AddOverhang(FluxionMeasure):
    """Attach a horizontal overhang to the façades of the model.

    Parameters
    ----------
    depth : float
        Overhang projection in metres (m). Must be within
        ``[MIN_OVERHANG_DEPTH_M, MAX_OVERHANG_DEPTH_M]`` (0.0–5.0 m).
        Default 1.0 m.
    height : float
        Mounting height above the base of the surface, in metres (m).
        Must be within ``[MIN_OVERHANG_HEIGHT_M, MAX_OVERHANG_HEIGHT_M]``
        (0.0–10.0 m). Default 2.5 m.
    orientation : str
        Façade orientation to shade: ``"north"``, ``"east"``,
        ``"south"``, ``"west"`` or ``"all"`` (every cardinal vertical
        façade). Case-insensitive; the native ``fluxion.Orientation``
        enum values are accepted too. Default ``"south"``.
    only_with_windows : bool
        If True, only attach the overhang to surfaces that already carry
        glazing (``window_area > 0``). Default False — apply to every
        façade of the requested orientation.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "depth",
                "type": "double",
                "default": DEFAULT_OVERHANG_DEPTH_M,
                "min": MIN_OVERHANG_DEPTH_M,
                "max": MAX_OVERHANG_DEPTH_M,
                "description": "Overhang depth (projection) in meters (m).",
            },
            {
                "name": "height",
                "type": "double",
                "default": DEFAULT_OVERHANG_HEIGHT_M,
                "min": MIN_OVERHANG_HEIGHT_M,
                "max": MAX_OVERHANG_HEIGHT_M,
                "description": "Mounting height above the surface base (m).",
            },
            {
                "name": "orientation",
                "type": "choice",
                "default": DEFAULT_ORIENTATION,
                "choices": list(ORIENTATION_CHOICES),
                "description": (
                    "Façade orientation to shade: north, east, south, west "
                    "or all (every cardinal vertical façade)."
                ),
            },
            {
                "name": "only_with_windows",
                "type": "bool",
                "default": False,
                "description": (
                    "If true, only attach the overhang to surfaces with "
                    "window_area > 0."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        # Validate BEFORE touching the model so an invalid request never
        # leaves a half-shaded model behind.
        depth = validate_depth(arguments.get("depth", DEFAULT_OVERHANG_DEPTH_M))
        height = validate_height(
            arguments.get("height", DEFAULT_OVERHANG_HEIGHT_M)
        )
        orientation = normalize_orientation(
            arguments.get("orientation", DEFAULT_ORIENTATION)
        )
        only_with_windows = bool(
            arguments.get("only_with_windows", False)
        )

        surfaces = read_surfaces(model)
        eligible = select_eligible(surfaces, orientation, only_with_windows)
        if not eligible:
            raise NoEligibleSurfaceError(
                "No surface of the model is eligible for the requested "
                f"overhang (orientation={orientation!r}, "
                f"only_with_windows={only_with_windows}). Refusing to "
                "report a shading change that never happened."
            )

        for surface in eligible:
            surface.add_overhang(depth=depth, height=height)

        persisted = persist_surfaces(model, surfaces)

        if persisted == "advisory":
            _logger.warning(
                "AddOverhang: the model exposes no 'set_surfaces' setter — "
                "the overhangs are advisory (visible on the surface "
                "snapshots, not persisted into the model)."
            )

        _logger.info(
            "AddOverhang: attached an overhang to %d surface(s) "
            "(depth=%.2f m, height=%.2f m, orientation=%s) — persisted via %s",
            len(eligible),
            depth,
            height,
            orientation,
            persisted,
        )
