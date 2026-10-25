"""SetWindowProperties — standard-library Fluxion measure (Issue #3995).

Sets the **glazing U-value** and/or the **solar heat gain coefficient**
(SHGC) of the model's window specs. This is the canonical *window upgrade*
parametric study in building energy modeling — "what does a single-pane →
low-e double-pane glazing package save?" — and the direct analogue of the
OpenStudio ``SetWindowToWallRatio``-family glazing measures, but on the
glazing *properties* rather than the glazing *area*.

Relationship to the sibling measures
------------------------------------
``measures/SetWindowToWallRatio.py`` resizes the glazing (area / WWR) and
leaves the glazing physics untouched. Nothing else in the standard library
can change the glazing *properties*: this measure is the natural sibling
that writes the two numbers the engine actually reads for the fenestration
— the conductive transmittance (``window_u_value``) and the solar
transmittance (``window_shgc``).

Definition
----------
The two fields are the ones the Rust schema carries on
``WindowSpec`` (``src/api/schema.rs``):

* ``window_u_value`` — the glazing U-value in W/m²·K (the whole-window
  thermal transmittance including the frame and the glazing layers).
* ``window_shgc`` — the solar heat gain coefficient, a dimensionless
  fraction of the incident solar radiation that enters the zone.

Each of the two arguments is **optional and independent**: passing only
``u_value_w_m2k`` leaves the model's SHGC untouched, and passing only
``shgc`` leaves the model's U-value untouched. At least one of them must
be provided — a request that changes neither is a no-op and fails loudly
instead of silently reporting a glazing upgrade.

Bounds
------
Both values are validated **before the model is touched**, so an invalid
request never leaves a half-mutated model behind:

* U-value — must be a finite number strictly greater than 0 and no
  greater than ``MAX_WINDOW_U_VALUE_W_M2K`` (10.0 W/m²·K). The lower
  bound is the Rust validator's rule (``src/api/schema.rs``:
  ``window_u_value <= 0`` is a validation error). The upper bound is a
  *measure-side* sanity cap: single-pane clear glass is ≈ 5.8 W/m²·K
  (``fluxion-core/src/ashrae_cases.rs``,
  ``WindowSpec::single_clear_glass``), double clear glass ≈ 2.1 W/m²·K
  (``WindowSpec::double_clear_glass``) and double low-e ≈ 2.0 W/m²·K
  (``WindowSpec::double_low_e``), so anything above 10 W/m²·K can only
  be a unit error (e.g. a value in BTU/h·ft²·°F).
* SHGC — must be a finite number in the half-open interval ``(0, 1]``,
  **exactly** the rule the Rust validator applies
  (``src/api/schema.rs``: ``!(0.0 < window_shgc && window_shgc <= 1.0)``
  is a validation error). A SHGC of 0 is not a window (it blocks all
  solar gain), and a SHGC above 1 is unphysical.

The U-value cap lives in this measure — the Rust engine has no glazing
U-value constant to borrow.

Finding the window specs
------------------------
The measure *probes* the model for the window specs it exposes, in this
documented order:

1. **A window / windows accessor on the model** — ``model.window_specs()``
   (canonical; the Rust ``WindowSpec`` collection) or ``model.windows()``
   (the shorter alias). Each element of the collection is a window spec
   carrying ``window_u_value`` / ``window_shgc``.
2. **Surface / zone objects carrying the window fields** —
   ``model.surfaces()`` and ``model.zones()``. A schema-shaped wrapper
   either carries the two fields directly or nests them in a ``window``
   attribute (the ``SurfaceSpec.window`` / ``ConstructionSpec.window``
   shape of ``src/api/schema.rs``); both spellings are recognised.
3. **The fields held directly on the model object** —
   ``model.window_u_value`` / ``model.window_shgc``.

The first two steps are *collection* probes: every element that carries
the window fields is a window spec and **all** of them are mutated — a
glazing upgrade applies to the model's glazing, not just to the first
spec that happens to be found. The third step is the last-resort
fallback for a model that holds the glazing fields itself.

If the model exposes **no** window spec at all, the measure raises
:class:`WindowNotFoundError` — a loud failure. Applying a glazing study
to a model without glazing and reporting success would put a glazing
delta on the provenance chain that belongs to a configuration the model
never had. This mirrors the ``LightingNotFoundError`` /
``EquipmentNotFoundError`` / ``HVACSystemNotFoundError`` pattern of the
other #3995 measures.

Persistence
-----------
The measure follows the snapshot / owned-value contract from
``docs/bindings.md``: every collection is read **once**, the window specs
in it are mutated in place, and the mutated collection is pushed back
through the matching setter when the binding exposes one:

* ``model.set_window_specs(...)`` / ``model.set_windows(...)`` — the
  mutated window-spec collection is written back.
* ``model.set_surfaces(...)`` / ``model.set_zones(...)`` — the mutated
  surface / zone collection is written back.
* neither — the mutation is **advisory**: the new glazing properties are
  visible on the snapshot the binding handed out, but they do not
  round-trip into the serialized model. This mirrors the documented
  advisory fields of ``ReplaceHVACWithVAV`` and of
  ``SetHVACEfficiency``'s COP fields.

On today's native ``fluxion.Model`` binding the measure is **advisory at
best**: the binding exposes no window-spec accessor and the surface
snapshot carries only ``area`` / ``u_value`` / ``orientation`` /
``window_area`` (``src/python/model_bindings/model.rs``), so a native
model exposes no window spec and the measure fails loudly with
:class:`WindowNotFoundError`. The measure is written against the
documented schema shape (``WindowSpec``) and round-trips as soon as the
binding grows a window-spec accessor.

Provenance (Issue #1816)
------------------------
When run through :func:`fluxion.measures.apply_measures` with an
``applied_deltas`` accumulator, a ``python_measure`` entry is appended
automatically. This measure itself performs no provenance bookkeeping —
the runner owns the chain.

Run with::

    fluxion apply-measures \\
        --model base.json \\
        --measures measures/ \\
        --measure-args args.json \\
        --output model.lowe.json

Where ``args.json`` is::

    {
        "SetWindowProperties": {
            "u_value_w_m2k": 1.7,
            "shgc": 0.35
        }
    }

Or from Python::

    from fluxion import Model
    from fluxion.measures import apply_measures
    from measures.SetWindowProperties import SetWindowProperties

    model = Model(num_zones=2)
    apply_measures(model, [SetWindowProperties],
                   {"SetWindowProperties": {"u_value_w_m2k": 1.7}})
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: The Rust field name of the glazing U-value (``WindowSpec`` in
#: ``src/api/schema.rs``).
U_VALUE_FIELD = "window_u_value"

#: The Rust field name of the solar heat gain coefficient (``WindowSpec``
#: in ``src/api/schema.rs``).
SHGC_FIELD = "window_shgc"

#: Largest glazing U-value the measure accepts (W/m²·K). Single-pane clear
#: glass is ≈ 5.8 W/m²·K (``fluxion-core/src/ashrae_cases.rs``); above
#: 10 W/m²·K the request can only be a unit error.
MAX_WINDOW_U_VALUE_W_M2K = 10.0

#: Largest solar heat gain coefficient the measure accepts (dimensionless
#: fraction). The Rust validator allows exactly ``(0, 1]``.
MAX_WINDOW_SHGC = 1.0

#: ``(getter, setter)`` pairs probed on the model for a *window spec
#: collection*, in order. ``window_specs`` is the canonical spelling (the
#: Rust ``WindowSpec`` collection); ``windows`` is the shorter alias.
WINDOW_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("window_specs", "set_window_specs"),
    ("windows", "set_windows"),
)

#: ``(getter, setter)`` pairs probed on the model for the *surface* and
#: *zone* collections, in order. A schema-shaped surface / zone carries the
#: glazing fields directly or nests them in a ``window`` attribute.
SURFACE_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("surfaces", "set_surfaces"),
    ("zones", "set_zones"),
)


class WindowNotFoundError(ValueError):
    """Raised when the model exposes no window spec to mutate.

    A glazing study applied to a model without glazing would report a
    change that never happened: the provenance chain would carry a
    glazing delta while the model kept its un-upgraded windows. The
    measure fails loudly instead.
    """


class WindowSpecGroup:
    """One collection of window specs found on the model.

    Attributes
    ----------
    path : str
        The accessor that produced the collection (``"window_specs"``,
        ``"windows"``, ``"surfaces"``, ``"zones"`` or
        ``"model_attribute"``).
    collection : Any
        The collection handed out by the accessor — pushed back through
        the setter when the binding exposes one.
    setter : str or None
        Name of the matching setter on the model, or ``None`` when the
        binding exposes none (the change is then advisory).
    specs : list
        The window specs of this collection (the objects that actually
        carry the glazing fields).
    """

    def __init__(
        self,
        path: str,
        collection: Any,
        setter: str | None,
        specs: list[Any] | None = None,
    ) -> None:
        self.path = path
        self.collection = collection
        self.setter = setter
        self.specs = list(specs) if specs is not None else []

    def __repr__(self) -> str:
        return (
            f"WindowSpecGroup(path={self.path!r}, setter={self.setter!r}, "
            f"specs={len(self.specs)})"
        )


def validate_u_value(
    value: Any,
    maximum: float = MAX_WINDOW_U_VALUE_W_M2K,
) -> float:
    """Return ``value`` as a validated glazing U-value (W/m²·K).

    The U-value must be a finite number strictly greater than 0 (the Rust
    validator's rule, ``src/api/schema.rs``) and no greater than
    ``maximum`` (the measure-side sanity cap). Anything else raises
    ``ValueError``.

    Pure function — unit-testable without the native bindings.
    """
    try:
        u_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Window U-value must be a number, got {value!r}"
        ) from exc

    if u_value != u_value:  # NaN
        raise ValueError("Window U-value must not be NaN")
    if u_value in (float("inf"), float("-inf")):
        raise ValueError("Window U-value must be finite")
    if u_value <= 0.0:
        raise ValueError(
            f"Window U-value must be > 0 W/m²·K, got {u_value}. The Rust "
            "schema validator rejects a non-positive glazing U-value "
            "(src/api/schema.rs)."
        )
    if u_value > float(maximum):
        raise ValueError(
            f"Window U-value must be <= {maximum} W/m²·K, got {u_value}. "
            "Single-pane clear glass is ≈ 5.8 W/m²·K — a larger value is "
            "a unit error."
        )
    return u_value


def validate_shgc(
    value: Any,
    maximum: float = MAX_WINDOW_SHGC,
) -> float:
    """Return ``value`` as a validated solar heat gain coefficient.

    The SHGC must be a finite number in the half-open interval
    ``(0, maximum]`` — exactly the rule the Rust validator applies
    (``src/api/schema.rs``: ``!(0.0 < window_shgc && window_shgc <= 1.0)``
    is a validation error). Anything else raises ``ValueError``.

    Pure function — unit-testable without the native bindings.
    """
    try:
        shgc = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Window SHGC must be a number, got {value!r}"
        ) from exc

    if shgc != shgc:  # NaN
        raise ValueError("Window SHGC must not be NaN")
    if shgc in (float("inf"), float("-inf")):
        raise ValueError("Window SHGC must be finite")
    if shgc <= 0.0:
        raise ValueError(
            f"Window SHGC must be > 0, got {shgc}. The Rust schema "
            "validator rejects a non-positive SHGC (src/api/schema.rs)."
        )
    if shgc > float(maximum):
        raise ValueError(
            f"Window SHGC must be <= {maximum}, got {shgc}. The Rust schema "
            "validator rejects a SHGC above 1 (src/api/schema.rs)."
        )
    return shgc


def _probe(model: Any, name: str) -> Any:
    """Return ``getattr(model, name)``, calling it when it is callable.

    Returns ``None`` when the attribute is missing or when calling it
    fails — a binding may expose an accessor it cannot serve on this
    model, in which case the probe moves on to the next candidate.
    """
    attribute = getattr(model, name, None)
    if attribute is None:
        return None
    if callable(attribute):
        try:
            return attribute()
        except (AttributeError, TypeError, NotImplementedError):
            return None
    return attribute


def _has_field(target: Any, name: str) -> bool:
    """True when ``target`` carries the field ``name``.

    Supports plain attributes (the ``fluxion`` snapshot style) and
    mapping keys (a ``dict``-shaped window spec).
    """
    if isinstance(target, dict):
        return name in target
    return hasattr(target, name)


def _carries_window_fields(target: Any) -> bool:
    """True when ``target`` looks like a window spec.

    A window spec carries the glazing U-value and/or the SHGC — the two
    fields of the Rust ``WindowSpec`` (``src/api/schema.rs``).
    """
    return _has_field(target, U_VALUE_FIELD) or _has_field(target, SHGC_FIELD)


def _as_sequence(value: Any) -> list[Any]:
    """Return ``value`` as a list of candidate window specs."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _window_spec_of(element: Any) -> Any:
    """Return the window spec carried by ``element``, or ``None``.

    A schema-shaped surface / zone carries the glazing fields either
    directly (``element.window_u_value``) or nested in a ``window``
    attribute (``element.window`` — the ``SurfaceSpec.window`` /
    ``ConstructionSpec.window`` shape of ``src/api/schema.rs``).
    """
    if _carries_window_fields(element):
        return element
    nested = element.get("window") if isinstance(element, dict) else getattr(
        element, "window", None
    )
    if nested is not None and _carries_window_fields(nested):
        return nested
    return None


def _collect_group(
    model: Any,
    getter_name: str,
    setter_name: str,
) -> WindowSpecGroup | None:
    """Return the window-spec group of one accessor, or ``None``.

    The accessor is probed once; every element that carries the glazing
    fields becomes a window spec of the returned group.
    """
    collection = _probe(model, getter_name)
    if collection is None:
        return None

    items = _as_sequence(collection)
    specs: list[Any] = []
    for item in items:
        spec = _window_spec_of(item)
        if spec is not None:
            specs.append(spec)

    if not specs:
        return None

    return WindowSpecGroup(
        path=getter_name,
        collection=collection,
        setter=setter_name,
        specs=specs,
    )


def resolve_window_specs(model: Any) -> list[WindowSpecGroup]:
    """Locate every window spec the model exposes.

    The probes run in this documented order:

    1. ``model.window_specs()`` / ``model.windows()`` — the window-spec
       collection of the model.
    2. ``model.surfaces()`` / ``model.zones()`` — surfaces / zones that
       carry the glazing fields directly or in a nested ``window``
       attribute.
    3. the fields held directly on the model object
       (``model.window_u_value`` / ``model.window_shgc``).

    Every collection that yields at least one window spec is returned —
    a glazing upgrade applies to all of the model's glazing, not just to
    the first spec that happens to be found.

    Raises :class:`WindowNotFoundError` when the model exposes no window
    spec at all.

    Pure function — unit-testable without the native bindings (it only
    reads the model).
    """
    groups: list[WindowSpecGroup] = []

    for getter_name, setter_name in WINDOW_ACCESSORS:
        group = _collect_group(model, getter_name, setter_name)
        if group is not None:
            groups.append(group)

    for getter_name, setter_name in SURFACE_ACCESSORS:
        group = _collect_group(model, getter_name, setter_name)
        if group is not None:
            groups.append(group)

    if groups:
        return groups

    # The model may hold the glazing fields itself (e.g. a schema-shaped
    # wrapper object).
    if _carries_window_fields(model):
        return [
            WindowSpecGroup(
                path="model_attribute",
                collection=model,
                setter=None,
                specs=[model],
            )
        ]

    raise WindowNotFoundError(
        "The model exposes no window spec to mutate — no 'window_specs' / "
        "'windows' accessor, no surface or zone carrying 'window_u_value' / "
        "'window_shgc', and no glazing field on the model itself. Refusing "
        "to report a glazing-property change that never happened."
    )


def require_window_fields(specs: list[Any], names: list[str]) -> None:
    """Check that every window spec carries the requested fields.

    A window spec that carries only one of the two glazing fields cannot
    take the other one. The check runs **before** any spec is mutated, so
    a request the model cannot satisfy never leaves a half-mutated set of
    window specs behind.

    Raises ``ValueError`` when a spec does not carry a requested field.
    """
    for spec in specs:
        for name in names:
            if not _has_field(spec, name):
                raise ValueError(
                    f"The window spec does not carry '{name}' — the glazing "
                    "properties of this model cannot be written. Refusing "
                    "to report a glazing-property change that never "
                    "happened."
                )


def set_window_field(target: Any, name: str, value: float) -> None:
    """Write ``value`` into the window field ``name`` of ``target``.

    Handles attribute-style and mapping-style targets.
    """
    if not _has_field(target, name):
        raise ValueError(
            f"The window spec does not carry '{name}' — the glazing "
            "properties of this model cannot be written. Refusing to "
            "report a glazing-property change that never happened."
        )
    if isinstance(target, dict) and not hasattr(target, name):
        target[name] = float(value)
        return
    setattr(target, name, float(value))


def persist_window_specs(
    model: Any,
    group: WindowSpecGroup,
) -> str:
    """Push the mutated window specs of ``group`` back into ``model``.

    Returns the persistence path that was taken:

    * the accessor's own path (``"window_specs"`` / ``"windows"`` /
      ``"surfaces"`` / ``"zones"``) — the matching setter exists and the
      mutated collection was written back.
    * ``"model_attribute"`` — the glazing fields live on the model
      itself; they were set in place.
    * ``"advisory"`` — no setter exists, so the new glazing properties
      are visible on the snapshot only.
    """
    if group.setter is None:
        return "model_attribute"

    setter = getattr(model, group.setter, None)
    if callable(setter):
        setter(group.collection)
        return group.path

    return "advisory"


class SetWindowProperties(FluxionMeasure):
    """Set the glazing U-value and/or SHGC of the model's window specs.

    Parameters
    ----------
    u_value_w_m2k : float or None
        Target glazing U-value in watts per square metre kelvin
        (W/m²·K). Must be > 0 and <= ``MAX_WINDOW_U_VALUE_W_M2K``
        (10.0). ``None`` (default) keeps the model's current glazing
        U-value.
    shgc : float or None
        Target solar heat gain coefficient (dimensionless fraction). Must
        be in ``(0, 1]``. ``None`` (default) keeps the model's current
        SHGC.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "u_value_w_m2k",
                "type": "double",
                "default": None,
                "min": 0.0,
                "max": MAX_WINDOW_U_VALUE_W_M2K,
                "description": (
                    "Target glazing U-value (W/m²·K). Must be > 0 and "
                    f"<= {MAX_WINDOW_U_VALUE_W_M2K}. null keeps the "
                    "model's current glazing U-value."
                ),
            },
            {
                "name": "shgc",
                "type": "double",
                "default": None,
                "min": 0.0,
                "max": MAX_WINDOW_SHGC,
                "description": (
                    "Target solar heat gain coefficient (fraction). Must be "
                    f"in (0, {MAX_WINDOW_SHGC}]. null keeps the model's "
                    "current SHGC."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        u_value = arguments.get("u_value_w_m2k", None)
        shgc = arguments.get("shgc", None)

        if u_value is None and shgc is None:
            raise ValueError(
                "SetWindowProperties requires at least one of "
                "'u_value_w_m2k' or 'shgc' — a glazing study that changes "
                "neither the U-value nor the SHGC is a no-op."
            )

        # Validate BEFORE touching the model so an invalid request never
        # leaves a half-mutated set of window specs behind.
        new_u_value = (
            None if u_value is None else validate_u_value(u_value)
        )
        new_shgc = None if shgc is None else validate_shgc(shgc)

        groups = resolve_window_specs(model)

        # Check that every window spec can take the requested fields
        # BEFORE any of them is mutated.
        required = []
        if new_u_value is not None:
            required.append(U_VALUE_FIELD)
        if new_shgc is not None:
            required.append(SHGC_FIELD)
        require_window_fields(
            [spec for group in groups for spec in group.specs], required
        )

        mutated = 0
        seen: set[int] = set()
        for group in groups:
            for spec in group.specs:
                if id(spec) in seen:
                    continue
                seen.add(id(spec))
                if new_u_value is not None:
                    set_window_field(spec, U_VALUE_FIELD, new_u_value)
                if new_shgc is not None:
                    set_window_field(spec, SHGC_FIELD, new_shgc)
                mutated += 1

            persisted = persist_window_specs(model, group)
            if persisted == "advisory":
                _logger.warning(
                    "SetWindowProperties: the model exposes no '%s' setter "
                    "— the glazing properties are advisory (visible on the "
                    "window-spec snapshot, not persisted into the model).",
                    group.setter,
                )

        _logger.info(
            "SetWindowProperties: %d window spec(s) updated — "
            "u_value=%s, shgc=%s",
            mutated,
            "unchanged" if new_u_value is None else f"{new_u_value:.3f}",
            "unchanged" if new_shgc is None else f"{new_shgc:.3f}",
        )
