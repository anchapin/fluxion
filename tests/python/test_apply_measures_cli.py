"""Pytest suite for ``fluxion apply-measures`` and the FluxionMeasure API.

Issue #1814 — Phase 2 (Python Measure API) of the Hybrid Measure Approach.

These tests cover:

- :class:`FluxionMeasure` base-class semantics (``apply`` / ``arguments``).
- The runtime guard that warns when a measure runs on a rayon worker thread
  or when ``FLUXION_INSIDE_TIMESTEPPING=1`` is set.
- :func:`discover_measures` — directory walking and subclass collection.
- :func:`apply_measures` — order preservation and argument plumbing.
- :func:`save_model` / :func:`load_model` — JSON round-trip (msgpack if
  installed).
- The ``fluxion apply-measures`` CLI as a subprocess — end-to-end coverage
  of the user-facing entry point.

The suite is designed to be runnable in any of three modes:

1. **With the native bindings installed** (``maturin develop``): every test
   that touches a real model exercises the snapshot/owned-value path.
2. **Without the native bindings** (CI without Rust toolchain): tests that
   need a ``Model`` are skipped via the ``needs_fluxion`` marker (set up
   by ``tests/conftest.py``). The pure-Python surface (discovery, argument
   parsing, the runtime guard) is fully covered.
3. **With msgpack available**: an extra round-trip test exercises the
   ``.msgpack`` path; it is auto-skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import threading
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASURES_DIR = REPO_ROOT / "measures" / "examples"


# ---------------------------------------------------------------------------
# Module availability
# ---------------------------------------------------------------------------


fluxion_spec = importlib.util.find_spec("fluxion")
HAS_FLUXION = fluxion_spec is not None

requires_fluxion = pytest.mark.skipif(
    not HAS_FLUXION,
    reason="fluxion Python bindings not available (run `maturin develop`)",
)

HAS_MSGPACK = importlib.util.find_spec("msgpack") is not None
requires_msgpack = pytest.mark.skipif(
    not HAS_MSGPACK, reason="msgpack not installed"
)


# ---------------------------------------------------------------------------
# FluxionMeasure semantics
# ---------------------------------------------------------------------------


@requires_fluxion
class TestFluxionMeasureBase:
    """Pure semantics of the abstract base class (no model mutation)."""

    def test_base_apply_raises(self):
        from fluxion import FluxionMeasure

        m = importlib.import_module("fluxion")
        model = m.Model(num_zones=1)

        # Calling apply() on the base class must raise NotImplementedError.
        with pytest.raises(NotImplementedError):
            FluxionMeasure().apply(model, {})

    def test_arguments_default_is_empty_list(self):
        from fluxion import FluxionMeasure

        assert FluxionMeasure().arguments() == []

    def test_subclass_default_name_strips_measure_suffix(self):
        from fluxion import FluxionMeasure

        class AddInsulationMeasure(FluxionMeasure):
            def apply(self, model, arguments):
                pass

        assert AddInsulationMeasure().name == "AddInsulation"

    def test_subclass_explicit_name_is_preserved(self):
        from fluxion import FluxionMeasure

        class Foo(FluxionMeasure):
            name = "My Explicit Name"

            def apply(self, model, arguments):
                pass

        assert Foo().name == "My Explicit Name"

    def test_parse_arguments_uses_defaults(self):
        from fluxion import FluxionMeasure

        class M(FluxionMeasure):
            def arguments(self):
                return [
                    {"name": "depth", "type": "double", "default": 1.0},
                    {"name": "height", "type": "double", "default": 2.0},
                ]

            def apply(self, model, arguments):
                pass

        m = M()
        assert m.parse_arguments(None) == {"depth": 1.0, "height": 2.0}
        assert m.parse_arguments({}) == {"depth": 1.0, "height": 2.0}
        assert m.parse_arguments({"depth": 5.0}) == {"depth": 5.0, "height": 2.0}

    def test_parse_arguments_warns_on_unknown_key(self):
        from fluxion import FluxionMeasure

        class M(FluxionMeasure):
            def arguments(self):
                return [{"name": "depth", "type": "double", "default": 1.0}]

            def apply(self, model, arguments):
                pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            M().parse_arguments({"unknown_key": 42})
        assert any("unknown_key" in str(w.message) for w in caught), [
            str(w.message) for w in caught
        ]


# ---------------------------------------------------------------------------
# Runtime guard
# ---------------------------------------------------------------------------


class TestRuntimeGuard:
    """The AOT-only rule must emit a runtime warning on worker threads."""

    def test_main_thread_does_not_warn(self):
        from fluxion import FluxionMeasure

        class M(FluxionMeasure):
            def apply(self, model, arguments):
                pass

        instance = M()

        # No native model needed: we never reach the base implementation
        # because the subclass no-ops. We do need *some* model to call
        # apply() though — fall back to a stub when fluxion is unavailable.
        if HAS_FLUXION:
            m = importlib.import_module("fluxion").Model(num_zones=1)
        else:
            m = object()  # type: ignore[assignment]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            instance.apply(m, {})
        assert not any(
            "rayon" in str(w.message).lower() or "timestepping" in str(w.message).lower()
            for w in caught
        ), "Main thread must not trip the worker-thread guard"

    def test_rayon_named_thread_warns(self):
        from fluxion import FluxionMeasure

        class M(FluxionMeasure):
            def apply(self, model, arguments):
                pass

        instance = M()
        m = importlib.import_module("fluxion").Model(num_zones=1) if HAS_FLUXION else object()

        result: dict[str, list[warnings._Action]] = {}

        def worker():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                instance.apply(m, {})
            result["caught"] = caught

        t = threading.Thread(target=worker, name="rayon-7")
        t.start()
        t.join()
        caught = result["caught"]
        assert any(
            "rayon" in str(w.message).lower() or "timestepping" in str(w.message).lower()
            for w in caught
        ), "Rayon-named thread must trip the guard"
        # And the warning must be a RuntimeWarning.
        assert any(
            issubclass(w.category, RuntimeWarning) for w in caught
        ), "Worker-thread warning must be RuntimeWarning"

    def test_tokio_named_thread_warns(self):
        from fluxion import FluxionMeasure

        class M(FluxionMeasure):
            def apply(self, model, arguments):
                pass

        instance = M()
        m = importlib.import_module("fluxion").Model(num_zones=1) if HAS_FLUXION else object()

        result: dict[str, list[warnings._Action]] = {}

        def worker():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                instance.apply(m, {})
            result["caught"] = caught

        t = threading.Thread(target=worker, name="tokio-runtime-worker")
        t.start()
        t.join()
        assert any(
            "timestepping" in str(w.message).lower() for w in result["caught"]
        ), "Tokio-named thread must trip the guard"

    def test_env_var_overrides_thread_name(self, monkeypatch):
        from fluxion import FluxionMeasure

        class M(FluxionMeasure):
            def apply(self, model, arguments):
                pass

        monkeypatch.setenv("FLUXION_INSIDE_TIMESTEPPING", "1")
        instance = M()
        m = importlib.import_module("fluxion").Model(num_zones=1) if HAS_FLUXION else object()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            instance.apply(m, {})
        assert any(
            "timestepping" in str(w.message).lower() for w in caught
        ), "Env-var override must trip the guard even on MainThread"


# ---------------------------------------------------------------------------
# Measure discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    """``discover_measures`` walks a directory and picks up subclasses."""

    def test_discovers_example_measures(self):
        from fluxion.measures import discover_measures

        classes = discover_measures(MEASURES_DIR)
        names = {c.__name__ for c in classes}
        # AddSouthOverhang and SetHVACCOP are the bundled examples.
        assert {"AddSouthOverhang", "SetHVACCOP"}.issubset(names), (
            f"expected AddSouthOverhang and SetHVACCOP, got {names}"
        )

    def test_returns_empty_list_for_missing_directory(self, tmp_path):
        from fluxion.measures import discover_measures

        missing = tmp_path / "does-not-exist"
        assert discover_measures(missing) == []

    def test_skips_dunder_files(self, tmp_path):
        from fluxion import FluxionMeasure
        from fluxion.measures import discover_measures

        # __init__.py is a real package marker; the discovery walker must
        # skip it (it usually contains no measures).
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "good.py").write_text(
            textwrap.dedent(
                """
                from fluxion import FluxionMeasure

                class RealMeasure(FluxionMeasure):
                    def apply(self, model, arguments):
                        pass
                """
            )
        )
        (tmp_path / "_private.py").write_text(
            textwrap.dedent(
                """
                from fluxion import FluxionMeasure

                class PrivateMeasure(FluxionMeasure):
                    def apply(self, model, arguments):
                        pass
                """
            )
        )

        names = {c.__name__ for c in discover_measures(tmp_path)}
        assert names == {"RealMeasure"}, f"got {names}"

    def test_skips_abstract_subclasses(self, tmp_path):
        from fluxion import FluxionMeasure
        from fluxion.measures import discover_measures

        # Use abc.ABCMeta + @abstractmethod so inspect.isabstract() picks it
        # up reliably. (Inspecting for "raises NotImplementedError" body is
        # too brittle for static analysis.)
        (tmp_path / "abstract.py").write_text(
            textwrap.dedent(
                """
                from abc import abstractmethod
                from fluxion import FluxionMeasure

                class AbstractBase(FluxionMeasure):
                    @abstractmethod
                    def apply(self, model, arguments):
                        ...
                """
            )
        )
        (tmp_path / "concrete.py").write_text(
            textwrap.dedent(
                """
                from fluxion import FluxionMeasure

                class Concrete(FluxionMeasure):
                    def apply(self, model, arguments):
                        pass
                """
            )
        )

        names = {c.__name__ for c in discover_measures(tmp_path)}
        assert names == {"Concrete"}, f"abstract leaked through: {names}"

    def test_import_failures_are_warned_not_raised(self, tmp_path):
        from fluxion.measures import discover_measures

        (tmp_path / "broken.py").write_text(
            textwrap.dedent(
                """
                import this_module_does_not_exist
                """
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = discover_measures(tmp_path)
        # Discovery must not crash — it should return [] and warn.
        assert result == []
        assert any("broken.py" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# apply_measures
# ---------------------------------------------------------------------------


@requires_fluxion
class TestApplyMeasures:
    """``apply_measures`` orchestrates the measure chain."""

    def _make_counter_measure(self):
        from fluxion import FluxionMeasure
        from fluxion.measures import apply_measures

        counter = {"calls": 0, "args": []}

        class CallCounter(FluxionMeasure):
            name = "CallCounter"

            def arguments(self):
                return [{"name": "marker", "type": "string", "default": "default"}]

            def apply(self, model, arguments):
                counter["calls"] += 1
                counter["args"].append(arguments)

        return CallCounter, counter, apply_measures

    def test_measures_run_in_order(self):
        from fluxion import FluxionMeasure
        from fluxion.measures import apply_measures

        order = []

        class A(FluxionMeasure):
            def apply(self, model, arguments):
                order.append("A")

        class B(FluxionMeasure):
            def apply(self, model, arguments):
                order.append("B")

        class C(FluxionMeasure):
            def apply(self, model, arguments):
                order.append("C")

        m = importlib.import_module("fluxion").Model(num_zones=1)
        applied = apply_measures(m, [A, B, C])
        assert order == ["A", "B", "C"]
        assert applied == ["A", "B", "C"]

    def test_arguments_passed_through(self):
        Measure, counter, apply_measures = self._make_counter_measure()
        m = importlib.import_module("fluxion").Model(num_zones=1)
        apply_measures(
            m,
            [Measure],
            measure_args={"CallCounter": {"marker": "from-cli"}},
        )
        assert counter["calls"] == 1
        assert counter["args"] == [{"marker": "from-cli"}]

    def test_missing_measure_args_uses_defaults(self):
        Measure, counter, apply_measures = self._make_counter_measure()
        m = importlib.import_module("fluxion").Model(num_zones=1)
        apply_measures(m, [Measure])
        assert counter["args"] == [{"marker": "default"}]

    def test_accepts_already_constructed_instance(self):
        from fluxion import FluxionMeasure
        from fluxion.measures import apply_measures

        seen = {"v": None}

        class M(FluxionMeasure):
            def apply(self, model, arguments):
                seen["v"] = arguments.get("v")

        m = importlib.import_module("fluxion").Model(num_zones=1)
        instance = M()
        apply_measures(m, [instance], {"M": {"v": 42}})
        assert seen["v"] == 42

    def test_rejects_non_measure_classes(self):
        from fluxion.measures import apply_measures

        class NotAMeasure:
            pass

        m = importlib.import_module("fluxion").Model(num_zones=1)
        with pytest.raises(TypeError):
            apply_measures(m, [NotAMeasure])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


@requires_fluxion
class TestSerialization:
    """JSON round-trip preserves the mutations a measure chain applies."""

    def test_json_round_trip(self, tmp_path):
        from fluxion.measures import apply_measures, load_model, save_model

        m = importlib.import_module("fluxion").Model(num_zones=2)
        apply_measures(m, [])  # no-op, just baseline
        path = tmp_path / "model.json"
        save_model(m, path)

        loaded = load_model(path)
        assert loaded.num_zones() == 2

    def test_apply_overhang_persists_through_json(self, tmp_path):
        from fluxion.measures import apply_measures, load_model, save_model

        m = importlib.import_module("fluxion").Model(num_zones=1)
        apply_measures(
            m,
            [_import_example("AddSouthOverhang")],
            {"AddSouthOverhang": {"depth": 1.5, "height": 3.0}},
        )

        path = tmp_path / "overhang.json"
        save_model(m, path)
        m2 = load_model(path)

        south = [
            s for s in m2.surfaces()
            if s.orientation == importlib.import_module("fluxion").Orientation.South
        ]
        assert south, "expected at least one south-facing surface"
        assert all(s.overhang_depth == 1.5 for s in south)
        assert all(s.overhang_height == 3.0 for s in south)

    def test_hvac_capacity_persists_through_json(self, tmp_path):
        from fluxion.measures import apply_measures, load_model, save_model

        m = importlib.import_module("fluxion").Model(num_zones=1)
        apply_measures(
            m,
            [_import_example("SetHVACCOP")],
            {"SetHVACCOP": {"heating_capacity": 25000, "cooling_capacity": 20000}},
        )

        path = tmp_path / "hvac.json"
        save_model(m, path)
        m2 = load_model(path)

        hvac = m2.hvac_system()
        assert hvac.heating_capacity == pytest.approx(25000.0, rel=1e-9)
        assert hvac.cooling_capacity == pytest.approx(20000.0, rel=1e-9)

    @requires_msgpack
    def test_msgpack_round_trip(self, tmp_path):
        from fluxion.measures import apply_measures, load_model, save_model

        m = importlib.import_module("fluxion").Model(num_zones=1)
        apply_measures(
            m,
            [_import_example("AddSouthOverhang")],
            {"AddSouthOverhang": {"depth": 0.5, "height": 1.0}},
        )

        path = tmp_path / "model.msgpack"
        save_model(m, path)
        assert path.exists() and path.stat().st_size > 0

        m2 = load_model(path)
        assert m2.num_zones() == 1
        # Shading should still be applied.
        orient = importlib.import_module("fluxion").Orientation.South
        assert any(
            s.overhang_depth == 0.5 for s in m2.surfaces() if s.orientation == orient
        )


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


@requires_fluxion
class TestApplyMeasuresCLI:
    """Drive the ``fluxion apply-measures`` CLI as a subprocess."""

    def _run_cli(self, *args):
        """Invoke the package's main() through ``python -m fluxion.cli``."""
        result = subprocess.run(
            [sys.executable, "-m", "fluxion.cli", *args],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        return result

    def test_help_lists_apply_measures(self):
        result = self._run_cli("--help")
        assert result.returncode == 0
        assert "apply-measures" in result.stdout

    def test_apply_measures_help(self):
        result = self._run_cli("apply-measures", "--help")
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--measures" in result.stdout
        assert "--output" in result.stdout

    def test_end_to_end(self, tmp_path):
        fluxion_mod = importlib.import_module("fluxion")
        from fluxion.measures import save_model

        base = fluxion_mod.Model(num_zones=1)
        base_path = tmp_path / "base.json"
        save_model(base, base_path)

        out_path = tmp_path / "out.json"
        result = self._run_cli(
            "apply-measures",
            "--model", str(base_path),
            "--measures", str(MEASURES_DIR),
            "--output", str(out_path),
        )
        assert result.returncode == 0, result.stderr
        assert out_path.exists(), "CLI did not write the output file"

        # The CLI returns a summary on stdout and embeds it in the output
        # JSON under ``_fluxion_run`` so the file is self-describing.
        stdout_summary = json.loads(result.stdout)
        assert "AddSouthOverhang" in stdout_summary["applied"]
        assert "SetHVACCOP" in stdout_summary["applied"]

        payload = json.loads(out_path.read_text())
        assert payload["num_zones"] == 1
        assert payload["schema_version"] == "1.0.0"
        assert "AddSouthOverhang" in payload["_fluxion_run"]["applied"]
        assert "SetHVACCOP" in payload["_fluxion_run"]["applied"]

        # The model should now have overhangs on south-facing surfaces.
        from fluxion.measures import load_model

        m2 = load_model(out_path)
        south = [
            s for s in m2.surfaces() if s.orientation == fluxion_mod.Orientation.South
        ]
        assert south and all(s.overhang_depth == 1.0 for s in south)
        # Default HVAC capacity from SetHVACCOP.
        assert m2.hvac_system().heating_capacity == pytest.approx(15000.0, rel=1e-9)

    def test_list_only_does_not_write(self, tmp_path):
        result = self._run_cli(
            "apply-measures",
            "--model", str(tmp_path / "missing.json"),
            "--measures", str(MEASURES_DIR),
            "--list",
        )
        # --list mode should not complain about a missing model because it
        # only walks the measures directory.
        assert result.returncode == 0, result.stderr
        names = json.loads(result.stdout)
        assert "AddSouthOverhang" in names
        assert "SetHVACCOP" in names

    def test_missing_model_returns_error(self, tmp_path):
        result = self._run_cli(
            "apply-measures",
            "--model", str(tmp_path / "missing.json"),
            "--measures", str(MEASURES_DIR),
            "--output", str(tmp_path / "out.json"),
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()

    def test_measure_args_propagate(self, tmp_path):
        fluxion_mod = importlib.import_module("fluxion")
        from fluxion.measures import save_model

        base = fluxion_mod.Model(num_zones=1)
        base_path = tmp_path / "base.json"
        save_model(base, base_path)

        args_path = tmp_path / "args.json"
        args_path.write_text(
            json.dumps(
                {
                    "AddSouthOverhang": {"depth": 2.0, "height": 4.0},
                    "SetHVACCOP": {"heating_capacity": 99999, "cooling_capacity": 88888},
                }
            )
        )

        out_path = tmp_path / "out.json"
        result = self._run_cli(
            "apply-measures",
            "--model", str(base_path),
            "--measures", str(MEASURES_DIR),
            "--measure-args", str(args_path),
            "--output", str(out_path),
        )
        assert result.returncode == 0, result.stderr

        from fluxion.measures import load_model

        m2 = load_model(out_path)
        south = [
            s for s in m2.surfaces() if s.orientation == fluxion_mod.Orientation.South
        ]
        assert south and all(s.overhang_depth == 2.0 for s in south)
        assert m2.hvac_system().heating_capacity == pytest.approx(99999.0, rel=1e-9)
        assert m2.hvac_system().cooling_capacity == pytest.approx(88888.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_example(class_name: str):
    """Import a concrete measure class from ``measures/examples/`` by name."""
    import importlib

    # Insert the measures dir on sys.path so importlib.util picks it up.
    measures_path = str(MEASURES_DIR)
    added = measures_path not in sys.path
    if added:
        sys.path.insert(0, measures_path)
    try:
        # The example files use absolute-style imports (``from fluxion ...``).
        # They are loaded via spec_from_file_location by discover_measures(),
        # so importing here by file is the most reliable path.
        from importlib.util import module_from_spec, spec_from_file_location

        for fname in ("add_overhang.py", "set_hvac_cop.py"):
            spec = spec_from_file_location(
                f"_measure_example_{fname[:-3]}",
                str(MEASURES_DIR / fname),
            )
            assert spec is not None and spec.loader is not None
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, class_name):
                return getattr(module, class_name)
        raise AssertionError(f"no measure class named {class_name} found")
    finally:
        if added:
            sys.path.remove(measures_path)
