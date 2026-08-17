import warnings

import pytest


def pytest_configure(config):
    pytest.fluxion_available = False
    pytest.fluxion_native_available = False
    pytest.fluxion_native_error = None
    try:
        import importlib.util

        if importlib.util.find_spec("fluxion") is not None:
            pytest.fluxion_available = True
            # Distinguish "Python wrapper is importable" from "native
            # extension actually loaded". `fluxion/__init__.py` exposes
            # `_NATIVE_IMPORT_ERROR` as the sentinel: it is ``None`` when
            # the compiled extension loaded successfully and an
            # ``ImportError`` instance otherwise. Tests that require the
            # native bindings should gate on this (e.g. via the
            # ``fluxion_module`` fixture below) rather than on the bare
            # presence of the wrapper module — otherwise the wrapper
            # imports cleanly but every binding test silently skips.
            # See Issue #2852.
            import fluxion as _fluxion_probe  # noqa: PLC0415

            _native_err = getattr(_fluxion_probe, "_NATIVE_IMPORT_ERROR", None)
            if _native_err is None:
                pytest.fluxion_native_available = True
            else:
                pytest.fluxion_native_error = _native_err
    except ImportError:
        pass

    # Issue #2852: emit a single loud warning at session start when the
    # native extension is missing so contributors see immediately that
    # their `pytest tests/python/` run is degraded — without this, the
    # 43+ native-binding tests below silently skip and a green pytest
    # pass looks indistinguishable from a real "all green" run.
    if pytest.fluxion_available and not pytest.fluxion_native_available:
        _err = pytest.fluxion_native_error
        warnings.warn(
            "fluxion native extension is unavailable"
            + (f" ({_err})" if _err is not None else "")
            + "; Python binding tests will SKIP. "
            "Run `maturin develop` (or `maturin develop --release`) to "
            "build and install the compiled extension before re-running "
            "`pytest tests/python/`. See Issue #2852.",
            stacklevel=1,
        )


def pytest_collection_modifyitems(config, items):
    if not getattr(pytest, "fluxion_available", False):
        skip_marker = pytest.mark.skip(reason="fluxion Python bindings not available")
        for item in items:
            if item.get_closest_marker("needs_fluxion"):
                item.add_marker(skip_marker)


@pytest.fixture(scope="module")
def fluxion_module():
    """
    Load and return the fluxion Python module.

    This fixture provides access to the fluxion module for integration tests.
    It imports the module once per test session (module scope) for efficiency.

    Tests that depend on this fixture should be marked with @pytest.mark.needs_fluxion
    to skip gracefully when fluxion is not installed.

    Returns:
        module: The fluxion Python module, providing access to BatchOracle, Model, VectorField, etc.

    Example:
        def test_example(fluxion_module):
            oracle = fluxion_module.BatchOracle()
            results = oracle.evaluate_population([[1.5, 20.0, 24.0]], False)
            assert len(results) == 1
    """
    import fluxion

    # Issue #2852: gate on the native-extension sentinel, not on the bare
    # presence of the wrapper module. Without this, the wrapper imports
    # cleanly but `fluxion.Model(...)` etc. raise ImportError from the
    # `__getattr__` shim, which causes the test to error mid-fixture
    # rather than skip with a clear message.
    _native_err = getattr(fluxion, "_NATIVE_IMPORT_ERROR", None)
    if _native_err is not None:
        pytest.skip(
            f"fluxion native extension is unavailable ({_native_err}); "
            "run `maturin develop` to enable this test."
        )
    return fluxion
