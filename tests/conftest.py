import pytest


def pytest_configure(config):
    pytest.fluxion_available = False
    try:
        import importlib.util

        if importlib.util.find_spec("fluxion") is not None:
            pytest.fluxion_available = True
    except ImportError:
        pass


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

    return fluxion
