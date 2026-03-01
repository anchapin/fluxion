import pytest
import sys
from pathlib import Path


def pytest_configure(config):
    pytest.fluxion_available = False
    try:
        import fluxion

        pytest.fluxion_available = True
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    if not getattr(pytest, "fluxion_available", False):
        skip_marker = pytest.mark.skip(reason="fluxion Python bindings not available")
        for item in items:
            if item.get_closest_marker("needs_fluxion"):
                item.add_marker(skip_marker)
