import pytest


def test_run_oracle_if_available():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    # Create an oracle and run a very small population to validate API and non-zero outputs
    oracle = fluxion.BatchOracle()
    pop = [[1.5, 21.0], [2.0, 22.0]]
    results = oracle.evaluate_population(pop, False)

    assert isinstance(results, list)
    assert len(results) == 2
    # Values should be numeric and non-negative
    assert all(isinstance(x, float) for x in results)
    assert all(x >= 0.0 for x in results)
