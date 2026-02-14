import pytest


def test_run_oracle_if_available():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    # Create an oracle and run a very small population to validate API and
    # non-zero outputs
    oracle = fluxion.BatchOracle()
    pop = [[1.5, 21.0, 27.0], [2.0, 22.0, 28.0]]
    results = oracle.evaluate_population(pop, False)

    assert isinstance(results, list)
    assert len(results) == 2
    # Values should be numeric and non-negative
    assert all(isinstance(x, float) for x in results)
    assert all(x >= 0.0 for x in results)


def test_batch_oracle_with_surrogates():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    oracle = fluxion.BatchOracle()
    pop = [[1.5, 21.0, 27.0], [2.0, 22.0, 28.0], [1.0, 23.0, 29.0]]
    results = oracle.evaluate_population(pop, True)

    assert len(results) == 3
    assert all(isinstance(x, float) for x in results)
    assert all(x >= 0.0 for x in results)


def test_batch_oracle_single_candidate():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    oracle = fluxion.BatchOracle()
    pop = [[1.5, 21.0, 27.0]]
    results = oracle.evaluate_population(pop, False)

    assert len(results) == 1
    assert isinstance(results[0], float)
    assert results[0] >= 0.0


def test_batch_oracle_large_population():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    oracle = fluxion.BatchOracle()
    # Create a larger population
    pop = [[0.5 + i * 0.1, 19.0 + i * 0.5, 25.0 + i * 0.5] for i in range(10)]
    results = oracle.evaluate_population(pop, False)

    assert len(results) == 10
    assert all(isinstance(x, float) for x in results)
    assert all(x >= 0.0 for x in results)


def test_model_single_building():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    model = fluxion.Model("config.json")
    result = model.simulate(1, False)

    assert isinstance(result, float)
    assert result >= 0.0


def test_model_with_surrogates():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    model = fluxion.Model("config.json")
    result = model.simulate(1, True)

    assert isinstance(result, float)
    assert result >= 0.0


def test_model_multiple_years():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    model = fluxion.Model("config.json")
    result_1yr = model.simulate(1, False)
    result_2yr = model.simulate(2, False)

    assert result_1yr >= 0.0
    assert result_2yr >= 0.0


def test_parameter_variations():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    oracle = fluxion.BatchOracle()

    # Test boundary values
    boundary_pop = [
        [0.5, 19.0, 25.0],  # Min values
        [3.0, 24.0, 30.0],  # Max values
        [1.5, 21.5, 27.5],  # Mid values
    ]
    results = oracle.evaluate_population(boundary_pop, False)

    assert len(results) == 3
    assert all(x >= 0.0 for x in results)


def test_empty_population_handling():
    try:
        import fluxion
    except Exception:
        pytest.skip("fluxion Python bindings not available; skip example test")

    oracle = fluxion.BatchOracle()
    pop = []
    results = oracle.evaluate_population(pop, False)

    assert isinstance(results, list)
    assert len(results) == 0
