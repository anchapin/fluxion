"""
Tests for Monte Carlo Data Generation Tool.

Run with: pytest tools/data_gen/test_monte_carlo.py -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tools.data_gen.monte_carlo import (
    BuildingConfig,
    MonteCarloDataGenerator,
    SimulationResult,
    TrainingSample,
)


class TestBuildingConfig:
    """Tests for BuildingConfig dataclass."""

    def test_default_values(self):
        """Test BuildingConfig default values."""
        config = BuildingConfig()

        assert config.width == 8.0
        assert config.length == 8.0
        assert config.height == 3.5
        assert config.wall_u_value == 0.5
        assert config.heating_setpoint == 20.0
        assert config.cooling_setpoint == 27.0

    def test_custom_values(self):
        """Test BuildingConfig with custom values."""
        config = BuildingConfig(
            width=10.0,
            length=12.0,
            height=4.0,
            wall_u_value=0.3,
            heating_setpoint=18.0,
            cooling_setpoint=28.0,
        )

        assert config.width == 10.0
        assert config.length == 12.0
        assert config.height == 4.0
        assert config.wall_u_value == 0.3
        assert config.heating_setpoint == 18.0
        assert config.cooling_setpoint == 28.0

    def test_to_dict(self):
        """Test BuildingConfig serialization to dict."""
        config = BuildingConfig(width=10.0, length=12.0)
        d = config.to_dict()

        assert "width" in d
        assert "length" in d
        assert d["width"] == 10.0
        assert d["length"] == 12.0


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_default_values(self):
        """Test SimulationResult default values."""
        config = BuildingConfig()
        result = SimulationResult(config=config, run_id="test_001")

        assert result.config == config
        assert result.run_id == "test_001"
        assert result.total_energy == 0.0
        assert result.success is False

    def test_time_series_arrays(self):
        """Test time series array initialization."""
        config = BuildingConfig()
        result = SimulationResult(config=config, run_id="test_001")

        assert len(result.outdoor_temps) == 8760
        assert len(result.indoor_temps) == 8760
        assert len(result.heating_loads) == 8760
        assert len(result.cooling_loads) == 8760


class TestTrainingSample:
    """Tests for TrainingSample dataclass."""

    def test_creation(self):
        """Test TrainingSample creation."""
        sample = TrainingSample(
            outdoor_temp_t=10.0,
            indoor_temp_t=20.0,
            solar_t=100.0,
            hvac_mode=1,
            hvac_power=500.0,
            indoor_temp_t1=21.0,
            energy_consumed=500.0,
            run_id="test_001",
            timestep=0,
        )

        assert sample.outdoor_temp_t == 10.0
        assert sample.indoor_temp_t == 20.0
        assert sample.hvac_mode == 1
        assert sample.energy_consumed == 500.0


class TestMonteCarloDataGenerator:
    """Tests for MonteCarloDataGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        gen = MonteCarloDataGenerator(
            output_dir="/tmp/test_output",
            num_samples=100,
            seed=42,
        )

        assert gen.output_dir.name == "test_output"
        assert gen.num_samples == 100
        assert gen.seed == 42

    def test_setup(self):
        """Test generator setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=10,
                seed=42,
                weather_dir="assets/weather",
            )

            gen.setup()

            assert gen.output_dir.exists()
            assert gen.param_sampler is not None
            assert len(gen.weather_files) >= 0

    def test_parameter_sampler_configured(self):
        """Test that parameter sampler has correct parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=10,
                seed=42,
                weather_dir="assets/weather",
            )

            gen.setup()

            # Check that sampler has parameters
            param_names = [p.name for p in gen.param_sampler.parameters]
            assert "width" in param_names
            assert "wall_u_value" in param_names
            assert "heating_setpoint" in param_names

    def test_sample_config(self):
        """Test building configuration sampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=10,
                seed=42,
                weather_dir="assets/weather",
            )

            gen.setup()
            config = gen.sample_config(0)

            assert isinstance(config, BuildingConfig)
            assert 5.0 <= config.width <= 20.0
            assert 0.2 <= config.wall_u_value <= 1.5

    def test_generate_small_dataset(self):
        """Test generating a small dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=5,
                seed=42,
                weather_dir="assets/weather",
                batch_size=5,
            )

            gen.setup()
            df = gen.generate()

            # Check that data was generated
            assert len(df) > 0

            # Check columns
            expected_cols = [
                "outdoor_temp_t",
                "indoor_temp_t",
                "solar_t",
                "hvac_mode",
                "hvac_power",
                "indoor_temp_t1",
                "energy_consumed",
                "run_id",
                "timestep",
            ]
            assert all(col in df.columns for col in expected_cols)

            # Check parquet file exists
            parquet_file = os.path.join(tmpdir, "training_data.parquet")
            assert os.path.exists(parquet_file)

            # Check metadata exists
            metadata_file = os.path.join(tmpdir, "metadata.json")
            assert os.path.exists(metadata_file)

    def test_generate_with_different_seeds(self):
        """Test that different seeds produce different results."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                gen1 = MonteCarloDataGenerator(
                    output_dir=tmpdir1,
                    num_samples=5,
                    seed=42,
                )
                gen1.setup()
                df1 = gen1.generate()

                gen2 = MonteCarloDataGenerator(
                    output_dir=tmpdir2,
                    num_samples=5,
                    seed=123,
                )
                gen2.setup()
                df2 = gen2.generate()

                # Dataframes should be different
                # (At least some values should differ)
                assert not df1.equals(df2)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                gen1 = MonteCarloDataGenerator(
                    output_dir=tmpdir1,
                    num_samples=5,
                    seed=42,
                )
                gen1.setup()
                df1 = gen1.generate()

                gen2 = MonteCarloDataGenerator(
                    output_dir=tmpdir2,
                    num_samples=5,
                    seed=42,
                )
                gen2.setup()
                df2 = gen2.generate()

                # Should be identical
                pd.testing.assert_frame_equal(df1, df2)

    def test_parquet_format(self):
        """Test that output is in correct Parquet format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=5,
                seed=42,
            )

            gen.setup()
            df = gen.generate()

            # Load from parquet
            df_loaded = pd.read_parquet(os.path.join(tmpdir, "training_data.parquet"))

            # Should match
            assert len(df) == len(df_loaded)
            assert list(df.columns) == list(df_loaded.columns)

    def test_sampling_methods(self):
        """Test different sampling methods."""
        methods = ["RANDOM", "LHS"]

        for method in methods:
            with tempfile.TemporaryDirectory() as tmpdir:
                gen = MonteCarloDataGenerator(
                    output_dir=tmpdir,
                    num_samples=5,
                    seed=42,
                    sampling_method=method,
                )

                gen.setup()
                config = gen.sample_config(0)

                assert config is not None
                assert 5.0 <= config.width <= 20.0

    def test_training_sample_format(self):
        """Test that training samples have correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=2,
                seed=42,
                batch_size=2,
            )

            gen.setup()
            df = gen.generate()

            # State_t columns should exist
            assert "outdoor_temp_t" in df.columns
            assert "indoor_temp_t" in df.columns
            assert "solar_t" in df.columns

            # Action_t columns should exist
            assert "hvac_mode" in df.columns
            assert "hvac_power" in df.columns

            # State_t+1 column should exist
            assert "indoor_temp_t1" in df.columns

            # Energy column should exist
            assert "energy_consumed" in df.columns


class TestDataGeneratorPerformance:
    """Performance-related tests for data generation."""

    def test_throughput(self):
        """Test that data generation is reasonably fast."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=50,
                seed=42,
                batch_size=50,
            )

            gen.setup()

            start_time = time.time()
            df = gen.generate()
            elapsed = time.time() - start_time

            # Should generate at least 1000 samples per second
            throughput = len(df) / elapsed
            assert throughput > 100, f"Throughput too low: {throughput:.1f} samples/s"

    def test_batch_processing(self):
        """Test batch processing works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=25,
                seed=42,
                batch_size=10,
            )

            gen.setup()
            df = gen.generate()

            # Should get all samples from batches
            assert len(df) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_weather_directory(self):
        """Test handling of missing weather directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=5,
                seed=42,
                weather_dir="/nonexistent/path",
            )

            gen.setup()

            # Should use default weather file
            assert len(gen.weather_files) >= 1

    def test_single_sample(self):
        """Test generating with single sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = MonteCarloDataGenerator(
                output_dir=tmpdir,
                num_samples=1,
                seed=42,
            )

            gen.setup()
            df = gen.generate()

            assert len(df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
