"""NumPy array validation tests for PyO3 bindings

Tests validate NumPy array handling across the FFI boundary including:
- Shape preservation
- Dtype conversion (f32 vs f64)
- Error handling
- Large array handling
"""

import numpy as np
import pytest


@pytest.mark.needs_fluxion
class TestNumPyArrays:
    """Validate NumPy array integration with PyO3 bindings"""

    def test_array_shape_validation(self, fluxion_module):
        """Validate NumPy array shapes are preserved across FFI boundary"""
        # Test 1D array
        arr_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vf = fluxion_module.VectorField(arr_1d.tolist())
        result = vf.to_numpy()
        assert result.shape == (5,), f"Expected shape (5,), got {result.shape}"
        assert np.allclose(result, arr_1d), "1D array values not preserved"

        # Test 2D array (flattened to 1D for VectorField)
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        vf_2d = fluxion_module.VectorField(arr_2d.flatten().tolist())
        result_2d = vf_2d.to_numpy()
        assert result_2d.shape == (6,), f"Expected shape (6,), got {result_2d.shape}"
        assert np.allclose(result_2d, arr_2d.flatten()), "2D array values not preserved"

        # Test 3D array (flattened to 1D for VectorField)
        arr_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        vf_3d = fluxion_module.VectorField(arr_3d.flatten().tolist())
        result_3d = vf_3d.to_numpy()
        assert result_3d.shape == (8,), f"Expected shape (8,), got {result_3d.shape}"
        assert np.allclose(result_3d, arr_3d.flatten()), "3D array values not preserved"

    def test_array_dtype_conversion(self, fluxion_module):
        """Validate f32 vs f64 dtype handling"""
        # Test f32 array (should convert to f64 internally)
        arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vf_f32 = fluxion_module.VectorField(arr_f32.tolist())
        result_f32 = vf_f32.to_numpy()

        assert result_f32.dtype == np.float64, f"Expected f64, got {result_f32.dtype}"
        assert result_f32.shape == (3,), f"Expected shape (3,), got {result_f32.shape}"

        # Test f64 array
        arr_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        vf_f64 = fluxion_module.VectorField(arr_f64.tolist())
        result_f64 = vf_f64.to_numpy()

        assert result_f64.dtype == np.float64, f"Expected f64, got {result_f64.dtype}"
        assert result_f64.shape == (3,), f"Expected shape (3,), got {result_f64.shape}"

        # Verify both produce same results
        assert np.allclose(
            result_f32, result_f64
        ), "f32 and f64 should produce same results"

    def test_large_numpy_array_handling(self, fluxion_module):
        """Validate large arrays don't cause FFI issues"""
        # Test with 10,000+ elements
        large_data = np.arange(10000, dtype=np.float64).tolist()
        vf = fluxion_module.VectorField(large_data)

        # Verify length is preserved
        assert vf.len() == 10000, f"Expected length 10000, got {vf.len()}"

        # Verify we can convert back to numpy without issues
        result = vf.to_numpy()
        assert result.shape == (10000,), f"Expected shape (10000,), got {result.shape}"
        assert result.dtype == np.float64, f"Expected f64, got {result.dtype}"

        # Verify values are preserved
        expected = np.arange(10000, dtype=np.float64)
        assert np.allclose(result, expected), "Large array values not preserved"

        # Verify integration works (no segfaults)
        integral = vf.integrate()
        assert integral > 0.0, f"Expected positive integral, got {integral}"
        assert np.isfinite(integral), f"Integral should be finite, got {integral}"

    def test_empty_array_handling(self, fluxion_module):
        """Validate empty arrays are handled gracefully"""
        # Empty array should work (VectorField can handle 0 elements)
        arr_empty = np.array([], dtype=np.float64)
        vf = fluxion_module.VectorField(arr_empty.tolist())

        assert vf.len() == 0, f"Expected length 0, got {vf.len()}"

        # Convert back to numpy
        result = vf.to_numpy()
        assert result.shape == (0,), f"Expected shape (0,), got {result.shape}"

        # Integration should be 0 for empty array
        integral = vf.integrate()
        assert integral == 0.0, f"Expected integral 0.0, got {integral}"

    def test_nan_array_handling(self, fluxion_module):
        """Validate NaN values are handled correctly"""
        # Create array with NaN values
        arr_with_nan = np.array([1.0, np.nan, 3.0, 4.0, np.inf, -np.inf])
        vf = fluxion_module.VectorField(arr_with_nan.tolist())

        # Verify length is preserved
        assert vf.len() == 6, f"Expected length 6, got {vf.len()}"

        # Convert back to numpy
        result = vf.to_numpy()
        assert result.shape == (6,), f"Expected shape (6,), got {result.shape}"

        # Verify NaN and Inf values are preserved
        assert result[0] == 1.0, "First element should be 1.0"
        assert np.isnan(result[1]), "Second element should be NaN"
        assert result[2] == 3.0, "Third element should be 3.0"
        assert result[3] == 4.0, "Fourth element should be 4.0"
        assert np.isinf(result[4]) and result[4] > 0, "Fifth element should be +Inf"
        assert np.isinf(result[5]) and result[5] < 0, "Sixth element should be -Inf"

        # Integration should handle NaN/Inf gracefully
        integral = vf.integrate()
        assert not np.isfinite(
            integral
        ), f"Integral with NaN/Inf should not be finite, got {integral}"
