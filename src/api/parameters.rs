// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Type-safe building parameters with validation for Python API.
//!
//! This module provides a [`BuildingParameters`] struct that wraps building design
//! parameters with type safety and validation. It can be used directly from Python
//! and provides conversion methods for backward compatibility with existing `Vec<f64>` APIs.

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::{exceptions::PyValueError, PyResult, Python};

/// Type-safe building parameters with validation.
///
/// This struct provides named fields for building design parameters, improving
/// type safety and reducing misuse compared to raw `Vec<f64>` parameter vectors.
///
/// # Python Example
/// ```python
/// import fluxion
///
/// # Create valid parameters
/// params = fluxion.BuildingParameters(
///     window_u_value=1.5,
///     heating_setpoint=20.0,
///     cooling_setpoint=24.0
/// )
///
/// # Access fields
/// print(params.window_u_value)  # 1.5
///
/// # Convert to vector for compatibility
/// vec = params.to_vec()
/// print(vec)  # [1.5, 20.0, 24.0]
/// ```
///
/// # Field Constraints
/// - `window_u_value`: 0.1–5.0 W/m²K
/// - `heating_setpoint`: 15.0–25.0 °C
/// - `cooling_setpoint`: 22.0–32.0 °C
/// - Heating setpoint must be less than cooling setpoint
#[cfg_attr(feature = "python-bindings", pyclass)]
#[derive(Clone, Debug, PartialEq)]
pub struct BuildingParameters {
    /// Window U-value (thermal transmittance) in W/m²K.
    ///
    /// Range: 0.1–5.0 W/m²K
    /// Typical values: Single glass (5.0) to triple-pane low-E (0.1)
    pub window_u_value: f64,

    /// Heating setpoint temperature in °C.
    ///
    /// Range: 15.0–25.0 °C
    /// Typical values: 20.0 °C for office buildings
    pub heating_setpoint: f64,

    /// Cooling setpoint temperature in °C.
    ///
    /// Range: 22.0–32.0 °C
    /// Typical values: 24.0 °C for office buildings
    pub cooling_setpoint: f64,
}

impl BuildingParameters {
    // Physical constraints (matching BatchOracle constants)
    const MIN_U_VALUE: f64 = 0.1; // Minimum realistic U-value (W/m²K)
    const MAX_U_VALUE: f64 = 5.0; // Maximum realistic U-value
    const MIN_HEATING_SETPOINT: f64 = 15.0; // Min heating setpoint (°C)
    const MAX_HEATING_SETPOINT: f64 = 25.0; // Max heating setpoint (°C)
    const MIN_COOLING_SETPOINT: f64 = 22.0; // Min cooling setpoint (°C)
    const MAX_COOLING_SETPOINT: f64 = 32.0; // Max cooling setpoint (°C)

    /// Validates building parameter values against physical constraints.
    ///
    /// # Returns
    /// `Ok(())` if all values are valid, `Err(String)` with descriptive message otherwise.
    pub fn validate(
        window_u_value: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> Result<(), String> {
        // Check for NaN and infinity
        if !window_u_value.is_finite() {
            return Err(format!(
                "window_u_value must be finite, got {}",
                window_u_value
            ));
        }
        if !heating_setpoint.is_finite() {
            return Err(format!(
                "heating_setpoint must be finite, got {}",
                heating_setpoint
            ));
        }
        if !cooling_setpoint.is_finite() {
            return Err(format!(
                "cooling_setpoint must be finite, got {}",
                cooling_setpoint
            ));
        }

        // Check ranges
        if !(Self::MIN_U_VALUE..=Self::MAX_U_VALUE).contains(&window_u_value) {
            return Err(format!(
                "window_u_value must be in range [{}, {}] W/m²K, got {}",
                Self::MIN_U_VALUE,
                Self::MAX_U_VALUE,
                window_u_value
            ));
        }
        if !(Self::MIN_HEATING_SETPOINT..=Self::MAX_HEATING_SETPOINT).contains(&heating_setpoint) {
            return Err(format!(
                "heating_setpoint must be in range [{}, {}]°C, got {}",
                Self::MIN_HEATING_SETPOINT,
                Self::MAX_HEATING_SETPOINT,
                heating_setpoint
            ));
        }
        if !(Self::MIN_COOLING_SETPOINT..=Self::MAX_COOLING_SETPOINT).contains(&cooling_setpoint) {
            return Err(format!(
                "cooling_setpoint must be in range [{}, {}]°C, got {}",
                Self::MIN_COOLING_SETPOINT,
                Self::MAX_COOLING_SETPOINT,
                cooling_setpoint
            ));
        }

        // Check heating/cooling relationship
        if heating_setpoint >= cooling_setpoint {
            return Err(format!(
                "heating_setpoint ({}) must be less than cooling_setpoint ({})",
                heating_setpoint, cooling_setpoint
            ));
        }

        Ok(())
    }

    /// Create new BuildingParameters with validation.
    ///
    /// # Returns
    /// `Ok(BuildingParameters)` if all values are valid, `Err(String)` otherwise.
    pub fn new(
        window_u_value: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> Result<Self, String> {
        Self::validate(window_u_value, heating_setpoint, cooling_setpoint)?;
        Ok(BuildingParameters {
            window_u_value,
            heating_setpoint,
            cooling_setpoint,
        })
    }

    /// Convert building parameters to a vector for backward compatibility.
    ///
    /// # Returns
    /// `Vec<f64>` in the format: `[window_u_value, heating_setpoint, cooling_setpoint]`
    ///
    /// # Example
    /// ```ignore
    /// let params = BuildingParameters::new(1.5, 20.0, 24.0)?;
    /// let vec = params.to_vec();
    /// assert_eq!(vec, vec![1.5, 20.0, 24.0]);
    /// ```
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.window_u_value,
            self.heating_setpoint,
            self.cooling_setpoint,
        ]
    }
}

impl Default for BuildingParameters {
    fn default() -> Self {
        BuildingParameters {
            window_u_value: 2.0,    // Typical double-glazed window
            heating_setpoint: 20.0, // Typical office heating setpoint
            cooling_setpoint: 24.0, // Typical office cooling setpoint
        }
    }
}

/// Try to convert a `Vec<f64>` parameter vector to `BuildingParameters`.
///
/// This provides backward compatibility with existing code that uses `Vec<f64>` parameter vectors.
///
/// # Arguments
/// * `vec` - Parameter vector with at least 3 elements:
///   - `[0]`: Window U-value (W/m²K)
///   - `[1]`: Heating setpoint (°C)
///   - `[2]`: Cooling setpoint (°C)
///
/// # Returns
/// `Ok(BuildingParameters)` if valid, `Err(String)` if invalid length or values.
///
/// # Example
/// ```ignore
/// use fluxion::lib::parameters::BuildingParameters;
/// let vec = vec![1.5, 20.0, 24.0];
/// let params = BuildingParameters::try_from(vec)?;
/// assert_eq!(params.window_u_value, 1.5);
/// ```
impl TryFrom<Vec<f64>> for BuildingParameters {
    type Error = String;

    fn try_from(vec: Vec<f64>) -> Result<Self, Self::Error> {
        if vec.len() < 3 {
            return Err(format!(
                "Parameter vector must have at least 3 elements, got {}",
                vec.len()
            ));
        }

        Self::new(vec[0], vec[1], vec[2])
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl BuildingParameters {
    /// Create a new BuildingParameters instance from Python.
    ///
    /// # Arguments
    /// * `window_u_value` - Window U-value in W/m²K (range: 0.1-5.0)
    /// * `heating_setpoint` - Heating setpoint in °C (range: 15.0-25.0)
    /// * `cooling_setpoint` - Cooling setpoint in °C (range: 22.0-32.0)
    ///
    /// # Returns
    /// `BuildingParameters` instance
    ///
    /// # Raises
    /// `ValueError` if any parameter is out of range or invalid
    #[new]
    fn new_py(window_u_value: f64, heating_setpoint: f64, cooling_setpoint: f64) -> PyResult<Self> {
        Self::new(window_u_value, heating_setpoint, cooling_setpoint)
            .map_err(|e| PyValueError::new_err(e))
    }

    /// Convert to Python list for backward compatibility.
    ///
    /// # Returns
    /// `list[float64]` in format `[window_u_value, heating_setpoint, cooling_setpoint]`
    fn to_vec_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.to_vec().into_pyobject(py)?.into_any().unbind())
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!(
            "BuildingParameters(window_u_value={}, heating_setpoint={}, cooling_setpoint={})",
            self.window_u_value, self.heating_setpoint, self.cooling_setpoint
        )
    }

    /// String representation for printing.
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(feature = "python-bindings")]
use pyo3::{Py, PyAny};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_building_parameters_valid() {
        let params = BuildingParameters::new(1.5, 20.0, 24.0).unwrap();
        assert_eq!(params.window_u_value, 1.5);
        assert_eq!(params.heating_setpoint, 20.0);
        assert_eq!(params.cooling_setpoint, 24.0);
    }

    #[test]
    fn test_building_parameters_default() {
        let params = BuildingParameters::default();
        assert_eq!(params.window_u_value, 2.0);
        assert_eq!(params.heating_setpoint, 20.0);
        assert_eq!(params.cooling_setpoint, 24.0);
    }

    #[test]
    fn test_building_parameters_to_vec() {
        let params = BuildingParameters::new(1.5, 20.0, 24.0).unwrap();
        let vec = params.to_vec();
        assert_eq!(vec, vec![1.5, 20.0, 24.0]);
    }

    #[test]
    fn test_building_parameters_from_vec() {
        let vec = vec![1.5, 20.0, 24.0];
        let params = BuildingParameters::try_from(vec).unwrap();
        assert_eq!(params.window_u_value, 1.5);
        assert_eq!(params.heating_setpoint, 20.0);
        assert_eq!(params.cooling_setpoint, 24.0);
    }

    #[test]
    fn test_building_parameters_invalid_u_value_low() {
        let result = BuildingParameters::new(0.05, 20.0, 24.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("window_u_value must be in range [0.1, 5]"));
    }

    #[test]
    fn test_building_parameters_invalid_u_value_high() {
        let result = BuildingParameters::new(6.0, 20.0, 24.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("window_u_value must be in range [0.1, 5]"));
    }

    #[test]
    fn test_building_parameters_invalid_heating_setpoint_low() {
        let result = BuildingParameters::new(1.5, 14.0, 24.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("heating_setpoint must be in range [15, 25]"));
    }

    #[test]
    fn test_building_parameters_invalid_heating_setpoint_high() {
        let result = BuildingParameters::new(1.5, 26.0, 24.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("heating_setpoint must be in range [15, 25]"));
    }

    #[test]
    fn test_building_parameters_invalid_cooling_setpoint_low() {
        let result = BuildingParameters::new(1.5, 20.0, 21.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cooling_setpoint must be in range [22, 32]"));
    }

    #[test]
    fn test_building_parameters_invalid_cooling_setpoint_high() {
        let result = BuildingParameters::new(1.5, 20.0, 33.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("cooling_setpoint must be in range [22, 32]"));
    }

    #[test]
    fn test_building_parameters_heating_equals_cooling() {
        let result = BuildingParameters::new(1.5, 22.0, 22.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("heating_setpoint (22) must be less than cooling_setpoint (22)"));
    }

    #[test]
    fn test_building_parameters_heating_greater_than_cooling() {
        let result = BuildingParameters::new(1.5, 25.0, 24.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("heating_setpoint (25) must be less than cooling_setpoint (24)"));
    }

    #[test]
    fn test_building_parameters_nan_window_u_value() {
        let result = BuildingParameters::new(f64::NAN, 20.0, 24.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("window_u_value must be finite"));
    }

    #[test]
    fn test_building_parameters_infinite_heating_setpoint() {
        let result = BuildingParameters::new(1.5, f64::INFINITY, 24.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("heating_setpoint must be finite"));
    }

    #[test]
    fn test_building_parameters_from_vec_too_short() {
        let vec = vec![1.5, 20.0];
        let result = BuildingParameters::try_from(vec);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Parameter vector must have at least 3 elements"));
    }

    #[test]
    fn test_building_parameters_clone() {
        let params1 = BuildingParameters::new(1.5, 20.0, 24.0).unwrap();
        let params2 = params1.clone();
        assert_eq!(params1, params2);
    }

    #[test]
    fn test_building_parameters_partial_eq() {
        let params1 = BuildingParameters::new(1.5, 20.0, 24.0).unwrap();
        let params2 = BuildingParameters::new(1.5, 20.0, 24.0).unwrap();
        assert_eq!(params1, params2);

        let params3 = BuildingParameters::new(2.0, 20.0, 24.0).unwrap();
        assert_ne!(params1, params3);
    }
}
