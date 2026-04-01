"""
Statistical Metrics for Component-Level Validation.

This module provides statistical metrics for comparing Fluxion and EnergyPlus
simulation results, including:
- RMSE (Root Mean Square Error)
- NMBE (Normalized Mean Bias Error)
- R² (Coefficient of Determination)
- CV(RMSE) (Coefficient of Variation of RMSE)
- Hourly error analysis

These metrics follow ASHRAE Guideline 14 and IPMVP standards.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MetricsResult:
    """Result of statistical metrics calculation."""

    rmse: float  # Root Mean Square Error
    rmse_units: str  # Units of RMSE (e.g., "W", "°C", "W/m²")
    nmbe: float  # Normalized Mean Bias Error (%)
    cv_rmse: float  # Coefficient of Variation of RMSE (%)
    r_squared: float  # Coefficient of determination (0-1)
    mean_error: float  # Mean error (bias)
    max_error: float  # Maximum absolute error
    min_error: float  # Minimum error
    mae: float  # Mean Absolute Error

    # Additional diagnostics
    n_points: int  # Number of data points
    reference_mean: float  # Mean of reference data
    reference_std: float  # Standard deviation of reference data

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rmse": self.rmse,
            "rmse_units": self.rmse_units,
            "nmbe_percent": self.nmbe,
            "cv_rmse_percent": self.cv_rmse,
            "r_squared": self.r_squared,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "min_error": self.min_error,
            "mae": self.mae,
            "n_points": self.n_points,
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
        }

    def passes_criteria(
        self,
        rmse_threshold: Optional[float] = None,
        nmbe_threshold: Optional[float] = None,
        r_squared_threshold: Optional[float] = None,
    ) -> bool:
        """Check if metrics pass acceptance criteria.

        Args:
            rmse_threshold: Maximum acceptable RMSE (absolute, in same units)
            nmbe_threshold: Maximum acceptable NMBE (%)
            r_squared_threshold: Minimum acceptable R²

        Returns:
            True if all specified criteria are met
        """
        if rmse_threshold is not None and self.rmse > rmse_threshold:
            return False
        if nmbe_threshold is not None and abs(self.nmbe) > nmbe_threshold:
            return False
        if r_squared_threshold is not None and self.r_squared < r_squared_threshold:
            return False
        return True


def calculate_rmse(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Square Error.

    Args:
        reference: Reference data (EnergyPlus)
        predicted: Predicted data (Fluxion)

    Returns:
        RMSE in same units as input data
    """
    if len(reference) != len(predicted):
        raise ValueError(f"Array length mismatch: {len(reference)} vs {len(predicted)}")

    return np.sqrt(np.mean((predicted - reference) ** 2))


def calculate_nmbe(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Normalized Mean Bias Error.

    NMBE = (mean(predicted - reference) / mean(reference)) × 100%

    Positive NMBE = Fluxion overestimates
    Negative NMBE = Fluxion underestimates

    Args:
        reference: Reference data
        predicted: Predicted data

    Returns:
        NMBE as percentage
    """
    mean_ref = np.mean(reference)
    if abs(mean_ref) < 1e-10:
        # Avoid division by zero - return raw bias
        return np.mean(predicted - reference) * 100.0

    mean_bias = np.mean(predicted - reference)
    return (mean_bias / mean_ref) * 100.0


def calculate_cv_rmse(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Coefficient of Variation of RMSE.

    CV(RMSE) = (RMSE / mean(reference)) × 100%

    Args:
        reference: Reference data
        predicted: Predicted data

    Returns:
        CV(RMSE) as percentage
    """
    mean_ref = np.mean(reference)
    if abs(mean_ref) < 1e-10:
        return 0.0

    rmse = calculate_rmse(reference, predicted)
    return (rmse / abs(mean_ref)) * 100.0


def calculate_r_squared(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot
    where:
        SS_res = Σ(predicted - reference)²
        SS_tot = Σ(reference - mean(reference))²

    R² = 1.0: Perfect fit
    R² = 0.0: No better than mean
    R² < 0.0: Worse than mean

    Args:
        reference: Reference data
        predicted: Predicted data

    Returns:
        R² value (can be negative for very poor fits)
    """
    ss_res = np.sum((predicted - reference) ** 2)
    ss_tot = np.sum((reference - np.mean(reference)) ** 2)

    if ss_tot < 1e-10:
        # No variance in reference data
        return 1.0 if ss_res < 1e-10 else 0.0

    return 1.0 - (ss_res / ss_tot)


def calculate_mae(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error.

    MAE = mean(|predicted - reference|)

    Args:
        reference: Reference data
        predicted: Predicted data

    Returns:
        MAE in same units as input data
    """
    return np.mean(np.abs(predicted - reference))


def calculate_all_metrics(
    reference: np.ndarray, predicted: np.ndarray, units: str = "W"
) -> MetricsResult:
    """Calculate all statistical metrics at once.

    Args:
        reference: Reference data (EnergyPlus)
        predicted: Predicted data (Fluxion)
        units: Units string for reporting

    Returns:
        MetricsResult dataclass with all metrics
    """
    # Ensure numpy arrays
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Calculate errors
    errors = predicted - reference

    # Calculate all metrics
    rmse = calculate_rmse(reference, predicted)
    nmbe = calculate_nmbe(reference, predicted)
    cv_rmse = calculate_cv_rmse(reference, predicted)
    r_squared = calculate_r_squared(reference, predicted)
    mae = calculate_mae(reference, predicted)

    return MetricsResult(
        rmse=rmse,
        rmse_units=units,
        nmbe=nmbe,
        cv_rmse=cv_rmse,
        r_squared=r_squared,
        mean_error=np.mean(errors),
        max_error=np.max(errors),
        min_error=np.min(errors),
        mae=mae,
        n_points=len(reference),
        reference_mean=np.mean(reference),
        reference_std=np.std(reference),
    )


def hourly_error_analysis(
    reference: np.ndarray,
    predicted: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> dict:
    """Perform hourly error analysis.

    Args:
        reference: Reference data
        predicted: Predicted data
        timestamps: Optional array of timestamps

    Returns:
        Dictionary with hourly error statistics
    """
    errors = predicted - reference
    abs_errors = np.abs(errors)

    # Basic statistics
    stats = {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "median_error": float(np.median(errors)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "max_abs_error": float(np.max(abs_errors)),
    }

    # Percentile analysis
    stats["p50_error"] = float(np.percentile(abs_errors, 50))
    stats["p90_error"] = float(np.percentile(abs_errors, 90))
    stats["p95_error"] = float(np.percentile(abs_errors, 95))
    stats["p99_error"] = float(np.percentile(abs_errors, 99))

    # Error distribution
    stats["n_overestimate"] = int(np.sum(errors > 0))
    stats["n_underestimate"] = int(np.sum(errors < 0))
    stats["n_zero_error"] = int(np.sum(np.abs(errors) < 1e-6))

    # Add timestamp info if provided
    if timestamps is not None:
        max_idx = np.argmax(abs_errors)
        stats["max_error_timestamp"] = str(timestamps[max_idx])

        min_idx = np.argmin(errors)
        stats["max_overestimate_timestamp"] = str(timestamps[min_idx])

        max_idx_pos = np.argmax(errors)
        stats["max_underestimate_timestamp"] = str(timestamps[max_idx_pos])

    return stats


def time_series_comparison(
    reference: np.ndarray,
    predicted: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    units: str = "W",
) -> dict:
    """Generate comprehensive time series comparison report.

    Args:
        reference: Reference data
        predicted: Predicted data
        timestamps: Optional timestamps
        units: Units string

    Returns:
        Dictionary with full comparison report
    """
    metrics = calculate_all_metrics(reference, predicted, units)
    hourly = hourly_error_analysis(reference, predicted, timestamps)

    return {
        "metrics": metrics.to_dict(),
        "hourly_analysis": hourly,
        "passes_ashrae_140": metrics.passes_criteria(
            rmse_threshold=10.0 if units == "W/m²" else 200.0,
            nmbe_threshold=10.0,
            r_squared_threshold=0.90,
        ),
    }


# ASHRAE Guideline 14 calibration thresholds
ASHRAE_140_THRESHOLDS = {
    "hourly": {
        "nmbe_max": 10.0,  # %
        "cv_rmse_max": 30.0,  # %
    },
    "monthly": {
        "nmbe_max": 5.0,  # %
        "cv_rmse_max": 15.0,  # %
    },
}


def check_ashrae_guideline_14(
    metrics: MetricsResult, frequency: str = "hourly"
) -> Tuple[bool, str]:
    """Check if model meets ASHRAE Guideline 14 calibration criteria.

    Args:
        metrics: Calculated metrics
        frequency: "hourly" or "monthly"

    Returns:
        (passes, message) tuple
    """
    thresholds = ASHRAE_140_THRESHOLDS.get(frequency, ASHRAE_140_THRESHOLDS["hourly"])

    issues = []
    if abs(metrics.nmbe) > thresholds["nmbe_max"]:
        issues.append(f"NMBE={metrics.nmbe:.1f}% exceeds {thresholds['nmbe_max']}%")
    if metrics.cv_rmse > thresholds["cv_rmse_max"]:
        issues.append(
            f"CV(RMSE)={metrics.cv_rmse:.1f}% exceeds {thresholds['cv_rmse_max']}%"
        )

    if issues:
        return False, f"Failed: {', '.join(issues)}"
    else:
        return True, "Passed ASHRAE Guideline 14 calibration criteria"


if __name__ == "__main__":
    # Example usage
    print("Statistical Metrics Module - Example Usage")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)
    reference = np.random.randn(8760) * 100 + 500  # EnergyPlus data
    predicted = reference + np.random.randn(8760) * 10  # Fluxion data with noise

    # Calculate metrics
    metrics = calculate_all_metrics(reference, predicted, units="W")

    print("\nMetrics:")
    print(f"  RMSE: {metrics.rmse:.2f} {metrics.rmse_units}")
    print(f"  NMBE: {metrics.nmbe:.2f}%")
    print(f"  CV(RMSE): {metrics.cv_rmse:.2f}%")
    print(f"  R²: {metrics.r_squared:.4f}")
    print(f"  MAE: {metrics.mae:.2f} {metrics.rmse_units}")

    # Check ASHRAE Guideline 14
    passes, msg = check_ashrae_guideline_14(metrics, "hourly")
    print(f"\nASHRAE Guideline 14 (hourly): {msg}")
