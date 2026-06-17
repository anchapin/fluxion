#!/usr/bin/env python3
"""
Surrogate Model Validation Script (v3.0)

Validates trained ONNX surrogate models against physics baselines.
Compares neural surrogate predictions with analytical/numerical physics calculations
to ensure RMSE < 2% and inference time < 1ms.

Usage:
    python scripts/validate_surrogate.py --component zone_thermal --model models/surrogate_zone_thermal.onnx
    python scripts/validate_surrogate.py --all-models --model-dir models/
    python scripts/validate_surrogate.py --physics-baseline data/physics_baseline.csv

Output:
    models/surrogate_{component}_validation.json - Detailed validation report
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    component: str
    model_path: str
    physics_baseline: str
    n_test_samples: int
    rmse: float
    rmse_normalized: float
    mae: float
    r2: float
    max_error: float
    inference_time_ms: float
    inference_time_p95_ms: float
    rmse_target_met: bool
    inference_target_met: bool
    all_targets_met: bool
    physics_output_mean: float
    physics_output_std: float
    neural_output_mean: float
    neural_output_std: float
    correlation: float
    validation_timestamp: str
    errors: List[str]


def generate_physics_baseline(
    component: str,
    n_samples: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate physics-based baseline predictions for validation.
    Uses first-principles equations to compute expected values.
    """
    np.random.seed(random_state)

    if component == "zone_thermal":
        exterior = np.random.uniform(-10, 40, n_samples)
        zone = np.random.uniform(18, 26, n_samples)
        solar = np.random.uniform(0, 800, n_samples)
        humidity = np.random.uniform(20, 80, n_samples)
        occupancy = np.random.uniform(0, 5, n_samples)
        hvac_mode = np.random.choice([0, 1, 2], n_samples)
        climate = np.random.uniform(0, 8, n_samples)

        X = np.column_stack([exterior, zone, solar, humidity, occupancy, hvac_mode, climate])

        delta_t = zone - exterior
        base_load = 0.5 * delta_t + 0.01 * solar + 0.1 * occupancy
        hvac_effect = np.where(hvac_mode == 1, 2.0, np.where(hvac_mode == 2, -1.5, 0.0))
        base_load = base_load + hvac_effect + np.random.normal(0, 0.1, n_samples)
        y = base_load.reshape(-1, 1)

    elif component == "solar_gain":
        lat = np.random.uniform(25, 50, n_samples)
        lon = np.random.uniform(-120, -70, n_samples)
        doy = np.random.randint(1, 366, n_samples)
        hour = np.random.randint(0, 24, n_samples)
        tilt = np.random.choice([0, 30, 45, 60, 90], n_samples)
        az = np.random.uniform(0, 360, n_samples)
        dni = np.random.uniform(0, 1000, n_samples)
        dhi = np.random.uniform(0, 300, n_samples)

        X = np.column_stack([lat, lon, doy, hour, tilt, az, dni, dhi])

        dec = 23.45 * np.sin(2 * np.pi * (doy - 81) / 365)
        hra = 15 * (hour - 12)
        cos_zenith = np.sin(np.radians(lat)) * np.sin(np.radians(dec)) + \
                    np.cos(np.radians(lat)) * np.cos(np.radians(dec)) * np.cos(np.radians(hra))
        zenith = np.arccos(np.clip(cos_zenith, -1, 1))
        effective_irrad = (dni * np.cos(zenith) + dhi) * np.cos(np.radians(tilt))
        gain = np.clip(effective_irrad * 0.85, 0, 1200) + np.random.normal(0, 5, n_samples)
        y = gain.reshape(-1, 1)

    elif component == "conduction":
        exterior = np.random.uniform(-10, 40, n_samples)
        interior = np.random.uniform(18, 26, n_samples)
        u_val = np.random.uniform(0.1, 2.5, n_samples)
        area = np.random.uniform(10, 100, n_samples)
        mass = np.random.uniform(50, 500, n_samples)
        emiss = np.random.uniform(0.7, 0.95, n_samples)

        X = np.column_stack([exterior, interior, u_val, area, mass, emiss])

        delta_t = interior - exterior
        q_conv = u_val * area * delta_t
        q_rad = 5.67e-8 * emiss * area * ((interior + 273)**4 - (exterior + 273)**4) * 1e-8
        total_flux = q_conv + q_rad * 0.1 + np.random.normal(0, 5, n_samples)
        y = total_flux.reshape(-1, 1)

    elif component == "ventilation":
        exterior = np.random.uniform(-10, 40, n_samples)
        interior = np.random.uniform(18, 26, n_samples)
        wind = np.random.uniform(0, 10, n_samples)
        pressure = np.random.uniform(99000, 101500, n_samples)
        vent_rate = np.random.uniform(0, 500, n_samples)
        ach = np.random.uniform(0, 10, n_samples)

        X = np.column_stack([exterior, interior, wind, pressure, vent_rate, ach])

        delta_t = interior - exterior
        sensible = 0.34 * vent_rate * delta_t / 3600
        latent = 0.68 * vent_rate * 0.01 * (exterior - interior) / 3600
        wind_effect = 0.05 * wind * vent_rate / 3600
        total_load = sensible + latent + wind_effect + np.random.normal(0, 2, n_samples)
        y = total_load.reshape(-1, 1)

    else:
        raise ValueError(f"Unknown component: {component}")

    return X, y


def run_neural_inference(
    model_path: Path,
    X: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Run neural network inference using onnxruntime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start = time.perf_counter()
    y_pred = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    inference_time = (time.perf_counter() - start) * 1000

    return y_pred, inference_time


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Compute validation metrics."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    max_error = np.max(np.abs(y_true - y_pred))

    y_range = y_true.max() - y_true.min()
    rmse_normalized = rmse / y_range if y_range > 0 else 0.0

    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    return {
        "rmse": float(rmse),
        "rmse_normalized": float(rmse_normalized),
        "mae": float(mae),
        "r2": float(r2),
        "max_error": float(max_error),
        "correlation": float(correlation),
    }


def validate_component(
    component: str,
    model_path: Path,
    n_test_samples: int = 1000,
    rmse_target: float = 0.02,
    inference_target_ms: float = 1.0,
) -> ValidationMetrics:
    """Validate a single component model."""
    logger.info(f"Validating {component}...")

    X, y_physics = generate_physics_baseline(component, n_test_samples)

    y_neural, inference_time = run_neural_inference(model_path, X)

    metrics = compute_metrics(y_physics, y_neural)

    inference_times = []
    for _ in range(10):
        _, t = run_neural_inference(model_path, X[:10])
        inference_times.append(t)
    mean_inference_time = np.mean(inference_times)
    inference_time_p95 = np.percentile(inference_times, 95)

    single_pred_time = []
    for i in range(100):
        single_X = X[i:i+1]
        start = time.perf_counter()
        _ = run_neural_inference(model_path, single_X)
        single_pred_time.append((time.perf_counter() - start) * 1000)
    single_pred_p95 = np.percentile(single_pred_time, 95)

    rmse_target_met = bool(metrics["rmse_normalized"] < rmse_target)
    inference_target_met = bool(single_pred_p95 < inference_target_ms)

    logger.info(f"  Physics mean: {y_physics.mean():.4f}, std: {y_physics.std():.4f}")
    logger.info(f"  Neural mean: {y_neural.mean():.4f}, std: {y_neural.std():.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  Normalized RMSE: {metrics['rmse_normalized']:.4f} ({metrics['rmse_normalized']*100:.2f}%)")
    logger.info(f"  R²: {metrics['r2']:.6f}")
    logger.info(f"  MAE: {metrics['mae']:.6f}")
    logger.info(f"  Max error: {metrics['max_error']:.6f}")
    logger.info(f"  Correlation: {metrics['correlation']:.6f}")
    logger.info(f"  Single pred P95: {single_pred_p95:.4f} ms")
    logger.info(f"  RMSE target < 2%: {'PASS' if rmse_target_met else 'FAIL'}")
    logger.info(f"  Inference < 1ms: {'PASS' if inference_target_met else 'FAIL'}")

    return ValidationMetrics(
        component=component,
        model_path=str(model_path),
        physics_baseline="synthetic_physics",
        n_test_samples=int(n_test_samples),
        rmse=float(metrics["rmse"]),
        rmse_normalized=float(metrics["rmse_normalized"]),
        mae=float(metrics["mae"]),
        r2=float(metrics["r2"]),
        max_error=float(metrics["max_error"]),
        inference_time_ms=float(mean_inference_time),
        inference_time_p95_ms=float(single_pred_p95),
        rmse_target_met=rmse_target_met,
        inference_target_met=inference_target_met,
        all_targets_met=bool(rmse_target_met and inference_target_met),
        physics_output_mean=float(y_physics.mean()),
        physics_output_std=float(y_physics.std()),
        neural_output_mean=float(y_neural.mean()),
        neural_output_std=float(y_neural.std()),
        correlation=float(metrics["correlation"]),
        validation_timestamp=datetime.now(timezone.utc).isoformat(),
        errors=[],
    )


def main():
    parser = argparse.ArgumentParser(description="Validate surrogate models against physics baseline")
    parser.add_argument("--component", type=str, help="Component to validate")
    parser.add_argument("--model", type=Path, help="Model file to validate")
    parser.add_argument("--all-models", action="store_true", help="Validate all models in directory")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Model directory")
    parser.add_argument("--n-samples", type=int, default=1000, help="Test samples")
    parser.add_argument("--rmse-target", type=float, default=0.02, help="RMSE target (normalized)")
    parser.add_argument("--inference-target-ms", type=float, default=1.0, help="Inference target (ms)")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory")

    args = parser.parse_args()

    if args.all_models:
        model_files = list(args.model_dir.glob("surrogate_*.onnx"))
        if not model_files:
            logger.error(f"No ONNX models found in {args.model_dir}")
            return 1
    elif args.model:
        model_files = [args.model]
    else:
        parser.error("--model or --all-models is required")
        return 1

    all_passed = True
    results = {}

    for model_path in model_files:
        component = args.component
        if not component:
            for comp in ["zone_thermal", "solar_gain", "conduction", "ventilation"]:
                if comp in model_path.stem:
                    component = comp
                    break

        if not component:
            logger.warning(f"Could not determine component for {model_path.name}")
            component = model_path.stem

        logger.info("=" * 60)
        logger.info(f"Validating: {model_path.name}")
        logger.info("=" * 60)

        try:
            metrics = validate_component(
                component,
                model_path,
                args.n_samples,
                args.rmse_target,
                args.inference_target_ms,
            )
            results[component] = metrics

            report_path = args.output_dir / f"surrogate_{component}_validation.json"
            with open(report_path, "w") as f:
                json.dump(asdict(metrics), f, indent=2)
            logger.info(f"Report saved: {report_path}")

            if not metrics.all_targets_met:
                all_passed = False

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            all_passed = False
            results[component] = ValidationMetrics(
                component=component,
                model_path=str(model_path),
                physics_baseline="",
                n_test_samples=0,
                rmse=0, rmse_normalized=0, mae=0, r2=0, max_error=0,
                inference_time_ms=0, inference_time_p95_ms=0,
                rmse_target_met=False, inference_target_met=False, all_targets_met=False,
                physics_output_mean=0, physics_output_std=0,
                neural_output_mean=0, neural_output_std=0, correlation=0,
                validation_timestamp=datetime.now(timezone.utc).isoformat(),
                errors=[str(e)],
            )

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Component':<20} {'RMSE Norm':<12} {'RMSE Target':<12} {'Inf P95':<12} {'Inf Target':<12} {'Status'}")
    logger.info("-" * 80)

    for component, metrics in results.items():
        rmse_status = "PASS" if metrics.rmse_target_met else "FAIL"
        inf_status = "PASS" if metrics.inference_target_met else "FAIL"
        overall = "PASS" if metrics.all_targets_met else "FAIL"

        logger.info(
            f"{component:<20} "
            f"{metrics.rmse_normalized:<12.4f} "
            f"{rmse_status:<12} "
            f"{metrics.inference_time_p95_ms:<12.4f} "
            f"{inf_status:<12} "
            f"{overall}"
        )

    passed = sum(1 for m in results.values() if m.all_targets_met)
    total = len(results)

    logger.info("-" * 80)
    logger.info(f"Passed: {passed}/{total}")

    if all_passed:
        logger.info("\nALL COMPONENTS PASSED VALIDATION")
        logger.info(f"  RMSE < {args.rmse_target*100}% and Inference < {args.inference_target_ms}ms")
    else:
        logger.warning("\nSOME COMPONENTS FAILED VALIDATION")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
