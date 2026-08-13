#!/usr/bin/env python3
"""
Surrogate Model Training Script (v3.0)

Trains compact MLP surrogate models per physics component using scikit-learn.
Each component (zone_thermal, solar_gain, conduction, ventilation) gets a dedicated
MLP model trained on synthetic data from generate_training_data.py.

Usage:
    python scripts/train_surrogate.py --component zone_thermal --data-dir data/synthetic/v2.1
    python scripts/train_surrogate.py --component solar_gain --data-dir data/synthetic/v2.1
    python scripts/train_surrogate.py --component conduction --data-dir data/synthetic/v2.1
    python scripts/train_surrogate.py --component ventilation --data-dir data/synthetic/v2.1
    python scripts/train_surrogate.py --all-components --data-dir data/synthetic/v2.1

Output:
    models/surrogate_{component}.onnx  - Trained ONNX model
    models/surrogate_{component}_metrics.json - Training metrics
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    component: str
    data_dir: Path
    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.001
    learning_rate: str = "adaptive"
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class TrainingMetrics:
    component: str
    model_version: str
    training_samples: int
    test_samples: int
    train_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    rmse_normalized: float
    training_time_seconds: float
    n_iterations: int
    scaler_mean: List[float]
    scaler_std: List[float]
    input_features: List[str]
    output_features: List[str]
    hidden_layer_sizes: Tuple[int, ...]
    onnx_opset: int = 17


COMPONENT_SCHEMAS = {
    "zone_thermal": {
        "input_features": [
            "exterior_temp",
            "zone_temp",
            "solar_rad",
            "humidity",
            "occupancy",
            "hvac_mode",
            "climate_zone_encoded",
        ],
        "output_features": ["thermal_load"],
        "target_rmse_normalized": 0.02,
    },
    "solar_gain": {
        "input_features": [
            "latitude",
            "longitude",
            "day_of_year",
            "hour_of_day",
            "surface_tilt",
            "surface_azimuth",
            "direct_normal_irradiance",
            "diffuse_horizontal_irradiance",
        ],
        "output_features": ["solar_gain_W"],
        "target_rmse_normalized": 0.02,
    },
    "conduction": {
        "input_features": [
            "exterior_temp",
            "interior_temp",
            "wall_u_value",
            "wall_area",
            "wall_mass",
            "surface_emissivity",
        ],
        "output_features": ["conduction_flux_W"],
        "target_rmse_normalized": 0.02,
    },
    "ventilation": {
        "input_features": [
            "exterior_temp",
            "interior_temp",
            "wind_speed",
            "indoor_pressure",
            "ventilation_rate",
            "ach",
        ],
        "output_features": ["ventilation_load_W"],
        "target_rmse_normalized": 0.02,
    },
}


def load_synthetic_data(
    component: str, data_dir: Path
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load synthetic training data for a component."""
    component_dir = data_dir / component
    train_file = component_dir / "train.parquet"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            f"Run: python scripts/generate_training_data.py --components {component} --output-dir {data_dir}"
        )

    try:
        import pandas as pd

        df = pd.read_parquet(train_file)
    except ImportError:
        raise ImportError(
            "pandas is required to load parquet files. Install with: pip install pandas pyarrow"
        )

    schema = COMPONENT_SCHEMAS.get(component)
    if schema is None:
        raise ValueError(
            f"Unknown component: {component}. Available: {list(COMPONENT_SCHEMAS.keys())}"
        )

    input_features = schema["input_features"]
    output_features = schema["output_features"]

    missing_inputs = [f for f in input_features if f not in df.columns]
    missing_outputs = [f for f in output_features if f not in df.columns]

    if missing_inputs:
        raise ValueError(
            f"Missing input features for {component}: {missing_inputs}. Available: {list(df.columns)}"
        )
    if missing_outputs:
        raise ValueError(
            f"Missing output features for {component}: {missing_outputs}. Available: {list(df.columns)}"
        )

    X = df[input_features].values.astype(np.float32)
    y = df[output_features].values.astype(np.float32)

    logger.info(f"Loaded {len(X)} samples for {component}")
    logger.info(f"  Input shape: {X.shape}, Output shape: {y.shape}")
    logger.info(f"  Input features: {input_features}")
    logger.info(f"  Output features: {output_features}")

    return X, y, input_features, output_features


def create_synthetic_data(
    component: str, n_samples: int = 10000, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Generate synthetic training data for a component when real data is not available.
    Uses physics-based relationships to create realistic training data.
    """
    np.random.seed(random_state)
    schema = COMPONENT_SCHEMAS.get(component)
    if schema is None:
        raise ValueError(f"Unknown component: {component}")

    input_features = schema["input_features"]
    output_features = schema["output_features"]

    X = []
    y = []

    if component == "zone_thermal":
        for _ in range(n_samples):
            exterior = np.random.uniform(-10, 40)
            zone = np.random.uniform(18, 26)
            solar = np.random.uniform(0, 800)
            humidity = np.random.uniform(20, 80)
            occupancy = np.random.uniform(0, 5)
            hvac_mode = np.random.choice([0, 1, 2])
            climate = np.random.uniform(0, 8)

            delta_t = zone - exterior
            base_load = (
                0.5 * delta_t
                + 0.01 * solar
                + 0.1 * occupancy
                + np.random.normal(0, 0.1)
            )
            if hvac_mode == 1:
                base_load += 2.0
            elif hvac_mode == 2:
                base_load -= 1.5

            X.append([exterior, zone, solar, humidity, occupancy, hvac_mode, climate])
            y.append([base_load])

    elif component == "solar_gain":
        for _ in range(n_samples):
            lat = np.random.uniform(25, 50)
            lon = np.random.uniform(-120, -70)
            doy = np.random.randint(1, 366)
            hour = np.random.randint(0, 24)
            tilt = np.random.choice([0, 30, 45, 60, 90])
            az = np.random.uniform(0, 360)
            dni = np.random.uniform(0, 1000)
            dhi = np.random.uniform(0, 300)

            dec = 23.45 * np.sin(2 * np.pi * (doy - 81) / 365)
            hra = 15 * (hour - 12)
            cos_zenith = np.sin(np.radians(lat)) * np.sin(np.radians(dec)) + np.cos(
                np.radians(lat)
            ) * np.cos(np.radians(dec)) * np.cos(np.radians(hra))
            zenith = np.arccos(np.clip(cos_zenith, -1, 1))
            effective_irrad = (dni * np.cos(zenith) + dhi) * np.cos(np.radians(tilt))
            gain = np.clip(effective_irrad * 0.85, 0, 1200) + np.random.normal(0, 5)

            X.append([lat, lon, doy, hour, tilt, az, dni, dhi])
            y.append([gain])

    elif component == "conduction":
        for _ in range(n_samples):
            exterior = np.random.uniform(-10, 40)
            interior = np.random.uniform(18, 26)
            u_val = np.random.uniform(0.1, 2.5)
            area = np.random.uniform(10, 100)
            mass = np.random.uniform(50, 500)
            emiss = np.random.uniform(0.7, 0.95)

            delta_t = interior - exterior
            q_conv = u_val * area * delta_t
            q_rad = (
                5.67e-8
                * emiss
                * area
                * ((interior + 273) ** 4 - (exterior + 273) ** 4)
                * 1e-8
            )
            total_flux = q_conv + q_rad * 0.1 + np.random.normal(0, 5)

            X.append([exterior, interior, u_val, area, mass, emiss])
            y.append([total_flux])

    elif component == "ventilation":
        for _ in range(n_samples):
            exterior = np.random.uniform(-10, 40)
            interior = np.random.uniform(18, 26)
            wind = np.random.uniform(0, 10)
            pressure = np.random.uniform(99000, 101500)
            vent_rate = np.random.uniform(0, 500)
            ach = np.random.uniform(0, 10)

            delta_t = interior - exterior
            sensible = 0.34 * vent_rate * delta_t / 3600
            latent = 0.68 * vent_rate * 0.01 * (exterior - interior) / 3600
            wind_effect = 0.05 * wind * vent_rate / 3600
            total_load = sensible + latent + wind_effect + np.random.normal(0, 2)

            X.append([exterior, interior, wind, pressure, vent_rate, ach])
            y.append([total_load])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"Generated {n_samples} synthetic samples for {component}")
    logger.info(f"  Input shape: {X.shape}, Output shape: {y.shape}")

    return X, y, input_features, output_features


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainingConfig,
) -> Tuple[MLPRegressor, StandardScaler, TrainingMetrics]:
    """Train an MLP regressor and compute metrics."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    logger.info(f"Training MLP with hidden layers: {config.hidden_layer_sizes}")
    logger.info(f"  Activation: {config.activation}, Solver: {config.solver}")
    logger.info(f"  Max iterations: {config.max_iter}")

    model = MLPRegressor(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        solver=config.solver,
        alpha=config.alpha,
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        early_stopping=config.early_stopping,
        validation_fraction=config.validation_fraction,
        n_iter_no_change=config.n_iter_no_change,
        random_state=config.random_state,
        verbose=True,
    )

    start_time = time.time()
    model.fit(X_train_scaled, y_train_scaled.ravel())
    training_time = time.time() - start_time

    y_pred_train = y_scaler.inverse_transform(
        model.predict(X_train_scaled).reshape(-1, 1)
    )
    y_pred_test = y_scaler.inverse_transform(
        model.predict(X_test_scaled).reshape(-1, 1)
    )

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    y_range = y_test.max() - y_test.min()
    if y_range > 0:
        rmse_normalized = test_rmse / y_range
    else:
        rmse_normalized = 0.0

    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"  Train RMSE: {train_rmse:.6f}")
    logger.info(f"  Test RMSE: {test_rmse:.6f}")
    logger.info(f"  Test MAE: {test_mae:.6f}")
    logger.info(f"  Test R²: {test_r2:.6f}")
    logger.info(
        f"  Normalized RMSE: {rmse_normalized:.4f} ({rmse_normalized * 100:.2f}%)"
    )
    logger.info(f"  Iterations: {model.n_iter_}")

    metrics = TrainingMetrics(
        component=config.component,
        model_version="3.0.0",
        training_samples=len(X_train),
        test_samples=len(X_test),
        train_rmse=float(train_rmse),
        test_rmse=float(test_rmse),
        test_mae=float(test_mae),
        test_r2=float(test_r2),
        rmse_normalized=float(rmse_normalized),
        training_time_seconds=training_time,
        n_iterations=model.n_iter_,
        scaler_mean=scaler.mean_.tolist(),
        scaler_std=scaler.scale_.tolist(),
        input_features=config.component
        and COMPONENT_SCHEMAS[config.component]["input_features"]
        or [],
        output_features=config.component
        and COMPONENT_SCHEMAS[config.component]["output_features"]
        or [],
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    return model, scaler, y_scaler, metrics


def export_to_onnx(
    model: MLPRegressor,
    scaler: StandardScaler,
    y_scaler: StandardScaler,
    config: TrainingConfig,
    output_path: Path,
) -> None:
    """
    Export trained MLP to ONNX format with opset 17 compatibility.
    Uses onnx directly创建模型结构 rather than skl2onnx.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    schema = COMPONENT_SCHEMAS[config.component]
    n_features = len(schema["input_features"])
    n_outputs = len(schema["output_features"])

    logger.info(f"Exporting ONNX model: {output_path}")
    logger.info(f"  Input features: {n_features}, Output features: {n_outputs}")

    input_tensor_name = "X"
    output_tensor_name = "Y"

    input_info = helper.make_tensor_value_info(
        input_tensor_name, TensorProto.FLOAT, [None, n_features]
    )
    output_info = helper.make_tensor_value_info(
        output_tensor_name, TensorProto.FLOAT, [None, n_outputs]
    )

    nodes = []
    initializers = []

    scale_in_init = numpy_helper.from_array(
        scaler.scale_.astype(np.float32), "scale_in"
    )
    bias_in_init = numpy_helper.from_array(
        (-scaler.mean_ * scaler.scale_).astype(np.float32), "bias_in"
    )
    scale_out_init = numpy_helper.from_array(
        (1.0 / y_scaler.scale_).astype(np.float32), "scale_out"
    )
    bias_out_init = numpy_helper.from_array(
        y_scaler.mean_.astype(np.float32), "bias_out"
    )

    initializers.extend([scale_in_init, bias_in_init, scale_out_init, bias_out_init])

    nodes.append(
        helper.make_node("Mul", [input_tensor_name, "scale_in"], ["scale_in_mul"])
    )
    nodes.append(helper.make_node("Add", ["scale_in_mul", "bias_in"], ["scaled_input"]))

    prev_out = "scaled_input"
    layer_sizes = [n_features] + list(config.hidden_layer_sizes) + [n_outputs]

    for i in range(len(layer_sizes) - 1):
        w_values = model.coefs_[i].astype(np.float32)
        b_values = model.intercepts_[i].astype(np.float32)

        w_init = numpy_helper.from_array(w_values, f"W_layer{i}")
        b_init = numpy_helper.from_array(b_values, f"b_layer{i}")
        initializers.extend([w_init, b_init])

        matmul_out = f"mm_out_{i}"
        add_out = f"add_out_{i}"

        nodes.append(
            helper.make_node("MatMul", [prev_out, f"W_layer{i}"], [matmul_out])
        )
        nodes.append(helper.make_node("Add", [matmul_out, f"b_layer{i}"], [add_out]))

        if i < len(layer_sizes) - 2:
            act_out = f"act_out_{i}"
            nodes.append(helper.make_node("Relu", [add_out], [act_out]))
            prev_out = act_out
        else:
            prev_out = add_out

    nodes.append(helper.make_node("Mul", [prev_out, "scale_out"], ["scale_out_mul"]))
    nodes.append(
        helper.make_node("Add", ["scale_out_mul", "bias_out"], [output_tensor_name])
    )

    graph = helper.make_graph(
        nodes,
        f"surrogate_{config.component}",
        [input_info],
        [output_info],
        initializers,
    )

    opset_imports = [helper.make_opsetid("", 17)]
    model_def = helper.make_model(graph, opset_imports=opset_imports)
    model_def.ir_version = 9
    model_def.producer_name = "fluxion.surrogate"
    model_def.producer_version = "3.0.0"

    onnx.save(model_def, str(output_path))
    logger.info(f"ONNX model saved: {output_path}")

    model_size = output_path.stat().st_size
    logger.info(f"  Model size: {model_size / 1024:.2f} KB")


def validate_onnx_model(model_path: Path) -> bool:
    """Validate ONNX model can be loaded and run with onnxruntime."""
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        test_input = np.random.randn(
            1,
            session.get_inputs()[0].shape[1] if session.get_inputs()[0].shape[1] else 8,
        ).astype(np.float32)
        result = session.run([output_name], {input_name: test_input})

        logger.info(f"ONNX model validation passed: {model_path}")
        logger.info(f"  Input shape: {test_input.shape}")
        logger.info(f"  Output shape: {result[0].shape}")
        return True

    except Exception as e:
        logger.error(f"ONNX model validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train surrogate model for a physics component"
    )
    parser.add_argument(
        "--component",
        type=str,
        help="Component to train (zone_thermal, solar_gain, conduction, ventilation)",
    )
    parser.add_argument(
        "--all-components", action="store_true", help="Train all components"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/synthetic/v2.1"),
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"), help="Output directory"
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="64,32",
        help="Hidden layer sizes (comma-separated)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=500, help="Max training iterations"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Synthetic samples (if no real data)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not args.component and not args.all_components:
        parser.error("--component or --all-components is required")

    components = [args.component] if args.component else list(COMPONENT_SCHEMAS.keys())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    hidden = tuple(int(x) for x in args.hidden_layers.split(","))

    results = {}
    for comp in components:
        logger.info("=" * 60)
        logger.info(f"Training {comp.upper()} component")
        logger.info("=" * 60)

        config = TrainingConfig(
            component=comp,
            data_dir=args.data_dir,
            hidden_layer_sizes=hidden,
            max_iter=args.max_iter,
            random_state=args.seed,
        )

        try:
            X, y, input_features, output_features = load_synthetic_data(
                comp, args.data_dir
            )
        except FileNotFoundError:
            logger.warning(f"Real data not found for {comp}, generating synthetic data")
            X, y, input_features, output_features = create_synthetic_data(
                comp, args.n_samples, args.seed
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )

        model, scaler, y_scaler, metrics = train_mlp(
            X_train, y_train, X_test, y_test, config
        )

        metrics.input_features = input_features
        metrics.output_features = output_features

        model_path = args.output_dir / f"surrogate_{comp}.onnx"
        export_to_onnx(model, scaler, y_scaler, config, model_path)

        if validate_onnx_model(model_path):
            logger.info("Model export validated successfully")

        metrics_path = args.output_dir / f"surrogate_{comp}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        results[comp] = {
            "status": "success",
            "rmse_normalized": metrics.rmse_normalized,
            "target_met": metrics.rmse_normalized < 0.02,
        }

    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for comp, result in results.items():
        status = "PASS" if result["target_met"] else "FAIL"
        logger.info(f"  {comp}: {status} (RMSE: {result['rmse_normalized']:.4f})")

    all_passed = all(r["target_met"] for r in results.values())
    if all_passed:
        logger.info("\nAll components meet RMSE < 2% target!")
    else:
        logger.warning("\nSome components did not meet RMSE < 2% target")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
