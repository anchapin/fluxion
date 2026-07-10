#!/usr/bin/env python3
"""
Train a Phase-2a surrogate against ThermalManifold metric tensors.

Issue #1463 introduces a new training path that replaces the legacy
``(heating, cooling)`` scalar targets with the full ``ThermalManifold``
geometric representation (4-D Riemannian metric, scalar temperature field,
gauge heat-flux connection). This script:

* Loads physics-extracted samples (or generates synthetic manifolds for
  the smoke-test path)
* Builds a ``GeometricSurrogate`` (three heads: metric / field / connection)
* Optimises with the new ``GaugeInvariantLoss`` from ``tools/piml_loss.py``
  which enforces:

  - Frobenius metric accuracy (the curvature geometry)
  - Field / connection data fidelity
  - **First-Law conservation** — heavy penalty if the predicted
    ``gauge_connection_sum`` exceeds the ground-truth sum (i.e. the model
    is hallucinating energy)
  - Parallel-transport consistency — the geometric flow
    ``T + dt * (M·T + A)`` must reproduce the next-step field
  - Dissipativity (metric symmetry, Kirchhoff reciprocity)

The legacy training scripts (``tools/train_surrogate.py``,
``tools/train_pinn.py``) are preserved unchanged for backward compatibility
with their (heating, cooling) output contract — see Issue #1463 acceptance
criteria.

Usage:

    # Smoke test (synthetic data, no EnergyPlus required):
    python tools/train_surrogate_manifold.py --synthetic --epochs 20

    # Production retrain (physics-extracted manifold tensors):
    python tools/train_surrogate_manifold.py \\
        --data-dir data/manifolds --output-dir models/manifold
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Make ``tools/`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from piml_loss import (  # noqa: E402  (sys.path edit above is intentional)
    GaugeInvariantConfig,
    GeometricSurrogate,
    ThermalManifoldBatch,
    generate_synthetic_manifold_data,
    train_geometric_surrogate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_manifold_batch(
    data_dir: Path,
) -> Tuple[torch.Tensor, ThermalManifoldBatch]:
    """Load ``ThermalManifold`` tensors saved by an upstream pipeline.

    The expected layout is::

        data_dir/
            features.npy         # (N, input_dim)
            metric.npy           # (N, D, D)
            field.npy            # (N, D)
            connection.npy       # (N, D)
            field_next.npy       # (N, D) — parallel-transport target
            meta.json            # {"manifold_dim": 4, "dt_seconds": 60.0, ...}

    Raises ``FileNotFoundError`` for any missing file with a precise message
    so the upstream pipeline can self-diagnose.
    """
    files = {
        "features": data_dir / "features.npy",
        "metric": data_dir / "metric.npy",
        "field": data_dir / "field.npy",
        "connection": data_dir / "connection.npy",
        "field_next": data_dir / "field_next.npy",
        "meta": data_dir / "meta.json",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"data_dir={data_dir} is missing required files: {missing}. "
            "Expected layout: features.npy, metric.npy, field.npy, "
            "connection.npy, field_next.npy, meta.json"
        )

    meta: Dict[str, Any] = json.loads(files["meta"].read_text())
    manifold_dim = int(meta.get("manifold_dim", 4))
    dt_seconds = float(meta.get("dt_seconds", 60.0))

    features = np.load(files["features"])
    metric = np.load(files["metric"])
    field = np.load(files["field"])
    connection = np.load(files["connection"])
    field_next = np.load(files["field_next"])

    if metric.shape[1:] != (manifold_dim, manifold_dim):
        raise ValueError(
            f"metric has trailing shape {metric.shape[1:]} but "
            f"manifold_dim={manifold_dim}"
        )
    if field.shape[1] != manifold_dim or connection.shape[1] != manifold_dim:
        raise ValueError(
            f"field/connection have last dim {field.shape[1]} / "
            f"{connection.shape[1]} but manifold_dim={manifold_dim}"
        )

    return (
        torch.from_numpy(np.asarray(features, dtype=np.float32)),
        ThermalManifoldBatch(
            metric=torch.from_numpy(np.asarray(metric, dtype=np.float32)),
            field=torch.from_numpy(np.asarray(field, dtype=np.float32)),
            connection=torch.from_numpy(np.asarray(connection, dtype=np.float32)),
            dt_seconds=dt_seconds,
            field_next=torch.from_numpy(np.asarray(field_next, dtype=np.float32)),
        ),
    )


def save_checkpoint(
    model: GeometricSurrogate,
    config: GaugeInvariantConfig,
    output_dir: Path,
    history: Dict[str, List[float]],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the trained model + config + history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "geometric_surrogate.pt")
    payload = {
        "config": config.__dict__,
        "history": history,
        "meta": extra_meta or {},
    }
    (output_dir / "training_meta.json").write_text(json.dumps(payload, indent=2))
    logger.info("Saved checkpoint to %s", output_dir)


def split_batch(
    features: torch.Tensor,
    batch: ThermalManifoldBatch,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[
    torch.Tensor,
    ThermalManifoldBatch,
    torch.Tensor,
    ThermalManifoldBatch,
]:
    """Deterministic train/val split by row index."""
    n = features.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    def _subset(t: Optional[torch.Tensor], idx: np.ndarray) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t[idx]

    train_batch = ThermalManifoldBatch(
        metric=_subset(batch.metric, train_idx),
        field=_subset(batch.field, train_idx),
        connection=_subset(batch.connection, train_idx),
        dt_seconds=batch.dt_seconds,
        field_next=_subset(batch.field_next, train_idx),
    )
    val_batch = ThermalManifoldBatch(
        metric=_subset(batch.metric, val_idx),
        field=_subset(batch.field, val_idx),
        connection=_subset(batch.connection, val_idx),
        dt_seconds=batch.dt_seconds,
        field_next=_subset(batch.field_next, val_idx),
    )
    return features[train_idx], train_batch, features[val_idx], val_batch


def energy_hallucination_check(
    pred_connection: torch.Tensor,
    target_connection: torch.Tensor,
) -> float:
    """Return the *fraction* of batch rows where the prediction injects more
    net power than the ground truth (``sum(A_pred) > sum(A_target)``).

    Surrogate training should drive this metric to zero — if it stays above
    ~5 % after convergence, the loss weights are misconfigured (typically
    ``conservation_weight`` is too low).
    """
    sum_pred = pred_connection.sum(dim=-1)
    sum_target = target_connection.sum(dim=-1)
    return float((sum_pred > sum_target).float().mean().item())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2a (issue #1463) surrogate training against "
        "ThermalManifold metric tensors."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory of pre-extracted manifold tensors "
        "(features.npy, metric.npy, field.npy, connection.npy, "
        "field_next.npy, meta.json).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip --data-dir and use a synthetic 5R1C manifold batch. "
        "Intended for smoke-testing the training path.",
    )
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/manifold",
        help="Where to save the trained GeometricSurrogate checkpoint.",
    )
    parser.add_argument(
        "--conservation-weight",
        type=float,
        default=10.0,
        help="Weight on the First-Law penalty (energy creation). "
        "Issue #1463 requires this to be heavy enough that the surrogate "
        "never hallucinates net energy generation.",
    )
    parser.add_argument(
        "--manifold-dim", type=int, default=4,
        help="Ambient dimension of the manifold. Pinned to 4 to match "
        "MANIFOLD_DIM in src/physics/geometry_tensor.rs.",
    )
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    if args.synthetic:
        if args.data_dir:
            logger.warning(
                "Both --synthetic and --data-dir set; using synthetic data."
            )
        logger.info(
            "[1/4] Generating synthetic manifold data (n=%d, dim=%d)",
            args.n_samples,
            args.manifold_dim,
        )
        features, batch = generate_synthetic_manifold_data(
            n_samples=args.n_samples,
            manifold_dim=args.manifold_dim,
            seed=args.seed,
        )
    else:
        if not args.data_dir:
            logger.error(
                "Either --synthetic or --data-dir is required. "
                "See --help for the expected layout."
            )
            return 2
        logger.info("[1/4] Loading manifold tensors from %s", args.data_dir)
        features, batch = load_manifold_batch(Path(args.data_dir))

    logger.info(
        "Loaded %d samples — metric=%s, field=%s, connection=%s",
        features.shape[0],
        tuple(batch.metric.shape),
        tuple(batch.field.shape),
        tuple(batch.connection.shape),
    )

    config = GaugeInvariantConfig(
        manifold_dim=args.manifold_dim,
        conservation_weight=args.conservation_weight,
        dt_seconds=float(batch.dt_seconds),
    )
    logger.info("[2/4] Gauge-invariant loss config: %s", config)

    train_features, train_batch, val_features, val_batch = split_batch(
        features, batch, seed=args.seed
    )
    logger.info(
        "Split: train=%d, val=%d", train_features.shape[0], val_features.shape[0]
    )

    model = GeometricSurrogate(
        input_dim=features.shape[1],
        manifold_dim=args.manifold_dim,
    ).to(device)

    logger.info("[3/4] Training GeometricSurrogate for %d epochs...", args.epochs)
    model, history = train_geometric_surrogate(
        model=model,
        train_features=train_features,
        train_batch=train_batch,
        config=config,
        val_features=val_features,
        val_batch=val_batch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    # Final evaluation: check that the model isn't hallucinating energy on
    # the validation set.
    model.eval()
    with torch.no_grad():
        pm, pf, pc = model(val_features.to(device))
        pc_cpu = pc.cpu()
        target_connection = val_batch.connection
        hallucination_rate = energy_hallucination_check(pc_cpu, target_connection)

    logger.info(
        "[4/4] Energy-hallucination rate on validation set: %.2f%% "
        "(target: 0%%; the model must not predict net energy creation)",
        hallucination_rate * 100.0,
    )

    save_checkpoint(
        model=model,
        config=config,
        output_dir=output_dir,
        history=history,
        extra_meta={
            "n_samples": features.shape[0],
            "device": str(device),
            "energy_hallucination_rate": hallucination_rate,
            "input_dim": features.shape[1],
        },
    )

    final_loss = history["total"][-1] if "total" in history else float("nan")
    final_val = history["val_loss"][-1] if history.get("val_loss") else float("nan")
    success = (final_loss == final_loss) and hallucination_rate < 0.05
    if success:
        logger.info(
            "SUCCESS: training converged (loss=%.4e, val=%.4e, "
            "energy_hallucination=%.2f%%)",
            final_loss,
            final_val,
            hallucination_rate * 100.0,
        )
        return 0
    logger.warning(
        "WARNING: training did not fully converge (loss=%.4e, val=%.4e, "
        "energy_hallucination=%.2f%%)",
        final_loss,
        final_val,
        hallucination_rate * 100.0,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())