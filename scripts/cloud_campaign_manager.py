#!/usr/bin/env python3
"""
Cloud Campaign Manager for Fluxion
==================================
Cloud-hosted orchestration for simulation campaigns.
Removes the "Local Tether" bottleneck by:
1. Storing campaign state in S3 (not local disk)
2. Workers push results directly to S3
3. Aggregator merges S3 results
4. SNS notification on completion

Usage
-----
# Start a new campaign
python scripts/cloud_campaign_manager.py --action create \
    --case 600 --params R_value,wall_thickness --sweep-type random --samples 50

# Check campaign status
python scripts/cloud_campaign_manager.py --action status --campaign-id <id>

# Wait for campaign completion
python scripts/cloud_campaign_manager.py --action wait --campaign-id <id>

# Trigger aggregation and get results
python scripts/cloud_campaign_manager.py --action aggregate --campaign-id <id>

Exit codes
----------
0 — Success
1 — Campaign failed
2 — Configuration error
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from state_store import (  # type: ignore[import-not-found]
        CampaignProgress,
        StateStore,
        TaskState,
        TaskStatus,
    )
except ImportError:  # pragma: no cover - defensive
    CampaignProgress = None  # type: ignore[assignment]
    StateStore = None  # type: ignore[assignment]
    TaskState = None  # type: ignore[assignment]
    TaskStatus = None  # type: ignore[assignment]


TRACE_BASE = Path(".sdd/traces/diagnostic")

CARGO_TEST_CMD = ["cargo", "test", "--test=ashrae_140_validation", "--", "--nocapture"]


class SweepType(Enum):
    GRID = "grid"
    RANDOM = "random"
    GRADIENT = "gradient"
    BINARY = "binary"
    LATIN_HYPERCUBE = "latin_hypercube"


@dataclass
class ParameterSpec:
    name: str
    default: float
    min_val: float
    max_val: float
    step: float
    unit: str = ""


@dataclass
class CampaignConfig:
    case_id: str
    sweep_type: SweepType
    parameters: list[ParameterSpec]
    max_iterations: int = 100
    samples_per_param: int = 10
    concurrent_workers: int = 4
    timeout_per_run: int = 300
    tolerance_mae: float = 5.0


@dataclass
class WorkUnit:
    work_unit_id: str
    campaign_id: str
    case_id: str
    parameters: dict[str, float]
    s3_result_prefix: str
    config: dict


@dataclass
class CampaignState:
    campaign_id: str
    config: dict
    work_units: list[dict]
    status: str
    start_time: str
    end_time: Optional[str] = None
    best_parameters: dict[str, float] = field(default_factory=dict)
    best_mae: float = 999.0
    completed_units: int = 0
    failed_units: int = 0
    results_uri: Optional[str] = None
    notification_sent: bool = False
    error_message: Optional[str] = None


def get_aws_clients():
    """Get configured boto3 clients."""
    if boto3 is None:
        raise RuntimeError("boto3 is required for cloud campaign. Install: pip install boto3")

    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    return {
        "s3": session.client("s3"),
        "sns": session.client("sns"),
        "dynamodb": session.client("dynamodb"),
        "sts": session.client("sts"),
    }


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def get_required_env(var: str) -> str:
    """Get required environment variable or fail."""
    value = os.environ.get(var)
    if not value:
        raise ValueError(f"Required environment variable not set: {var}")
    return value


def _resolve_state_store(selection: str) -> Optional["StateStore"]:
    """Translate the ``--state-store`` CLI flag into a concrete store.

    ``auto`` consults ``FLUXION_STATE_STORE`` (with backward-compatible
    fall-through to ``FLUXION_CAMPAIGN_TABLE`` for pre-T7.3 deployments
    that only configured the legacy DynamoDB table name).
    ``memory`` returns an :class:`InMemoryStateStore` (handy for tests
    and smoke checks that should not touch AWS).
    """
    if StateStore is None:
        return None
    selection = (selection or "auto").strip().lower()
    if selection == "auto":
        env = (os.environ.get("FLUXION_STATE_STORE") or "").strip().lower()
        if env:
            selection = env
        elif os.environ.get("FLUXION_CAMPAIGN_TABLE") or os.environ.get("FLUXION_REDIS_URL"):
            selection = (
                "dynamodb"
                if os.environ.get("FLUXION_CAMPAIGN_TABLE")
                else "redis"
            )
        else:
            return None
    if selection in ("memory", "inmemory"):
        return StateStore.create("memory")
    if selection in ("dynamodb", "redis"):
        try:
            return StateStore.create(selection)
        except Exception as exc:  # pragma: no cover - transport errors
            print(
                f"[WARN] Could not build {selection} state store: {exc}",
                file=sys.stderr,
            )
            return None
    return None


def ensure_s3_prefix(s3_client, bucket: str, prefix: str) -> None:
    """Ensure S3 prefix exists (create empty object if needed)."""
    if prefix.endswith("/"):
        prefix = prefix[:-1]
    try:
        s3_client.head_object(Bucket=bucket, Key=prefix + "/_placeholder")
    except ClientError:
        s3_client.put_object(Bucket=bucket, Key=prefix + "/_placeholder", Body=b"")


def generate_grid_points(specs: list[ParameterSpec]) -> list[dict[str, float]]:
    """Generate full factorial grid of parameter combinations."""
    import itertools

    grids = []
    for spec in specs:
        points = []
        val = spec.min_val
        while val <= spec.max_val + 1e-9:
            points.append(val)
            val += spec.step
        grids.append(points)

    combinations = list(itertools.product(*grids))
    return [
        {specs[i].name: combo[i] for i in range(len(specs))} for combo in combinations
    ]


def generate_random_points(
    specs: list[ParameterSpec], samples: int
) -> list[dict[str, float]]:
    """Generate random parameter samples within bounds."""
    return [
        {spec.name: random.uniform(spec.min_val, spec.max_val) for spec in specs}
        for _ in range(samples)
    ]


def create_campaign(
    config: CampaignConfig,
    s3_bucket: str,
    s3_prefix: str,
    sns_topic_arn: Optional[str] = None,
) -> CampaignState:
    """Create a new campaign and upload initial state to S3."""
    clients = get_aws_clients()

    campaign_id = f"fluxion-{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"[*] Creating campaign {campaign_id}")

    if config.sweep_type == SweepType.GRID:
        param_combinations = generate_grid_points(config.parameters)
    elif config.sweep_type == SweepType.RANDOM:
        param_combinations = generate_random_points(
            config.parameters, config.samples_per_param
        )
    else:
        param_combinations = generate_random_points(
            config.parameters, config.max_iterations
        )

    param_combinations = param_combinations[: config.max_iterations]

    print(f"[*] Generating {len(param_combinations)} work units")

    work_units = []
    for i, params in enumerate(param_combinations):
        work_unit_id = f"{campaign_id}-wu-{i:04d}"
        work_unit = WorkUnit(
            work_unit_id=work_unit_id,
            campaign_id=campaign_id,
            case_id=config.case_id,
            parameters=params,
            s3_result_prefix=s3_prefix,
            config=asdict(config),
        )
        work_units.append(asdict(work_unit))

        unit_key = f"{s3_prefix}/work-units/{work_unit_id}.json"
        clients["s3"].put_object(
            Bucket=s3_bucket,
            Key=unit_key,
            Body=json.dumps(work_unit, indent=2),
            ContentType="application/json",
        )

    config_dict = {
        "case_id": config.case_id,
        "sweep_type": config.sweep_type.value,
        "parameters": [
            {
                "name": s.name,
                "default": s.default,
                "min": s.min_val,
                "max": s.max_val,
                "step": s.step,
                "unit": s.unit,
            }
            for s in config.parameters
        ],
        "max_iterations": config.max_iterations,
        "samples_per_param": config.samples_per_param,
        "tolerance_mae": config.tolerance_mae,
    }

    state = CampaignState(
        campaign_id=campaign_id,
        config=config_dict,
        work_units=work_units,
        status="created",
        start_time=datetime.now(timezone.utc).isoformat(),
    )

    state_uri = f"s3://{s3_bucket}/{s3_prefix}/campaigns/{campaign_id}/state.json"
    clients["s3"].put_object(
        Bucket=s3_bucket,
        Key=f"{s3_prefix}/campaigns/{campaign_id}/state.json",
        Body=json.dumps(asdict(state), indent=2),
        ContentType="application/json",
    )

    print(f"[*] Campaign state: {state_uri}")
    print(f"[*] Work units: {len(work_units)}")

    if sns_topic_arn:
        print(f"[*] SNS notifications will be sent to: {sns_topic_arn}")

    return state


def get_campaign_state(campaign_id: str, s3_bucket: str, s3_prefix: str) -> Optional[CampaignState]:
    """Retrieve campaign state from S3."""
    clients = get_aws_clients()

    try:
        response = clients["s3"].get_object(
            Bucket=s3_bucket,
            Key=f"{s3_prefix}/campaigns/{campaign_id}/state.json",
        )
        data = json.loads(response["Body"].read().decode("utf-8"))
        return CampaignState(**data)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return None
        raise


def update_campaign_state(state: CampaignState, s3_bucket: str, s3_prefix: str) -> None:
    """Update campaign state in S3."""
    clients = get_aws_clients()
    clients["s3"].put_object(
        Bucket=s3_bucket,
        Key=f"{s3_prefix}/campaigns/{state.campaign_id}/state.json",
        Body=json.dumps(asdict(state), indent=2),
        ContentType="application/json",
    )


def check_campaign_progress(
    state: CampaignState,
    s3_bucket: str,
    s3_prefix: str,
    *,
    state_store: Optional["StateStore"] = None,
) -> CampaignState:
    """Check progress by aggregating worker state-store entries.

    Resolution order
    ----------------
    1. ``state_store`` argument (preferred — tests / local pipeline).
    2. ``FLUXION_STATE_STORE`` env (``dynamodb`` or ``redis``).
    3. Legacy S3 listing — counts ``results/*.json`` blobs produced by
       older worker versions that predate the state-store refactor.

    The state-store aggregation is authoritative when available because
    workers now publish per-task completion directly to it, including
    metrics and error_message that the S3-listing path cannot recover.
    """
    progress: Optional[CampaignProgress] = None
    if state_store is not None and CampaignProgress is not None:
        progress = state_store.aggregate_progress(
            state.campaign_id, total=len(state.work_units)
        )
    elif StateStore is not None:
        backend = (os.environ.get("FLUXION_STATE_STORE") or "").strip().lower()
        if backend in ("dynamodb", "redis"):
            try:
                store = StateStore.from_env()
                progress = store.aggregate_progress(
                    state.campaign_id, total=len(state.work_units)
                )
            except Exception as exc:  # pragma: no cover - transport errors
                print(
                    f"[WARN] StateStore aggregate failed: {exc}",
                    file=sys.stderr,
                )

    if progress is not None:
        state.completed_units = progress.completed
        state.failed_units = progress.failed
        total = len(state.work_units)
        if state.status == "created" and progress.in_flight > 0:
            state.status = "running"
        if progress.is_complete and total > 0:
            state.status = "completed"
        print(
            f"[*] Campaign {state.campaign_id} (state-store): "
            f"{progress.completed}/{total} completed, "
            f"{progress.failed} failed, {progress.in_flight} in-flight "
            f"({progress.progress_pct:.1f}%)"
        )
        return state

    # --- Legacy S3 listing fallback -----------------------------------
    clients = get_aws_clients()

    completed = 0
    failed = 0

    results_prefix = f"{s3_prefix}/results"

    try:
        paginator = clients["s3"].get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=f"{results_prefix}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".json") and not key.endswith("_placeholder"):
                    if "failed" in key:
                        failed += 1
                    else:
                        completed += 1
    except ClientError:
        pass

    state.completed_units = completed
    state.failed_units = failed

    total = len(state.work_units)
    progress_pct = (completed + failed) / total * 100 if total > 0 else 0

    print(f"[*] Campaign {state.campaign_id}: {completed}/{total} completed, {failed} failed ({progress_pct:.1f}%)")

    if state.status == "created" and (completed + failed) > 0:
        state.status = "running"
    if completed + failed >= total and total > 0:
        state.status = "completed"

    return state


def wait_for_completion(
    campaign_id: str,
    s3_bucket: str,
    s3_prefix: str,
    poll_interval: int = 30,
    *,
    state_store: Optional["StateStore"] = None,
) -> CampaignState:
    """Poll until campaign is complete."""
    print(f"[*] Waiting for campaign {campaign_id} to complete...")

    while True:
        state = get_campaign_state(campaign_id, s3_bucket, s3_prefix)
        if state is None:
            print(f"[ERROR] Campaign {campaign_id} not found")
            sys.exit(1)

        state = check_campaign_progress(
            state, s3_bucket, s3_prefix, state_store=state_store
        )
        update_campaign_state(state, s3_bucket, s3_prefix)

        if state.status in ("completed", "failed"):
            return state

        time.sleep(poll_interval)


def send_completion_notification(
    state: CampaignState, s3_bucket: str, s3_prefix: str, sns_topic_arn: str
) -> None:
    """Send SNS notification with campaign results."""
    if state.notification_sent:
        print("[*] Notification already sent, skipping")
        return

    clients = get_aws_clients()

    account_id = clients["sts"].get_caller_identity()["Account"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    results_uri = f"https://{s3_bucket}.s3.{region}.amazonaws.com/{s3_prefix}/campaigns/{state.campaign_id}/results/"

    message = {
        "campaign_id": state.campaign_id,
        "status": state.status,
        "start_time": state.start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(state.work_units),
        "completed_runs": state.completed_units,
        "failed_runs": state.failed_units,
        "best_mae": state.best_mae,
        "best_parameters": state.best_parameters,
        "results_uri": results_uri,
    }

    clients["sns"].publish(
        TopicArn=sns_topic_arn,
        Subject=f"Fluxion Campaign {state.campaign_id} {'Completed' if state.status == 'completed' else 'Failed'}",
        Message=json.dumps(message, indent=2),
    )

    state.notification_sent = True
    update_campaign_state(state, s3_bucket, s3_prefix)

    print(f"[*] Notification sent to: {sns_topic_arn}")


def trigger_aggregator(
    campaign_id: str, s3_bucket: str, s3_prefix: str, aggregator_function_name: Optional[str] = None
) -> str:
    """Trigger the aggregator Lambda or ECS task."""
    clients = get_aws_clients()

    results_uri = f"s3://{s3_bucket}/{s3_prefix}/campaigns/{campaign_id}/results/"

    if aggregator_function_name:
        lambda_client = clients.get("lambda") or boto3.client("lambda")
        lambda_client.invoke(
            FunctionName=aggregator_function_name,
            InvocationType="Event",
            Payload=json.dumps({
                "campaign_id": campaign_id,
                "s3_bucket": s3_bucket,
                "s3_prefix": s3_prefix,
                "results_uri": results_uri,
            }),
        )
        print(f"[*] Aggregator Lambda triggered: {aggregator_function_name}")
    else:
        print("[*] No aggregator configured - results remain in S3")
        print(f"[*] Results URI: {results_uri}")

    return results_uri


def build_default_params() -> dict[str, ParameterSpec]:
    """Build default parameter specs for ASHRAE cases."""
    return {
        "R_value": ParameterSpec(
            "R_value", default=2.0, min_val=1.0, max_val=5.0, step=0.5, unit="m²K/W"
        ),
        "wall_thickness": ParameterSpec(
            "wall_thickness",
            default=0.15,
            min_val=0.05,
            max_val=0.30,
            step=0.05,
            unit="m",
        ),
        "thermal_mass": ParameterSpec(
            "thermal_mass", default=1.0, min_val=0.5, max_val=2.0, step=0.1, unit=""
        ),
        "h_tr_is": ParameterSpec(
            "h_tr_is", default=8.29, min_val=5.0, max_val=15.0, step=1.0, unit="W/m²K"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cloud Campaign Manager for Fluxion")
    parser.add_argument(
        "--action",
        type=str,
        choices=["create", "status", "wait", "aggregate", "notify"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        help="Campaign ID (required for status, wait, aggregate, notify actions)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="600",
        help="ASHRAE 140 case ID",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Comma-separated parameter names to sweep",
    )
    parser.add_argument(
        "--sweep-type",
        type=str,
        choices=["grid", "random", "gradient", "binary", "latin_hypercube"],
        default="random",
        help="Sweep strategy",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples for random sweep",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.environ.get("FLUXION_S3_BUCKET"),
        help="S3 bucket for campaign data",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=os.environ.get("FLUXION_S3_PREFIX", "fluxion-campaigns"),
        help="S3 prefix for campaign data",
    )
    parser.add_argument(
        "--sns-topic",
        type=str,
        default=os.environ.get("FLUXION_SNS_TOPIC_ARN"),
        help="SNS topic ARN for notifications",
    )
    parser.add_argument(
        "--aggregator-function",
        type=str,
        help="Lambda function name for aggregation",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Poll interval in seconds for wait action",
    )
    parser.add_argument(
        "--state-store",
        type=str,
        choices=["dynamodb", "redis", "memory", "auto"],
        default="auto",
        help=(
            "State-store backend for per-task progress (T7.3). "
            "'auto' uses FLUXION_STATE_STORE when set, otherwise falls "
            "back to legacy S3 listing."
        ),
    )
    args = parser.parse_args()

    state_store = _resolve_state_store(args.state_store)

    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix.rstrip("/")

    if args.action == "create":
        if not s3_bucket:
            parser.error("--s3-bucket is required (or set FLUXION_S3_BUCKET env var)")

        param_names = (
            args.params.split(",") if args.params else ["R_value", "wall_thickness"]
        )
        default_params = build_default_params()

        specs = []
        for name in param_names:
            if name in default_params:
                specs.append(default_params[name])
            else:
                specs.append(
                    ParameterSpec(name, default=1.0, min_val=0.1, max_val=10.0, step=0.1)
                )

        config = CampaignConfig(
            case_id=args.case,
            sweep_type=SweepType(args.sweep_type),
            parameters=specs,
            max_iterations=args.max_iterations,
            samples_per_param=args.samples,
        )

        state = create_campaign(config, s3_bucket, s3_prefix, args.sns_topic)

        print(f"[*] Campaign created: {state.campaign_id}")
        print(f"[*] Work units: {len(state.work_units)}")
        print(f"[*] Run workers with:")
        print(f"    python scripts/s3_worker.py --param-file s3://{s3_bucket}/{s3_prefix}/work-units/{{work_unit_id}}.json")

        return 0

    if args.action == "status":
        if not args.campaign_id:
            parser.error("--campaign-id is required for status action")
        if not s3_bucket and state_store is None:
            parser.error(
                "--s3-bucket is required (or set FLUXION_S3_BUCKET env var) "
                "or pass --state-store"
            )

        if state_store is None:
            state = get_campaign_state(args.campaign_id, s3_bucket, s3_prefix)
        else:
            # State-store-only mode: synthesize a minimal CampaignState
            # from the campaign definition stored in S3 (if any) or just
            # use the live aggregate.
            state = (
                get_campaign_state(args.campaign_id, s3_bucket, s3_prefix)
                if s3_bucket
                else CampaignState(
                    campaign_id=args.campaign_id,
                    config={},
                    work_units=[],
                    status="created",
                    start_time=datetime.now(timezone.utc).isoformat(),
                )
            )
            if state is None:
                print(f"[ERROR] Campaign {args.campaign_id} not found")
                return 1

        state = check_campaign_progress(
            state, s3_bucket, s3_prefix, state_store=state_store
        )

        print(f"\nCampaign: {state.campaign_id}")
        print(f"Status: {state.status}")
        print(f"Start: {state.start_time}")
        print(f"Completed: {state.completed_units}/{len(state.work_units) or '?'}")
        print(f"Failed: {state.failed_units}")
        print(f"Best MAE: {state.best_mae:.2f}%")

        return 0

    if args.action == "wait":
        if not args.campaign_id:
            parser.error("--campaign-id is required for wait action")
        if not s3_bucket and state_store is None:
            parser.error(
                "--s3-bucket is required (or set FLUXION_S3_BUCKET env var) "
                "or pass --state-store"
            )

        if s3_bucket:
            state = wait_for_completion(
                args.campaign_id,
                s3_bucket,
                s3_prefix,
                args.poll_interval,
                state_store=state_store,
            )
        else:
            # State-store-only wait loop
            print(f"[*] Waiting for campaign {args.campaign_id} to complete...")
            while True:
                progress = (
                    state_store.aggregate_progress(args.campaign_id, total=0)
                    if state_store is not None
                    else None
                )
                if progress is None:
                    print("[ERROR] No state store available")
                    return 1
                print(
                    f"[*] {progress.completed} completed, "
                    f"{progress.failed} failed, "
                    f"{progress.in_flight} in-flight"
                )
                if progress.is_complete:
                    state = CampaignState(
                        campaign_id=args.campaign_id,
                        config={},
                        work_units=[],
                        status="completed",
                        start_time=datetime.now(timezone.utc).isoformat(),
                        completed_units=progress.completed,
                        failed_units=progress.failed,
                    )
                    break
                time.sleep(args.poll_interval)

        print(f"\n[*] Campaign {args.campaign_id} is now: {state.status}")
        print(f"    Completed: {state.completed_units}")
        print(f"    Failed: {state.failed_units}")

        if args.sns_topic and state.status == "completed":
            send_completion_notification(state, s3_bucket, s3_prefix, args.sns_topic)

        return 0

    if args.action == "aggregate":
        if not args.campaign_id:
            parser.error("--campaign-id is required for aggregate action")
        if not s3_bucket:
            parser.error("--s3-bucket is required (or set FLUXION_S3_BUCKET env var)")

        results_uri = trigger_aggregator(args.campaign_id, s3_bucket, s3_prefix, args.aggregator_function)
        print(f"[*] Aggregation triggered")
        print(f"[*] Results: {results_uri}")

        return 0

    if args.action == "notify":
        if not args.campaign_id:
            parser.error("--campaign-id is required for notify action")
        if not s3_bucket:
            parser.error("--s3-bucket is required (or set FLUXION_S3_BUCKET env var)")
        if not args.sns_topic:
            parser.error("--sns-topic is required for notify action")

        state = get_campaign_state(args.campaign_id, s3_bucket, s3_prefix)
        if state is None:
            print(f"[ERROR] Campaign {args.campaign_id} not found")
            return 1

        send_completion_notification(state, s3_bucket, s3_prefix, args.sns_topic)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
