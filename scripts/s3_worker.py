#!/usr/bin/env python3
"""
S3 Worker Script for Fluxion Campaigns
======================================
Executes individual simulation runs and pushes KPIs directly to S3.
Designed to run on remote machines (Hetzner, EC2, etc.) without
needing to stream results back to a local machine.

Usage
-----
# Run a single work unit (typically called by the campaign manager or scheduler)
python scripts/s3_worker.py --campaign-id <id> --work-unit-id <id> --param-file <s3://...>

# As a daemon listening for SQS messages (alternative)
python scripts/s3_worker.py --mode sqs --queue-url <sqs-queue-url>

Exit codes
----------
0 — Work unit completed successfully
1 — Work unit failed
2 — Configuration error
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception


TRACE_BASE = Path(".sdd/traces/diagnostic")
CARGO_TEST_CMD = ["cargo", "test", "--test=ashrae_140_validation", "--", "--nocapture"]


@dataclass
class WorkUnit:
    work_unit_id: str
    campaign_id: str
    case_id: str
    parameters: dict[str, float]
    s3_result_prefix: str
    config: dict


@dataclass
class WorkUnitResult:
    work_unit_id: str
    campaign_id: str
    run_id: str
    case_id: str
    parameters: dict[str, float]
    heating_mae: float
    cooling_mae: float
    peak_heating_mae: float
    peak_cooling_mae: float
    temperature_mae: float
    overall_pass: bool
    duration_ms: int
    timestamp: str
    error_message: Optional[str] = None


@dataclass
class KPIResult:
    """
    Aggregated KPI result emitted by workers before S3 sync.

    KPI Schema
    ----------
    All MAE (Mean Absolute Error) values are percentages (%).
    Statistics are computed across multiple runs of the same work unit.

    Fields
    ------
    work_unit_id : str
        Unique work unit identifier.
    campaign_id : str
        Campaign this work unit belongs to.
    run_id : str
        Execution run identifier.
    case_id : str
        ASHRAE 140 case identifier (e.g., "600", "600_00001").
    parameters : dict[str, float]
        Input parameters swept for this work unit.
    num_runs : int
        Number of simulation runs aggregated into this KPI.
    timestamp : str
        ISO 8601 timestamp of result emission.

    Aggregated MAE Statistics
    -------------------------
    heating_mae_mean : float
        Mean heating MAE across runs (%).
    heating_mae_std : float
        Standard deviation of heating MAE across runs (%).
    heating_mae_min : float
        Minimum heating MAE across runs (%).
    heating_mae_max : float
        Maximum heating MAE across runs (%).

    cooling_mae_mean : float
        Mean cooling MAE across runs (%).
    cooling_mae_std : float
        Standard deviation of cooling MAE across runs (%).
    cooling_mae_min : float
        Minimum cooling MAE across runs (%).
    cooling_mae_max : float
        Maximum cooling MAE across runs (%).

    peak_heating_mae_mean : float
        Mean peak heating MAE across runs (%).
    peak_cooling_mae_mean : float
        Mean peak cooling MAE across runs (%).
    temperature_mae_mean : float
        Mean temperature MAE across runs (%).

    Pass Rate
    ---------
    overall_pass_rate : float
        Fraction of runs that passed (0.0 to 1.0).

    Performance
    -----------
    duration_ms_mean : float
        Mean execution duration across runs (ms).

    Error Handling
    --------------
    error_message : Optional[str]
        Error message if all runs failed, None otherwise.

    Raw Results
    -----------
    raw_results : Optional[list[WorkUnitResult]]
        Individual run results if --emit-raw was specified, None otherwise.
    """
    work_unit_id: str
    campaign_id: str
    run_id: str
    case_id: str
    parameters: dict[str, float]
    num_runs: int
    timestamp: str
    heating_mae_mean: float
    heating_mae_std: float
    heating_mae_min: float
    heating_mae_max: float
    cooling_mae_mean: float
    cooling_mae_std: float
    cooling_mae_min: float
    cooling_mae_max: float
    peak_heating_mae_mean: float
    peak_cooling_mae_mean: float
    temperature_mae_mean: float
    overall_pass_rate: float
    duration_ms_mean: float
    error_message: Optional[str] = None
    raw_results: Optional[list] = None


def get_aws_clients():
    """Get configured boto3 clients."""
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 worker. Install: pip install boto3")

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
    }


def download_work_unit(param_file: str, clients: dict) -> WorkUnit:
    """Download work unit parameters from S3."""
    if param_file.startswith("s3://"):
        bucket, key = parse_s3_uri(param_file)
        response = clients["s3"].get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8"))
    else:
        with open(param_file) as f:
            data = json.load(f)

    return WorkUnit(
        work_unit_id=data["work_unit_id"],
        campaign_id=data["campaign_id"],
        case_id=data.get("case_id", "600"),
        parameters=data["parameters"],
        s3_result_prefix=data["s3_result_prefix"],
        config=data.get("config", {}),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def push_result_to_s3(result: WorkUnitResult | KPIResult, s3_prefix: str, clients: dict) -> str:
    """Push work unit result or KPI result directly to S3. Returns the S3 URI."""
    bucket, prefix = parse_s3_uri(s3_prefix)
    result_key = f"{prefix}/results/{result.work_unit_id}.json"

    clients["s3"].put_object(
        Bucket=bucket,
        Key=result_key,
        Body=json.dumps(asdict(result), indent=2),
        ContentType="application/json",
    )

    return f"s3://{bucket}/{result_key}"


def update_campaign_progress(
    campaign_id: str, work_unit_id: str, status: str, clients: dict
) -> None:
    """Update campaign progress in DynamoDB or S3."""
    table_name = os.environ.get("FLUXION_CAMPAIGN_TABLE")
    if table_name:
        dynamodb = clients["dynamodb"]
        try:
            dynamodb.update_item(
                TableName=table_name,
                Key={"campaign_id": {"S": campaign_id}},
                UpdateExpression="SET work_unit_statuses.#wid = :status, last_updated = :now",
                ExpressionAttributeNames={"#wid": work_unit_id},
                ExpressionAttributeValues={
                    ":status": {"S": status},
                    ":now": {"S": datetime.now(timezone.utc).isoformat()},
                },
            )
        except ClientError as e:
            print(f"[WARN] Failed to update DynamoDB: {e}", file=sys.stderr)
    else:
        s3_state_prefix = os.environ.get("FLUXION_CAMPAIGN_STATE_PREFIX", "")
        if s3_state_prefix:
            bucket, prefix = parse_s3_uri(s3_state_prefix)
            state_key = f"{prefix}/{campaign_id}/progress/{work_unit_id}.json"
            clients["s3"].put_object(
                Bucket=bucket,
                Key=state_key,
                Body=json.dumps(
                    {
                        "work_unit_id": work_unit_id,
                        "status": status,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                ContentType="application/json",
            )


def run_simulation(work_unit: WorkUnit) -> tuple[dict[str, Any], str]:
    """Run the ASHRAE validation for a single work unit."""
    try:
        result = subprocess.run(
            CARGO_TEST_CMD,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, **{f"FLUXION_PARAM_{k.upper()}": str(v) for k, v in work_unit.parameters.items()}},
        )
        output = result.stdout + result.stderr
        metrics = parse_cargo_output(output)
        return metrics, output
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}, ""
    except Exception as e:
        return {"error": str(e)}, ""


def parse_cargo_output(output: str) -> dict[str, Any]:
    """Parse ASHRAE 140 validation output for MAE values."""
    import re

    metrics: dict[str, Any] = {
        "heating_mae": 0.0,
        "cooling_mae": 0.0,
        "peak_heating_mae": 0.0,
        "peak_cooling_mae": 0.0,
        "temperature_mae": 0.0,
        "overall_pass": False,
    }

    mae_pattern = re.compile(r"Mean\s+Absolute\s+Error:\s*([\d.]+)%", re.IGNORECASE)
    for match in mae_pattern.finditer(output):
        val = float(match.group(1))
        if metrics["heating_mae"] == 0.0:
            metrics["heating_mae"] = val

    case_pattern = re.compile(
        r"Case\s+(\d+[A-Z0-9_]*)\s*[:\-]\s*"
        r"Heating\s*=\s*([\d.]+)\s*\(Ref:\s*([\d.+-]+)\s*-\s*([\d.+-]+)\),\s*"
        r"Cooling\s*=\s*([\d.]+)\s*\(Ref:\s*([\d.+-]+)\s*-\s*([\d.+-]+)\)"
    )

    heating_errors = []
    cooling_errors = []
    for match in case_pattern.finditer(output):
        case = match.group(1)
        if work_unit.case_id and (case.startswith(work_unit.case_id) or work_unit.case_id in case):
            ref_heat = (float(match.group(3)) + float(match.group(4))) / 2
            ref_cool = (float(match.group(6)) + float(match.group(7))) / 2
            sim_heat = float(match.group(2))
            sim_cool = float(match.group(5))
            if ref_heat > 0:
                heating_errors.append(abs(sim_heat - ref_heat) / ref_heat * 100)
            if ref_cool > 0:
                cooling_errors.append(abs(sim_cool - ref_cool) / ref_cool * 100)

    if heating_errors:
        metrics["heating_mae"] = sum(heating_errors) / len(heating_errors)
    if cooling_errors:
        metrics["cooling_mae"] = sum(cooling_errors) / len(cooling_errors)

    summary_pattern = re.compile(
        r"Pass\s+Rate:\s*([\d.]+)%.*?Passed:\s*(\d+).*?Failed:\s*(\d+)",
        re.DOTALL | re.IGNORECASE,
    )
    summary_match = summary_pattern.search(output)
    if summary_match:
        pass_rate = float(summary_match.group(1))
        metrics["overall_pass"] = pass_rate >= 80.0

    return metrics


def compute_kpi_stats(values: list[float]) -> tuple[float, float, float, float]:
    """Compute mean, std, min, max from a list of values."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean_val = sum(values) / len(values)
    std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5 if len(values) > 1 else 0.0
    return mean_val, std_val, min(values), max(values)


def run_worker(work_unit: WorkUnit, emit_raw: bool = False, num_runs: int = 1) -> KPIResult:
    """
    Execute a work unit and emit aggregated KPIs to S3.

    Parameters
    ----------
    work_unit : WorkUnit
        Work unit to execute.
    emit_raw : bool
        If True, include individual run results in output.
    num_runs : int
        Number of simulation runs to aggregate (default 1).

    Returns
    -------
    KPIResult
        Aggregated KPI result with statistics.
    """
    clients = get_aws_clients()
    run_id = str(uuid.uuid4())[:8]

    print(f"[*] Starting work unit {work_unit.work_unit_id}")
    print(f"[*] Campaign: {work_unit.campaign_id}")
    print(f"[*] Parameters: {work_unit.parameters}")
    print(f"[*] Running {num_runs} simulation(s) for aggregation")

    update_campaign_progress(
        work_unit.campaign_id, work_unit.work_unit_id, "running", clients
    )

    raw_results: list[WorkUnitResult] = []
    heating_maes = []
    cooling_maes = []
    peak_heating_maes = []
    peak_cooling_maes = []
    temperature_maes = []
    pass_count = 0
    durations = []

    for run_idx in range(num_runs):
        start = time.time()
        metrics, raw_output = run_simulation(work_unit)
        duration_ms = int((time.time() - start) * 1000)
        durations.append(duration_ms)

        if "error" in metrics:
            raw_results.append(WorkUnitResult(
                work_unit_id=work_unit.work_unit_id,
                campaign_id=work_unit.campaign_id,
                run_id=f"{run_id}-{run_idx}",
                case_id=work_unit.case_id,
                parameters=work_unit.parameters,
                heating_mae=999.0,
                cooling_mae=999.0,
                peak_heating_mae=999.0,
                peak_cooling_mae=999.0,
                temperature_mae=999.0,
                overall_pass=False,
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error_message=str(metrics.get("error", "unknown")),
            ))
        else:
            heating = metrics.get("heating_mae", 999.0)
            cooling = metrics.get("cooling_mae", 999.0)
            peak_heat = metrics.get("peak_heating_mae", 999.0)
            peak_cool = metrics.get("peak_cooling_mae", 999.0)
            temp = metrics.get("temperature_mae", 999.0)
            passed = metrics.get("overall_pass", False)

            heating_maes.append(heating)
            cooling_maes.append(cooling)
            peak_heating_maes.append(peak_heat)
            peak_cooling_maes.append(peak_cool)
            temperature_maes.append(temp)
            if passed:
                pass_count += 1

            raw_results.append(WorkUnitResult(
                work_unit_id=work_unit.work_unit_id,
                campaign_id=work_unit.campaign_id,
                run_id=f"{run_id}-{run_idx}",
                case_id=work_unit.case_id,
                parameters=work_unit.parameters,
                heating_mae=heating,
                cooling_mae=cooling,
                peak_heating_mae=peak_heat,
                peak_cooling_mae=peak_cool,
                temperature_mae=temp,
                overall_pass=passed,
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

    # Compute aggregated KPIs
    h_mean, h_std, h_min, h_max = compute_kpi_stats(heating_maes)
    c_mean, c_std, c_min, c_max = compute_kpi_stats(cooling_maes)
    ph_mean, _, _, _ = compute_kpi_stats(peak_heating_maes)
    pc_mean, _, _, _ = compute_kpi_stats(peak_cooling_maes)
    t_mean, _, _, _ = compute_kpi_stats(temperature_maes)
    dur_mean, _, _, _ = compute_kpi_stats(durations)
    pass_rate = pass_count / num_runs if num_runs > 0 else 0.0

    has_error = any(r.error_message is not None for r in raw_results)
    error_msg = None
    if has_error and all(r.error_message is not None for r in raw_results):
        error_msg = "; ".join(set(r.error_message for r in raw_results if r.error_message))

    result = KPIResult(
        work_unit_id=work_unit.work_unit_id,
        campaign_id=work_unit.campaign_id,
        run_id=run_id,
        case_id=work_unit.case_id,
        parameters=work_unit.parameters,
        num_runs=num_runs,
        timestamp=datetime.now(timezone.utc).isoformat(),
        heating_mae_mean=h_mean,
        heating_mae_std=h_std,
        heating_mae_min=h_min,
        heating_mae_max=h_max,
        cooling_mae_mean=c_mean,
        cooling_mae_std=c_std,
        cooling_mae_min=c_min,
        cooling_mae_max=c_max,
        peak_heating_mae_mean=ph_mean,
        peak_cooling_mae_mean=pc_mean,
        temperature_mae_mean=t_mean,
        overall_pass_rate=pass_rate,
        duration_ms_mean=dur_mean,
        error_message=error_msg,
        raw_results=raw_results if emit_raw else None,
    )

    result_uri = push_result_to_s3(result, work_unit.s3_result_prefix, clients)
    print(f"[*] KPI result pushed to: {result_uri}")

    update_campaign_progress(
        work_unit.campaign_id, work_unit.work_unit_id, "completed", clients
    )

    return result


def run_sqs_worker(queue_url: str, emit_raw: bool = False, num_runs: int = 1) -> None:
    """Long-running SQS worker that processes messages from a queue."""
    clients = get_aws_clients()
    sqs = clients["sqs"] if "sqs" in clients else boto3.client("sqs")

    print(f"[*] Listening on SQS queue: {queue_url}")
    print(f"[*] Aggregation: {num_runs} runs, emit_raw={emit_raw}")

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
            )

            messages = response.get("Messages", [])
            for message in messages:
                body = json.loads(message["Body"])
                work_unit = download_work_unit(body["param_file"], clients)

                try:
                    run_worker(work_unit, emit_raw=emit_raw, num_runs=num_runs)
                    sqs.delete_message(
                        QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"]
                    )
                except Exception as e:
                    print(f"[ERROR] Work unit failed: {e}", file=sys.stderr)
                    update_campaign_progress(
                        work_unit.campaign_id,
                        work_unit.work_unit_id,
                        f"failed: {e}",
                        clients,
                    )

        except KeyboardInterrupt:
            print("[*] Shutting down worker...")
            break
        except Exception as e:
            print(f"[ERROR] Worker error: {e}", file=sys.stderr)
            time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description="S3 Worker for Fluxion Campaigns")
    parser.add_argument(
        "--campaign-id",
        type=str,
        help="Campaign ID",
    )
    parser.add_argument(
        "--work-unit-id",
        type=str,
        help="Work unit ID",
    )
    parser.add_argument(
        "--param-file",
        type=str,
        help="S3 URI or local path to work unit parameters",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "sqs"],
        default="single",
        help="Worker mode: single work unit or SQS listener",
    )
    parser.add_argument(
        "--queue-url",
        type=str,
        help="SQS queue URL (for sqs mode)",
    )
    parser.add_argument(
        "--emit-raw",
        action="store_true",
        help="Include individual run results in output (increases data transfer)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of simulation runs to aggregate per work unit (default: 1)",
    )
    args = parser.parse_args()

    if args.mode == "sqs":
        if not args.queue_url:
            parser.error("--queue-url is required for sqs mode")
        run_sqs_worker(args.queue_url, emit_raw=args.emit_raw, num_runs=args.num_runs)
        return 0

    if not args.param_file:
        parser.error("--param-file is required for single mode")

    work_unit = download_work_unit(args.param_file, get_aws_clients())
    result = run_worker(work_unit, emit_raw=args.emit_raw, num_runs=args.num_runs)

    print(f"[*] Work unit complete: {result.work_unit_id}")
    print(f"    Aggregated KPIs over {result.num_runs} run(s):")
    print(f"    Heating MAE: {result.heating_mae_mean:.2f}% (std: {result.heating_mae_std:.2f}%)")
    print(f"    Cooling MAE: {result.cooling_mae_mean:.2f}% (std: {result.cooling_mae_std:.2f}%)")
    print(f"    Pass rate: {result.overall_pass_rate * 100:.1f}%")

    return 0 if result.error_message is None else 1


if __name__ == "__main__":
    sys.exit(main())
