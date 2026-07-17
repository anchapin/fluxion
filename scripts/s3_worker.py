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

try:
    import zstandard as zstd
except ImportError:
    zstd = None


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


def push_result_to_s3(result: WorkUnitResult, s3_prefix: str, clients: dict) -> str:
    """Push work unit result directly to S3. Returns the S3 URI."""
    bucket, prefix = parse_s3_uri(s3_prefix)
    result_key = f"{prefix}/results/{result.work_unit_id}.json.zst"

    json_body = json.dumps(asdict(result), indent=2)

    if zstd is not None:
        body = zstd.compress(json_body.encode("utf-8"))
        content_type = "application/zstd"
    else:
        body = json_body.encode("utf-8")
        content_type = "application/json"

    clients["s3"].put_object(
        Bucket=bucket,
        Key=result_key,
        Body=body,
        ContentType=content_type,
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


def run_worker(work_unit: WorkUnit) -> WorkUnitResult:
    """Execute a single work unit and push result to S3."""
    clients = get_aws_clients()
    run_id = str(uuid.uuid4())[:8]

    print(f"[*] Starting work unit {work_unit.work_unit_id}")
    print(f"[*] Campaign: {work_unit.campaign_id}")
    print(f"[*] Parameters: {work_unit.parameters}")

    update_campaign_progress(
        work_unit.campaign_id, work_unit.work_unit_id, "running", clients
    )

    start = time.time()
    metrics, raw_output = run_simulation(work_unit)
    duration_ms = int((time.time() - start) * 1000)

    if "error" in metrics:
        result = WorkUnitResult(
            work_unit_id=work_unit.work_unit_id,
            campaign_id=work_unit.campaign_id,
            run_id=run_id,
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
        )
    else:
        result = WorkUnitResult(
            work_unit_id=work_unit.work_unit_id,
            campaign_id=work_unit.campaign_id,
            run_id=run_id,
            case_id=work_unit.case_id,
            parameters=work_unit.parameters,
            heating_mae=metrics.get("heating_mae", 999.0),
            cooling_mae=metrics.get("cooling_mae", 999.0),
            peak_heating_mae=metrics.get("peak_heating_mae", 999.0),
            peak_cooling_mae=metrics.get("peak_cooling_mae", 999.0),
            temperature_mae=metrics.get("temperature_mae", 999.0),
            overall_pass=metrics.get("overall_pass", False),
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    result_uri = push_result_to_s3(result, work_unit.s3_result_prefix, clients)
    print(f"[*] Result pushed to: {result_uri}")

    update_campaign_progress(
        work_unit.campaign_id, work_unit.work_unit_id, "completed", clients
    )

    return result


def run_sqs_worker(queue_url: str) -> None:
    """Long-running SQS worker that processes messages from a queue."""
    clients = get_aws_clients()
    sqs = clients["sqs"] if "sqs" in clients else boto3.client("sqs")

    print(f"[*] Listening on SQS queue: {queue_url}")

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
                    run_worker(work_unit)
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
    args = parser.parse_args()

    if args.mode == "sqs":
        if not args.queue_url:
            parser.error("--queue-url is required for sqs mode")
        run_sqs_worker(args.queue_url)
        return 0

    if not args.param_file:
        parser.error("--param-file is required for single mode")

    work_unit = download_work_unit(args.param_file, get_aws_clients())
    result = run_worker(work_unit)

    print(f"[*] Work unit complete: {result.work_unit_id}")
    print(f"    MAE: {result.heating_mae:.2f}% heating, {result.cooling_mae:.2f}% cooling")

    return 0 if result.error_message is None else 1


if __name__ == "__main__":
    sys.exit(main())
