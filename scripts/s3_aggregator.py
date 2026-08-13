#!/usr/bin/env python3
"""
S3 Aggregator for Fluxion Campaigns
===================================
Merges individual work unit results from S3 into a final dataset.
Can run as a standalone script or be triggered by SNS/Lambda.

Usage
-----
# Aggregate results for a campaign
python scripts/s3_aggregator.py --campaign-id <id> --s3-bucket <bucket> --s3-prefix <prefix>

# As a Lambda handler
python scripts/s3_aggregator.py --mode lambda

Exit codes
----------
0 — Aggregation successful
1 — Aggregation failed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


@dataclass
class AggregatedResult:
    work_unit_id: str
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
class AggregationReport:
    campaign_id: str
    aggregation_time: str
    total_work_units: int
    successful_runs: int
    failed_runs: int
    pass_rate: float
    best_mae: float
    best_parameters: dict[str, float]
    results: list[dict]
    convergence_data: list[dict]


def get_aws_clients():
    """Get configured boto3 clients."""
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 aggregator. Install: pip install boto3")

    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    return {
        "s3": session.client("s3"),
        "dynamodb": session.client("dynamodb"),
    }


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def collect_results_from_s3(s3_client, bucket: str, results_prefix: str) -> list[AggregatedResult]:
    """Collect all result JSON files from S3 results prefix."""
    results: list[AggregatedResult] = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{results_prefix}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".json.zst") and not key.endswith("_placeholder"):
                    try:
                        response = s3_client.get_object(Bucket=bucket, Key=key)
                        body_bytes = response["Body"].read()
                        if zstd is not None:
                            body_bytes = zstd.decompress(body_bytes)
                        data = json.loads(body_bytes.decode("utf-8"))
                        results.append(AggregatedResult(**data))
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"[WARN] Failed to parse {key}: {e}", file=sys.stderr)
                elif key.endswith(".json") and not key.endswith("_placeholder"):
                    try:
                        response = s3_client.get_object(Bucket=bucket, Key=key)
                        data = json.loads(response["Body"].read().decode("utf-8"))
                        results.append(AggregatedResult(**data))
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"[WARN] Failed to parse {key}: {e}", file=sys.stderr)
    except ClientError as e:
        print(f"[ERROR] Failed to list results: {e}", file=sys.stderr)

    return results


def aggregate_results(
    campaign_id: str,
    s3_bucket: str,
    s3_prefix: str,
    output_format: str = "json",
) -> AggregationReport:
    """Aggregate all work unit results into a final report."""
    clients = get_aws_clients()
    s3_client = clients["s3"]

    results_prefix = f"{s3_prefix}/results"
    print(f"[*] Collecting results from s3://{s3_bucket}/{results_prefix}/")

    results = collect_results_from_s3(s3_client, s3_bucket, results_prefix)

    if not results:
        print("[WARN] No results found")
        return AggregationReport(
            campaign_id=campaign_id,
            aggregation_time=datetime.now(timezone.utc).isoformat(),
            total_work_units=0,
            successful_runs=0,
            failed_runs=0,
            pass_rate=0.0,
            best_mae=999.0,
            best_parameters={},
            results=[],
            convergence_data=[],
        )

    successful = [r for r in results if r.error_message is None]
    failed = [r for r in results if r.error_message is not None]

    pass_rate = sum(1 for r in successful if r.overall_pass) / len(successful) * 100 if successful else 0.0

    best_result = min(successful, key=lambda r: r.heating_mae + r.cooling_mae) if successful else None
    best_mae = (best_result.heating_mae + best_result.cooling_mae) if best_result else 999.0
    best_parameters = best_result.parameters if best_result else {}

    convergence_data = []
    for r in sorted(results, key=lambda x: x.timestamp):
        if r.error_message is None:
            convergence_data.append({
                "iteration": len(convergence_data) + 1,
                "timestamp": r.timestamp,
                "heating_mae": r.heating_mae,
                "cooling_mae": r.cooling_mae,
                "total_mae": r.heating_mae + r.cooling_mae,
            })

    report = AggregationReport(
        campaign_id=campaign_id,
        aggregation_time=datetime.now(timezone.utc).isoformat(),
        total_work_units=len(results),
        successful_runs=len(successful),
        failed_runs=len(failed),
        pass_rate=pass_rate,
        best_mae=best_mae,
        best_parameters=best_parameters,
        results=[asdict(r) for r in results],
        convergence_data=convergence_data,
    )

    report_key = f"{s3_prefix}/campaigns/{campaign_id}/results/aggregation_report.json"
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=report_key,
        Body=json.dumps(asdict(report), indent=2),
        ContentType="application/json",
    )

    print(f"[*] Aggregation complete: {len(results)} work units")
    print(f"[*] Successful: {len(successful)}, Failed: {len(failed)}")
    print(f"[*] Pass rate: {pass_rate:.1f}%")
    print(f"[*] Best MAE: {best_mae:.2f}%")
    print(f"[*] Report: s3://{s3_bucket}/{report_key}")

    if output_format == "csv":
        csv_key = f"{s3_prefix}/campaigns/{campaign_id}/results/convergence_data.csv"
        csv_content = "iteration,timestamp,heating_mae,cooling_mae,total_mae\n"
        for cd in convergence_data:
            csv_content += f"{cd['iteration']},{cd['timestamp']},{cd['heating_mae']},{cd['cooling_mae']},{cd['total_mae']}\n"
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=csv_key,
            Body=csv_content,
            ContentType="text/csv",
        )
        print(f"[*] CSV: s3://{s3_bucket}/{csv_key}")

    return report


def generate_download_link(s3_client, bucket: str, key: str, expiration: int = 3600) -> str:
    """Generate a pre-signed URL for downloading results."""
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration,
        )
        return url
    except ClientError:
        return f"s3://{bucket}/{key}"


def lambda_handler(event: dict, context: Any) -> dict:
    """AWS Lambda handler for SNS-triggered aggregation."""
    print(f"[*] Lambda triggered with event: {json.dumps(event)}")

    campaign_id = event.get("campaign_id")
    s3_bucket = event.get("s3_bucket")
    s3_prefix = event.get("s3_prefix", "fluxion-campaigns")

    if not all([campaign_id, s3_bucket]):
        print("[ERROR] Missing required parameters")
        return {"statusCode": 400, "body": "Missing required parameters"}

    try:
        report = aggregate_results(campaign_id, s3_bucket, s3_prefix)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "campaign_id": campaign_id,
                "status": "aggregated",
                "total_work_units": report.total_work_units,
                "successful_runs": report.successful_runs,
                "failed_runs": report.failed_runs,
                "pass_rate": report.pass_rate,
                "best_mae": report.best_mae,
            }),
        }
    except Exception as e:
        print(f"[ERROR] Aggregation failed: {e}")
        return {"statusCode": 500, "body": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="S3 Aggregator for Fluxion Campaigns")
    parser.add_argument(
        "--campaign-id",
        type=str,
        required=True,
        help="Campaign ID",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.environ.get("FLUXION_S3_BUCKET"),
        help="S3 bucket",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=os.environ.get("FLUXION_S3_PREFIX", "fluxion-campaigns"),
        help="S3 prefix",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv", "both"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standalone", "lambda"],
        default="standalone",
        help="Run mode",
    )
    parser.add_argument(
        "--download-link-expiration",
        type=int,
        default=3600,
        help="Pre-signed URL expiration in seconds",
    )
    args = parser.parse_args()

    if args.mode == "lambda":
        print("[*] Running in Lambda mode - waiting for event")
        return 0

    if not args.s3_bucket:
        parser.error("--s3-bucket is required (or set FLUXION_S3_BUCKET env var)")

    report = aggregate_results(args.campaign_id, args.s3_bucket, args.s3_prefix, args.output_format)

    clients = get_aws_clients()
    s3_client = clients["s3"]

    report_key = f"{args.s3_prefix}/campaigns/{args.campaign_id}/results/aggregation_report.json"
    download_url = generate_download_link(
        s3_client, args.s3_bucket, report_key, args.download_link_expiration
    )

    print("\n[*] Results available at:")
    print(f"    {download_url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
