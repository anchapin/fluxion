# Cloud Campaign Manager for Fluxion

Direct-to-S3 campaign orchestration that removes the "Local Tether" bottleneck.

## Overview

OSimFlow's original campaign submission relied on the user's local machine to manage the execution loop. If a user disconnected or their laptop went to sleep, the remote simulation would drop.

This cloud-hosted system decouples campaign execution from local machines by:

1. **Campaign State in S3** — All campaign state is stored in S3, not local disk
2. **Workers Push to S3** — Individual simulation workers push KPIs directly to S3
3. **Cloud Aggregator** — Merges S3 result files into a final dataset
4. **SNS Notifications** — Email/SMS notification upon campaign completion

## Webhook Notification (T7.4 — Issue #1788)

On 100% completion, the coordinator fans a notification out to **both**
user-configured channels (each optional, both default-on-if-supplied):

| Channel | CLI flag | Env var |
|---------|----------|---------|
| SNS     | `--sns-topic`   | `FLUXION_SNS_TOPIC_ARN` |
| Webhook | `--webhook-url` | `FLUXION_WEBHOOK_URL`   |

The webhook is POSTed as JSON with the following payload (mirrors the
SNS message body so T7.5 email-fallback consumers can rely on a single
schema):

```json
{
  "campaign_id": "fluxion-abc123",
  "status": "completed",
  "start_time": "2026-07-18T00:00:00+00:00",
  "end_time":   "2026-07-18T01:00:00+00:00",
  "total_runs": 50,
  "completed_runs": 48,
  "failed_runs": 2,
  "best_mae": 4.7,
  "best_parameters": {"R_value": 1.5, "wall_thickness": 0.15},
  "results_uri": "https://bucket.s3.us-east-1.amazonaws.com/fluxion-campaigns/campaigns/fluxion-abc123/results/"
}
```

Notification config (`webhook_url`, `sns_topic_arn`) is persisted on the
campaign's `state.json`, so a coordinator restart can still fire the
configured channels without the user re-supplying CLI flags.

```bash
# Create campaign with both channels configured
python scripts/cloud_campaign_manager.py --action create \
    --case 600 --params R_value,wall_thickness --sweep-type random --samples 50 \
    --sns-topic arn:aws:sns:us-east-1:000:topic \
    --webhook-url https://hooks.example.com/fluxion-campaign

# Wait action fires both channels at 100% completion
python scripts/cloud_campaign_manager.py --action wait \
    --campaign-id fluxion-abc123def456
```

## State Store (T7.3 — Issue #1787)

Workers publish per-task completion to a **state store** (DynamoDB or Redis)
instead of relying solely on result-file presence in S3. The coordinator then
**aggregates** state-store entries to compute overall campaign progress.
The state-store path is the authoritative one; the S3 listing in
`check_campaign_progress` remains as a fallback for pre-T7.3 deployments.

### Backends

| Backend | Use case                                    | Configuration                                  |
|---------|---------------------------------------------|------------------------------------------------|
| DynamoDB| Serverless-native, no extra infra           | `FLUXION_STATE_STORE=dynamodb` + `FLUXION_CAMPAIGN_TABLE` (or `boto3` env) |
| Redis   | Sub-second polling for very large campaigns | `FLUXION_STATE_STORE=redis` + `FLUXION_REDIS_URL` |
| Memory  | Local-dev / tests                           | `FLUXION_STATE_STORE=memory`                   |

### Schema

- **DynamoDB**
  - Partition key: `campaign_id` (S)
  - Sort key:     `work_unit_id` (S)
  - Attributes:   `status` (S), `timestamp` (S), `error_message` (S, optional),
    `metrics` (M, optional).
- **Redis**
  - Hash per task:   `fluxion:campaign:{campaign_id}:task:{work_unit_id}`
  - Sorted set per campaign: `fluxion:campaign:{campaign_id}:tasks`
    (member = `work_unit_id`, score = epoch-ms of last update).

### Aggregation

`StateStore.aggregate_progress(campaign_id, total)` returns a
`CampaignProgress` snapshot with `pending`, `running`, `completed`, `failed`,
`progress_pct`, and `is_complete`. The coordinator uses this to update the
campaign state on every poll without ever listing S3.

### CLI

```bash
# Use DynamoDB explicitly
python scripts/cloud_campaign_manager.py --action status \
    --campaign-id fluxion-abc --state-store dynamodb

# Use Redis explicitly
python scripts/cloud_campaign_manager.py --action status \
    --campaign-id fluxion-abc --state-store redis

# Memory (local dev)
python scripts/cloud_campaign_manager.py --action status \
    --campaign-id fluxion-abc --state-store memory

# Auto (consults FLUXION_STATE_STORE, then falls back to legacy S3 listing)
python scripts/cloud_campaign_manager.py --action status \
    --campaign-id fluxion-abc --state-store auto
```

## Architecture

```
┌──────────────────┐     ┌─────────────┐     ┌─────────────────┐
│ Cloud Campaign   │────▶│  S3 Bucket  │◀────│  S3 Worker      │
│ Manager          │     │             │     │  (Remote VM/EC2)│
└──────────────────┘     │  work-units │     └─────────────────┘
       │                 │  kpi-results/│              │
       │                 │  state.json  │              │
       │                 └─────────────┘              │
       ▼                                                │
┌──────────────────┐                                   │
│ SNS Notification │◀──────────────────────────────────┘
│ (Email/SMS)     │         (on completion)
└──────────────────┘
```

**Worker-side KPI Aggregation (T8.3):** Workers compute and emit only aggregated KPIs
before S3 sync, cutting transfer volume. Raw results are included only when
`--emit-raw` is specified.

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install boto3

# Configure AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

# Set campaign parameters
export FLUXION_S3_BUCKET=your-bucket-name
export FLUXION_S3_PREFIX=fluxion-campaigns
export FLUXION_SNS_TOPIC_ARN=arn:aws:sns:region:account:topic
```

### Create a Campaign

```bash
python scripts/cloud_campaign_manager.py \
  --action create \
  --case 600 \
  --params R_value,wall_thickness \
  --sweep-type random \
  --samples 50 \
  --s3-bucket your-bucket-name \
  --s3-prefix fluxion-campaigns
```

This creates:
- Campaign state in `s3://bucket/prefix/campaigns/{id}/state.json`
- Work units in `s3://bucket/prefix/work-units/{id}.json`

### Run Workers

On each remote machine (Hetzner, EC2, etc.):

```bash
# Process a single work unit
python scripts/s3_worker.py \
  --param-file s3://bucket/prefix/work-units/campaign-wu-0000.json

# Or run as an SQS listener for auto-scaling
python scripts/s3_worker.py \
  --mode sqs \
  --queue-url https://sqs.region.amazonaws.com/account/queue-name
```

### Monitor Progress

```bash
# Check status
python scripts/cloud_campaign_manager.py \
  --action status \
  --campaign-id fluxion-abc123def456

# Wait for completion
python scripts/cloud_campaign_manager.py \
  --action wait \
  --campaign-id fluxion-abc123def456
```

### Aggregate Results

```bash
python scripts/s3_aggregator.py \
  --campaign-id fluxion-abc123def456 \
  --s3-bucket your-bucket-name \
  --s3-prefix fluxion-campaigns
```

This creates:
- `s3://bucket/prefix/campaigns/{id}/results/aggregation_report.json`
- `s3://bucket/prefix/campaigns/{id}/results/convergence_data.csv`

### Send Notification

```bash
python scripts/cloud_campaign_manager.py \
  --action notify \
  --campaign-id fluxion-abc123def456 \
  --sns-topic arn:aws:sns:region:account:topic
```

## Workflow Components

### `cloud_campaign_manager.py`

Main orchestration script. Actions:
- `create` — Create a new campaign with work units
- `status` — Check campaign progress
- `wait` — Poll until campaign completes
- `aggregate` — Trigger result aggregation
- `notify` — Send SNS notification

### `s3_worker.py`

Worker script for remote execution. Supports:
- **Single mode** — Process one work unit and exit
- **SQS mode** — Long-running worker that processes messages from an SQS queue

### `s3_aggregator.py`

Merges individual work unit results into a final dataset. Can run:
- Standalone script
- AWS Lambda function (triggered by SNS or S3 events)

## AWS Setup

### S3 Bucket

```bash
# Create bucket
aws s3 mb s3://your-bucket-name --region us-east-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
  --bucket your-bucket-name \
  --versioning-configuration Status=Enabled
```

## KPI Schema

Worker-side aggregation emits `KPIResult` JSON files with the following structure:

```json
{
  "work_unit_id": "wu-0001",
  "campaign_id": "fluxion-abc123",
  "run_id": "a1b2c3d4",
  "case_id": "600",
  "parameters": {"R_value": 2.5, "wall_thickness": 0.15},
  "num_runs": 5,
  "timestamp": "2026-07-17T12:00:00Z",

  "heating_mae_mean": 5.2,
  "heating_mae_std": 0.8,
  "heating_mae_min": 4.1,
  "heating_mae_max": 6.3,
  "cooling_mae_mean": 4.8,
  "cooling_mae_std": 0.6,
  "cooling_mae_min": 3.9,
  "cooling_mae_max": 5.7,

  "peak_heating_mae_mean": 8.2,
  "peak_cooling_mae_mean": 7.1,
  "temperature_mae_mean": 3.4,

  "overall_pass_rate": 0.8,
  "duration_ms_mean": 45200,

  "error_message": null,
  "raw_results": null
}
```

**Fields:**
- All MAE values are percentages (%)
- `num_runs` indicates how many simulation runs were aggregated
- `raw_results` is only populated when `--emit-raw` is specified
- `overall_pass_rate` is the fraction of runs that passed (0.0 to 1.0)

### SNS Topic

```bash
# Create topic
aws sns create-topic --name fluxion-campaign-notifications

# Subscribe email
aws sns subscribe \
  --topic-arn arn:aws:sns:region:account:fluxion-campaign-notifications \
  --protocol email \
  --notification-endpoint your@email.com
```

### IAM Role (for EC2/ECS)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sns:Publish"
      ],
      "Resource": "arn:aws:sns:region:account:fluxion-campaign-notifications"
    }
  ]
}
```

## GitHub Actions Integration

Use the `cloud_campaign.yml` workflow to run campaigns from GitHub Actions:

1. Add AWS secrets to your repository:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_SESSION_TOKEN` (for temporary credentials)

2. Set repository variables:
   - `AWS_REGION`

3. Dispatch the workflow with your campaign parameters.

## Remote Worker Setup

### Hetzner VM

```bash
# Provision runner (from scripts/provision-hetzner-runner.sh)
./scripts/provision-hetzner-runner.sh \
  --github-repo anchapin/fluxion \
  --github-token $RUNNER_TOKEN \
  --hcloud-ssh-key your-ssh-key

# Install campaign dependencies
ssh root@$VM_IP 'pip install boto3'

# Start worker (processes work from SQS queue)
ssh root@$VM_IP 'python scripts/s3_worker.py --mode sqs --queue-url $QUEUE_URL'
```

### AWS EC2 (Auto Scaling)

Use the provided CloudFormation template or Terraform module to provision:
- Auto Scaling group with worker instances
- SQS queue for work distribution
- IAM role with minimal permissions

## Troubleshooting

### Worker fails with "boto3 not found"
```bash
pip install boto3
```

### Campaign stuck in "created" status
- Ensure workers can reach S3
- Check work units exist in `s3://bucket/prefix/work-units/`
- Verify AWS credentials are valid

### SNS notification not received
- Confirm SNS topic subscription is confirmed
- Check email spam folder
- Verify IAM permissions for `sns:Publish`

## Migration from Local Campaigns

The new system is API-compatible with the existing `autonomous_parameter_sweep.py` for the parameter specification format. To migrate:

1. Set up AWS credentials and S3 bucket
2. Use `cloud_campaign_manager.py --action create` instead of running the sweep locally
3. Deploy workers on remote machines
4. Use `s3_aggregator.py` to collect results

The local `autonomous_parameter_sweep.py` can still be used for:
- Quick local debugging
- Small campaigns (< 10 runs)
- CI/local development
