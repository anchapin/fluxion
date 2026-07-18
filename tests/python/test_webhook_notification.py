"""
Unit tests for Issue #1788 / T7.4 — webhook notification on campaign
completion.

Covers the new ``webhook_url`` field on :class:`CampaignState`, the
``send_webhook_notification`` HTTP delivery path, and the unified
``send_completion_notification`` that fans out to both webhook and SNS
channels.

Tests are hermetic: no boto3 calls, no real HTTP traffic.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CLOUD_MANAGER_SCRIPT = SCRIPTS_DIR / "cloud_campaign_manager.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cloud_manager_module():
    return _load_module(CLOUD_MANAGER_SCRIPT, "cloud_campaign_manager_under_test")


@pytest.fixture(scope="module")
def state_store_module():
    spec = importlib.util.spec_from_file_location(
        "state_store_under_test", SCRIPTS_DIR / "state_store.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["state_store_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _make_state(cloud_manager_module, **overrides) -> object:
    """Build a CampaignState in the 'completed' terminal state."""
    base = dict(
        campaign_id="camp-webhook",
        config={"case_id": "600"},
        work_units=[{"work_unit_id": "wu-0001"}, {"work_unit_id": "wu-0002"}],
        status="completed",
        start_time="2026-07-18T00:00:00+00:00",
        end_time="2026-07-18T01:00:00+00:00",
        completed_units=2,
        failed_units=0,
        best_mae=2.5,
        best_parameters={"R_value": 1.5},
    )
    base.update(overrides)
    return cloud_manager_module.CampaignState(**base)


# ---------------------------------------------------------------------------
# CampaignState — webhook_url field plumbing
# ---------------------------------------------------------------------------


def test_campaign_state_has_webhook_url_field(cloud_manager_module):
    assert "webhook_url" in cloud_manager_module.CampaignState.__dataclass_fields__
    assert "sns_topic_arn" in cloud_manager_module.CampaignState.__dataclass_fields__


def test_campaign_state_webhook_url_defaults_to_none(cloud_manager_module):
    state = cloud_manager_module.CampaignState(
        campaign_id="x",
        config={},
        work_units=[],
        status="created",
        start_time=datetime.now(timezone.utc).isoformat(),
    )
    assert state.webhook_url is None
    assert state.sns_topic_arn is None


def test_campaign_state_round_trip_preserves_webhook_url(cloud_manager_module):
    state = _make_state(cloud_manager_module, webhook_url="https://example.test/hook")
    payload = asdict(state)
    rebuilt = cloud_manager_module.CampaignState(**payload)
    assert rebuilt.webhook_url == "https://example.test/hook"


# ---------------------------------------------------------------------------
# build_completion_payload
# ---------------------------------------------------------------------------


def test_build_completion_payload_includes_campaign_id_and_result_uri(
    cloud_manager_module,
):
    state = _make_state(cloud_manager_module)
    payload = cloud_manager_module.build_completion_payload(
        state, s3_bucket="my-bucket", s3_prefix="fluxion-campaigns"
    )
    assert payload["campaign_id"] == "camp-webhook"
    assert payload["status"] == "completed"
    assert payload["results_uri"].endswith(
        "/fluxion-campaigns/campaigns/camp-webhook/results/"
    )
    assert payload["results_uri"].startswith("https://my-bucket.s3.")
    assert payload["total_runs"] == 2
    assert payload["completed_runs"] == 2


# ---------------------------------------------------------------------------
# send_webhook_notification — urllib happy path
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self._status_code = status_code

    def getcode(self) -> int:
        return self._status_code

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_send_webhook_notification_posts_payload_with_campaign_id(
    cloud_manager_module,
):
    state = _make_state(cloud_manager_module)
    captured: dict = {}

    def fake_urlopen(request, timeout=10.0):
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return _FakeResponse(204)

    with patch.object(
        cloud_manager_module.urllib.request, "urlopen", side_effect=fake_urlopen
    ):
        ok = cloud_manager_module.send_webhook_notification(
            state,
            s3_bucket="bucket",
            s3_prefix="prefix",
            webhook_url="https://hooks.example.test/campaign",
            timeout=5.0,
        )

    assert ok is True
    assert captured["url"] == "https://hooks.example.test/campaign"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 5.0
    assert captured["body"]["campaign_id"] == "camp-webhook"
    assert captured["body"]["status"] == "completed"
    assert captured["body"]["results_uri"].endswith(
        "/prefix/campaigns/camp-webhook/results/"
    )
    assert captured["headers"].get("Content-type") == "application/json"
    assert captured["headers"].get("User-agent", "").startswith("fluxion-cloud-campaign")


def test_send_webhook_notification_returns_false_on_http_error(
    cloud_manager_module,
):
    state = _make_state(cloud_manager_module)

    def fake_urlopen(request, timeout=10.0):
        raise cloud_manager_module.urllib.error.HTTPError(
            request.full_url, 500, "Internal Server Error", {}, io.BytesIO(b"")
        )

    with patch.object(
        cloud_manager_module.urllib.request, "urlopen", side_effect=fake_urlopen
    ):
        ok = cloud_manager_module.send_webhook_notification(
            state, "bucket", "prefix", "https://hooks.example.test/campaign"
        )
    assert ok is False


def test_send_webhook_notification_returns_false_on_network_error(
    cloud_manager_module,
):
    state = _make_state(cloud_manager_module)

    def fake_urlopen(request, timeout=10.0):
        raise cloud_manager_module.urllib.error.URLError("connection refused")

    with patch.object(
        cloud_manager_module.urllib.request, "urlopen", side_effect=fake_urlopen
    ):
        ok = cloud_manager_module.send_webhook_notification(
            state, "bucket", "prefix", "https://hooks.example.test/campaign"
        )
    assert ok is False


# ---------------------------------------------------------------------------
# send_completion_notification — fan-out to webhook + SNS
# ---------------------------------------------------------------------------


def test_send_completion_notification_only_webhook(cloud_manager_module):
    state = _make_state(cloud_manager_module)

    def fake_urlopen(request, timeout=10.0):
        return _FakeResponse(200)

    with patch.object(
        cloud_manager_module.urllib.request, "urlopen", side_effect=fake_urlopen
    ), patch.object(
        cloud_manager_module, "update_campaign_state", lambda *a, **k: None
    ):
        cloud_manager_module.send_completion_notification(
            state,
            s3_bucket="bucket",
            s3_prefix="prefix",
            sns_topic_arn=None,
            webhook_url="https://hooks.example.test/campaign",
        )

    assert state.notification_sent is True


def test_send_completion_notification_only_sns(cloud_manager_module):
    state = _make_state(cloud_manager_module)
    sns_client = MagicMock()
    clients = {"sns": sns_client, "sts": MagicMock()}

    with patch.object(
        cloud_manager_module, "get_aws_clients", lambda: clients
    ), patch.object(
        cloud_manager_module, "update_campaign_state", lambda *a, **k: None
    ):
        cloud_manager_module.send_completion_notification(
            state,
            s3_bucket="bucket",
            s3_prefix="prefix",
            sns_topic_arn="arn:aws:sns:us-east-1:000:topic",
            webhook_url=None,
        )

    sns_client.publish.assert_called_once()
    assert state.notification_sent is True


def test_send_completion_notification_fans_out_to_both_channels(
    cloud_manager_module,
):
    state = _make_state(cloud_manager_module)
    sns_client = MagicMock()
    clients = {"sns": sns_client, "sts": MagicMock()}

    with patch.object(
        cloud_manager_module.urllib.request, "urlopen", return_value=_FakeResponse(200)
    ), patch.object(
        cloud_manager_module, "get_aws_clients", lambda: clients
    ), patch.object(
        cloud_manager_module, "update_campaign_state", lambda *a, **k: None
    ):
        cloud_manager_module.send_completion_notification(
            state,
            s3_bucket="bucket",
            s3_prefix="prefix",
            sns_topic_arn="arn:aws:sns:us-east-1:000:topic",
            webhook_url="https://hooks.example.test/campaign",
        )

    sns_client.publish.assert_called_once()
    assert state.notification_sent is True
    # SNS message body and webhook payload share the same campaign_id
    sns_call = sns_client.publish.call_args
    sns_body = json.loads(sns_call.kwargs["Message"])
    assert sns_body["campaign_id"] == "camp-webhook"
    assert sns_body["status"] == "completed"


def test_send_completion_notification_is_idempotent(cloud_manager_module):
    state = _make_state(
        cloud_manager_module, notification_sent=True, webhook_url="https://x"
    )
    sns_client = MagicMock()
    clients = {"sns": sns_client}

    with patch.object(
        cloud_manager_module.urllib.request, "urlopen"
    ) as urlopen_mock, patch.object(
        cloud_manager_module, "get_aws_clients", lambda: clients
    ):
        cloud_manager_module.send_completion_notification(
            state,
            "bucket",
            "prefix",
            sns_topic_arn="arn:aws:sns:us-east-1:000:topic",
            webhook_url="https://hooks.example.test/campaign",
        )

    urlopen_mock.assert_not_called()
    sns_client.publish.assert_not_called()


def test_send_completion_notification_no_channel_warns(cloud_manager_module):
    state = _make_state(cloud_manager_module)
    with patch.object(
        cloud_manager_module, "update_campaign_state", lambda *a, **k: None
    ):
        # Should not raise even with neither channel configured.
        cloud_manager_module.send_completion_notification(
            state, "bucket", "prefix", sns_topic_arn=None, webhook_url=None
        )


# ---------------------------------------------------------------------------
# create_campaign — webhook URL persisted in state.json
# ---------------------------------------------------------------------------


def test_create_campaign_persists_webhook_url_in_state(cloud_manager_module):
    captured: dict = {}

    class _FakeS3:
        def put_object(self, **kwargs):
            captured["Key"] = kwargs.get("Key", "")
            captured["Body"] = kwargs.get("Body", b"")
            return {"ETag": "fake"}

    with patch.object(
        cloud_manager_module, "get_aws_clients",
        lambda: {"s3": _FakeS3(), "sns": object()},
    ):
        state = cloud_manager_module.create_campaign(
            config=cloud_manager_module.CampaignConfig(
                case_id="600",
                sweep_type=cloud_manager_module.SweepType.RANDOM,
                parameters=[
                    cloud_manager_module.ParameterSpec(
                        name="R_value",
                        default=1.0,
                        min_val=0.5,
                        max_val=3.0,
                        step=0.1,
                    )
                ],
                max_iterations=2,
                samples_per_param=2,
            ),
            s3_bucket="bucket",
            s3_prefix="prefix",
            sns_topic_arn=None,
            webhook_url="https://hooks.example.test/campaign",
        )

    assert state.webhook_url == "https://hooks.example.test/campaign"
    body = json.loads(captured["Body"])
    assert body["webhook_url"] == "https://hooks.example.test/campaign"


# ---------------------------------------------------------------------------
# wait_for_completion end-to-end — webhook fires on 100% completion
# ---------------------------------------------------------------------------


def test_wait_for_completion_fires_webhook_at_full_completion(
    cloud_manager_module, state_store_module
):
    """Full coordinator loop: state-store hits 100%, webhook POSTs, idempotent re-entry."""
    store = state_store_module.InMemoryStateStore()
    for i in range(3):
        store.set_state(
            state_store_module.TaskState.now("camp-loop", f"wu-{i:03d}", "completed")
        )

    state = cloud_manager_module.CampaignState(
        campaign_id="camp-loop",
        config={"case_id": "600"},
        work_units=[{"work_unit_id": f"wu-{i:03d}"} for i in range(3)],
        status="created",
        start_time=datetime.now(timezone.utc).isoformat(),
        webhook_url="https://hooks.example.test/campaign",
    )

    captured: list[dict] = []

    def fake_urlopen(request, timeout=10.0):
        captured.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse(200)

    with patch.object(
        cloud_manager_module.urllib.request, "urlopen", side_effect=fake_urlopen
    ), patch.object(
        cloud_manager_module, "update_campaign_state", lambda *a, **k: None
    ):
        result = cloud_manager_module.check_campaign_progress(
            state, s3_bucket="bucket", s3_prefix="prefix", state_store=store
        )
        assert result.status == "completed"
        cloud_manager_module.send_completion_notification(
            result,
            s3_bucket="bucket",
            s3_prefix="prefix",
            sns_topic_arn=None,
            webhook_url=result.webhook_url,
        )

    assert len(captured) == 1
    assert captured[0]["campaign_id"] == "camp-loop"
    assert captured[0]["status"] == "completed"
    assert captured[0]["results_uri"].endswith(
        "/prefix/campaigns/camp-loop/results/"
    )
    assert state.notification_sent is True
