"""
Unit tests for the campaign state-store (Issue #1787 / T7.3).

Covers:
- :mod:`scripts.state_store` for all three backends (in-memory, DynamoDB
  via ``moto``, and Redis via a self-contained stub that satisfies the
  small surface area the real ``redis-py`` client exposes).
- The worker's :func:`update_campaign_progress` and :func:`run_worker`
  end-to-end paths through the new state-store.
- The coordinator's :func:`check_campaign_progress` aggregation path
  when a state-store is provided.

The tests intentionally avoid invoking ``cargo`` or hitting AWS — they
exercise the pure data path so they stay fast and CI-friendly. A live
DynamoDB / Redis integration test would belong in a separate suite
gated on ``FLUXION_RUN_CLOUD_INTEGRATION=1``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
STATE_STORE_SCRIPT = SCRIPTS_DIR / "state_store.py"
S3_WORKER_SCRIPT = SCRIPTS_DIR / "s3_worker.py"
CLOUD_MANAGER_SCRIPT = SCRIPTS_DIR / "cloud_campaign_manager.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def state_store_module():
    return _load_module(STATE_STORE_SCRIPT, "state_store")


@pytest.fixture(scope="module")
def s3_worker_module():
    return _load_module(S3_WORKER_SCRIPT, "s3_worker_under_test")


@pytest.fixture(scope="module")
def cloud_manager_module():
    return _load_module(CLOUD_MANAGER_SCRIPT, "cloud_campaign_manager_under_test")


# ---------------------------------------------------------------------------
# Backend-agnostic behavioural tests — run against every store implementation.
# ---------------------------------------------------------------------------


def _exercise_store(store, state_store_module):
    """Push a campaign through the store and assert observable invariants."""
    state_store_module.TaskState.now("camp-A", "wu-1", "running")
    entries = [
        ("wu-1", "running", None, None),
        ("wu-2", "completed", {"heating_mae": 1.0, "cooling_mae": 0.5}, None),
        ("wu-3", "failed", None, "timeout"),
        ("wu-4", "pending", None, None),
    ]
    for wid, status, metrics, err in entries:
        store.set_state(
            state_store_module.TaskState.now(
                "camp-A", wid, status, metrics=metrics, error_message=err
            )
        )

    progress = store.aggregate_progress("camp-A", total=4)
    assert progress.completed == 1
    assert progress.failed == 1
    assert progress.running == 1
    assert progress.pending == 1
    assert progress.total == 4
    assert progress.progress_pct == pytest.approx(50.0, abs=0.01)
    assert not progress.is_complete

    # Mark wu-1 and wu-4 as terminal; campaign becomes complete.
    store.set_state(state_store_module.TaskState.now("camp-A", "wu-1", "completed"))
    store.set_state(state_store_module.TaskState.now("camp-A", "wu-4", "failed", error_message="boom"))
    progress = store.aggregate_progress("camp-A", total=4)
    assert progress.completed == 2
    assert progress.failed == 2
    assert progress.is_complete
    assert progress.progress_pct == pytest.approx(100.0, abs=0.01)

    # get_state round-trip on the only entry that has metrics.
    fetched = store.get_state("camp-A", "wu-2")
    assert fetched is not None
    assert fetched.status == state_store_module.TaskStatus.COMPLETED
    assert fetched.metrics == {"heating_mae": 1.0, "cooling_mae": 0.5}

    # Re-setting wu-2 with no metrics should clear the optional field.
    store.set_state(state_store_module.TaskState.now("camp-A", "wu-2", "completed"))
    assert store.get_state("camp-A", "wu-2").metrics is None

    # Different campaigns are isolated.
    store.set_state(
        state_store_module.TaskState.now("camp-B", "wu-1", "completed")
    )
    progress_b = store.aggregate_progress("camp-B", total=1)
    assert progress_b.completed == 1
    assert progress_b.is_complete


def test_in_memory_store_round_trip(state_store_module):
    store = state_store_module.InMemoryStateStore()
    assert store.backend_name == "memory"
    _exercise_store(store, state_store_module)


def test_in_memory_store_list_is_independent_of_insertion_order(state_store_module):
    store = state_store_module.InMemoryStateStore()
    for i in range(5):
        store.set_state(
            state_store_module.TaskState.now("camp", f"wu-{i}", "pending")
        )
    assert len(store.list_states("camp")) == 5


def test_task_state_validation(state_store_module):
    with pytest.raises(ValueError):
        state_store_module.TaskState(
            campaign_id="",
            work_unit_id="wu",
            status="running",
            timestamp="now",
        )
    with pytest.raises(ValueError):
        state_store_module.TaskState(
            campaign_id="c",
            work_unit_id="",
            status="running",
            timestamp="now",
        )
    with pytest.raises(ValueError):
        state_store_module.TaskState(
            campaign_id="c",
            work_unit_id="wu",
            status="not-a-status",
            timestamp="now",
        )


def test_task_state_to_from_dict(state_store_module):
    original = state_store_module.TaskState(
        campaign_id="c1",
        work_unit_id="wu-1",
        status="completed",
        timestamp="2026-01-01T00:00:00+00:00",
        metrics={"heating_mae": 1.0},
        error_message=None,
    )
    payload = original.to_dict()
    restored = state_store_module.TaskState.from_dict(payload)
    assert restored.campaign_id == original.campaign_id
    assert restored.work_unit_id == original.work_unit_id
    assert restored.status == original.status
    assert restored.timestamp == original.timestamp
    assert restored.metrics == original.metrics


def test_dynamodb_store_round_trip(state_store_module):
    moto = pytest.importorskip("moto")
    import boto3

    with moto.mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        store = state_store_module.DynamoDBStateStore(
            dynamodb_client=ddb, table_name="fluxion-state-test"
        )
        store.ensure_table()
        _exercise_store(store, state_store_module)


def test_dynamodb_store_ensure_table_is_idempotent(state_store_module):
    moto = pytest.importorskip("moto")
    import boto3

    with moto.mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        store = state_store_module.DynamoDBStateStore(
            dynamodb_client=ddb, table_name="fluxion-state-test"
        )
        store.ensure_table()
        store.ensure_table()
        # Should not raise; querying should be empty.
        assert store.list_states("nope") == []


# ---------------------------------------------------------------------------
# Redis backend — fake client satisfying the redis-py surface area we use.
# ---------------------------------------------------------------------------


class _FakePipe:
    def __init__(self, parent: "_FakeRedis") -> None:
        self.parent = parent
        self.queue: list[tuple] = []

    def hset(self, key, mapping=None, **_kw):
        self.queue.append(("hset", key, dict(mapping or {})))
        return self

    def hdel(self, key, *fields):
        self.queue.append(("hdel", key, list(fields)))
        return self

    def zadd(self, key, mapping):
        self.queue.append(("zadd", key, dict(mapping)))
        return self

    def hgetall(self, key):
        self.queue.append(("hgetall", key))
        return self

    def delete(self, key):
        self.queue.append(("delete", key))
        return self

    def execute(self):
        results = []
        for cmd in self.queue:
            results.append(self.parent._run(cmd))
        self.queue.clear()
        return results


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    def pipeline(self):
        return _FakePipe(self)

    # Direct-call APIs (used by .clear() and for cleanliness in tests).
    def hset(self, key, mapping=None, **_kw):
        self.hashes.setdefault(key, {}).update(mapping or {})
        return 1

    def hdel(self, key, *fields):
        existing = self.hashes.get(key, {})
        removed = sum(1 for f in fields if existing.pop(f, None) is not None)
        return removed

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(
            {
                m.decode() if isinstance(m, bytes) else m: float(s)
                for m, s in mapping.items()
            }
        )
        return 1

    def zrange(self, key, start, end):
        items = self.zsets.get(key, {})
        ordered = sorted(items.items(), key=lambda kv: kv[1])
        if end == -1:
            end = len(ordered) - 1
        return [k for k, _ in ordered[start : end + 1]]

    def delete(self, key):
        existed = key in self.hashes or key in self.zsets
        self.hashes.pop(key, None)
        self.zsets.pop(key, None)
        return 1 if existed else 0

    # Internal helper used by the pipeline.
    def _run(self, cmd):
        kind = cmd[0]
        if kind == "hset":
            _, key, mapping = cmd
            self.hset(key, mapping=mapping)
            return 1
        if kind == "hdel":
            _, key, fields = cmd
            return self.hdel(key, *fields)
        if kind == "zadd":
            _, key, mapping = cmd
            self.zadd(key, mapping)
            return 1
        if kind == "hgetall":
            _, key = cmd
            return self.hgetall(key)
        if kind == "delete":
            _, key = cmd
            return self.delete(key)
        raise AssertionError(f"unknown fake-redis cmd: {kind}")


@pytest.fixture
def fake_redis():
    return _FakeRedis()


def test_redis_store_round_trip(state_store_module, fake_redis):
    store = state_store_module.RedisStateStore(redis_client=fake_redis)
    assert store.backend_name == "redis"
    _exercise_store(store, state_store_module)


def test_redis_store_clear(state_store_module, fake_redis):
    store = state_store_module.RedisStateStore(redis_client=fake_redis)
    store.set_state(state_store_module.TaskState.now("camp", "wu-1", "completed"))
    store.clear("camp")
    assert store.list_states("camp") == []
    assert store.aggregate_progress("camp", total=1).progress_pct == 0.0


def test_redis_store_handles_unicode_work_unit_ids(state_store_module, fake_redis):
    store = state_store_module.RedisStateStore(redis_client=fake_redis)
    store.set_state(
        state_store_module.TaskState.now("camp-ü", "wu-日本語", "completed")
    )
    fetched = store.get_state("camp-ü", "wu-日本語")
    assert fetched is not None
    assert fetched.status == state_store_module.TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Factory / from_env tests.
# ---------------------------------------------------------------------------


def test_create_factory_dispatch(state_store_module, fake_redis):
    assert isinstance(
        state_store_module.StateStore.create("memory"), state_store_module.InMemoryStateStore
    )
    redis_store = state_store_module.StateStore.create("redis", redis_client=fake_redis)
    assert isinstance(redis_store, state_store_module.RedisStateStore)


def test_create_factory_rejects_unknown_backend(state_store_module):
    with pytest.raises(ValueError):
        state_store_module.StateStore.create("not-a-backend")


def test_from_env_respects_explicit_backend(state_store_module, monkeypatch):
    monkeypatch.setenv("FLUXION_STATE_STORE", "memory")
    store = state_store_module.StateStore.from_env()
    assert store.backend_name == "memory"


def test_from_env_unknown_backend_raises(state_store_module, monkeypatch):
    monkeypatch.setenv("FLUXION_STATE_STORE", "invalid")
    with pytest.raises(ValueError):
        state_store_module.StateStore.from_env()


# ---------------------------------------------------------------------------
# Worker integration: update_campaign_progress + run_worker
# ---------------------------------------------------------------------------


def _stub_clients():
    class _FakeS3:
        def __init__(self):
            self.objects: list[tuple[str, bytes]] = []

        def put_object(self, **kwargs):
            self.objects.append((kwargs.get("Key", ""), kwargs.get("Body", b"")))
            return {"ETag": "fake"}

        def get_object(self, **_kwargs):
            class _Body:
                def read(self_inner):
                    return b"{}"

            return {"Body": _Body()}

    s3 = _FakeS3()
    return {"s3": s3, "dynamodb": object(), "sns": object()}


def test_update_campaign_progress_uses_injected_store(state_store_module, s3_worker_module):
    store = state_store_module.InMemoryStateStore()
    clients = _stub_clients()
    s3_worker_module.update_campaign_progress(
        "camp-A", "wu-1", "completed", clients, store=store,
        metrics={"heating_mae": 2.0},
    )
    fetched = store.get_state("camp-A", "wu-1")
    assert fetched is not None
    assert fetched.status == state_store_module.TaskStatus.COMPLETED
    assert fetched.metrics == {"heating_mae": 2.0}


def test_update_campaign_progress_failure_path_writes_error(state_store_module, s3_worker_module):
    store = state_store_module.InMemoryStateStore()
    clients = _stub_clients()
    s3_worker_module.update_campaign_progress(
        "camp-A", "wu-1", "failed", clients, store=store, error_message="kaboom",
    )
    fetched = store.get_state("camp-A", "wu-1")
    assert fetched.status == state_store_module.TaskStatus.FAILED
    assert fetched.error_message == "kaboom"


def test_run_worker_success_path(state_store_module, s3_worker_module):
    store = state_store_module.InMemoryStateStore()
    clients = _stub_clients()

    wu = s3_worker_module.WorkUnit(
        work_unit_id="wu-1",
        campaign_id="camp-A",
        case_id="600",
        parameters={"R_value": 2.0},
        s3_result_prefix="s3://bucket/prefix",
        config={},
    )

    def fake_simulation(_wu):
        return (
            {
                "heating_mae": 1.0,
                "cooling_mae": 0.5,
                "peak_heating_mae": 2.0,
                "peak_cooling_mae": 1.0,
                "temperature_mae": 0.7,
                "overall_pass": True,
            },
            "",
        )

    with patch.object(s3_worker_module, "get_aws_clients", lambda: clients), \
         patch.object(s3_worker_module, "push_result_to_s3", lambda *a, **k: "s3://bucket/prefix/wu-1.json"), \
         patch.object(s3_worker_module, "run_simulation", fake_simulation):
        result = s3_worker_module.run_worker(wu, state_store=store)

    assert result.error_message is None
    # KPIResult aggregates stats across runs: heating_mae_mean / _std /
    # _min / _max (single-run == rated). The legacy flat field
    # ``heating_mae`` no longer exists on KPIResult.
    assert result.heating_mae_mean == pytest.approx(1.0)
    state = store.get_state("camp-A", "wu-1")
    assert state.status == state_store_module.TaskStatus.COMPLETED
    assert state.metrics["heating_mae_mean"] == pytest.approx(1.0)


def test_run_worker_failure_path(state_store_module, s3_worker_module):
    store = state_store_module.InMemoryStateStore()
    clients = _stub_clients()

    wu = s3_worker_module.WorkUnit(
        work_unit_id="wu-fail",
        campaign_id="camp-A",
        case_id="600",
        parameters={"R_value": 2.0},
        s3_result_prefix="s3://bucket/prefix",
        config={},
    )

    def fake_simulation(_wu):
        return ({"error": "subprocess timeout"}, "")

    with patch.object(s3_worker_module, "get_aws_clients", lambda: clients), \
         patch.object(s3_worker_module, "push_result_to_s3", lambda *a, **k: "s3://bucket/prefix/wu-fail.json"), \
         patch.object(s3_worker_module, "run_simulation", fake_simulation):
        result = s3_worker_module.run_worker(wu, state_store=store)

    assert result.error_message == "subprocess timeout"
    state = store.get_state("camp-A", "wu-fail")
    assert state.status == state_store_module.TaskStatus.FAILED
    assert state.error_message == "subprocess timeout"


# ---------------------------------------------------------------------------
# Coordinator integration: check_campaign_progress uses state-store.
# ---------------------------------------------------------------------------


def test_check_campaign_progress_uses_state_store(state_store_module, cloud_manager_module):
    store = state_store_module.InMemoryStateStore()
    for wid, status in [
        ("wu-1", "running"),
        ("wu-2", "completed"),
        ("wu-3", "failed"),
    ]:
        store.set_state(
            state_store_module.TaskState.now("camp-X", wid, status)
        )
    state = cloud_manager_module.CampaignState(
        campaign_id="camp-X",
        config={"case_id": "600"},
        work_units=[{"work_unit_id": w} for w in ("wu-1", "wu-2", "wu-3")],
        status="created",
        start_time=datetime.now(timezone.utc).isoformat(),
    )

    # Patch the AWS clients out so the S3 fallback cannot fire.
    with patch.object(cloud_manager_module, "get_aws_clients") as fake_clients:
        fake_clients.return_value = _stub_clients()
        # Even if the S3 listing ran, it would observe 0 — the assertion
        # below must therefore come from the state-store path.
        updated = cloud_manager_module.check_campaign_progress(
            state, s3_bucket="x", s3_prefix="y", state_store=store
        )

    assert updated.completed_units == 1
    assert updated.failed_units == 1
    # wu-1 is running so the campaign should be marked running, not yet completed.
    assert updated.status == "running"

    # Now finish the running task.
    store.set_state(state_store_module.TaskState.now("camp-X", "wu-1", "completed"))
    with patch.object(cloud_manager_module, "get_aws_clients") as fake_clients:
        fake_clients.return_value = _stub_clients()
        updated = cloud_manager_module.check_campaign_progress(
            state, s3_bucket="x", s3_prefix="y", state_store=store
        )
    assert updated.status == "completed"
    assert updated.completed_units == 2
    assert updated.failed_units == 1


def test_resolve_state_store_memory_short_circuits(cloud_manager_module):
    store = cloud_manager_module._resolve_state_store("memory")
    assert store is not None
    assert store.backend_name == "memory"


def test_resolve_state_store_auto_defaults_to_none(cloud_manager_module, monkeypatch):
    monkeypatch.delenv("FLUXION_STATE_STORE", raising=False)
    monkeypatch.delenv("FLUXION_CAMPAIGN_TABLE", raising=False)
    monkeypatch.delenv("FLUXION_REDIS_URL", raising=False)
    assert cloud_manager_module._resolve_state_store("auto") is None


def test_resolve_state_store_auto_picks_dynamodb_from_legacy_env(
    cloud_manager_module, monkeypatch
):
    monkeypatch.delenv("FLUXION_STATE_STORE", raising=False)
    monkeypatch.delenv("FLUXION_REDIS_URL", raising=False)
    monkeypatch.setenv("FLUXION_CAMPAIGN_TABLE", "fluxion-campaign-state")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    store = cloud_manager_module._resolve_state_store("auto")
    assert store is not None
    assert store.backend_name == "dynamodb"


def test_full_worker_coordinator_loop(state_store_module, s3_worker_module, cloud_manager_module):
    """End-to-end: workers push to the store, coordinator aggregates."""
    store = state_store_module.InMemoryStateStore()
    clients = _stub_clients()

    work_units = [
        s3_worker_module.WorkUnit(
            work_unit_id=f"wu-{i:03d}",
            campaign_id="camp-loop",
            case_id="600",
            parameters={"R_value": 1.0 + i * 0.1},
            s3_result_prefix="s3://bucket/prefix",
            config={},
        )
        for i in range(4)
    ]

    def fake_simulation(wu):
        if wu.work_unit_id == "wu-002":
            return {"error": "synthetic failure"}, ""
        return (
            {
                "heating_mae": 1.0,
                "cooling_mae": 0.5,
                "peak_heating_mae": 2.0,
                "peak_cooling_mae": 1.0,
                "temperature_mae": 0.7,
                "overall_pass": True,
            },
            "",
        )

    with patch.object(s3_worker_module, "get_aws_clients", lambda: clients), \
         patch.object(s3_worker_module, "push_result_to_s3", lambda *a, **k: ""), \
         patch.object(s3_worker_module, "run_simulation", fake_simulation):
        for wu in work_units:
            s3_worker_module.run_worker(wu, state_store=store)

    state = cloud_manager_module.CampaignState(
        campaign_id="camp-loop",
        config={"case_id": "600"},
        work_units=[{"work_unit_id": wu.work_unit_id} for wu in work_units],
        status="created",
        start_time=datetime.now(timezone.utc).isoformat(),
    )

    with patch.object(cloud_manager_module, "get_aws_clients", lambda: _stub_clients()):
        progress = cloud_manager_module.check_campaign_progress(
            state, s3_bucket="x", s3_prefix="y", state_store=store
        )

    assert progress.completed_units == 3
    assert progress.failed_units == 1
    assert progress.status == "completed"
    # Sanity-check the underlying aggregate too.
    assert store.aggregate_progress("camp-loop", total=4).is_complete
