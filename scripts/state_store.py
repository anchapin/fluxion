#!/usr/bin/env python3
"""
State Store for Fluxion Campaigns (Issue #1787 / T7.3)
======================================================

Workers write per-task completion directly to a cloud state-store
(DynamoDB or Redis) rather than back to the client. The coordinator
aggregates state-store entries to compute overall campaign progress.

Backends
--------
- :class:`DynamoDBStateStore`  — Atomic ``UpdateItem`` with a composite key
  (``campaign_id`` PK + ``work_unit_id`` SK). Status / metrics are stored as
  typed attributes so the coordinator can ``Query`` per campaign without a
  full table scan.
- :class:`RedisStateStore`     — One hash per task plus a sorted set per
  campaign for cheap ordering and progress queries.
- :class:`InMemoryStateStore`  — Process-local, useful for unit tests and
  the local-dev path that has no cloud credentials.

Selection is driven by ``FLUXION_STATE_STORE`` (values: ``dynamodb``,
``redis``, ``memory``) plus backend-specific configuration. All backends
share the :class:`StateStore` interface so workers and the coordinator
can be backend-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

LOGGER = logging.getLogger("fluxion.state_store")

# DynamoDB limits work-unit IDs up to 2048 bytes; we stay well below.
MAX_WORK_UNIT_ID_LEN = 512
MAX_CAMPAIGN_ID_LEN = 256


class TaskStatus(str, Enum):
    """Lifecycle of a single work unit within a campaign."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def coerce(cls, value: Any) -> "TaskStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            for member in cls:
                if member.value == key:
                    return member
        raise ValueError(f"Unknown TaskStatus: {value!r}")


@dataclass
class TaskState:
    """A single per-task entry written by a worker."""

    campaign_id: str
    work_unit_id: str
    status: TaskStatus
    timestamp: str
    error_message: Optional[str] = None
    metrics: Optional[dict[str, float]] = None

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id is required")
        if not self.work_unit_id:
            raise ValueError("work_unit_id is required")
        if len(self.campaign_id) > MAX_CAMPAIGN_ID_LEN:
            raise ValueError(f"campaign_id exceeds {MAX_CAMPAIGN_ID_LEN} chars")
        if len(self.work_unit_id) > MAX_WORK_UNIT_ID_LEN:
            raise ValueError(f"work_unit_id exceeds {MAX_WORK_UNIT_ID_LEN} chars")
        self.status = TaskStatus.coerce(self.status)
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @classmethod
    def now(
        cls,
        campaign_id: str,
        work_unit_id: str,
        status: TaskStatus | str,
        *,
        error_message: Optional[str] = None,
        metrics: Optional[dict[str, float]] = None,
    ) -> "TaskState":
        return cls(
            campaign_id=campaign_id,
            work_unit_id=work_unit_id,
            status=TaskStatus.coerce(status),
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_message=error_message,
            metrics=metrics,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "work_unit_id": self.work_unit_id,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        return cls(
            campaign_id=str(data["campaign_id"]),
            work_unit_id=str(data["work_unit_id"]),
            status=TaskStatus.coerce(data["status"]),
            timestamp=str(data.get("timestamp") or ""),
            error_message=data.get("error_message"),
            metrics=data.get("metrics"),
        )


@dataclass
class CampaignProgress:
    """Aggregated campaign-level progress computed from state-store entries."""

    campaign_id: str
    total: int = 0
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    work_unit_ids: list[str] = field(default_factory=list)

    @property
    def finished(self) -> int:
        """Tasks in a terminal state (completed + failed)."""
        return self.completed + self.failed

    @property
    def in_flight(self) -> int:
        return self.running

    @property
    def is_complete(self) -> bool:
        """``True`` iff every expected task has reached a terminal state."""
        return self.total > 0 and self.finished >= self.total

    @property
    def progress_pct(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(100.0, self.finished * 100.0 / self.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "total": self.total,
            "pending": self.pending,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "finished": self.finished,
            "in_flight": self.in_flight,
            "is_complete": self.is_complete,
            "progress_pct": round(self.progress_pct, 2),
            "work_unit_ids": list(self.work_unit_ids),
        }


class StateStore(ABC):
    """Abstract state-store interface shared by all backends."""

    backend_name: str = "abstract"

    @abstractmethod
    def set_state(self, state: TaskState) -> None:
        """Idempotently write or update the state of a single task."""

    @abstractmethod
    def get_state(self, campaign_id: str, work_unit_id: str) -> Optional[TaskState]:
        """Return the current state of a single task, or ``None``."""

    @abstractmethod
    def list_states(self, campaign_id: str) -> list[TaskState]:
        """Return all known task states for a campaign (any status)."""

    def aggregate_progress(
        self, campaign_id: str, total: Optional[int] = None
    ) -> CampaignProgress:
        """Aggregate per-task states into a campaign-wide progress snapshot.

        ``total`` is the number of tasks the campaign manager expects; when
        omitted it is inferred from the number of entries seen so far.
        """
        states = self.list_states(campaign_id)
        progress = CampaignProgress(campaign_id=campaign_id)
        for s in states:
            progress.work_unit_ids.append(s.work_unit_id)
            if s.status == TaskStatus.COMPLETED:
                progress.completed += 1
            elif s.status == TaskStatus.FAILED:
                progress.failed += 1
            elif s.status == TaskStatus.RUNNING:
                progress.running += 1
            else:
                progress.pending += 1
        # Tasks may have been claimed by a worker before the coordinator
        # knew about them — prefer the larger of the two counts.
        progress.total = max(total or 0, len(states))
        return progress

    # --- Convenience constructors -----------------------------------------
    @classmethod
    def from_env(cls) -> "StateStore":
        backend = (os.environ.get("FLUXION_STATE_STORE") or "").strip().lower()
        if backend == "dynamodb":
            return DynamoDBStateStore.from_env()
        if backend == "redis":
            return RedisStateStore.from_env()
        if backend in ("memory", "inmemory", ""):
            return InMemoryStateStore()
        raise ValueError(f"Unknown FLUXION_STATE_STORE backend: {backend!r}")

    @classmethod
    def create(
        cls,
        backend: str,
        *,
        dynamodb_client: Any = None,
        redis_client: Any = None,
        table_name: Optional[str] = None,
        key_prefix: Optional[str] = None,
        redis_url: Optional[str] = None,
    ) -> "StateStore":
        backend = (backend or "memory").strip().lower()
        if backend == "dynamodb":
            client = dynamodb_client
            if client is None:
                # Defer import / construction so this remains test-friendly.
                return DynamoDBStateStore.from_env()
            return DynamoDBStateStore(
                dynamodb_client=client,
                table_name=table_name
                or os.environ.get("FLUXION_CAMPAIGN_TABLE", "fluxion-campaign-state"),
            )
        if backend == "redis":
            if redis_client is not None:
                return RedisStateStore(
                    redis_client=redis_client,
                    key_prefix=key_prefix
                    or os.environ.get("FLUXION_REDIS_PREFIX", "fluxion:campaign"),
                )
            return RedisStateStore(
                key_prefix=key_prefix
                or os.environ.get("FLUXION_REDIS_PREFIX", "fluxion:campaign"),
                redis_url=redis_url or os.environ.get("FLUXION_REDIS_URL"),
            )
        if backend in ("memory", "inmemory", ""):
            return InMemoryStateStore()
        raise ValueError(f"Unknown state-store backend: {backend!r}")


class InMemoryStateStore(StateStore):
    """Process-local store useful for unit tests and the local-dev path."""

    backend_name = "memory"

    def __init__(self) -> None:
        # {campaign_id: {work_unit_id: TaskState}}
        self._states: dict[str, dict[str, TaskState]] = {}

    def set_state(self, state: TaskState) -> None:
        self._states.setdefault(state.campaign_id, {})[state.work_unit_id] = state

    def get_state(self, campaign_id: str, work_unit_id: str) -> Optional[TaskState]:
        return self._states.get(campaign_id, {}).get(work_unit_id)

    def list_states(self, campaign_id: str) -> list[TaskState]:
        return list(self._states.get(campaign_id, {}).values())

    def clear(self, campaign_id: Optional[str] = None) -> None:
        """Test helper. Removes entries for a campaign (or all campaigns)."""
        if campaign_id is None:
            self._states.clear()
        else:
            self._states.pop(campaign_id, None)


class DynamoDBStateStore(StateStore):
    """DynamoDB-backed state store.

    Schema
    ------
    Partition key : ``campaign_id`` (S)
    Sort key      : ``work_unit_id`` (S)

    Item attributes::

        status        (S) — pending | running | completed | failed
        timestamp     (S) — ISO 8601 UTC
        error_message (S, optional)
        metrics       (M, optional) — {heating_mae, cooling_mae, ...}

    The table is expected to be created out-of-band by IaC. The helper
    :meth:`ensure_table` will create it on demand for tests and local dev.
    """

    backend_name = "dynamodb"

    def __init__(self, dynamodb_client: Any, table_name: str) -> None:
        if not dynamodb_client:
            raise ValueError("dynamodb_client is required for DynamoDBStateStore")
        if not table_name:
            raise ValueError("table_name is required for DynamoDBStateStore")
        self.client = dynamodb_client
        self.table_name = table_name

    @classmethod
    def from_env(cls) -> "DynamoDBStateStore":
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "boto3 is required for DynamoDBStateStore; install with `pip install boto3`"
            ) from exc
        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        return cls(
            dynamodb_client=session.client("dynamodb"),
            table_name=os.environ.get(
                "FLUXION_CAMPAIGN_TABLE", "fluxion-campaign-state"
            ),
        )

    def ensure_table(self) -> None:
        """Create the DynamoDB table if it does not already exist (test helper)."""
        try:
            self.client.describe_table(TableName=self.table_name)
            return
        except Exception:
            pass
        try:
            from botocore.exceptions import ClientError  # type: ignore
        except ImportError:  # pragma: no cover - import guard
            ClientError = Exception

        try:
            self.client.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {"AttributeName": "campaign_id", "KeyType": "HASH"},
                    {"AttributeName": "work_unit_id", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "campaign_id", "AttributeType": "S"},
                    {"AttributeName": "work_unit_id", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
        except ClientError as exc:  # pragma: no cover - best-effort
            LOGGER.warning("ensure_table: create_table failed: %s", exc)

    def set_state(self, state: TaskState) -> None:
        set_parts = ["#s = :s", "#t = :t"]
        remove_parts: list[str] = []
        expr_names: dict[str, str] = {"#s": "status", "#t": "timestamp"}
        expr_values: dict[str, Any] = {
            ":s": {"S": state.status.value},
            ":t": {"S": state.timestamp},
        }
        if state.error_message is not None:
            set_parts.append("#e = :e")
            expr_names["#e"] = "error_message"
            expr_values[":e"] = {"S": state.error_message}
        else:
            remove_parts.append("#e")
            expr_names["#e"] = "error_message"
        if state.metrics is not None:
            set_parts.append("#m = :m")
            expr_names["#m"] = "metrics"
            expr_values[":m"] = {"M": _to_dynamodb_map(state.metrics)}
        else:
            remove_parts.append("#m")
            expr_names["#m"] = "metrics"

        update_expr = "SET " + ", ".join(set_parts)
        if remove_parts:
            update_expr += " REMOVE " + ", ".join(remove_parts)

        self.client.update_item(
            TableName=self.table_name,
            Key={
                "campaign_id": {"S": state.campaign_id},
                "work_unit_id": {"S": state.work_unit_id},
            },
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )

    def get_state(self, campaign_id: str, work_unit_id: str) -> Optional[TaskState]:
        response = self.client.get_item(
            TableName=self.table_name,
            Key={
                "campaign_id": {"S": campaign_id},
                "work_unit_id": {"S": work_unit_id},
            },
        )
        item = response.get("Item")
        if not item:
            return None
        return _from_dynamodb_item(item)

    def list_states(self, campaign_id: str) -> list[TaskState]:
        states: list[TaskState] = []
        kwargs: dict[str, Any] = {
            "TableName": self.table_name,
            "KeyConditionExpression": "campaign_id = :cid",
            "ExpressionAttributeValues": {":cid": {"S": campaign_id}},
        }
        while True:
            response = self.client.query(**kwargs)
            for item in response.get("Items", []):
                states.append(_from_dynamodb_item(item))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key
        return states


class RedisStateStore(StateStore):
    """Redis-backed state store.

    Layout
    ------
    ``fluxion:campaign:{campaign_id}:task:{work_unit_id}`` — hash with fields:
        status        (str)
        timestamp     (str — ISO 8601)
        error_message (str, optional)
        metrics       (JSON-encoded string)

    ``fluxion:campaign:{campaign_id}:tasks`` — sorted set whose members are
    work-unit IDs scored by epoch-ms, used for cheap enumeration.
    """

    backend_name = "redis"

    def __init__(
        self,
        redis_client: Any = None,
        key_prefix: str = "fluxion:campaign",
        redis_url: Optional[str] = None,
    ) -> None:
        self.key_prefix = key_prefix
        if redis_client is None:
            try:
                import redis  # type: ignore
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "redis is required for RedisStateStore; install with `pip install redis`"
                ) from exc
            redis_client = redis.Redis.from_url(
                redis_url
                or os.environ.get("FLUXION_REDIS_URL", "redis://localhost:6379/0")
            )
        self.client = redis_client

    @classmethod
    def from_env(cls) -> "RedisStateStore":
        return cls(
            key_prefix=os.environ.get("FLUXION_REDIS_PREFIX", "fluxion:campaign")
        )

    def _task_key(self, campaign_id: str, work_unit_id: str) -> str:
        return f"{self.key_prefix}:{campaign_id}:task:{work_unit_id}"

    def _index_key(self, campaign_id: str) -> str:
        return f"{self.key_prefix}:{campaign_id}:tasks"

    @staticmethod
    def _epoch_ms(timestamp: str) -> float:
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.timestamp() * 1000.0
        except Exception:
            return 0.0

    def set_state(self, state: TaskState) -> None:
        mapping: dict[str, str] = {
            "status": state.status.value,
            "timestamp": state.timestamp,
        }
        task_key = self._task_key(state.campaign_id, state.work_unit_id)
        pipe = self.client.pipeline()

        if state.error_message is not None:
            mapping["error_message"] = state.error_message
        else:
            pipe.hdel(task_key, "error_message")
        if state.metrics is not None:
            mapping["metrics"] = json.dumps(state.metrics)
        else:
            pipe.hdel(task_key, "metrics")

        pipe.hset(task_key, mapping=mapping)
        pipe.zadd(
            self._index_key(state.campaign_id),
            {state.work_unit_id: self._epoch_ms(state.timestamp)},
        )
        pipe.execute()

    def get_state(self, campaign_id: str, work_unit_id: str) -> Optional[TaskState]:
        raw = self.client.hgetall(self._task_key(campaign_id, work_unit_id))
        if not raw:
            return None
        decoded = _decode_redis_hash(raw)
        metrics_raw = decoded.pop("metrics", None)
        metrics = json.loads(metrics_raw) if metrics_raw else None
        return TaskState(
            campaign_id=campaign_id,
            work_unit_id=work_unit_id,
            status=TaskStatus.coerce(decoded.get("status", "pending")),
            timestamp=decoded.get("timestamp", ""),
            error_message=decoded.get("error_message"),
            metrics=metrics,
        )

    def list_states(self, campaign_id: str) -> list[TaskState]:
        ids = self.client.zrange(self._index_key(campaign_id), 0, -1)
        if not ids:
            return []
        pipe = self.client.pipeline()
        for wid in ids:
            pipe.hgetall(self._task_key(campaign_id, wid))
        results = pipe.execute()
        out: list[TaskState] = []
        for raw_work_unit_id, raw in zip(ids, results):
            if not raw:
                continue
            decoded = _decode_redis_hash(raw)
            metrics_raw = decoded.pop("metrics", None)
            metrics = json.loads(metrics_raw) if metrics_raw else None
            out.append(
                TaskState(
                    campaign_id=campaign_id,
                    work_unit_id=_ensure_str(raw_work_unit_id),
                    status=TaskStatus.coerce(decoded.get("status", "pending")),
                    timestamp=decoded.get("timestamp", ""),
                    error_message=decoded.get("error_message"),
                    metrics=metrics,
                )
            )
        return out

    def clear(self, campaign_id: Optional[str] = None) -> None:
        """Test helper: drop all task entries for a campaign (or all)."""
        if campaign_id is None:
            # Caller is responsible for full cleanup; do nothing destructive.
            return
        ids = self.client.zrange(self._index_key(campaign_id), 0, -1)
        pipe = self.client.pipeline()
        for wid in ids:
            pipe.delete(self._task_key(campaign_id, wid))
        pipe.delete(self._index_key(campaign_id))
        pipe.execute()


# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------


def _to_dynamodb_map(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat metric dict to DynamoDB's typed-map format."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = {"BOOL": value}
        elif isinstance(value, (int, float)):
            out[key] = {"N": str(value)}
        elif isinstance(value, str):
            out[key] = {"S": value}
        else:
            out[key] = {"S": json.dumps(value)}
    return out


def _from_dynamodb_item(item: dict[str, Any]) -> TaskState:
    metrics: Optional[dict[str, float]] = None
    raw_metrics = item.get("metrics")
    if isinstance(raw_metrics, dict) and "M" in raw_metrics:
        metrics_map = raw_metrics["M"]
        metrics = {}
        for k, v in metrics_map.items():
            if "N" in v:
                try:
                    metrics[k] = float(v["N"])
                except ValueError:
                    continue
            elif "S" in v:
                metrics[k] = v["S"]
    return TaskState(
        campaign_id=item["campaign_id"]["S"],
        work_unit_id=item["work_unit_id"]["S"],
        status=TaskStatus.coerce(item.get("status", {}).get("S", "pending")),
        timestamp=item.get("timestamp", {}).get("S", ""),
        error_message=item.get("error_message", {}).get("S"),
        metrics=metrics,
    )


def _decode_redis_hash(raw: Any) -> dict[str, str]:
    """Decode a redis-py hash (bytes/str keys) into ``str -> str``."""
    decoded: dict[str, str] = {}
    for k, v in raw.items():
        key = _ensure_str(k)
        value = _ensure_str(v)
        decoded[key] = value
    return decoded


def _ensure_str(value: Any) -> str:
    """Decode ``bytes`` (or bytearray) to ``str``; pass-through otherwise."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


__all__ = [
    "TaskStatus",
    "TaskState",
    "CampaignProgress",
    "StateStore",
    "InMemoryStateStore",
    "DynamoDBStateStore",
    "RedisStateStore",
]


if __name__ == "__main__":  # pragma: no cover
    # Tiny smoke test: pick a backend and round-trip one task.
    logging.basicConfig(level=logging.INFO)
    store = StateStore.from_env()
    print(f"Using backend: {store.backend_name}", file=sys.stderr)
    sample = TaskState.now(
        campaign_id="smoke-test",
        work_unit_id="wu-0001",
        status=TaskStatus.RUNNING,
    )
    store.set_state(sample)
    fetched = store.get_state(sample.campaign_id, sample.work_unit_id)
    assert fetched is not None and fetched.status == TaskStatus.RUNNING
    progress = store.aggregate_progress(sample.campaign_id, total=1)
    print(progress.to_dict())
