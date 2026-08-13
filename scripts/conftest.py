# scripts/conftest.py
# -------------------
# Shared pytest fixtures and path configuration for the OSimFlow Python
# orchestration test suite (Issue #1847).
#
# Goals:
#   * Make `scripts/` importable without per-test sys.path gymnastics.
#   * Provide deterministic, no-I/O default parameter specs.
#   * Provide hermetic fakes for the AWS / cloud modules so tests do not
#     require credentials, network, DynamoDB, Redis, or K8s.
#
# Coverage must remain ≥60% line coverage on each of:
#   - scripts/cloud_campaign_manager.py
#   - scripts/autonomous_parameter_sweep.py
#   - scripts/ashrae_benchmark_harness.py

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup: make `scripts/` importable.
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Environment hygiene.
# ---------------------------------------------------------------------------

# Prevent any test from accidentally reaching the real cloud: clear every
# Fluxion/Cloud env var that would trigger a live roundtrip.  Tests that
# need to exercise env-driven code paths should set them locally via
# `monkeypatch.setenv` so the cleanup is automatic.
_CLOUD_ENV_VARS = (
    "FLUXION_S3_BUCKET",
    "FLUXION_S3_PREFIX",
    "FLUXION_SNS_TOPIC_ARN",
    "FLUXION_WEBHOOK_URL",
    "FLUXION_EMAIL_FROM",
    "FLUXION_EMAIL_TO",
    "FLUXION_EMAIL_CC",
    "FLUXION_EMAIL_API_ENDPOINT",
    "FLUXION_EMAIL_API_AUTH",
    "FLUXION_EMAIL_DOWNLOAD_URL",
    "FLUXION_STATE_STORE",
    "FLUXION_CAMPAIGN_TABLE",
    "FLUXION_REDIS_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "DWAVE_API_TOKEN",
)


@pytest.fixture(autouse=True)
def _scrub_cloud_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no test leaks real AWS / DWAVE / state-store credentials."""
    for var in _CLOUD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Stub AWS clients and S3/DynamoDB/Redis backends.
# ---------------------------------------------------------------------------


@dataclass
class _FakeS3Client:
    """In-memory S3 mock covering the operations used by cloud_campaign_manager."""

    bucket_objects: dict[str, dict[str, bytes]] = field(default_factory=dict)
    listed_objects: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    head_should_fail: bool = False

    def put_object(
        self, Bucket: str, Key: str, Body: bytes = b"", **_: Any
    ) -> dict[str, Any]:
        self.bucket_objects.setdefault(Bucket, {})[Key] = (
            Body if isinstance(Body, (bytes, bytearray)) else str(Body).encode("utf-8")
        )
        return {"ETag": '"deadbeef"'}

    def get_object(self, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        body = self.bucket_objects.get(Bucket, {}).get(Key)
        if body is None:
            # ``get_campaign_state`` only suppresses ClientError when the
            # Error.Code is exactly the string ``"404"``.  Real S3 returns
            # ``"NoSuchKey"``; we mirror the literal that the helper checks
            # for.  Other tests can patch the bucket_objects to simulate
            # real ``NoSuchKey`` propagation.
            raise _make_client_error_for_get(404, "404", Key)
        return {"Body": _FakeBody(body)}

    def head_object(self, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        if self.head_should_fail:
            raise _make_client_error(404, "NotFound", Key)
        if Key not in self.bucket_objects.get(Bucket, {}):
            raise _make_client_error(404, "NoSuchKey", Key)
        return {}

    def get_paginator(self, name: str) -> "_FakeS3Paginator":
        assert name == "list_objects_v2"
        return _FakeS3Paginator(self.listed_objects)


@dataclass
class _FakeBody:
    data: bytes

    def read(self) -> bytes:
        return self.data


@dataclass
class _FakeS3Paginator:
    listed_objects: dict[str, list[dict[str, Any]]]

    def paginate(self, Bucket: str, Prefix: str, **_):  # noqa: A003  (boto3 kw)
        objs = self.listed_objects.get(Bucket, [])
        filtered = [o for o in objs if o.get("Key", "").startswith(Prefix)]
        yield {"Contents": filtered}


class _FakeSNSClient:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    def publish(self, **kwargs: Any) -> dict[str, Any]:
        self.published.append(kwargs)
        return {"MessageId": "fake-message-id"}


def _make_client_error(code: int, error_code: str, key: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": error_code, "Message": f"missing {key}"},
            "ResponseMetadata": {"HTTPStatusCode": code},
        },
        "HeadObject",
    )


def _make_client_error_for_get(code: int, error_code: str, key: str) -> Exception:
    """Like ``_make_client_error`` but stamps the operation as ``GetObject``."""
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": error_code, "Message": f"missing {key}"},
            "ResponseMetadata": {"HTTPStatusCode": code},
        },
        "GetObject",
    )


@pytest.fixture
def fake_aws_clients(monkeypatch: pytest.MonkeyPatch):
    """Return a dict of fake AWS clients and patch ``get_aws_clients`` to use them.

    The cloud_campaign_manager code calls ``get_aws_clients()`` at the top
    of every public function.  We override the helper to return our dict so
    tests never touch the real boto3 session.
    """
    s3 = _FakeS3Client()
    sns = _FakeSNSClient()
    clients = {
        "s3": s3,
        "sns": sns,
        "dynamodb": MagicMock(name="dynamodb"),
        "sts": MagicMock(name="sts"),
        "lambda": MagicMock(name="lambda"),
    }

    # `cloud_campaign_manager.py` is imported as a top-level module — there is
    # no `scripts` package (no `scripts/__init__.py`).  Patch the symbol in
    # the module's own namespace.
    import cloud_campaign_manager as _ccm

    monkeypatch.setattr(_ccm, "get_aws_clients", lambda: clients)
    return clients


# ---------------------------------------------------------------------------
# state_store: in-memory backend reused across tests (no Redis / DynamoDB).
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_state_store():
    """Construct an InMemoryStateStore.  Requires the local state_store module."""
    from state_store import InMemoryStateStore

    return InMemoryStateStore()


@pytest.fixture
def populated_state_store(in_memory_state_store):
    """InMemoryStateStore pre-populated with 5 work units: 3 completed, 1 failed, 1 pending."""
    from state_store import TaskState, TaskStatus

    store = in_memory_state_store
    for i, status in enumerate(
        [
            TaskStatus.COMPLETED,
            TaskStatus.COMPLETED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.PENDING,
        ]
    ):
        wu = f"wu-{i:04d}"
        store.set_state(
            TaskState(
                work_unit_id=wu,
                status=status,
                campaign_id="fluxion-camp-test",
                metrics={"heating_mae": 1.2 + i * 0.1}
                if status == TaskStatus.COMPLETED
                else {},
                timestamp="2026-01-01T00:00:00Z",
            )
        )
    return store


# ---------------------------------------------------------------------------
# urlopen stub (used by webhook + email notification paths).
# ---------------------------------------------------------------------------


@dataclass
class _FakeHTTPResponse:
    status: int = 200

    def __enter__(self):  # noqa: D401 - context manager protocol
        return self

    def __exit__(self, *exc):  # noqa: D401 - context manager protocol
        return False

    def getcode(self) -> int:
        return self.status


@pytest.fixture
def fake_urlopen(monkeypatch: pytest.MonkeyPatch):
    """Patch ``urllib.request.urlopen`` with a callable factory.

    Usage:
        fake_urlopen(status=500)             # always returns 500
        fake_urlopen(raises=urllib.error.URLError("boom"))  # raises
        fake_urlopen(capture=callback)        # callback(request) -> response
    """
    state: dict[str, Any] = {"calls": []}

    def _install(
        status: int = 200, raises: Exception | None = None, capture: Any | None = None
    ) -> _FakeHTTPResponse:
        if capture is not None:

            def _factory(request, *args, **kwargs):
                state["calls"].append(request)
                return capture(request)
        elif raises is not None:

            def _factory(request, *args, **kwargs):
                state["calls"].append(request)
                raise raises
        else:

            def _factory(request, *args, **kwargs):
                state["calls"].append(request)
                return _FakeHTTPResponse(status=status)

        monkeypatch.setattr("urllib.request.urlopen", _factory)
        return _FakeHTTPResponse(status=status)

    state["install"] = _install
    state["calls"] = []
    return state


# ---------------------------------------------------------------------------
# subprocess stub (used by sweep + harness scripts).
# ---------------------------------------------------------------------------


@dataclass
class _FakeCompletedProcess:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch):
    """Patch ``subprocess.run`` / ``subprocess.check_output`` for OSimFlow scripts.

    Usage:
        fake_subprocess(run_return=_FakeCompletedProcess(0, stdout="...", stderr=""))
        fake_subprocess(check_output_return="abc1234")
    """
    state: dict[str, Any] = {"run_calls": [], "check_output_calls": []}

    def _install(
        run_return: _FakeCompletedProcess | Exception | None = None,
        check_output_return: str | bytes | Exception | None = None,
    ) -> None:
        def _fake_run(*args, **kwargs):
            state["run_calls"].append((args, kwargs))
            if isinstance(run_return, Exception):
                raise run_return
            return (
                run_return
                if run_return is not None
                else _FakeCompletedProcess(0, "", "")
            )

        def _fake_check_output(*args, **kwargs):
            state["check_output_calls"].append((args, kwargs))
            if isinstance(check_output_return, Exception):
                raise check_output_return
            return check_output_return if check_output_return is not None else ""

        monkeypatch.setattr("subprocess.run", _fake_run)
        monkeypatch.setattr("subprocess.check_output", _fake_check_output)

    state["install"] = _install
    return state


# ---------------------------------------------------------------------------
# Sweep / campaign spec fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def fluxion_model_spec() -> dict[str, Any]:
    """A deterministic FluxionModel spec dict for sweep/aggregation tests."""
    return {
        "case_id": "600",
        "sweep_type": "random",
        "parameters": [
            {
                "name": "R_value",
                "default": 2.0,
                "min": 1.0,
                "max": 5.0,
                "step": 0.5,
                "unit": "m²K/W",
            },
            {
                "name": "wall_thickness",
                "default": 0.15,
                "min": 0.05,
                "max": 0.30,
                "step": 0.05,
                "unit": "m",
            },
        ],
        "max_iterations": 4,
        "samples_per_param": 4,
        "tolerance_mae": 5.0,
    }


@pytest.fixture
def temp_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch trace directory; rerouted .sdd/traces/diagnostic to tmp via cwd."""
    trace = tmp_path / ".sdd" / "traces" / "diagnostic"
    trace.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return trace


@pytest.fixture
def temp_campaign_db(tmp_path: Path) -> Path:
    """Scratch location to mimic a campaign-state DB file (we use JSON here)."""
    db = tmp_path / "campaign_state.json"
    db.write_text("{}")
    return db


# ---------------------------------------------------------------------------
# Wrapper helpers for common assertion patterns.
# ---------------------------------------------------------------------------


@contextmanager
def raises_in_env(env_var: str) -> Iterator[None]:
    """Assert that ``os.environ.get(env_var)`` returns falsy in the with-block."""
    saved = os.environ.pop(env_var, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[env_var] = saved
