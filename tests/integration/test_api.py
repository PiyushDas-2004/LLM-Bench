from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from llmbench.api.app import create_app
from llmbench.config.settings import Settings

KEY = "test-api-key-not-a-secret"


@pytest.fixture
async def client(tmp_path: Path, repo_root: Path) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(_env_file=None, api_key=SecretStr(KEY), workspace=str(repo_root))  # type: ignore[call-arg]
    app = create_app(settings, db_url=f"sqlite:///{tmp_path / 'api.db'}", workers=1)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def _config(accuracy: float = 0.85, seed: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "name": "api-test",
        "dataset": "datasets/qa_basic.jsonl",
        "task": "qa",
        "trials": 2,
        "sampling": {"n": 12},
        "models": [
            {
                "id": "bot",
                "provider": "mock",
                "model": "mock-strong",
                "params": {"accuracy": accuracy, "temperature": 0.7, "mock_seed": seed},
            }
        ],
        "prompts": [{"id": "v1", "template": "prompts/qa_v1.j2"}],
        "metrics": ["exact_match", "consistency"],
        **extra,
    }


async def _wait(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    for _ in range(400):
        body: dict[str, Any] = (
            await client.get(f"/runs/{job_id}", headers={"x-api-key": KEY})
        ).json()
        if body.get("status") in {"completed", "failed"}:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError("job did not finish")


async def test_health_and_metrics_are_open(client: httpx.AsyncClient) -> None:
    r = await client.get("/health", headers={"x-request-id": "abc"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.headers["x-request-id"] == "abc"
    m = await client.get("/metrics")
    assert m.status_code == 200
    assert "llmbench_http_requests_total" in m.text


async def test_auth_is_required(client: httpx.AsyncClient) -> None:
    assert (await client.get("/runs")).status_code == 401
    bad = await client.get("/runs", headers={"x-api-key": "wrong"})
    assert bad.status_code == 401
    assert bad.json()["request_id"]


async def test_submit_poll_results_and_compare(client: httpx.AsyncClient) -> None:
    headers = {"x-api-key": KEY}
    r1 = await client.post(
        "/runs", json={"config": _config(), "tags": {"src": "api"}}, headers=headers
    )
    assert r1.status_code == 202
    first = await _wait(client, r1.json()["job_id"])
    assert first["status"] == "completed", first
    assert first["done"] == first["total"] == 24
    r2 = await client.post(
        "/runs", json={"config": _config(0.3, 1), "use_cache": False}, headers=headers
    )
    second = await _wait(client, r2.json()["job_id"])
    runs = (await client.get("/runs", headers=headers)).json()
    assert len(runs) == 2
    info = (await client.get(f"/runs/{first['run_id']}", headers=headers)).json()
    assert info["status"] == "completed"
    results = (await client.get(f"/runs/{r1.json()['job_id']}/results", headers=headers)).json()
    assert results["configs"][0]["metrics"]["exact_match"]["ci"]["n_samples"] == 12
    cmp = await client.post(
        "/compare",
        json={"baseline": first["run_id"], "candidate": second["run_id"], "metrics": ["pass"]},
        headers=headers,
    )
    assert cmp.status_code == 200
    assert cmp.json()["groups"][0]["metrics"][0]["metric"] == "pass"
    jobs = (await client.get("/jobs", headers=headers)).json()
    assert len(jobs) == 2
    metrics = (await client.get("/metrics")).text
    assert "llmbench_runs_total" in metrics
    assert "llmbench_provider_requests_total" in metrics


async def test_errors(client: httpx.AsyncClient) -> None:
    headers = {"x-api-key": KEY}
    missing = await client.get("/runs/does-not-exist", headers=headers)
    assert missing.status_code == 404
    invalid = await client.post("/runs", json={"config": {"name": "x"}}, headers=headers)
    assert invalid.status_code == 400
    escape = await client.post(
        "/runs", json={"config": _config(dataset="../../etc/passwd")}, headers=headers
    )
    assert escape.status_code == 400
    assert "workspace" in escape.json()["error"]
    absolute = _config()
    absolute["prompts"] = [{"id": "v1", "template": "/etc/hosts"}]
    assert (
        await client.post("/runs", json={"config": absolute}, headers=headers)
    ).status_code == 400
    failing = await client.post(
        "/runs", json={"config": _config(dataset="datasets/missing.jsonl")}, headers=headers
    )
    job = await _wait(client, failing.json()["job_id"])
    assert job["status"] == "failed"
    assert "not found" in job["error"]
    bad_cmp = await client.post(
        "/compare", json={"baseline": "a", "candidate": "b"}, headers=headers
    )
    assert bad_cmp.status_code == 404


async def test_insecure_mode_allows_requests(tmp_path: Path, repo_root: Path) -> None:
    settings = Settings(_env_file=None, workspace=str(repo_root))  # type: ignore[call-arg]
    app = create_app(settings, db_url=f"sqlite:///{tmp_path / 'open.db'}", require_auth=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c,
    ):
        assert (await c.get("/runs")).status_code == 200


async def test_cors_for_hosted_dashboard(tmp_path: Path, repo_root: Path) -> None:
    origin = "https://example.github.io"
    settings = Settings(
        _env_file=None, api_key=SecretStr(KEY), workspace=str(repo_root), cors_origins=origin
    )  # type: ignore[call-arg]
    app = create_app(settings, db_url=f"sqlite:///{tmp_path / 'cors.db'}")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c,
    ):
        pre = await c.options(
            "/runs",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert pre.status_code == 200
        assert pre.headers["access-control-allow-origin"] == origin
        other = await c.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers
