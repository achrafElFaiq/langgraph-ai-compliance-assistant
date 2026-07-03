"""Integration tests for the health/readiness endpoints."""
import pytest

pytestmark = pytest.mark.integration


async def test_health_liveness_ok(client):
    """GET /health returns 200 with status 'ok' — liveness needs no dependencies."""
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_ready_returns_503_when_db_down(client, monkeypatch):
    """GET /health/ready returns 503 when the DB is unreachable.

    Spec (task): readiness returns 200 ONLY when the DB is live, 503 otherwise —
    so an orchestrator won't route traffic to an instance that can't serve. This
    test fails if readiness masks a dead DB behind a 200.
    """
    import src.api.routes.health as health_mod

    class DownStore:
        async def retrieve(self, *a, **k):
            raise RuntimeError("database connection is not initialized")

    monkeypatch.setattr(health_mod, "store", DownStore())

    r = await client.get("/health/ready")
    assert r.status_code == 503


async def test_ready_returns_200_when_db_live(client, live_store):
    """GET /health/ready returns 200 when the DB connection is live.

    Requires a reachable test DB (skips otherwise). Matters as the positive half
    of the readiness contract.
    """
    r = await client.get("/health/ready")
    assert r.status_code == 200
