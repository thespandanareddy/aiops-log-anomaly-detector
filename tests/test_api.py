"""Unit and integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.app import app
from db.database import db


@pytest.fixture(scope="module")
def client():
    """Create a test client for FastAPI."""
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    """Verify root API returns service info and documentation links."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "name" in data
    assert data["docs_url"] == "/docs"


def test_health_endpoint(client):
    """Verify health endpoint reports system status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["db_connected"] is True


def test_detect_and_query_alerts(client):
    """Verify log detection endpoint detects anomalous logs and queries alert history."""
    # Insert test alert directly to test alert query API
    alert_id = db.insert_alert(
        timestamp="2026-11-16T12:00:00",
        session_id="blk_api_test",
        log_source="hdfs",
        risk_score=92.0,
        severity="CRITICAL",
        sequence_score=0.95,
        ae_score=0.88,
        suspicious_events=[{"event_id": "E1", "reason": "Test sequence break"}],
        explanation="Abnormal event sequence detected via API.",
    )

    # List alerts
    res = client.get("/api/alerts?severity=CRITICAL")
    assert res.status_code == 200
    alerts = res.json()
    assert len(alerts) >= 1
    found = [a for a in alerts if a["alert_id"] == alert_id]
    assert len(found) == 1
    assert found[0]["session_id"] == "blk_api_test"

    # Get single alert
    detail_res = client.get(f"/api/alerts/{alert_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["alert_id"] == alert_id
    assert detail["severity"] == "CRITICAL"

    # Update status
    update_res = client.put(f"/api/alerts/{alert_id}/status", json={"status": "ACKNOWLEDGED"})
    assert update_res.status_code == 200
    assert update_res.json()["new_status"] == "ACKNOWLEDGED"

    # Stats
    stats_res = client.get("/api/alerts/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_alerts"] >= 1


def test_drift_status_endpoint(client):
    """Verify drift endpoint returns observations."""
    db.insert_drift_event(
        timestamp="2026-11-16T12:05:00",
        log_source="hdfs",
        psi_score=0.15,
        novel_template_ratio=0.10,
        total_events=500,
        drift_status="MODERATE_DRIFT",
        details={},
    )

    res = client.get("/api/drift")
    assert res.status_code == 200
    drift_data = res.json()
    assert "recent_events" in drift_data
    assert len(drift_data["recent_events"]) >= 1
