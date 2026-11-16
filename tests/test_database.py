"""Unit tests for SQLite Database operations."""

import os
import tempfile
from pathlib import Path
import pytest

from db.database import DatabaseManager


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = Path(f.name)
    
    db_mgr = DatabaseManager(db_path=temp_db_path)
    yield db_mgr
    
    # Cleanup after test
    try:
        if temp_db_path.exists():
            os.remove(temp_db_path)
    except Exception:
        pass


def test_db_initialization(test_db):
    """Verify that tables are created upon initialization."""
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "alerts" in tables
        assert "drift_events" in tables
        assert "model_metadata" in tables


def test_insert_and_get_alert(test_db):
    """Verify inserting and querying alerts."""
    suspicious = [{"event_id": "E12", "template": "Packet responder failed", "line": 42}]
    alert_id = test_db.insert_alert(
        timestamp="2026-11-16T10:00:00",
        session_id="blk_-12345678",
        log_source="hdfs",
        risk_score=88.5,
        severity="HIGH",
        sequence_score=0.92,
        ae_score=0.85,
        suspicious_events=suspicious,
        explanation="Abnormal transition: unexpected failure packet received.",
        drift_flag=False,
        status="NEW"
    )
    assert alert_id > 0

    alert = test_db.get_alert_by_id(alert_id)
    assert alert is not None
    assert alert["session_id"] == "blk_-12345678"
    assert alert["severity"] == "HIGH"
    assert alert["risk_score"] == pytest.approx(88.5, 0.01)
    assert len(alert["suspicious_events"]) == 1
    assert alert["suspicious_events"][0]["event_id"] == "E12"


def test_alert_filtering_and_status_update(test_db):
    """Verify filtering alerts by severity, search query, and status update."""
    test_db.insert_alert(
        timestamp="2026-11-16T10:01:00",
        session_id="blk_111",
        log_source="hdfs",
        risk_score=95.0,
        severity="CRITICAL",
        sequence_score=0.98,
        ae_score=0.92,
        suspicious_events=[],
        explanation="Fatal kernel exception",
    )
    test_db.insert_alert(
        timestamp="2026-11-16T10:02:00",
        session_id="blk_222",
        log_source="bgl",
        risk_score=45.0,
        severity="LOW",
        sequence_score=0.48,
        ae_score=0.42,
        suspicious_events=[],
        explanation="Minor delay observed",
    )

    critical_alerts = test_db.get_alerts(severity="CRITICAL")
    assert len(critical_alerts) == 1
    assert critical_alerts[0]["session_id"] == "blk_111"

    search_alerts = test_db.get_alerts(search_query="kernel")
    assert len(search_alerts) == 1
    assert search_alerts[0]["session_id"] == "blk_111"

    updated = test_db.update_alert_status(critical_alerts[0]["id"], "RESOLVED")
    assert updated is True

    resolved_alerts = test_db.get_alerts(status="RESOLVED")
    assert len(resolved_alerts) == 1
    assert resolved_alerts[0]["id"] == critical_alerts[0]["id"]


def test_drift_and_model_metadata(test_db):
    """Verify drift event logging and model version recording."""
    drift_id = test_db.insert_drift_event(
        timestamp="2026-11-16T11:00:00",
        log_source="hdfs",
        psi_score=0.25,
        novel_template_ratio=0.18,
        total_events=1200,
        drift_status="SIGNIFICANT_DRIFT",
        details={"top_drifted_templates": ["E5", "E18"]}
    )
    assert drift_id > 0

    drift_events = test_db.get_drift_events(limit=5)
    assert len(drift_events) == 1
    assert drift_events[0]["drift_status"] == "SIGNIFICANT_DRIFT"
    assert drift_events[0]["details"]["top_drifted_templates"] == ["E5", "E18"]

    model_id = test_db.save_model_metadata(
        version="v1.0.0",
        model_type="baseline",
        total_templates=28,
        training_samples=1500,
        val_f1=0.932,
        val_precision=0.950,
        val_recall=0.915,
        val_fpr=0.045
    )
    assert model_id > 0

    active_model = test_db.get_active_model_metadata()
    assert active_model is not None
    assert active_model["version"] == "v1.0.0"
    assert active_model["val_f1"] == pytest.approx(0.932, 0.001)
