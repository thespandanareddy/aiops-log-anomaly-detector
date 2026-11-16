"""
Thread-safe SQLite Database Operations for AIOps Log Anomaly Detector.
Manages persistent alert history, drift logs, and model metadata.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.config import DATABASE_PATH, DB_DIR


class DatabaseManager:
    """Manages SQLite connection lifecycle and query execution."""

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection with Row factory and WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def init_db(self) -> None:
        """Initialize database tables and indexes from schema.sql."""
        schema_file = DB_DIR / "schema.sql"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_file}")

        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    # -------------------------------------------------------------
    # Alert Management
    # -------------------------------------------------------------
    def insert_alert(
        self,
        timestamp: str,
        session_id: str,
        log_source: str,
        risk_score: float,
        severity: str,
        sequence_score: float,
        ae_score: float,
        suspicious_events: List[Dict[str, Any]],
        explanation: str,
        drift_flag: bool = False,
        status: str = "NEW"
    ) -> int:
        """Persist a newly detected anomaly alert."""
        query = """
            INSERT INTO alerts (
                timestamp, session_id, log_source, risk_score, severity,
                sequence_score, ae_score, suspicious_events_json, explanation,
                drift_flag, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        suspicious_json = json.dumps(suspicious_events, default=str)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    timestamp, session_id, log_source, float(risk_score), severity,
                    float(sequence_score), float(ae_score), suspicious_json,
                    explanation, 1 if drift_flag else 0, status
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_alerts(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        log_source: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query alert history with optional filtering and pagination."""
        clauses = []
        params: List[Any] = []

        if severity:
            clauses.append("severity = ?")
            params.append(severity.upper())
        if status:
            clauses.append("status = ?")
            params.append(status.upper())
        if log_source:
            clauses.append("log_source = ?")
            params.append(log_source.lower())
        if search_query:
            clauses.append("(explanation LIKE ? OR session_id LIKE ? OR suspicious_events_json LIKE ?)")
            term = f"%{search_query}%"
            params.extend([term, term, term])

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM alerts
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                try:
                    item["suspicious_events"] = json.loads(item["suspicious_events_json"])
                except Exception:
                    item["suspicious_events"] = []
                results.append(item)
            return results

    def get_alert_by_id(self, alert_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full details of a specific alert."""
        query = "SELECT * FROM alerts WHERE id = ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (alert_id,))
            row = cursor.fetchone()
            if not row:
                return None
            item = dict(row)
            try:
                item["suspicious_events"] = json.loads(item["suspicious_events_json"])
            except Exception:
                item["suspicious_events"] = []
            return item

    def update_alert_status(self, alert_id: int, new_status: str) -> bool:
        """Update the workflow status of an alert (e.g. ACKNOWLEDGED, RESOLVED)."""
        valid_statuses = {"NEW", "ACKNOWLEDGED", "RESOLVED"}
        if new_status.upper() not in valid_statuses:
            raise ValueError(f"Invalid status: {new_status}. Expected one of {valid_statuses}")

        query = "UPDATE alerts SET status = ? WHERE id = ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (new_status.upper(), alert_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Aggregate summary counts of alerts by severity and status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT severity, COUNT(*) FROM alerts GROUP BY severity")
            severity_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status")
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT AVG(risk_score) FROM alerts")
            avg_risk = cursor.fetchone()[0] or 0.0

            return {
                "total_alerts": total_alerts,
                "severity_counts": severity_counts,
                "status_counts": status_counts,
                "average_risk_score": round(float(avg_risk), 2),
            }

    # -------------------------------------------------------------
    # Drift Event Tracking
    # -------------------------------------------------------------
    def insert_drift_event(
        self,
        timestamp: str,
        log_source: str,
        psi_score: float,
        novel_template_ratio: float,
        total_events: int,
        drift_status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Log a concept drift evaluation observation."""
        query = """
            INSERT INTO drift_events (
                timestamp, log_source, psi_score, novel_template_ratio,
                total_events_analyzed, drift_status, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        details_str = json.dumps(details or {}, default=str)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    timestamp, log_source, float(psi_score), float(novel_template_ratio),
                    int(total_events), drift_status, details_str
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_drift_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent drift logs."""
        query = "SELECT * FROM drift_events ORDER BY id DESC LIMIT ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                try:
                    item["details"] = json.loads(item["details_json"])
                except Exception:
                    item["details"] = {}
                results.append(item)
            return results

    # -------------------------------------------------------------
    # Model Metadata & Versioning
    # -------------------------------------------------------------
    def save_model_metadata(
        self,
        version: str,
        model_type: str,
        total_templates: int,
        training_samples: int,
        val_f1: Optional[float] = None,
        val_precision: Optional[float] = None,
        val_recall: Optional[float] = None,
        val_fpr: Optional[float] = None,
        threshold_sequence: Optional[float] = None,
        threshold_ae: Optional[float] = None
    ) -> int:
        """Record model training / refresh checkpoint metadata."""
        now_str = datetime.now().isoformat()
        query = """
            INSERT OR REPLACE INTO model_metadata (
                version, model_type, trained_at, total_templates,
                training_samples, val_f1, val_precision, val_recall,
                val_fpr, threshold_sequence, threshold_ae, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Deactivate previous active models
            conn.execute("UPDATE model_metadata SET status = 'ARCHIVED' WHERE status = 'ACTIVE'")
            cursor.execute(
                query,
                (
                    version, model_type, now_str, total_templates, training_samples,
                    val_f1, val_precision, val_recall, val_fpr,
                    threshold_sequence, threshold_ae
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_active_model_metadata(self) -> Optional[Dict[str, Any]]:
        """Get the currently active model's metadata."""
        query = "SELECT * FROM model_metadata WHERE status = 'ACTIVE' ORDER BY id DESC LIMIT 1"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            return dict(row) if row else None


# Module singleton instance
db = DatabaseManager()
