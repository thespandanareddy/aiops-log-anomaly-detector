-- SQLite Schema for AIOps Log Anomaly Detector

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    log_source TEXT NOT NULL,
    risk_score REAL NOT NULL,
    severity TEXT NOT NULL,
    sequence_score REAL NOT NULL,
    ae_score REAL NOT NULL,
    suspicious_events_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    drift_flag INTEGER DEFAULT 0,
    status TEXT DEFAULT 'NEW',
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);

CREATE TABLE IF NOT EXISTS drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    log_source TEXT NOT NULL,
    psi_score REAL NOT NULL,
    novel_template_ratio REAL NOT NULL,
    total_events_analyzed INTEGER NOT NULL,
    drift_status TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_drift_status ON drift_events(drift_status);

CREATE TABLE IF NOT EXISTS model_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    total_templates INTEGER NOT NULL,
    training_samples INTEGER NOT NULL,
    val_f1 REAL,
    val_precision REAL,
    val_recall REAL,
    val_fpr REAL,
    threshold_sequence REAL,
    threshold_ae REAL,
    status TEXT DEFAULT 'ACTIVE',
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
