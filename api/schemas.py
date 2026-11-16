"""
Pydantic Request and Response Schemas for the FastAPI REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    active_model_version: Optional[str] = None
    is_trained: bool
    total_templates: int
    db_connected: bool


class TrainRequest(BaseModel):
    dataset_name: str = Field(default="hdfs", description="Dataset identifier ('hdfs', 'bgl', or 'custom')")
    max_lines: Optional[int] = Field(default=2000, description="Max lines to ingest for training")
    lstm_epochs: int = Field(default=8, ge=1, le=50)
    ae_epochs: int = Field(default=10, ge=1, le=50)


class TrainResponse(BaseModel):
    status: str
    dataset: str
    total_records: int
    total_sessions: int
    train_sessions: int
    val_sessions: int
    test_sessions: int
    vocab_size: int
    calibrated_threshold: float
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]


class DetectRequest(BaseModel):
    log_text: Optional[str] = Field(default=None, description="Raw multiline log text")
    log_type: str = Field(default="custom", description="'hdfs', 'bgl', or 'custom'")
    persist_alerts: bool = Field(default=True, description="Whether to persist detected alerts to SQLite")


class SuspiciousEventItem(BaseModel):
    step: int
    event_id: str
    template: str
    timestamp: str
    level: str
    raw_log: str
    reason: str


class AlertResponseItem(BaseModel):
    alert_id: Optional[int]
    session_id: str
    timestamp: str
    log_source: str
    risk_score: float
    severity: str
    sequence_score: Optional[float] = None
    ae_score: Optional[float] = None
    explanation: str
    suspicious_events: List[Dict[str, Any]] = []
    drift_flag: bool = False
    status: str = "NEW"


class AlertStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="'NEW', 'ACKNOWLEDGED', or 'RESOLVED'")


class DriftResponse(BaseModel):
    recent_events: List[Dict[str, Any]]
    current_status: str
    latest_psi: float
    latest_novel_ratio: float


class RefreshRequest(BaseModel):
    log_text: Optional[str] = Field(default=None, description="Operational logs to adapt model")
    log_type: str = Field(default="custom")
    epochs: int = Field(default=5, ge=1, le=20)


class RefreshResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
    pseudo_normal_added: Optional[int] = None
    suspicious_excluded: Optional[int] = None
    total_training_samples: Optional[int] = None
    calibrated_threshold: Optional[float] = None
    reason: Optional[str] = None
