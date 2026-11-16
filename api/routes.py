"""
FastAPI REST API Endpoints for AIOps Log Anomaly Detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from api.schemas import (
    AlertResponseItem,
    AlertStatusUpdateRequest,
    DetectRequest,
    DriftResponse,
    HealthResponse,
    RefreshRequest,
    RefreshResponse,
    TrainRequest,
    TrainResponse,
)
from db.database import db
from src.config import RAW_DATA_DIR
from src.ingestion.dataset_downloader import ensure_benchmarks_available
from src.ingestion.log_loader import LogRecord, get_log_loader
from src.pipeline import pipeline

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return system and model readiness status."""
    active_model = db.get_active_model_metadata()
    return HealthResponse(
        status="HEALTHY",
        active_model_version=active_model["version"] if active_model else "NONE",
        is_trained=pipeline.is_trained,
        total_templates=len(pipeline.drain_parser.get_all_templates()),
        db_connected=True,
    )


@router.post("/train", response_model=TrainResponse)
def train_pipeline(req: TrainRequest) -> TrainResponse:
    """
    Train detection models on specified benchmark dataset using strict chronological split.
    """
    # Ensure datasets exist
    benchmarks = ensure_benchmarks_available()
    d_name = req.dataset_name.lower()

    if d_name == "hdfs":
        loader = get_log_loader("hdfs", labels_file=benchmarks["hdfs_labels"])
        records = loader.load_file(benchmarks["hdfs_2k"], max_lines=req.max_lines)
    elif d_name == "bgl":
        loader = get_log_loader("bgl")
        records = loader.load_file(benchmarks["bgl_2k"], max_lines=req.max_lines)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset: {req.dataset_name}. Choose 'hdfs' or 'bgl'."
        )

    if not records:
        raise HTTPException(status_code=500, detail="No log records could be ingested.")

    train_res = pipeline.train_on_records(
        records=records,
        log_type=d_name,
        train_ratio=0.60,
        val_ratio=0.20,
        lstm_epochs=req.lstm_epochs,
        ae_epochs=req.ae_epochs,
    )

    return TrainResponse(
        status="SUCCESS",
        dataset=d_name,
        total_records=train_res["total_records"],
        total_sessions=train_res["total_sessions"],
        train_sessions=train_res["train_sessions"],
        val_sessions=train_res["val_sessions"],
        test_sessions=train_res["test_sessions"],
        vocab_size=train_res["vocab_size"],
        calibrated_threshold=train_res["calibrated_threshold"],
        val_metrics=train_res["val_metrics"],
        test_metrics=train_res["test_metrics"],
    )


@router.post("/detect", response_model=List[AlertResponseItem])
def detect_anomalies(req: DetectRequest) -> List[AlertResponseItem]:
    """
    Run real-time anomaly detection on submitted log text.
    Returns detected alerts with plain-English explanations.
    """
    if not pipeline.is_trained:
        # Auto-train on small sample if not yet initialized
        benchmarks = ensure_benchmarks_available()
        loader = get_log_loader("hdfs", labels_file=benchmarks["hdfs_labels"])
        recs = loader.load_file(benchmarks["hdfs_2k"], max_lines=200)
        pipeline.train_on_records(recs, log_type="hdfs", lstm_epochs=3, ae_epochs=3)

    if not req.log_text or not req.log_text.strip():
        raise HTTPException(status_code=400, detail="log_text parameter cannot be empty.")

    loader = get_log_loader(req.log_type)
    lines = req.log_text.strip().splitlines()
    records = [loader.parse_line(line, line_num=idx + 1) for idx, line in enumerate(lines) if line.strip()]

    alerts = pipeline.detect_stream(
        records=records, log_type=req.log_type, persist_alerts=req.persist_alerts
    )

    response = []
    for a in alerts:
        response.append(
            AlertResponseItem(
                alert_id=a.get("alert_id"),
                session_id=a["session_id"],
                timestamp=a["timestamp"],
                log_source=a["log_source"],
                risk_score=a["risk_score"],
                severity=a["severity"],
                explanation=a["explanation"],
                suspicious_events=a["suspicious_events"],
                drift_flag=a["drift_flag"],
                status="NEW",
            )
        )
    return response


@router.post("/detect/upload", response_model=List[AlertResponseItem])
async def detect_upload(
    file: UploadFile = File(...),
    log_type: str = Form("custom"),
    persist_alerts: bool = Form(True),
) -> List[AlertResponseItem]:
    """Upload a custom log file and execute anomaly detection."""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    return detect_anomalies(
        DetectRequest(log_text=text, log_type=log_type, persist_alerts=persist_alerts)
    )


@router.get("/alerts", response_model=List[AlertResponseItem])
def list_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: LOW, MEDIUM, HIGH, CRITICAL"),
    status: Optional[str] = Query(None, description="Filter by status: NEW, ACKNOWLEDGED, RESOLVED"),
    log_source: Optional[str] = Query(None, description="Filter by log source: hdfs, bgl, custom"),
    search: Optional[str] = Query(None, description="Search keyword in explanation or session ID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> List[AlertResponseItem]:
    """Retrieve persisted alerts with multi-criteria filtering."""
    rows = db.get_alerts(
        severity=severity,
        status=status,
        log_source=log_source,
        search_query=search,
        limit=limit,
        offset=offset,
    )
    items = []
    for r in rows:
        items.append(
            AlertResponseItem(
                alert_id=r["id"],
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                log_source=r["log_source"],
                risk_score=r["risk_score"],
                severity=r["severity"],
                sequence_score=r.get("sequence_score"),
                ae_score=r.get("ae_score"),
                explanation=r["explanation"],
                suspicious_events=r.get("suspicious_events", []),
                drift_flag=bool(r.get("drift_flag")),
                status=r.get("status", "NEW"),
            )
        )
    return items


@router.get("/alerts/stats")
def get_alert_statistics() -> Dict[str, Any]:
    """Get aggregated alert metrics and severity breakdown."""
    return db.get_alert_statistics()


@router.get("/alerts/{alert_id}", response_model=AlertResponseItem)
def get_alert_details(alert_id: int) -> AlertResponseItem:
    """Retrieve individual alert details and suspicious event traces."""
    alert = db.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert with id {alert_id} not found.")
    return AlertResponseItem(
        alert_id=alert["id"],
        session_id=alert["session_id"],
        timestamp=alert["timestamp"],
        log_source=alert["log_source"],
        risk_score=alert["risk_score"],
        severity=alert["severity"],
        sequence_score=alert.get("sequence_score"),
        ae_score=alert.get("ae_score"),
        explanation=alert["explanation"],
        suspicious_events=alert.get("suspicious_events", []),
        drift_flag=bool(alert.get("drift_flag")),
        status=alert.get("status", "NEW"),
    )


@router.put("/alerts/{alert_id}/status")
def update_alert_status(alert_id: int, req: AlertStatusUpdateRequest) -> Dict[str, Any]:
    """Update workflow status of an alert (NEW, ACKNOWLEDGED, RESOLVED)."""
    try:
        updated = db.update_alert_status(alert_id, req.status)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
        return {"status": "SUCCESS", "alert_id": alert_id, "new_status": req.status.upper()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/drift", response_model=DriftResponse)
def get_drift_status(limit: int = Query(20, ge=1, le=100)) -> DriftResponse:
    """Get historical drift observations and latest PSI score."""
    events = db.get_drift_events(limit=limit)
    latest_psi = float(events[0]["psi_score"]) if events else 0.0
    latest_novel = float(events[0]["novel_template_ratio"]) if events else 0.0
    cur_status = events[0]["drift_status"] if events else "STABLE"

    return DriftResponse(
        recent_events=events,
        current_status=cur_status,
        latest_psi=round(latest_psi, 4),
        latest_novel_ratio=round(latest_novel, 4),
    )


@router.post("/model/refresh", response_model=RefreshResponse)
def refresh_model(req: RefreshRequest) -> RefreshResponse:
    """
    Trigger unsupervised model refresh on new operational logs.
    Filters pseudo-normal sessions to adapt models without labels.
    """
    if not req.log_text or not req.log_text.strip():
        raise HTTPException(status_code=400, detail="log_text cannot be empty.")

    loader = get_log_loader(req.log_type)
    lines = req.log_text.strip().splitlines()
    recs = [loader.parse_line(l, line_num=i + 1) for i, l in enumerate(lines) if l.strip()]

    refresh_res = pipeline.refresh_model(recs, log_type=req.log_type)
    return RefreshResponse(
        status=refresh_res.get("status", "COMPLETED"),
        model_version=refresh_res.get("model_version"),
        pseudo_normal_added=refresh_res.get("pseudo_normal_added"),
        suspicious_excluded=refresh_res.get("suspicious_excluded"),
        total_training_samples=refresh_res.get("total_training_samples"),
        calibrated_threshold=refresh_res.get("calibrated_threshold"),
        reason=refresh_res.get("reason"),
    )


@router.get("/metrics")
def get_model_metrics() -> Dict[str, Any]:
    """Retrieve active model metadata and evaluation benchmarks."""
    active_model = db.get_active_model_metadata()
    if not active_model:
        return {"model_status": "UNINITIALIZED"}
    return active_model
