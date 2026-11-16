"""Unit tests for Explainability Engine and Concept Drift Monitoring."""

import pytest
from src.detection.autoencoder import AutoencoderDetector
from src.detection.calibrator import Calibrator
from src.detection.lstm_detector import LSTMAnomalyDetector
from src.drift.monitor import DriftMonitor
from src.drift.refresher import ModelRefresher
from src.explainability.explainer import AnomalyExplainer
from src.ingestion.log_loader import LogRecord
from src.parsing.drain_parser import DrainParser
from src.preprocessing.sessionizer import LogSession


def test_explainability_generation():
    """Verify plain-English diagnostic generation for sequential and volumetric issues."""
    parser = DrainParser(persistence_path=None)
    parser.parse_message("Receiving block", update_miner=True)
    parser.parse_message("PacketResponder terminating", update_miner=True)
    parser.parse_message("Verification failed checksum mismatch", update_miner=True)

    explainer = AnomalyExplainer(drain_parser=parser)

    records = [
        LogRecord("Receiving block", timestamp="2026-11-16 10:00:00", level="INFO"),
        LogRecord("Verification failed checksum mismatch", timestamp="2026-11-16 10:00:01", level="ERROR"),
    ]
    session = LogSession("blk_test", ["E1", "E3"], records, is_anomaly=True)

    score_data = {
        "sequence_score": 0.50,
        "ae_score": 0.40,
        "calibrated_risk": 82.5,
        "seq_diagnostics": [
            {
                "step_index": 1,
                "actual_event": "E3",
                "actual_prob": 0.001,
                "expected_topk": [{"event_id": "E2", "prob": 0.95}],
            }
        ],
        "ae_diagnostics": [
            {
                "event_id": "E3",
                "actual_count": 5.0,
                "expected_count": 0.1,
                "residual": 0.45,
            }
        ],
    }

    explanation_res = explainer.generate_explanation(session, score_data)

    assert "Abnormal Event Sequence" in explanation_res["summary_text"]
    assert "Volumetric Discrepancy" in explanation_res["summary_text"]
    assert len(explanation_res["suspicious_events"]) > 0
    assert explanation_res["suspicious_events"][0]["event_id"] == "E3"
    assert explanation_res["suspicious_events"][0]["level"] == "ERROR"


def test_drift_monitor_psi_and_novel_templates():
    """Verify PSI calculation and novel template ratio."""
    baseline = ["E1", "E2", "E1", "E2", "E3"] * 20
    monitor = DriftMonitor(baseline_events=baseline)

    # Identical distribution -> Stable (PSI near 0)
    stable_eval = monitor.evaluate_window(baseline, log_source="test", record_db=False)
    assert stable_eval["psi_score"] < 0.05
    assert stable_eval["drift_status"] == "STABLE"

    # Heavily shifted distribution with new templates -> Significant drift
    drifted_events = ["E99", "E100", "E101"] * 20
    drift_eval = monitor.evaluate_window(drifted_events, log_source="test", record_db=False)
    assert drift_eval["novel_template_ratio"] > 0.5
    assert drift_eval["drift_status"] == "SIGNIFICANT_DRIFT"


def test_unsupervised_model_refresher():
    """Verify unsupervised refresh filters pseudo-normal sessions."""
    parser = DrainParser(persistence_path=None)
    lstm = LSTMAnomalyDetector(window_size=3, top_k=2, model_path=None)
    ae = AutoencoderDetector(model_path=None, hidden_dims=[8, 4])
    calibrator = Calibrator()
    monitor = DriftMonitor()

    refresher = ModelRefresher(
        drain_parser=parser,
        lstm_detector=lstm,
        ae_detector=ae,
        calibrator=calibrator,
        drift_monitor=monitor,
        confidence_threshold=40.0,
    )

    # Historical training sequences
    normal_seqs = [["E1", "E2", "E3"]] * 20
    lstm.fit(normal_seqs, epochs=2)
    ae.fit(normal_seqs, epochs=2)
    calibrator.decision_threshold = 0.50

    # Incoming sessions: some normal, some anomalous
    incoming = [
        LogSession(f"norm_{i}", ["E1", "E2", "E3"], [], is_anomaly=False)
        for i in range(10)
    ] + [
        LogSession("anom_1", ["E99", "E99", "E99"], [], is_anomaly=True)
    ]

    res = refresher.refresh(incoming_sessions=incoming, epochs=2)
    assert res["status"] == "SUCCESS"
    assert res["pseudo_normal_added"] >= 5
    assert "model_version" in res
