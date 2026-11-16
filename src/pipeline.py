"""
Unified End-to-End AIOps Log Anomaly Detection Pipeline.
Coordinates ingestion, Drain3 parsing, semantic matching, dual model detection,
plain-English explainability, drift tracking, and SQLite persistence.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.detection.autoencoder import AutoencoderDetector
from src.detection.calibrator import Calibrator
from src.detection.lstm_detector import LSTMAnomalyDetector
from src.drift.monitor import DriftMonitor
from src.drift.refresher import ModelRefresher
from src.explainability.explainer import AnomalyExplainer
from src.ingestion.log_loader import LogRecord, get_log_loader
from src.parsing.drain_parser import DrainParser
from src.parsing.semantic_matcher import SemanticMatcher
from src.preprocessing.sessionizer import LogSession, Sessionizer
from db.database import db

logger = logging.getLogger(__name__)


class AIOpsPipeline:
    """End-to-End Log Anomaly Detection Engine."""

    def __init__(
        self,
        drain_parser: Optional[DrainParser] = None,
        lstm_detector: Optional[LSTMAnomalyDetector] = None,
        ae_detector: Optional[AutoencoderDetector] = None,
        calibrator: Optional[Calibrator] = None,
        drift_monitor: Optional[DriftMonitor] = None,
    ):
        self.drain_parser = drain_parser or DrainParser()
        self.semantic_matcher = SemanticMatcher(drain_parser=self.drain_parser)
        self.lstm = lstm_detector or LSTMAnomalyDetector()
        self.ae = ae_detector or AutoencoderDetector()
        self.calibrator = calibrator or Calibrator()
        self.explainer = AnomalyExplainer(drain_parser=self.drain_parser)
        self.drift_monitor = drift_monitor or DriftMonitor()
        self.refresher = ModelRefresher(
            drain_parser=self.drain_parser,
            lstm_detector=self.lstm,
            ae_detector=self.ae,
            calibrator=self.calibrator,
            drift_monitor=self.drift_monitor,
        )
        self.historical_normal_sessions: List[LogSession] = []
        self.validation_sessions: List[LogSession] = []
        self.is_trained = False

    def train_on_records(
        self,
        records: List[LogRecord],
        log_type: str = "hdfs",
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        lstm_epochs: int = 8,
        ae_epochs: int = 10,
    ) -> Dict[str, Any]:
        """
        Train end-to-end models using strict chronological split:
        - Oldest 60%: Train (learns templates, fits LSTM and AE on normal sessions)
        - Next 20%: Validation (calibrates thresholds for F1 and FPR)
        - Newest 20%: Test split (held out for evaluation)
        """
        # 1. Build chronological sessions
        if log_type.lower() == "hdfs":
            sessions = Sessionizer.build_block_sessions(
                records, self.drain_parser, update_miner=True
            )
        else:
            sessions = Sessionizer.build_time_window_sessions(
                records, self.drain_parser, update_miner=True
            )

        self.semantic_matcher.rebuild_index()

        # 2. Strict chronological split (NEVER random shuffle)
        train_sessions, val_sessions, test_sessions = Sessionizer.chronological_split(
            sessions, train_ratio=train_ratio, val_ratio=val_ratio
        )

        self.validation_sessions = val_sessions

        # 3. Filter normal sessions from training split for unsupervised learning
        normal_train = [s for s in train_sessions if not s.is_anomaly]
        if not normal_train:
            normal_train = train_sessions

        self.historical_normal_sessions = normal_train
        train_seqs = [s.events for s in normal_train]

        # 4. Fit LSTM and Autoencoder on normal train sequences
        lstm_info = self.lstm.fit(train_seqs, epochs=lstm_epochs)
        ae_info = self.ae.fit(train_seqs, epochs=ae_epochs)

        # 5. Set baseline for drift monitoring
        all_train_events = [e for seq in train_seqs for e in seq]
        self.drift_monitor.set_baseline(all_train_events)

        # 6. Calibrate decision threshold on validation split
        val_metrics = self.calibrator.calibrate(val_sessions, self.lstm, self.ae)

        # 7. Evaluate on held-out test split
        test_metrics = self.calibrator.evaluate(test_sessions, self.lstm, self.ae)

        self.is_trained = True

        # 8. Record initial baseline model metadata in SQLite
        try:
            db.save_model_metadata(
                version="v1.0.0-baseline",
                model_type="baseline",
                total_templates=self.lstm.vocab_size,
                training_samples=len(normal_train),
                val_f1=val_metrics.get("val_f1"),
                val_precision=val_metrics.get("val_precision"),
                val_recall=val_metrics.get("val_recall"),
                val_fpr=val_metrics.get("val_fpr"),
                threshold_sequence=self.calibrator.decision_threshold,
                threshold_ae=self.calibrator.ae_scale_factor,
            )
        except Exception as e:
            logger.error(f"Failed to record baseline model metadata: {e}")

        return {
            "status": "TRAINED",
            "total_records": len(records),
            "total_sessions": len(sessions),
            "train_sessions": len(train_sessions),
            "val_sessions": len(val_sessions),
            "test_sessions": len(test_sessions),
            "vocab_size": self.lstm.vocab_size,
            "calibrated_threshold": self.calibrator.decision_threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    def detect_stream(
        self,
        records: List[LogRecord],
        log_type: str = "custom",
        persist_alerts: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process incoming log records, detect anomalies, generate explanations,
        and persist alerts to SQLite.
        """
        if log_type.lower() == "hdfs":
            sessions = Sessionizer.build_block_sessions(
                records, self.drain_parser, update_miner=False
            )
        else:
            sessions = Sessionizer.build_fixed_window_sessions(
                records, self.drain_parser, window_size=20, update_miner=False
            )

        # Run drift evaluation on all incoming events in this batch
        all_incoming_events = [e for s in sessions for e in s.events]
        drift_eval = self.drift_monitor.evaluate_window(
            all_incoming_events, log_source=log_type, record_db=True
        )

        detected_alerts = []

        for s in sessions:
            # Check for novel/semantic matches
            novel_templates = []
            for r in s.records:
                match_res = self.semantic_matcher.match(r.content, allow_semantic_fallback=True)
                if match_res["is_novel"]:
                    novel_templates.append(str(match_res["template"]))

            score_data = self.calibrator.compute_session_scores(s, self.lstm, self.ae)
            risk = score_data["calibrated_risk"]
            severity = Calibrator.get_severity(risk)

            is_alert = (
                severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
                or score_data["combined_score"] >= self.calibrator.decision_threshold
                or len(novel_templates) > 0
            )

            if is_alert:
                explanation_data = self.explainer.generate_explanation(
                    session=s,
                    score_data=score_data,
                    is_novel_present=len(novel_templates) > 0,
                    novel_templates=novel_templates,
                )

                ts = s.timestamp or datetime.now().isoformat()
                alert_id = None
                if persist_alerts:
                    alert_id = db.insert_alert(
                        timestamp=ts,
                        session_id=s.session_id,
                        log_source=log_type,
                        risk_score=risk,
                        severity=severity if severity != "NORMAL" else "LOW",
                        sequence_score=score_data["sequence_score"],
                        ae_score=score_data["ae_score"],
                        suspicious_events=explanation_data["suspicious_events"],
                        explanation=explanation_data["summary_text"],
                        drift_flag=drift_eval["is_drift_detected"],
                        status="NEW",
                    )

                detected_alerts.append({
                    "alert_id": alert_id,
                    "session_id": s.session_id,
                    "timestamp": ts,
                    "log_source": log_type,
                    "risk_score": risk,
                    "severity": severity if severity != "NORMAL" else "LOW",
                    "explanation": explanation_data["summary_text"],
                    "suspicious_events": explanation_data["suspicious_events"],
                    "drift_flag": drift_eval["is_drift_detected"],
                    "ground_truth_anomaly": s.is_anomaly,
                })

        return detected_alerts

    def refresh_model(
        self,
        new_records: List[LogRecord],
        log_type: str = "custom",
    ) -> Dict[str, Any]:
        """Trigger unsupervised model refresh on new operational logs."""
        if log_type.lower() == "hdfs":
            sessions = Sessionizer.build_block_sessions(
                new_records, self.drain_parser, update_miner=True
            )
        else:
            sessions = Sessionizer.build_fixed_window_sessions(
                new_records, self.drain_parser, window_size=20, update_miner=True
            )

        self.semantic_matcher.rebuild_index()

        refresh_res = self.refresher.refresh(
            incoming_sessions=sessions,
            historical_normal_sessions=self.historical_normal_sessions,
            val_sessions=self.validation_sessions,
        )
        return refresh_res


# Global pipeline instance
pipeline = AIOpsPipeline()
