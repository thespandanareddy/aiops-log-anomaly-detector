"""
Unsupervised Model Refresh Engine.
Adapts detection models to concept drift using high-confidence pseudo-normal sessions
without requiring manual anomaly labels.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from src.config import REFRESH_CONFIDENCE_THRESHOLD, REFRESH_MIN_NEW_SAMPLES
from src.detection.autoencoder import AutoencoderDetector
from src.detection.calibrator import Calibrator
from src.detection.lstm_detector import LSTMAnomalyDetector
from src.drift.monitor import DriftMonitor
from src.parsing.drain_parser import DrainParser
from src.preprocessing.sessionizer import LogSession
from db.database import db

logger = logging.getLogger(__name__)


class ModelRefresher:
    """Orchestrates autonomous unsupervised model retraining and adaptation."""

    def __init__(
        self,
        drain_parser: DrainParser,
        lstm_detector: LSTMAnomalyDetector,
        ae_detector: AutoencoderDetector,
        calibrator: Calibrator,
        drift_monitor: DriftMonitor,
        confidence_threshold: float = REFRESH_CONFIDENCE_THRESHOLD,
    ):
        self.drain_parser = drain_parser
        self.lstm = lstm_detector
        self.ae = ae_detector
        self.calibrator = calibrator
        self.drift_monitor = drift_monitor
        self.confidence_threshold = confidence_threshold
        self.refresh_count = 0

    def refresh(
        self,
        incoming_sessions: List[LogSession],
        historical_normal_sessions: Optional[List[LogSession]] = None,
        val_sessions: Optional[List[LogSession]] = None,
        epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform unsupervised model refresh:
        1. Select high-confidence normal sessions (calibrated risk < confidence_threshold).
        2. Combine with historical baseline to prevent catastrophic forgetting.
        3. Update vocabulary and fine-tune LSTM and Autoencoder.
        4. Re-calibrate thresholds and record metadata.
        """
        if not incoming_sessions:
            return {"status": "SKIPPED", "reason": "No incoming sessions provided"}

        pseudo_normal_sessions = []
        excluded_count = 0

        for s in incoming_sessions:
            score_data = self.calibrator.compute_session_scores(s, self.lstm, self.ae)
            if score_data["calibrated_risk"] < self.confidence_threshold:
                pseudo_normal_sessions.append(s)
            else:
                excluded_count += 1

        logger.info(
            f"Filtered {len(pseudo_normal_sessions)} high-confidence pseudo-normal sessions "
            f"({excluded_count} suspicious sessions excluded)."
        )

        if len(pseudo_normal_sessions) < 5:
            return {
                "status": "SKIPPED",
                "reason": f"Insufficient pseudo-normal samples ({len(pseudo_normal_sessions)} < 5)",
                "excluded_count": excluded_count,
            }

        # Combine historical normal sessions with new pseudo-normal sessions
        all_train_sessions = list(pseudo_normal_sessions)
        if historical_normal_sessions:
            # Replay historical normal samples (sample up to 200 to keep CPU training fast)
            all_train_sessions.extend(historical_normal_sessions[:200])

        train_seqs = [s.events for s in all_train_sessions]

        # Retrain / fine-tune models
        lstm_metrics = self.lstm.fit(train_seqs, epochs=epochs)
        ae_metrics = self.ae.fit(train_seqs, epochs=epochs)

        # Update drift monitor baseline with refreshed normal distribution
        flat_events = [e for seq in train_seqs for e in seq]
        self.drift_monitor.set_baseline(flat_events)

        # Re-calibrate if validation sessions available
        calib_metrics = {}
        if val_sessions:
            calib_metrics = self.calibrator.calibrate(val_sessions, self.lstm, self.ae)

        self.refresh_count += 1
        new_version = f"v{self.refresh_count + 1}.0.0-refresh"

        # Record new model in SQLite
        try:
            db.save_model_metadata(
                version=new_version,
                model_type="refreshed",
                total_templates=self.lstm.vocab_size,
                training_samples=len(all_train_sessions),
                val_f1=calib_metrics.get("val_f1"),
                val_precision=calib_metrics.get("val_precision"),
                val_recall=calib_metrics.get("val_recall"),
                val_fpr=calib_metrics.get("val_fpr"),
                threshold_sequence=self.calibrator.decision_threshold,
                threshold_ae=self.calibrator.ae_scale_factor,
            )
        except Exception as e:
            logger.error(f"Failed to record model metadata: {e}")

        return {
            "status": "SUCCESS",
            "model_version": new_version,
            "pseudo_normal_added": len(pseudo_normal_sessions),
            "suspicious_excluded": excluded_count,
            "total_training_samples": len(all_train_sessions),
            "new_vocab_size": self.lstm.vocab_size,
            "calibrated_threshold": self.calibrator.decision_threshold,
            "calib_metrics": calib_metrics,
        }
