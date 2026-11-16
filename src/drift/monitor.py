"""
Concept Drift Monitoring Engine.
Computes Population Stability Index (PSI) and novel template ratio
over sliding observation windows to detect operational shifts.
"""

from collections import Counter
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import (
    NOVEL_TEMPLATE_RATE_THRESHOLD,
    PSI_MODERATE_DRIFT,
    PSI_SIGNIFICANT_DRIFT,
)
from db.database import db

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitors distributional shift and unseen template emergence across log streams."""

    def __init__(
        self,
        baseline_events: Optional[List[str]] = None,
        psi_moderate: float = PSI_MODERATE_DRIFT,
        psi_significant: float = PSI_SIGNIFICANT_DRIFT,
        novel_threshold: float = NOVEL_TEMPLATE_RATE_THRESHOLD,
    ):
        self.psi_moderate = psi_moderate
        self.psi_significant = psi_significant
        self.novel_threshold = novel_threshold
        self.baseline_vocab: set = set()
        self.baseline_distribution: Dict[str, float] = {}
        if baseline_events:
            self.set_baseline(baseline_events)

    def set_baseline(self, baseline_events: List[str]) -> None:
        """Establish baseline template frequency distribution from normal training logs."""
        if not baseline_events:
            return
        self.baseline_vocab = set(baseline_events)
        counts = Counter(baseline_events)
        total = float(len(baseline_events))
        self.baseline_distribution = {k: v / total for k, v in counts.items()}

    def compute_psi(
        self,
        current_events: List[str],
        epsilon: float = 1e-4
    ) -> float:
        """
        Compute Population Stability Index (PSI) between baseline and current distribution.
        PSI = sum((q_i - p_i) * ln(q_i / p_i))
        """
        if not self.baseline_distribution or not current_events:
            return 0.0

        current_counts = Counter(current_events)
        total_current = float(len(current_events))

        # Union of all observed templates
        all_templates = sorted(set(self.baseline_distribution.keys()).union(set(current_counts.keys())))

        psi_total = 0.0
        for tmpl in all_templates:
            # Smoothed probabilities
            p = self.baseline_distribution.get(tmpl, 0.0) + epsilon
            q = (current_counts.get(tmpl, 0) / total_current) + epsilon

            psi_total += (q - p) * np.log(q / p)

        return float(max(0.0, psi_total))

    def evaluate_window(
        self,
        window_events: List[str],
        log_source: str = "stream",
        record_db: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate an observation window for concept drift.
        Returns metrics and persists drift observation to SQLite.
        """
        if not window_events:
            return {
                "psi_score": 0.0,
                "novel_template_ratio": 0.0,
                "drift_status": "STABLE",
                "total_events": 0,
            }

        psi = self.compute_psi(window_events)

        # Calculate fraction of novel templates
        novel_events = [e for e in window_events if e not in self.baseline_vocab]
        novel_ratio = len(novel_events) / max(len(window_events), 1)

        # Determine overall drift status
        if psi >= self.psi_significant or novel_ratio >= self.novel_threshold:
            status = "SIGNIFICANT_DRIFT"
        elif psi >= self.psi_moderate or novel_ratio >= (self.novel_threshold / 2.0):
            status = "MODERATE_DRIFT"
        else:
            status = "STABLE"

        now_str = datetime.now().isoformat()
        novel_unique = list(set(novel_events))

        details = {
            "top_novel_templates": novel_unique[:5],
            "novel_count": len(novel_events),
            "baseline_vocab_size": len(self.baseline_vocab),
        }

        if record_db:
            try:
                db.insert_drift_event(
                    timestamp=now_str,
                    log_source=log_source,
                    psi_score=psi,
                    novel_template_ratio=novel_ratio,
                    total_events=len(window_events),
                    drift_status=status,
                    details=details,
                )
            except Exception as e:
                logger.error(f"Failed to record drift event in DB: {e}")

        return {
            "timestamp": now_str,
            "psi_score": round(psi, 4),
            "novel_template_ratio": round(novel_ratio, 4),
            "drift_status": status,
            "total_events": len(window_events),
            "novel_unique_templates": novel_unique,
            "is_drift_detected": status != "STABLE",
        }
