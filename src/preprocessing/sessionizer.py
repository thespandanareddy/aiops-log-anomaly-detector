"""
Session Building and Strict Chronological Splitting Module.
Groups raw log streams into block sessions (HDFS) or time/sliding windows (BGL/custom).
"""

from collections import defaultdict, Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from src.ingestion.log_loader import LogRecord
from src.parsing.drain_parser import DrainParser


class LogSession:
    """Represents a coherent sequence of log events for anomaly detection."""

    def __init__(
        self,
        session_id: str,
        events: List[str],
        records: List[Any],
        is_anomaly: bool = False,
        label: str = "Normal"
    ):
        self.session_id = session_id
        self.events = events
        self.records = records
        self.is_anomaly = is_anomaly
        self.label = label
        self.timestamp = records[0].timestamp if records else ""

    def get_event_counts(self, all_event_ids: List[str]) -> np.ndarray:
        """Construct a fixed-dimension count vector over the global vocabulary."""
        counts = Counter(self.events)
        return np.array([counts.get(eid, 0) for eid in all_event_ids], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.events)

    def to_dict(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "event_count": len(self.events),
            "events": self.events,
            "is_anomaly": self.is_anomaly,
            "label": self.label,
            "timestamp": self.timestamp,
        }


class Sessionizer:
    """Builds structured sessions from parsed log streams."""

    @staticmethod
    def build_block_sessions(
        records: List[Any],
        drain_parser: DrainParser,
        update_miner: bool = True
    ) -> List[LogSession]:
        """
        Group records by block ID (HDFS paradigm) preserving chronological sequence.
        """
        grouped_records = defaultdict(list)
        block_labels = {}

        for rec in records:
            key = rec.session_key or "unknown_block"
            grouped_records[key].append(rec)
            if rec.is_anomaly or (rec.label and rec.label.lower() not in {"normal", "-"}):
                block_labels[key] = rec.label or "Anomaly"

        sessions: List[LogSession] = []
        for session_id, block_recs in grouped_records.items():
            if not block_recs:
                continue

            events = []
            for r in block_recs:
                eid, _, _ = drain_parser.parse_message(r.content, update_miner=update_miner)
                events.append(eid)

            is_anom = session_id in block_labels or any(r.is_anomaly for r in block_recs)
            label = block_labels.get(session_id, "Normal")

            sessions.append(
                LogSession(
                    session_id=session_id,
                    events=events,
                    records=block_recs,
                    is_anomaly=is_anom,
                    label=label
                )
            )

        return sessions

    @staticmethod
    def build_time_window_sessions(
        records: List[Any],
        drain_parser: DrainParser,
        window_seconds: int = 300,
        step_seconds: int = 60,
        update_miner: bool = True
    ) -> List[LogSession]:
        """
        Group records into sliding time windows (BGL paradigm).
        """
        # Parse timestamps into epoch seconds
        timed_records = []
        for r in records:
            epoch = 0
            if r.timestamp:
                try:
                    dt = datetime.fromisoformat(r.timestamp)
                    epoch = int(dt.timestamp())
                except Exception:
                    epoch = 0
            timed_records.append((epoch, r))

        # Sort chronologically
        timed_records.sort(key=lambda x: x[0])

        if not timed_records:
            return []

        start_epoch = timed_records[0][0]
        end_epoch = timed_records[-1][0]
        sessions: List[LogSession] = []

        cur_start = start_epoch
        win_idx = 0

        while cur_start <= end_epoch:
            cur_end = cur_start + window_seconds
            window_recs = [r for ep, r in timed_records if cur_start <= ep < cur_end]
            if window_recs:
                events = [
                    drain_parser.parse_message(r.content, update_miner=update_miner)[0]
                    for r in window_recs
                ]
                is_anom = any(r.is_anomaly for r in window_recs)
                anom_labels = [r.label for r in window_recs if r.is_anomaly]
                label = anom_labels[0] if anom_labels else "Normal"

                session_id = f"win_{win_idx}_{cur_start}"
                sessions.append(
                    LogSession(
                        session_id=session_id,
                        events=events,
                        records=window_recs,
                        is_anomaly=is_anom,
                        label=label
                    )
                )
                win_idx += 1
            cur_start += step_seconds

        return sessions

    @staticmethod
    def build_fixed_window_sessions(
        records: List[Any],
        drain_parser: DrainParser,
        window_size: int = 20,
        update_miner: bool = True
    ) -> List[LogSession]:
        """
        Chunk log sequence into fixed-length event windows.
        """
        sessions: List[LogSession] = []
        for i in range(0, len(records), window_size):
            chunk = records[i: i + window_size]
            events = [
                drain_parser.parse_message(r.content, update_miner=update_miner)[0]
                for r in chunk
            ]
            is_anom = any(r.is_anomaly for r in chunk)
            label = "Anomaly" if is_anom else "Normal"
            sessions.append(
                LogSession(
                    session_id=f"seq_win_{i // window_size}",
                    events=events,
                    records=chunk,
                    is_anomaly=is_anom,
                    label=label
                )
            )
        return sessions

    @staticmethod
    def chronological_split(
        sessions: List[LogSession],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20
    ) -> Tuple[List[LogSession], List[LogSession], List[LogSession]]:
        """
        Strictly partition sessions chronologically:
          - Oldest 60%: Training
          - Next 20%: Validation / Calibration
          - Newest 20%: Testing
        NEVER randomly shuffles data.
        """
        n = len(sessions)
        if n == 0:
            return [], [], []

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_set = sessions[:train_end]
        val_set = sessions[train_end:val_end]
        test_set = sessions[val_end:]

        return train_set, val_set, test_set
