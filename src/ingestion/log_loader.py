"""
Data Ingestion Module for HDFS, BGL, and Custom Log Files.
Parses raw unstructured log streams into standardized structured records.
"""

import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from src.preprocessing.maskers import masker


class LogRecord:
    """Standardized representation of an ingested log line."""

    def __init__(
        self,
        raw_text: str,
        timestamp: Optional[str] = None,
        level: str = "INFO",
        component: str = "SYSTEM",
        content: str = "",
        session_key: str = "",
        label: Optional[str] = None,
        line_num: int = 0
    ):
        self.raw_text = raw_text.strip()
        self.timestamp = timestamp or ""
        self.level = level.upper()
        self.component = component
        self.content = content.strip() if content else raw_text.strip()
        self.session_key = session_key
        self.label = label  # '-' or 'Normal' if normal, otherwise anomaly tag
        self.line_num = line_num
        self.is_anomaly = bool(label and label != "-" and label.lower() != "normal")

    def to_dict(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "line_num": self.line_num,
            "timestamp": self.timestamp,
            "level": self.level,
            "component": self.component,
            "content": self.content,
            "session_key": self.session_key,
            "label": self.label or "-",
            "is_anomaly": self.is_anomaly,
            "raw_text": self.raw_text,
        }


class HDFSLoader:
    """
    Parser for Hadoop Distributed File System (HDFS) logs.
    Format: <Date> <Time> <Pid> <Level> <Component>: <Content>
    Example: 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_1001 src: ...
    """
    HDFS_PATTERN = re.compile(
        r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>[A-Z]+)\s+(?P<component>[^:]+):\s+(?P<content>.*)$"
    )

    def __init__(self, labels_file: Optional[Path] = None):
        self.labels_map: Dict[str, str] = {}
        if labels_file and Path(labels_file).exists():
            self._load_labels(Path(labels_file))

    def _load_labels(self, labels_file: Path) -> None:
        """Load ground truth anomaly labels mapping: BlockId -> Label ('Normal'/'Anomaly')."""
        with open(labels_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self.labels_map[row[0].strip()] = row[1].strip()

    def parse_line(self, line: str, line_num: int = 0) -> LogRecord:
        """Parse a single HDFS log line into a LogRecord."""
        match = self.HDFS_PATTERN.match(line.strip())
        if match:
            d = match.groupdict()
            date_str = f"20{d['date'][:2]}-{d['date'][2:4]}-{d['date'][4:6]}"
            time_str = f"{d['time'][:2]}:{d['time'][2:4]}:{d['time'][4:6]}"
            ts = f"{date_str} {time_str}"
            level = d["level"]
            component = d["component"]
            content = d["content"]
        else:
            ts = ""
            level = "INFO"
            component = "HDFS"
            content = line.strip()

        block_id = masker.extract_block_id(content)
        label = self.labels_map.get(block_id, "Normal" if block_id else "-")

        return LogRecord(
            raw_text=line,
            timestamp=ts,
            level=level,
            component=component,
            content=content,
            session_key=block_id,
            label=label,
            line_num=line_num
        )

    def load_file(self, file_path: Path, max_lines: Optional[int] = None) -> List[LogRecord]:
        """Load and parse an HDFS log file."""
        records = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                records.append(self.parse_line(line, line_num=i))
                if max_lines and len(records) >= max_lines:
                    break
        return records


class BGLLoader:
    """
    Parser for Blue Gene/L (BGL) supercomputer logs.
    Format: <Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Component> <Level> <Content>
    Label '-' indicates normal; non-dash values indicate anomaly alerts.
    """
    BGL_PATTERN = re.compile(
        r"^(?P<label>\S+)\s+(?P<sec_ts>\d+)\s+(?P<date>\S+)\s+(?P<node>\S+)\s+(?P<time>\S+)\s+(?P<node_repeat>\S+)\s+(?P<subsystem>\S+)\s+(?P<component>\S+)\s+(?P<level>\S+)\s+(?P<content>.*)$"
    )

    def parse_line(self, line: str, line_num: int = 0) -> LogRecord:
        """Parse a single BGL log line into a LogRecord."""
        match = self.BGL_PATTERN.match(line.strip())
        if match:
            d = match.groupdict()
            label = d["label"]
            try:
                ts = datetime.fromtimestamp(int(d["sec_ts"])).isoformat()
            except Exception:
                ts = f"{d['date']} {d['time']}"
            level = d["level"]
            component = d["component"]
            content = d["content"]
            session_key = d["node"]
        else:
            label = "-"
            ts = ""
            level = "INFO"
            component = "BGL"
            content = line.strip()
            session_key = ""

        return LogRecord(
            raw_text=line,
            timestamp=ts,
            level=level,
            component=component,
            content=content,
            session_key=session_key,
            label=label,
            line_num=line_num
        )

    def load_file(self, file_path: Path, max_lines: Optional[int] = None) -> List[LogRecord]:
        """Load and parse a BGL log file."""
        records = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                records.append(self.parse_line(line, line_num=i))
                if max_lines and len(records) >= max_lines:
                    break
        return records


class CustomLogLoader:
    """
    Flexible generic log parser for arbitrary user logs.
    Handles syslog, application logs (Apache/Nginx/Java), and CSV/TSV streams.
    """
    # Regex for standard ISO/syslog timestamp prefixes
    TIMESTAMP_PREFIX = re.compile(
        r"^(\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s+\d{2}:\d{2}:\d{2})"
    )
    LEVEL_PATTERN = re.compile(r"\b(DEBUG|INFO|WARN(?:ING)?|ERR(?:OR)?|FATAL|CRITICAL)\b", re.IGNORECASE)

    def parse_line(self, line: str, line_num: int = 0) -> LogRecord:
        """Heuristically extract timestamp, level, and content from custom log."""
        ts = ""
        ts_match = self.TIMESTAMP_PREFIX.search(line)
        if ts_match:
            ts = ts_match.group(0)
            rest = line[ts_match.end():].strip()
        else:
            rest = line.strip()

        level = "INFO"
        lvl_match = self.LEVEL_PATTERN.search(rest)
        if lvl_match:
            level = lvl_match.group(0).upper()

        block_id = masker.extract_block_id(line)
        content = rest

        return LogRecord(
            raw_text=line,
            timestamp=ts,
            level=level,
            component="CUSTOM",
            content=content,
            session_key=block_id if block_id else f"seq_{line_num // 20}",
            label="-",
            line_num=line_num
        )

    def load_file(self, file_path: Path, max_lines: Optional[int] = None) -> List[LogRecord]:
        """Load and parse a custom log file."""
        records = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                records.append(self.parse_line(line, line_num=i))
                if max_lines and len(records) >= max_lines:
                    break
        return records


def get_log_loader(log_type: str, labels_file: Optional[Path] = None):
    """Factory function for log loaders."""
    t = log_type.lower()
    if t == "hdfs":
        return HDFSLoader(labels_file=labels_file)
    elif t == "bgl":
        return BGLLoader()
    else:
        return CustomLogLoader()
