"""Unit tests for Sessionizer and Chronological Splitting."""

import pytest
from src.ingestion.log_loader import LogRecord
from src.parsing.drain_parser import DrainParser
from src.preprocessing.sessionizer import Sessionizer


def test_block_sessionizer():
    """Verify grouping logs by block ID into LogSession objects."""
    parser = DrainParser(persistence_path=None)
    records = [
        LogRecord("Line 1", session_key="blk_1", content="Receiving block blk_1"),
        LogRecord("Line 2", session_key="blk_1", content="Terminating block blk_1"),
        LogRecord("Line 3", session_key="blk_2", content="Receiving block blk_2"),
        LogRecord("Line 4", session_key="blk_2", content="Error on block blk_2", label="Anomaly"),
    ]

    sessions = Sessionizer.build_block_sessions(records, parser)
    assert len(sessions) == 2

    s1 = next(s for s in sessions if s.session_id == "blk_1")
    s2 = next(s for s in sessions if s.session_id == "blk_2")

    assert len(s1.events) == 2
    assert s1.is_anomaly is False

    assert len(s2.events) == 2
    assert s2.is_anomaly is True


def test_chronological_split():
    """Verify strict 60/20/20 chronological splitting without shuffling."""
    records = [
        LogRecord(f"Line {i}", session_key=f"blk_{i}", content=f"Event {i}")
        for i in range(100)
    ]
    parser = DrainParser(persistence_path=None)
    sessions = Sessionizer.build_block_sessions(records, parser)
    assert len(sessions) == 100

    train, val, test = Sessionizer.chronological_split(sessions, train_ratio=0.60, val_ratio=0.20)

    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20

    # Strict chronological order check: blk_0..59 in train, blk_60..79 in val, blk_80..99 in test
    assert train[0].session_id == "blk_0"
    assert train[-1].session_id == "blk_59"
    assert val[0].session_id == "blk_60"
    assert val[-1].session_id == "blk_79"
    assert test[0].session_id == "blk_80"
    assert test[-1].session_id == "blk_99"
