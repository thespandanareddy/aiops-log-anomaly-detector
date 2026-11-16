"""Unit tests for Log Ingestion and Parameter Masking."""

import pytest
from src.ingestion.log_loader import HDFSLoader, BGLLoader, CustomLogLoader
from src.preprocessing.maskers import LogMasker


def test_parameter_masking():
    """Verify masking of block IDs, IPs, hex addresses, paths, and numbers."""
    masker = LogMasker()
    
    sample_hdfs = "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    masked = masker.mask(sample_hdfs)
    assert "blk_-1608999687919862906" not in masked
    assert "<*BLK*>" in masked
    assert "<*IP*>" in masked

    sample_hex = "Instruction cache parity error at 0x7fff89ab with 63543 double-hummer exceptions"
    masked_hex = masker.mask(sample_hex)
    assert "0x7fff89ab" not in masked_hex
    assert "<*HEX*>" in masked_hex
    assert "<*NUM*>" in masked_hex


def test_hdfs_loader_parsing():
    """Verify parsing single HDFS line into standardized LogRecord."""
    loader = HDFSLoader()
    line = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_1001 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    record = loader.parse_line(line, line_num=1)

    assert record.level == "INFO"
    assert record.component == "dfs.DataNode$DataXceiver"
    assert record.session_key == "blk_1001"
    assert record.timestamp == "2008-11-09 20:35:18"
    assert record.is_anomaly is False


def test_bgl_loader_parsing():
    """Verify parsing BGL line with alert category and normal line."""
    loader = BGLLoader()
    
    # Normal BGL line
    normal_line = "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
    normal_rec = loader.parse_line(normal_line, line_num=1)
    assert normal_rec.is_anomaly is False
    assert normal_rec.label == "-"
    assert normal_rec.level == "INFO"

    # Anomalous BGL line
    anom_line = "KERNDTLB 1117842440 2005.06.03 R23-M0-NE-C:J05-U01 2005-06-03-16.47.20.730545 R23-M0-NE-C:J05-U01 RAS KERNEL FATAL data TLB error interrupt"
    anom_rec = loader.parse_line(anom_line, line_num=2)
    assert anom_rec.is_anomaly is True
    assert anom_rec.label == "KERNDTLB"
    assert anom_rec.level == "FATAL"


def test_custom_loader_parsing():
    """Verify parsing arbitrary custom log line."""
    loader = CustomLogLoader()
    line = "2026-11-16 12:00:00 [ERROR] Database connection timed out after 30000ms"
    rec = loader.parse_line(line, line_num=5)
    assert rec.level == "ERROR"
    assert "Database connection timed out" in rec.content
