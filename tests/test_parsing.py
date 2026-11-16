"""Unit tests for Drain3 parsing and Semantic Matcher."""

import os
import tempfile
from pathlib import Path
import pytest

from src.parsing.drain_parser import DrainParser
from src.parsing.semantic_matcher import SemanticMatcher


def test_drain_parsing_and_clustering():
    """Verify Drain3 clusters logs into templates."""
    parser = DrainParser(persistence_path=None)

    line1 = "Receiving block blk_1001 src: /10.250.19.102:54106"
    line2 = "Receiving block blk_1002 src: /10.250.19.103:54107"

    eid1, tmpl1, is_new1 = parser.parse_message(line1, update_miner=True)
    eid2, tmpl2, is_new2 = parser.parse_message(line2, update_miner=True)

    assert is_new1 is True
    assert eid1 == eid2  # Same template after masking variables!
    assert is_new2 is False
    assert "<*BLK*>" in tmpl1


def test_semantic_matching_fallback():
    """Verify semantic matcher handles subtle phrasing variations."""
    parser = DrainParser(persistence_path=None)

    # Train on canonical template
    canonical = "Connection to host terminated abruptly due to socket error"
    eid, tmpl, _ = parser.parse_message(canonical, update_miner=True)

    matcher = SemanticMatcher(drain_parser=parser, similarity_threshold=0.65)
    matcher.rebuild_index()

    # Variant with slight wording modification
    variant = "Connection to server terminated abruptly due to socket error"
    match_res = matcher.match(variant, allow_semantic_fallback=True)

    assert match_res["is_novel"] is False
    assert match_res["match_type"] in ["EXACT", "SEMANTIC"]
    assert match_res["confidence"] >= 0.65

    # Completely unrelated log message
    unrelated = "Memory allocation failed for kernel heap buffer"
    unrelated_res = matcher.match(unrelated, allow_semantic_fallback=True)
    assert unrelated_res["is_novel"] is True
    assert unrelated_res["match_type"] == "NOVEL"
