"""
Log Masking and Variable Normalization Engine.
Extracts and masks dynamic tokens (IPs, Block IDs, hex values, numbers, paths)
to help Drain3 group semantically equivalent log messages into common templates.
"""

import re
from typing import List, Tuple

# Pre-compiled regex patterns for performance
MASK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # HDFS block identifiers (e.g. blk_-1608999687919862906, blk_1001)
    (re.compile(r"blk_-?\d+"), "<*BLK*>"),
    # IPv4 addresses with optional ports
    (re.compile(r"/(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::\d+)?"), "<*IP*>"),
    (re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::\d+)?\b"), "<*IP*>"),
    # Hexadecimal memory addresses & hashes
    (re.compile(r"0x[0-9a-fA-F]+"), "<*HEX*>"),
    # Standard file paths (e.g. /var/log/syslog, /mnt/hdfs/data)
    (re.compile(r"(?:/[a-zA-Z0-9_\.\-]+){2,}"), "<*PATH*>"),
    # Timestamps in ISO/common formats inside message content
    (re.compile(r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?"), "<*TIME*>"),
    # Floating point numbers
    (re.compile(r"\b\d+\.\d+\b"), "<*NUM*>"),
    # Integers (numbers with 1 or more digits)
    (re.compile(r"\b\d+\b"), "<*NUM*>"),
]


class LogMasker:
    """Masks dynamic variables and extracts block identifiers from raw log messages."""

    def __init__(self, patterns: List[Tuple[re.Pattern, str]] = MASK_PATTERNS):
        self.patterns = patterns
        self.block_pattern = re.compile(r"blk_-?\d+")

    def mask(self, content: str) -> str:
        """
        Replace variable parameters with wildcard markers.
        Preserves structural syntax words and error indicators.
        """
        masked = content
        for pattern, replacement in self.patterns:
            masked = pattern.sub(replacement, masked)
        # Normalize multiple spaces
        return " ".join(masked.split())

    def extract_block_id(self, log_line: str) -> str:
        """Extract HDFS block identifier (blk_*) if present, else empty string."""
        match = self.block_pattern.search(log_line)
        return match.group(0) if match else ""


# Global singleton instance
masker = LogMasker()
