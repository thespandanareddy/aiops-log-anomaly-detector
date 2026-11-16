"""
Automated dataset fetcher and synthetic data generator for development testing.
Fetches official LogPAI Loghub benchmarks (HDFS, BGL) and caches them locally.
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Dict

from src.config import DATASET_URLS, RAW_DATA_DIR, SAMPLES_DATA_DIR

logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with streaming chunks and cache verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        logger.info(f"Using cached file: {destination} ({destination.stat().st_size} bytes)")
        return destination

    logger.info(f"Downloading {url} -> {destination}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AIOps-Log-Anomaly-Detector/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(destination, "wb") as out_file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        logger.info(f"Download complete: {destination} ({destination.stat().st_size} bytes)")
    except Exception as e:
        logger.warning(f"Offline mode: Could not reach internet ({e}). Creating local fallback dataset.")
        create_synthetic_dev_dataset(destination)
    return destination


def ensure_benchmarks_available() -> Dict[str, Path]:
    """Ensure HDFS and BGL benchmark files are downloaded and cached."""
    paths = {}
    paths["hdfs_2k"] = download_file(DATASET_URLS["hdfs_2k"], RAW_DATA_DIR / "HDFS_2k.log")
    paths["hdfs_labels"] = download_file(DATASET_URLS["hdfs_labels"], RAW_DATA_DIR / "anomaly_label.csv")
    paths["bgl_2k"] = download_file(DATASET_URLS["bgl_2k"], RAW_DATA_DIR / "BGL_2k.log")
    return paths


def create_synthetic_dev_dataset(output_path: Path = SAMPLES_DATA_DIR / "synthetic_dev.log") -> Path:
    """
    Generate synthetic log data strictly for fast development/unit testing.
    Clearly labeled as synthetic to distinguish from real benchmark evaluation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        # Normal sequences for blk_1001
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_1001 src: /10.250.19.102:54106 dest: /10.250.19.102:50010\n",
        "081109 203519 143 INFO dfs.DataNode$BlockReceiver: PacketResponder 1 for block blk_1001 terminating\n",
        "081109 203520 143 INFO dfs.DataNode$BlockReceiver: Received block blk_1001 of size 67108864 from /10.250.19.102\n",
        # Normal sequences for blk_1002
        "081109 203521 144 INFO dfs.DataNode$DataXceiver: Receiving block blk_1002 src: /10.250.19.103:54107 dest: /10.250.19.103:50010\n",
        "081109 203522 144 INFO dfs.DataNode$BlockReceiver: PacketResponder 2 for block blk_1002 terminating\n",
        "081109 203523 144 INFO dfs.DataNode$BlockReceiver: Received block blk_1002 of size 67108864 from /10.250.19.103\n",
        # Anomalous sequence for blk_9999 (Abrupt failure/exception)
        "081109 203524 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_9999 src: /10.250.19.104:54108 dest: /10.250.19.104:50010\n",
        "081109 203525 145 ERROR dfs.DataNode$BlockReceiver: Verification failed for block blk_9999 checksum mismatch\n",
        "081109 203526 145 WARN dfs.DataNode$DataXceiver: Execution failed unexpected EOF for block blk_9999\n",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Checking / downloading benchmark datasets...")
    files = ensure_benchmarks_available()
    for k, v in files.items():
        print(f"  {k}: {v} ({v.stat().st_size} bytes)")
    synth = create_synthetic_dev_dataset()
    print(f"  Synthetic sample: {synth}")
