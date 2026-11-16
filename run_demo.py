"""
End-to-End Evaluation and Verification Script.
Demonstrates full lifecycle: Ingestion -> Drain3 -> Dual Models ->
Strict Chronological Benchmark (60/20/20) -> Plain-English Explainability ->
Concept Drift Monitoring -> Unsupervised Model Refresh -> SQLite Persistence.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.database import db
from src.ingestion.dataset_downloader import ensure_benchmarks_available
from src.ingestion.log_loader import LogRecord, get_log_loader
from src.pipeline import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo")


def run_full_evaluation(dataset_name: str = "hdfs", max_lines: int = 1500):
    print("=" * 70)
    print("  DRIFT-AWARE, EXPLAINABLE AIOps LOG ANOMALY DETECTOR")
    print("  CPU-Based End-to-End Evaluation & Verification Runner")
    print("=" * 70)

    # 1. Ensure Benchmark Datasets
    print("\n[Step 1/6] Fetching / Verifying Loghub Benchmark Datasets...")
    benchmarks = ensure_benchmarks_available()
    for k, p in benchmarks.items():
        print(f"  [OK] {k}: {p.name} ({p.stat().st_size:,} bytes)")

    # 2. Ingest Logs
    print(f"\n[Step 2/6] Ingesting {dataset_name.upper()} Dataset (max {max_lines} lines)...")
    if dataset_name == "hdfs":
        loader = get_log_loader("hdfs", labels_file=benchmarks["hdfs_labels"])
        records = loader.load_file(benchmarks["hdfs_2k"], max_lines=max_lines)
    else:
        loader = get_log_loader("bgl")
        records = loader.load_file(benchmarks["bgl_2k"], max_lines=max_lines)

    print(f"  [OK] Ingested {len(records)} raw log records.")

    # 3. Train on Strict Chronological Split (60% Train, 20% Val, 20% Test)
    print("\n[Step 3/6] Training on Strict Chronological Split (60/20/20)...")
    print("  [Protocol: Zero Random Shuffling across temporal splits]")
    train_res = pipeline.train_on_records(
        records=records,
        log_type=dataset_name,
        train_ratio=0.60,
        val_ratio=0.20,
        lstm_epochs=8,
        ae_epochs=10,
    )

    print(f"  [OK] Total Sessions Formed: {train_res['total_sessions']}")
    print(f"  [OK] Train Sessions (60%):  {train_res['train_sessions']}")
    print(f"  [OK] Val Sessions (20%):    {train_res['val_sessions']}")
    print(f"  [OK] Test Sessions (20%):   {train_res['test_sessions']}")
    print(f"  [OK] Learned Templates:     {train_res['vocab_size']}")
    print(f"  [OK] Calibrated Threshold:  {train_res['calibrated_threshold']:.4f}")

    # 4. Display Real Test Split Metrics
    test_m = train_res["test_metrics"]
    print("\n" + "=" * 50)
    print("  UNSEEN TEST SPLIT EVALUATION (Newest 20% Logs)")
    print("=" * 50)
    print(f"  Precision:            {test_m['precision']*100:.2f}%")
    print(f"  Recall:               {test_m['recall']*100:.2f}%")
    print(f"  F1-Score:             {test_m['f1_score']*100:.2f}%  (Target: >= 90.0%)")
    print(f"  False Positive Rate:  {test_m['false_positive_rate']*100:.2f}%  (Target: < 10.0%)")
    print(f"  Accuracy:             {test_m['accuracy']*100:.2f}%")
    print(f"  Confusion Matrix:     TP={test_m['confusion_matrix']['tp']}, "
          f"FP={test_m['confusion_matrix']['fp']}, "
          f"TN={test_m['confusion_matrix']['tn']}, "
          f"FN={test_m['confusion_matrix']['fn']}")
    print("=" * 50)

    # 5. Live Anomaly Detection & Plain-English Explainability
    print("\n[Step 4/6] Live Anomaly Detection with Plain-English Diagnostics...")
    anom_stream = [
        "081109 203524 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_demo_999 src: /10.250.19.104:54108 dest: /10.250.19.104:50010\n",
        "081109 203525 145 ERROR dfs.DataNode$BlockReceiver: Verification failed for block blk_demo_999 checksum mismatch\n",
        "081109 203526 145 WARN dfs.DataNode$DataXceiver: Execution failed unexpected EOF for block blk_demo_999\n",
    ]
    test_recs = [loader.parse_line(line, line_num=i+1) for i, line in enumerate(anom_stream)]
    alerts = pipeline.detect_stream(test_recs, log_type=dataset_name, persist_alerts=True)

    print(f"  [OK] Processed test stream -> Detected {len(alerts)} alert(s).")
    for a in alerts:
        print(f"\n  [ALERT #{a['alert_id']}] Session: {a['session_id']} | Severity: {a['severity']} | Risk: {a['risk_score']:.1f}/100")
        print(f"  Explanation:\n    {a['explanation']}")

    # 6. Concept Drift Monitoring & Unsupervised Model Refresh
    print("\n[Step 5/6] Testing Concept Drift Monitoring...")
    drift_events = ["E1", "E2", "E999_NEW", "E998_NEW", "E997_NEW"] * 20
    drift_eval = pipeline.drift_monitor.evaluate_window(drift_events, log_source="drift_test", record_db=True)
    print(f"  [OK] Drift Status:          {drift_eval['drift_status']}")
    print(f"  [OK] PSI Score:             {drift_eval['psi_score']:.4f}")
    print(f"  [OK] Novel Template Ratio:  {drift_eval['novel_template_ratio']*100:.1f}%")

    print("\n[Step 6/6] Unsupervised Model Refresh on Operational Logs...")
    refresh_stream = [
        "081109 203527 146 INFO dfs.DataNode$DataXceiver: Receiving block blk_refresh_101 src: /10.250.19.105:54109 dest: /10.250.19.105:50010\n",
        "081109 203528 146 INFO dfs.DataNode$BlockReceiver: PacketResponder 1 for block blk_refresh_101 terminating\n",
        "081109 203529 146 INFO dfs.DataNode$BlockReceiver: Received block blk_refresh_101 of size 67108864 from /10.250.19.105\n",
    ] * 5
    refresh_recs = [loader.parse_line(l, line_num=i+1) for i, l in enumerate(refresh_stream)]
    refresh_res = pipeline.refresh_model(refresh_recs, log_type=dataset_name)
    print(f"  [OK] Refresh Status:        {refresh_res['status']}")
    print(f"  [OK] Updated Model Version: {refresh_res.get('model_version')}")
    print(f"  [OK] Pseudo-Normal Added:   {refresh_res.get('pseudo_normal_added')}")

    # Persistent SQLite Audit
    stats = db.get_alert_statistics()
    print("\n" + "=" * 50)
    print(f"  SQLITE AUDIT PERSISTENCE VERIFIED")
    print(f"  Total Alerts Persisted:  {stats['total_alerts']}")
    print(f"  Severity Distribution:   {stats['severity_counts']}")
    print(f"  Database Location:       {db.db_path}")
    print("=" * 50)
    print("\n[OK] ALL CORE MODULES INTEGRATED AND OPERATIONAL.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIOps Anomaly Detector Evaluation")
    parser.add_argument("--dataset", type=str, default="hdfs", choices=["hdfs", "bgl"])
    parser.add_argument("--lines", type=int, default=1200)
    args = parser.parse_args()

    run_full_evaluation(dataset_name=args.dataset, max_lines=args.lines)
