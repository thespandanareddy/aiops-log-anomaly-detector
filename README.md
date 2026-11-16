# Drift-Aware, Explainable AIOps Log Anomaly Detector

A CPU-friendly, intelligent log anomaly detection and explainability system for enterprise AIOps.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26%2B-orange)
![Drain3](https://img.shields.io/badge/Drain3-Streaming-purple)

---

## 1. Project Overview

Modern cloud systems generate millions of log messages daily. Traditional rule-based alerts fail when log volumes surge or when software upgrades introduce new log templates (concept drift). 

This system provides an end-to-end unsupervised pipeline that:
- **Learns Normal Execution Patterns**: Uses PyTorch LSTM next-event prediction and dense Autoencoder volumetric reconstruction.
- **Understands Log Variations**: Employs Drain3 prefix-tree clustering with a TF-IDF semantic fallback matcher.
- **Explains Every Anomaly in Plain English**: Highlights unexpected transitions, candidate alternatives, probability gaps, and frequency spikes.
- **Monitors Concept Drift**: Tracks template distribution shifts using the Population Stability Index (PSI).
- **Refreshes Models Without Labels**: Automatically incorporates drifted normal patterns via high-confidence pseudo-normal windows.
- **Maintains Audit History**: Persists every alert, its raw suspicious logs, severity, and status in SQLite.

---

## 2. Architecture & Modules

```
aiops-log-anomaly-detector/
├── api/                    # FastAPI REST backend
│   ├── app.py              # Application factory & middleware
│   ├── routes.py           # REST endpoints
│   └── schemas.py          # Pydantic data schemas
├── dashboard/              # Streamlit interactive UI
│   ├── app.py              # Main dashboard entrypoint
│   └── pages/              # Overview, Analyzer, Alerts, Drift, Retrain, Benchmark
├── data/                   # Data storage
│   ├── raw/                # HDFS / BGL / Custom raw logs
│   ├── processed/          # Preprocessed sequences and sessions
│   └── samples/            # Pre-packaged sample logs for testing
├── db/                     # SQLite persistence
│   ├── database.py         # Thread-safe SQLite repository
│   └── schema.sql          # Schema DDL (alerts, drift, models)
├── models/                 # Checkpoints & Drain3 persistence
├── src/                    # Core algorithms
│   ├── ingestion/          # Loaders for HDFS, BGL & custom files
│   ├── preprocessing/      # Regex masking & sessionizing (block/time)
│   ├── parsing/            # Drain3 parser & semantic matcher
│   ├── detection/          # PyTorch LSTM & Autoencoder models, calibration
│   ├── explainability/     # Natural language explanation generator
│   └── drift/              # PSI monitor & unsupervised refresher
├── tests/                  # Pytest test suite
├── requirements.txt        # Pinned dependencies
├── run_demo.py             # One-click verification & benchmark runner
└── README.md
```

---

## 3. Strict Chronological Evaluation Protocol

Evaluations on Loghub datasets (HDFS and BGL) enforce a strict temporal split:
- **Oldest 60%**: Training split (learns Drain3 templates and normal baseline behavior)
- **Next 20%**: Validation split (tunes decision thresholds for target $\ge 90\%$ F1 and $< 10\%$ FPR)
- **Newest 20%**: Test split (unseen operational logs evaluated for Precision, Recall, F1, and FPR)

*Data is strictly partitioned chronologically; random shuffling across splits is prohibited to prevent data leakage.*

---

## 4. Quick Start

### Installation

```bash
# Clone / navigate to project directory
cd aiops-log-anomaly-detector

# Install dependencies (CPU PyTorch and required packages)
pip install -r requirements.txt
```

### Running the Services

1. **Start FastAPI Backend**:
   ```bash
   python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
   ```
   *Interactive API Docs: `http://localhost:8000/docs`*

2. **Start Streamlit Dashboard**:
   ```bash
   streamlit run dashboard/app.py --server.port 8501
   ```
   *Dashboard UI: `http://localhost:8501`*

3. **Run Automated Test Suite**:
   ```bash
   pytest tests/ -v
   ```

4. **Run End-to-End Evaluation & Demo**:
   ```bash
   python run_demo.py --dataset hdfs
   ```
