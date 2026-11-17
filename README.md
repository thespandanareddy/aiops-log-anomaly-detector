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
AIOps_Log_Anomaly_Detector/
â”œâ”€â”€ api/                    # FastAPI REST backend
â”‚   â”œâ”€â”€ app.py              # Application factory & middleware
â”‚   â”œâ”€â”€ routes.py           # REST endpoints
â”‚   â””â”€â”€ schemas.py          # Pydantic data schemas
â”œâ”€â”€ dashboard/              # Streamlit interactive UI
â”‚   â”œâ”€â”€ app.py              # Main dashboard entrypoint
â”‚   â””â”€â”€ pages/              # Overview, Analyzer, Alerts, Drift, Retrain, Benchmark
â”œâ”€â”€ data/                   # Data storage
â”‚   â”œâ”€â”€ raw/                # HDFS / BGL / Custom raw logs
â”‚   â”œâ”€â”€ processed/          # Preprocessed sequences and sessions
â”‚   â””â”€â”€ samples/            # Pre-packaged sample logs for testing
â”œâ”€â”€ db/                     # SQLite persistence
â”‚   â”œâ”€â”€ database.py         # Thread-safe SQLite repository
â”‚   â””â”€â”€ schema.sql          # Schema DDL (alerts, drift, models)
â”œâ”€â”€ models/                 # Checkpoints & Drain3 persistence
â”œâ”€â”€ src/                    # Core algorithms
â”‚   â”œâ”€â”€ ingestion/          # Loaders for HDFS, BGL & custom files
â”‚   â”œâ”€â”€ preprocessing/      # Regex masking & sessionizing (block/time)
â”‚   â”œâ”€â”€ parsing/            # Drain3 parser & semantic matcher
â”‚   â”œâ”€â”€ detection/          # PyTorch LSTM & Autoencoder models, calibration
â”‚   â”œâ”€â”€ explainability/     # Natural language explanation generator
â”‚   â””â”€â”€ drift/              # PSI monitor & unsupervised refresher
â”œâ”€â”€ tests/                  # Pytest test suite
â”œâ”€â”€ requirements.txt        # Pinned dependencies
â”œâ”€â”€ run_demo.py             # One-click verification & benchmark runner
â””â”€â”€ README.md
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
cd F:\AIOps_Log_Anomaly_Detector

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

