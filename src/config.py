"""
Configuration settings, hyperparameters, and directory paths for the
Drift-Aware, Explainable AIOps Log Anomaly Detector.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
DB_DIR = PROJECT_ROOT / "db"
DATABASE_PATH = DB_DIR / "aiops_alerts.db"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DATA_DIR, MODELS_DIR, DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset URLs (LogPAI Loghub official sources)
DATASET_URLS = {
    "hdfs_2k": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    "hdfs_labels": "https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/anomaly_label.csv",
    "bgl_2k": "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log",
}

# Drain3 Log Parsing Parameters
DRAIN_SIM_TH = 0.5          # Similarity threshold
DRAIN_DEPTH = 4             # Max tree depth
DRAIN_MAX_CHILDREN = 100    # Max children per node
DRAIN_PARAM_STR = "<*>"     # Wildcard token
DRAIN_STATE_FILE = MODELS_DIR / "drain3_state.bin"

# Semantic Matcher Settings
SEMANTIC_SIM_THRESHOLD = 0.70  # Threshold to link unseen message to known template

# Preprocessing & Session Settings
HDFS_BLOCK_REGEX = r"(blk_-?\d+)"
BGL_WINDOW_SIZE_SECS = 300     # 5-minute sliding window for time-based sessioning
BGL_STEP_SIZE_SECS = 60        # 1-minute step
DEFAULT_FIXED_WINDOW = 20      # Window size for sequence modeling

# Model Hyperparameters (CPU-Friendly)
# Sequence-based LSTM
LSTM_EMBED_DIM = 32
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_WINDOW_SIZE = 10          # Number of previous events to predict next event
LSTM_TOP_K = 5                 # Acceptable top-k candidates for normal transition
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 64
LSTM_LR = 0.005

# Volumetric Autoencoder
AE_HIDDEN_DIMS = [64, 32, 16]  # Symmetric: Input -> 64 -> 32 -> 16 -> 32 -> 64 -> Input
AE_EPOCHS = 15
AE_BATCH_SIZE = 32
AE_LR = 0.003

# Anomaly Scoring & Fusion Weights
SCORE_WEIGHT_SEQUENCE = 0.55
SCORE_WEIGHT_AUTOENCODER = 0.45

# Risk Severity Thresholds (Calibrated risk 0 - 100)
SEVERITY_THRESHOLDS = {
    "LOW": 40.0,
    "MEDIUM": 65.0,
    "HIGH": 80.0,
    "CRITICAL": 90.0,
}

# Concept Drift (PSI & Template Drift)
PSI_BINS = 10
PSI_MODERATE_DRIFT = 0.10      # PSI between 0.10 and 0.20 indicates slight drift
PSI_SIGNIFICANT_DRIFT = 0.20   # PSI > 0.20 indicates substantial population shift
NOVEL_TEMPLATE_RATE_THRESHOLD = 0.15 # >15% unseen templates triggers drift alert

# Model Refresh Configuration
REFRESH_CONFIDENCE_THRESHOLD = 30.0  # Sessions with calibrated risk < 30 treated as pseudo-normal
REFRESH_MIN_NEW_SAMPLES = 50         # Minimum new normal samples before retraining
