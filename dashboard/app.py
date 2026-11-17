"""
Enterprise AIOps Log Anomaly Detection Platform.
Drift-Aware, Explainable, CPU-Optimized Observability Dashboard.
"""

from datetime import datetime
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.database import db
from src.config import DATASET_URLS, RAW_DATA_DIR, SAMPLES_DATA_DIR
from src.detection.calibrator import Calibrator
from src.ingestion.dataset_downloader import ensure_benchmarks_available, create_synthetic_dev_dataset
from src.ingestion.log_loader import LogRecord, get_log_loader
from src.pipeline import pipeline

# -------------------------------------------------------------
# Streamlit Page Config & Theme Settings
# -------------------------------------------------------------
st.set_page_config(
    page_title="AIOps Log Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Dark Theme CSS
CUSTOM_DARK_CSS = """
<style>
/* Global Canvas & Typography */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #101827 !important;
    color: #F1F5F9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #1B2738 !important;
    border-right: 1px solid #243347 !important;
}

[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}

/* Card and Container Styling */
.metric-card {
    background-color: #1B2738;
    border: 1px solid #243347;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.kpi-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-bottom: 4px;
}

.kpi-val {
    font-size: 26px;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 11.5px;
    color: #64748B;
    margin-top: 4px;
}

/* Alert Severity Badges */
.badge-critical { background-color: #EF4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; letter-spacing: 0.04em; }
.badge-high { background-color: #F97316; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; letter-spacing: 0.04em; }
.badge-medium { background-color: #F59E0B; color: black; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; letter-spacing: 0.04em; }
.badge-low { background-color: #3B82F6; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; letter-spacing: 0.04em; }
.badge-normal { background-color: #10B981; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; letter-spacing: 0.04em; }

/* Explanation Box */
.explanation-box {
    background-color: #141E2D;
    border: 1px solid #243347;
    border-left: 4px solid #3B82F6;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 13.5px;
    line-height: 1.55;
    color: #E2E8F0;
    margin-top: 10px;
    margin-bottom: 12px;
}

/* Streamlit Inputs & Selectors */
[data-baseweb="select"] > div {
    background-color: #1B2738 !important;
    border-color: #243347 !important;
    color: #F1F5F9 !important;
}

textarea, input[type="text"] {
    background-color: #141E2D !important;
    border: 1px solid #243347 !important;
    color: #F1F5F9 !important;
    border-radius: 6px !important;
}

/* Table container styling */
div[data-testid="stTable"], div[data-testid="stDataFrame"] {
    background-color: #1B2738;
    border: 1px solid #243347;
    border-radius: 8px;
    padding: 4px;
}

/* Button overrides */
button[kind="primary"] {
    background-color: #3B82F6 !important;
    border-color: #3B82F6 !important;
    color: white !important;
    font-weight: 600 !important;
}

button[kind="primary"]:hover {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
}

button[kind="secondary"] {
    background-color: #1B2738 !important;
    border: 1px solid #243347 !important;
    color: #F1F5F9 !important;
}
</style>
"""
st.markdown(CUSTOM_DARK_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# Initialize DB & Model on Startup
# -------------------------------------------------------------
@st.cache_resource
def initialize_system():
    db.init_db()
    create_synthetic_dev_dataset()
    active = db.get_active_model_metadata()
    if not active or not pipeline.is_trained:
        try:
            benchmarks = ensure_benchmarks_available()
            loader = get_log_loader("hdfs", labels_file=benchmarks["hdfs_labels"])
            records = loader.load_file(benchmarks["hdfs_2k"], max_lines=600)
            pipeline.train_on_records(records, log_type="hdfs", lstm_epochs=5, ae_epochs=6)
        except Exception as e:
            st.warning(f"Initial baseline training warning: {e}")
    return True

initialize_system()


def render_table(df, use_container_width=True):
    """Safely render tables with st.dataframe and fallback if needed."""
    try:
        st.dataframe(df, use_container_width=use_container_width)
    except Exception:
        try:
            st.table(df)
        except Exception:
            st.write(df)


def format_percent(val, default="N/A", digits=1):
    """Safely format a ratio (0.0-1.0) as percentage or return default when None or invalid."""
    if val is None:
        return default
    try:
        f = float(val)
        return f"{f * 100:.{digits}f}%"
    except (ValueError, TypeError):
        return default


def format_float(val, default="N/A", digits=2):
    """Safely format a float or return default when None or invalid."""
    if val is None:
        return default
    try:
        f = float(val)
        return f"{f:.{digits}f}"
    except (ValueError, TypeError):
        return default


def apply_dark_plotly_theme(fig, height=320, title=None):
    """Applies enterprise dark theme to any Plotly chart."""
    fig.update_layout(
        paper_bgcolor="#1B2738",
        plot_bgcolor="#1B2738",
        font=dict(color="#F1F5F9", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12),
        title=dict(text=title, font=dict(color="#F1F5F9", size=14, family="sans-serif")) if title else None,
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        height=height,
        legend=dict(
            font=dict(color="#94A3B8", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        gridcolor="#243347",
        linecolor="#243347",
        tickfont=dict(color="#94A3B8", size=11),
        title_font=dict(color="#94A3B8", size=12),
    )
    fig.update_yaxes(
        gridcolor="#243347",
        linecolor="#243347",
        tickfont=dict(color="#94A3B8", size=11),
        title_font=dict(color="#94A3B8", size=12),
    )
    return fig


def render_header(title: str, description: str, status_badge: str = "SYSTEM HEALTHY", badge_color: str = "#10B981"):
    """Render unified top header bar for each page."""
    st.markdown(
        f"""
        <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 22px; padding-bottom: 16px; border-bottom: 1px solid #243347;'>
            <div>
                <div style='font-size: 22px; font-weight: 700; color: #F1F5F9; letter-spacing: -0.02em; margin-bottom: 4px;'>{title}</div>
                <div style='font-size: 13.5px; color: #94A3B8;'>{description}</div>
            </div>
            <div style='display: flex; align-items: center; gap: 8px; background-color: #1B2738; padding: 6px 14px; border-radius: 6px; border: 1px solid #243347;'>
                <span style='height: 8px; width: 8px; border-radius: 50%; background-color: {badge_color}; display: inline-block;'></span>
                <span style='font-size: 11px; font-weight: 700; color: #F1F5F9; letter-spacing: 0.06em; text-transform: uppercase;'>{status_badge}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------
# Sidebar Navigation & Telemetry
# -------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='margin-bottom: 16px;'>
        <div style='font-size: 17px; font-weight: 700; color: #F1F5F9; letter-spacing: -0.01em;'>AIOps Log Monitor</div>
        <div style='font-size: 12px; color: #94A3B8;'>Drift-Aware, Explainable CPU Platform</div>
    </div>
    """,
    unsafe_allow_html=True
)

active_model = db.get_active_model_metadata()
active_ver = active_model["version"] if active_model else "v1.0.0-baseline"
template_count = len(pipeline.drain_parser.get_all_templates())

# Model and Engine telemetry pills
st.sidebar.markdown(
    f"""
    <div style='display: flex; gap: 8px; margin-bottom: 16px;'>
        <span style='background-color: #141E2D; border: 1px solid #243347; color: #3B82F6; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px;'>{active_ver}</span>
        <span style='background-color: #141E2D; border: 1px solid #243347; color: #14B8A6; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px;'>{template_count} Templates</span>
    </div>
    """,
    unsafe_allow_html=True
)

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "System Overview",
        "Live Log Analyzer",
        "Alert History & Audit",
        "Concept Drift Monitor",
        "Model Retraining & Refresh",
        "Benchmark Evaluation (Loghub)",
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# System Health & Component Status Panel
st.sidebar.markdown(
    """
    <div style='background-color: #141E2D; border: 1px solid #243347; border-radius: 6px; padding: 12px; font-size: 11.5px;'>
        <div style='font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;'>System Posture</div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
            <span style='color: #94A3B8;'>Inference Engine:</span>
            <span style='color: #10B981; font-weight: 600;'>PyTorch CPU</span>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
            <span style='color: #94A3B8;'>Parser Core:</span>
            <span style='color: #3B82F6; font-weight: 600;'>Drain3 Streaming</span>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
            <span style='color: #94A3B8;'>Audit Storage:</span>
            <span style='color: #F1F5F9; font-weight: 600;'>SQLite WAL</span>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #94A3B8;'>Drift Guard:</span>
            <span style='color: #14B8A6; font-weight: 600;'>Active (PSI)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------------
# View 1: System Overview
# -------------------------------------------------------------
if nav_choice == "System Overview":
    render_header(
        title="System Overview & Telemetry",
        description="Real-time operational health, alert distribution, and cluster anomaly posture.",
        status_badge="ONLINE - CPU READY",
        badge_color="#10B981"
    )

    stats = db.get_alert_statistics()
    drift_events = db.get_drift_events(limit=1)
    cur_drift = drift_events[0]["drift_status"] if drift_events else "STABLE"

    drift_color = "#10B981" if cur_drift == "STABLE" else "#F59E0B" if "MODERATE" in cur_drift else "#EF4444"

    # Top KPI Metrics Row
    k1, k2, k3, k4, k5 = st.columns(5)
    sev_counts = stats.get("severity_counts", {})
    crit_high = sev_counts.get("CRITICAL", 0) + sev_counts.get("HIGH", 0)

    with k1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='kpi-title'>Total Alerts Recorded</div>
                <div class='kpi-val'>{stats.get('total_alerts', 0)}</div>
                <div class='kpi-sub'>All-time persistent traces</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""
            <div class='metric-card' style='border-top: 3px solid #EF4444;'>
                <div class='kpi-title'>Critical & High Alerts</div>
                <div class='kpi-val' style='color: #EF4444;'>{crit_high}</div>
                <div class='kpi-sub'>Immediate triage required</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        avg_risk_disp = format_float(stats.get('average_risk_score'), default="0.0", digits=1)
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='kpi-title'>Average Risk Score</div>
                <div class='kpi-val'>{avg_risk_disp}<span style='font-size: 14px; color: #94A3B8;'> / 100</span></div>
                <div class='kpi-sub'>Calibrated sequential & volumetric</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='kpi-title'>Active Model Version</div>
                <div class='kpi-val' style='font-size: 20px; color: #3B82F6;'>{active_ver}</div>
                <div class='kpi-sub'>{template_count} extracted templates</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k5:
        st.markdown(
            f"""
            <div class='metric-card' style='border-top: 3px solid {drift_color};'>
                <div class='kpi-title'>Concept Drift Status</div>
                <div class='kpi-val' style='font-size: 19px; color: {drift_color};'>{cur_drift}</div>
                <div class='kpi-sub'>Population Stability Index</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Chart Section
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        if sev_counts:
            df_sev = pd.DataFrame(list(sev_counts.items()), columns=["Severity", "Count"])
            color_map = {
                "CRITICAL": "#EF4444",
                "HIGH": "#F97316",
                "MEDIUM": "#F59E0B",
                "LOW": "#3B82F6",
            }
            fig_pie = px.pie(
                df_sev,
                values="Count",
                names="Severity",
                color="Severity",
                color_discrete_map=color_map,
                hole=0.55,
            )
            apply_dark_plotly_theme(fig_pie, height=320, title="Anomaly Severity Breakdown")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No alert records detected yet.")

    with c_chart2:
        status_counts = stats.get("status_counts", {})
        if status_counts:
            df_status = pd.DataFrame(list(status_counts.items()), columns=["Status", "Count"])
            status_colors = {
                "NEW": "#3B82F6",
                "ACKNOWLEDGED": "#F59E0B",
                "RESOLVED": "#10B981"
            }
            fig_bar = px.bar(
                df_status,
                x="Status",
                y="Count",
                color="Status",
                color_discrete_map=status_colors,
            )
            apply_dark_plotly_theme(fig_bar, height=320, title="Operational Triage Workflow Status")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No alert statuses recorded yet.")

    # Recent Operational Alerts Section
    st.markdown("<div style='font-size: 16px; font-weight: 700; color: #F1F5F9; margin-top: 10px; margin-bottom: 12px;'>Recent Operational Alerts</div>", unsafe_allow_html=True)
    recent_alerts = db.get_alerts(limit=5)
    if recent_alerts:
        for a in recent_alerts:
            sev = a["severity"]
            sev_badge_class = (
                "badge-critical" if sev == "CRITICAL"
                else "badge-high" if sev == "HIGH"
                else "badge-medium" if sev == "MEDIUM"
                else "badge-low"
            )
            border_color = (
                "#EF4444" if sev == "CRITICAL"
                else "#F97316" if sev == "HIGH"
                else "#F59E0B" if sev == "MEDIUM"
                else "#3B82F6"
            )

            risk_val = a.get("risk_score")
            risk_disp = format_float(risk_val, default="0.0", digits=1)
            sess_id = a.get("session_id", "Unknown")
            ts = a.get("timestamp", "")

            with st.expander(f"[{sev}] Session {sess_id} | Risk: {risk_disp}/100 | {ts}"):
                st.markdown(
                    f"""
                    <div style='background-color: #1B2738; border: 1px solid {border_color}; border-left: 4px solid {border_color}; border-radius: 6px; padding: 14px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span class='{sev_badge_class}'>{sev}</span>
                                <span style='font-weight: 600; color: #F1F5F9; margin-left: 8px;'>Session <code>{sess_id}</code></span>
                                <span style='font-size: 12px; color: #94A3B8; margin-left: 8px;'>Source: {a.get('log_source', '')}</span>
                            </div>
                            <div style='font-size: 14px; font-weight: 700; color: {border_color};'>Risk Score: {risk_disp}/100</div>
                        </div>
                        <div class='explanation-box'>
                            <div style='font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #3B82F6; margin-bottom: 4px;'>Diagnostic Summary</div>
                            <div>{a.get('explanation', '')}</div>
                        </div>
                        <div style='font-size: 11.5px; color: #64748B;'>
                            Sequence Score: {format_float(a.get('sequence_score'), default='N/A', digits=3)} | Autoencoder Score: {format_float(a.get('ae_score'), default='N/A', digits=3)} | Status: {a.get('status', 'NEW')} | Logged: {ts}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("No recent alerts logged. The system is operating normally.")


# -------------------------------------------------------------
# View 2: Live Log Analyzer & Upload
# -------------------------------------------------------------
elif nav_choice == "Live Log Analyzer":
    render_header(
        title="Live Log Analyzer & Real-Time Diagnostics",
        description="Stream raw log events, inspect multi-model sequential & volumetric signals, and view explainable diagnostics.",
        status_badge="INFERENCE ENGINE READY",
        badge_color="#3B82F6"
    )

    scenarios = {
        "HDFS Checksum Corruption Anomaly (Failure Scenario)": {
            "format": "hdfs",
            "logs": (
                "081109 203524 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_-999999999999999999 src: /10.250.19.104:54108 dest: /10.250.19.104:50010\n"
                "081109 203525 145 ERROR dfs.DataNode$BlockReceiver: Verification failed for block blk_-999999999999999999 checksum mismatch\n"
                "081109 203526 145 WARN dfs.DataNode$DataXceiver: Execution failed unexpected EOF for block blk_-999999999999999999\n"
            ),
        },
        "HDFS Normal Flow (Baseline Healthy Block)": {
            "format": "hdfs",
            "logs": (
                "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010\n"
                "081109 203519 143 INFO dfs.DataNode$BlockReceiver: PacketResponder 1 for block blk_-1608999687919862906 terminating\n"
                "081109 203520 143 INFO dfs.DataNode$BlockReceiver: Received block blk_-1608999687919862906 of size 67108864 from /10.250.19.102\n"
            ),
        },
        "BGL Kernel Panic Anomaly (Failure Scenario)": {
            "format": "bgl",
            "logs": (
                "KERNDTLB 1117842440 2005.06.03 R23-M0-NE-C:J05-U01 2005-06-03-16.47.20.730545 R23-M0-NE-C:J05-U01 RAS KERNEL FATAL data TLB error interrupt\n"
                "KERNDTLB 1117842441 2005.06.03 R23-M0-NE-C:J05-U01 2005-06-03-16.47.21.112233 R23-M0-NE-C:J05-U01 RAS KERNEL FATAL instruction address: 0x00004320\n"
            ),
        },
        "Concept Drift / Unseen Error Scenario": {
            "format": "custom",
            "logs": (
                "2026-11-16 14:00:00 [ERROR] OutOfMemoryError: Metaspace exhausted in JVM container\n"
                "2026-11-16 14:00:01 [CRITICAL] Kernel watchdog killed worker thread PID 9821\n"
            ),
        },
        "Custom Blank Input / Upload Your Own": {
            "format": "custom",
            "logs": "",
        },
    }

    if "live_scenario" not in st.session_state:
        st.session_state.live_scenario = "HDFS Checksum Corruption Anomaly (Failure Scenario)"
    if "live_log_text" not in st.session_state:
        st.session_state.live_log_text = scenarios[st.session_state.live_scenario]["logs"]
    if "live_format" not in st.session_state:
        st.session_state.live_format = scenarios[st.session_state.live_scenario]["format"]

    def on_scenario_selected():
        chosen = st.session_state.live_scenario
        st.session_state.live_log_text = scenarios[chosen]["logs"]
        st.session_state.live_format = scenarios[chosen]["format"]

    col_cfg1, col_cfg2 = st.columns([2, 1])
    with col_cfg1:
        st.selectbox(
            "1. Select Pre-Configured Test Scenario:",
            list(scenarios.keys()),
            key="live_scenario",
            on_change=on_scenario_selected,
        )
    with col_cfg2:
        format_options = ["hdfs", "bgl", "custom"]
        fmt_idx = format_options.index(st.session_state.live_format) if st.session_state.live_format in format_options else 0
        log_type = st.selectbox(
            "2. Log Parser Target Grammar:",
            format_options,
            index=fmt_idx,
            key="live_format",
        )

    uploaded_file = st.file_uploader("Upload External Log File (.log, .txt, .csv):", type=["log", "txt", "csv"])
    if uploaded_file is not None:
        st.session_state.live_log_text = uploaded_file.read().decode("utf-8", errors="ignore")

    log_input = st.text_area(
        "Raw Log Stream Buffer (Inspect or modify before detection):",
        key="live_log_text",
        height=180,
    )

    if st.button("Run Real-Time Detection", type="primary"):
        if not log_input.strip():
            st.error("Please provide log lines to analyze. Select a scenario above or upload a log file.")
        else:
            with st.spinner("Processing stream through Drain3, LSTM, Autoencoder, and Drift Monitor..."):
                loader = get_log_loader(log_type)
                lines = log_input.strip().splitlines()
                records = [loader.parse_line(l, line_num=i + 1) for i, l in enumerate(lines) if l.strip()]

                alerts = pipeline.detect_stream(records, log_type=log_type, persist_alerts=True)

            if alerts:
                st.markdown(f"<div style='font-size: 16px; font-weight: 700; color: #EF4444; margin-top: 14px; margin-bottom: 12px;'>Detected {len(alerts)} Anomaly Alert(s)</div>", unsafe_allow_html=True)
                for a in alerts:
                    sev = a.get("severity", "LOW")
                    risk_val = a.get("risk_score")
                    risk_disp = format_float(risk_val, default="0.0", digits=1)
                    color = "#EF4444" if sev in ["CRITICAL", "HIGH"] else "#F59E0B" if sev == "MEDIUM" else "#3B82F6"
                    badge_class = (
                        "badge-critical" if sev == "CRITICAL"
                        else "badge-high" if sev == "HIGH"
                        else "badge-medium" if sev == "MEDIUM"
                        else "badge-low"
                    )

                    st.markdown(
                        f"""
                        <div style='background-color: #1B2738; border: 1px solid {color}; border-left: 4px solid {color}; border-radius: 8px; padding: 18px; margin-bottom: 16px;'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div>
                                    <span class='{badge_class}'>{sev}</span>
                                    <span style='font-weight: 700; font-size: 15px; color: #F1F5F9; margin-left: 10px;'>Alert: Session <code>{a.get('session_id', 'Unknown')}</code></span>
                                </div>
                                <div>
                                    <span style='font-size: 20px; font-weight: 700; color: {color};'>{risk_disp}</span>
                                    <span style='font-size: 12px; color: #94A3B8;'> / 100 Risk Score</span>
                                </div>
                            </div>
                            <div class='explanation-box'>
                                <div style='font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #3B82F6; margin-bottom: 4px;'>Plain-English Diagnostic</div>
                                <div>{a.get('explanation', '')}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if a["suspicious_events"]:
                        st.markdown("<div style='font-size: 13px; font-weight: 600; color: #94A3B8; margin-bottom: 6px;'>Suspicious Transition Events:</div>", unsafe_allow_html=True)
                        df_susp = pd.DataFrame(a["suspicious_events"])
                        cols_to_show = [c for c in ["step", "level", "raw_log", "reason"] if c in df_susp.columns]
                        render_table(df_susp[cols_to_show], use_container_width=True)
            else:
                st.markdown(
                    """
                    <div style='background-color: #141E2D; border: 1px solid #10B981; border-left: 4px solid #10B981; border-radius: 6px; padding: 16px; margin-top: 14px;'>
                        <div style='font-weight: 700; color: #10B981;'>Normal System Behavior</div>
                        <div style='font-size: 13px; color: #94A3B8; margin-top: 4px;'>No transition violations or volumetric count discrepancies detected across the dual neural model detectors.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# -------------------------------------------------------------
# View 3: Alert History & Audit Trail
# -------------------------------------------------------------
elif nav_choice == "Alert History & Audit":
    render_header(
        title="Persistent Alert History & Audit Trail",
        description="Filter, search, inspect, and update the operational lifecycle of all alerts stored in SQLite.",
        status_badge="AUDIT LOG ACTIVE",
        badge_color="#3B82F6"
    )

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        sev_filter = st.selectbox("Severity Filter:", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col_f2:
        status_filter = st.selectbox("Lifecycle Status:", ["ALL", "NEW", "ACKNOWLEDGED", "RESOLVED"])
    with col_f3:
        source_filter = st.selectbox("Log Source:", ["ALL", "hdfs", "bgl", "custom"])
    with col_f4:
        search_query = st.text_input("Search Logs / Trace:", placeholder="Keyword (e.g. blk_, checksum, timeout)")

    sev_val = None if sev_filter == "ALL" else sev_filter
    stat_val = None if status_filter == "ALL" else status_filter
    src_val = None if source_filter == "ALL" else source_filter
    search_val = search_query.strip() if search_query.strip() else None

    alerts_list = db.get_alerts(
        severity=sev_val,
        status=stat_val,
        log_source=src_val,
        search_query=search_val,
        limit=100
    )

    st.markdown(f"<div style='font-size: 12px; color: #64748B; margin-bottom: 8px;'>Matching Records: <strong>{len(alerts_list)}</strong></div>", unsafe_allow_html=True)

    if alerts_list:
        table_data = []
        for r in alerts_list:
            risk_val = r.get("risk_score")
            risk_disp = format_float(risk_val, default="0.0", digits=1)
            table_data.append({
                "ID": r["id"],
                "Timestamp": r.get("timestamp", ""),
                "Session ID": r.get("session_id", "Unknown"),
                "Source": r.get("log_source", ""),
                "Severity": r.get("severity", "LOW"),
                "Risk Score": risk_disp,
                "Status": r.get("status", "NEW"),
            })
        render_table(pd.DataFrame(table_data), use_container_width=True)

        st.markdown("<div style='font-size: 16px; font-weight: 700; color: #F1F5F9; margin-top: 18px; margin-bottom: 10px;'>Alert Deep-Dive Inspector</div>", unsafe_allow_html=True)
        alert_ids = [r["id"] for r in alerts_list]
        selected_id = st.selectbox("Select Alert Record ID:", alert_ids)

        chosen = next((r for r in alerts_list if r["id"] == selected_id), None)
        if chosen:
            sev = chosen["severity"]
            accent = "#EF4444" if sev in ["CRITICAL", "HIGH"] else "#F59E0B" if sev == "MEDIUM" else "#3B82F6"

            st.markdown(
                f"""
                <div style='background-color: #1B2738; border: 1px solid #243347; border-left: 4px solid {accent}; border-radius: 8px; padding: 18px; margin-bottom: 14px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='font-size: 16px; font-weight: 700; color: #F1F5F9;'>Alert #{chosen['id']} — Session <code>{chosen['session_id']}</code></div>
                        <div><span style='background-color: {accent}; color: white; padding: 3px 10px; border-radius: 4px; font-weight: 700; font-size: 11px;'>{sev}</span></div>
                    </div>
                    <div class='explanation-box'>
                        <div style='font-size: 11px; font-weight: 700; text-transform: uppercase; color: #3B82F6; margin-bottom: 4px;'>Diagnostic Narrative</div>
                        <div>{chosen['explanation']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Metrics Row
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                risk_disp = format_float(chosen.get('risk_score'), default="N/A", digits=1)
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='kpi-title'>Calibrated Risk</div>
                        <div class='kpi-val' style='color: {accent};'>{risk_disp}<span style='font-size: 13px; color: #94A3B8;'> / 100</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with d2:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='kpi-title'>Lifecycle Status</div>
                        <div class='kpi-val' style='font-size: 18px;'>{chosen.get('status', 'NEW')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with d3:
                seq_disp = format_float(chosen.get('sequence_score'), default="N/A", digits=3)
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='kpi-title'>Sequence Score (LSTM)</div>
                        <div class='kpi-val'>{seq_disp}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with d4:
                ae_disp = format_float(chosen.get('ae_score'), default="N/A", digits=3)
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='kpi-title'>Autoencoder Score (MSE)</div>
                        <div class='kpi-val'>{ae_disp}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<div style='font-size: 13px; font-weight: 600; color: #94A3B8; margin-top: 10px; margin-bottom: 6px;'>Suspicious Transition Events in Trace:</div>", unsafe_allow_html=True)
            if chosen.get("suspicious_events"):
                render_table(pd.DataFrame(chosen["suspicious_events"]), use_container_width=True)
            else:
                st.info("No specific single-line transition violations logged for this trace.")

            # Operational Action Buttons
            st.markdown("<div style='font-size: 13px; font-weight: 600; color: #94A3B8; margin-top: 14px; margin-bottom: 8px;'>Operational Lifecycle Actions:</div>", unsafe_allow_html=True)
            col_b1, col_b2, _ = st.columns([1.5, 1.5, 4])
            with col_b1:
                if st.button("Mark Acknowledged", key=f"ack_{chosen['id']}"):
                    db.update_alert_status(chosen["id"], "ACKNOWLEDGED")
                    st.success(f"Alert #{chosen['id']} acknowledged.")
                    st.rerun()
            with col_b2:
                if st.button("Mark Resolved", key=f"res_{chosen['id']}"):
                    db.update_alert_status(chosen["id"], "RESOLVED")
                    st.success(f"Alert #{chosen['id']} marked as resolved.")
                    st.rerun()
    else:
        st.info("No alert records match the specified query filters.")


# -------------------------------------------------------------
# View 4: Concept Drift Monitor
# -------------------------------------------------------------
elif nav_choice == "Concept Drift Monitor":
    render_header(
        title="Concept Drift Monitoring & Distribution Shift",
        description="Tracks Population Stability Index (PSI) and novel template ratio over sliding observation windows.",
        status_badge="STATISTICAL MONITOR ACTIVE",
        badge_color="#14B8A6"
    )

    drift_records = db.get_drift_events(limit=50)

    if drift_records:
        latest = drift_records[0]
        cur_status = latest.get("drift_status", "STABLE")
        status_color = "#10B981" if cur_status == "STABLE" else "#F59E0B" if "MODERATE" in cur_status else "#EF4444"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class='metric-card' style='border-top: 3px solid {status_color};'>
                    <div class='kpi-title'>Latest Drift Status</div>
                    <div class='kpi-val' style='font-size: 18px; color: {status_color};'>{cur_status}</div>
                    <div class='kpi-sub'>Observation window classification</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            psi_disp = format_float(latest.get('psi_score'), default="0.0000", digits=4)
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Population Stability Index</div>
                    <div class='kpi-val'>{psi_disp}</div>
                    <div class='kpi-sub'>Threshold: 0.10 Mod / 0.20 Sig</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            nov_disp = format_percent(latest.get('novel_template_ratio'), default="0.0%", digits=1)
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Novel Template Ratio</div>
                    <div class='kpi-val'>{nov_disp}</div>
                    <div class='kpi-sub'>Unseen log pattern frequency</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Evaluations Audited</div>
                    <div class='kpi-val'>{len(drift_records)}</div>
                    <div class='kpi-sub'>Historical observation windows</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Plotly Time-Series Chart
        df_drift = pd.DataFrame(drift_records).iloc[::-1]  # Chronological order

        fig_psi = go.Figure()
        fig_psi.add_trace(go.Scatter(
            x=df_drift["timestamp"],
            y=df_drift["psi_score"],
            mode="lines+markers",
            name="Observed PSI",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=6, color="#3B82F6")
        ))
        fig_psi.add_hline(y=0.10, line_dash="dash", line_color="#F59E0B", annotation_text="Moderate Drift (0.10)")
        fig_psi.add_hline(y=0.20, line_dash="dash", line_color="#EF4444", annotation_text="Significant Drift (0.20)")
        apply_dark_plotly_theme(fig_psi, height=340, title="Population Stability Index (PSI) Historical Progression")
        st.plotly_chart(fig_psi, use_container_width=True)

        st.markdown("<div style='font-size: 15px; font-weight: 700; color: #F1F5F9; margin-top: 14px; margin-bottom: 8px;'>Historical Drift Evaluations Log</div>", unsafe_allow_html=True)
        render_table(
            df_drift[["id", "timestamp", "log_source", "psi_score", "novel_template_ratio", "drift_status"]],
            use_container_width=True
        )
    else:
        st.info("No drift observations recorded yet. Live log streams will populate drift history.")


# -------------------------------------------------------------
# View 5: Model Management & Unsupervised Refresh
# -------------------------------------------------------------
elif nav_choice == "Model Retraining & Refresh":
    render_header(
        title="Model Management & Unsupervised Adaptation",
        description="Adapt neural models to concept drift using high-confidence pseudo-normal sessions without manual labels.",
        status_badge="SELF-TRAINING ENGINE READY",
        badge_color="#3B82F6"
    )

    active_meta = db.get_active_model_metadata()
    if active_meta:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            ver_disp = active_meta.get("version") or "v1.0.0-baseline"
            trained_disp = active_meta.get("trained_at") or "Baseline"
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Active Model Version</div>
                    <div class='kpi-val' style='font-size: 20px; color: #3B82F6;'>{ver_disp}</div>
                    <div class='kpi-sub'>Trained: {trained_disp}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            mtype_disp = (active_meta.get("model_type") or "BASELINE").upper()
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Architecture</div>
                    <div class='kpi-val' style='font-size: 20px;'>{mtype_disp}</div>
                    <div class='kpi-sub'>DeepLog LSTM + Volumetric AE</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            vocab_disp = str(active_meta.get("total_templates")) if active_meta.get("total_templates") is not None else "N/A"
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Vocabulary Dimension</div>
                    <div class='kpi-val'>{vocab_disp}</div>
                    <div class='kpi-sub'>Unique Drain3 templates</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            th_seq_disp = format_float(active_meta.get("threshold_sequence"), default="N/A", digits=3)
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Calibrated Threshold</div>
                    <div class='kpi-val'>{th_seq_disp}</div>
                    <div class='kpi-sub'>Optimal F1 decision boundary</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='font-size: 14px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 14px; margin-bottom: 8px;'>Validation Benchmarks</div>", unsafe_allow_html=True)
        mb1, mb2, mb3, mb4 = st.columns(4)

        val_f1_disp = format_percent(active_meta.get("val_f1"), default="N/A", digits=1)
        val_prec_disp = format_percent(active_meta.get("val_precision"), default="N/A", digits=1)
        val_rec_disp = format_percent(active_meta.get("val_recall"), default="N/A", digits=1)
        val_fpr_disp = format_percent(active_meta.get("val_fpr"), default="N/A", digits=1)

        f1_color = "#10B981" if val_f1_disp != "N/A" else "#94A3B8"
        fpr_color = "#3B82F6" if val_fpr_disp != "N/A" else "#94A3B8"

        with mb1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Validation F1-Score</div>
                    <div class='kpi-val' style='color: {f1_color};'>{val_f1_disp}</div>
                    <div class='kpi-sub'>Supervised benchmark</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with mb2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Validation Precision</div>
                    <div class='kpi-val'>{val_prec_disp}</div>
                    <div class='kpi-sub'>Positive predictive value</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with mb3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Validation Recall</div>
                    <div class='kpi-val'>{val_rec_disp}</div>
                    <div class='kpi-sub'>True positive rate</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with mb4:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>False Positive Rate</div>
                    <div class='kpi-val' style='color: {fpr_color};'>{val_fpr_disp}</div>
                    <div class='kpi-sub'>False alarm frequency</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div style='background-color: #141E2D; border: 1px solid #243347; border-radius: 6px; padding: 14px 18px; margin-bottom: 14px;'>
                <div style='font-size: 14px; font-weight: 600; color: #94A3B8;'>No Active Model Metadata Available</div>
                <div style='font-size: 12.5px; color: #64748B; margin-top: 2px;'>Baseline model initialization is in progress or awaiting configuration.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background-color: #1B2738; border: 1px solid #243347; border-radius: 8px; padding: 18px;'>
            <div style='font-size: 15px; font-weight: 700; color: #F1F5F9; margin-bottom: 4px;'>Trigger Unsupervised Model Adaptation</div>
            <div style='font-size: 12.5px; color: #94A3B8; margin-bottom: 12px;'>
                Upload or paste operational logs. Sessions with calibrated risk below the threshold are retained as pseudo-normal data.
                The system updates the template dictionary, replays historical normal data to prevent catastrophic forgetting, and fine-tunes both detectors.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    refresh_text = st.text_area(
        "Operational Log Stream for Model Adaptation:",
        height=130,
        placeholder="Paste newly arrived operational logs here...",
    )
    conf_th = st.slider("Pseudo-Normal Confidence Threshold (Risk < X):", min_value=10.0, max_value=50.0, value=30.0)

    if st.button("Execute Unsupervised Model Refresh", type="primary"):
        if not refresh_text.strip():
            st.error("Please provide operational log lines for model adaptation.")
        else:
            with st.spinner("Filtering pseudo-normal sessions and updating neural weights..."):
                loader = get_log_loader("custom")
                lines = refresh_text.strip().splitlines()
                recs = [loader.parse_line(l, line_num=i+1) for i, l in enumerate(lines) if l.strip()]
                refresh_res = pipeline.refresh_model(recs, log_type="custom")

            if refresh_res.get("status") == "SUCCESS":
                st.success(
                    f"Model Refreshed Successfully. New Version: {refresh_res['model_version']}. "
                    f"Incorporated {refresh_res['pseudo_normal_added']} pseudo-normal sessions "
                    f"({refresh_res['suspicious_excluded']} suspicious sessions safely excluded)."
                )
                st.rerun()
            else:
                st.warning(f"Refresh not completed: {refresh_res.get('reason')}")


# -------------------------------------------------------------
# View 6: Benchmark Evaluation (Loghub)
# -------------------------------------------------------------
elif nav_choice == "Benchmark Evaluation (Loghub)":
    render_header(
        title="Loghub Benchmark Dataset Evaluation",
        description="Rigorous evaluation on Loghub HDFS and BGL benchmarks using strict chronological split (60% Train, 20% Val, 20% Test) without random shuffling.",
        status_badge="BENCHMARK READY",
        badge_color="#3B82F6"
    )

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        eval_dataset = st.selectbox("Benchmark Dataset:", ["hdfs", "bgl"])
    with col_b2:
        max_lines_eval = st.number_input("Sample Evaluation Lines:", min_value=500, max_value=2000, value=1200, step=200)
    with col_b3:
        st.write("")
        st.write("")
        run_bench_btn = st.button("Run Chronological Benchmark", type="primary")

    if run_bench_btn:
        with st.spinner(f"Executing authentic chronological benchmark evaluation on {eval_dataset.upper()}..."):
            benchmarks = ensure_benchmarks_available()
            if eval_dataset == "hdfs":
                loader = get_log_loader("hdfs", labels_file=benchmarks["hdfs_labels"])
                recs = loader.load_file(benchmarks["hdfs_2k"], max_lines=int(max_lines_eval))
            else:
                loader = get_log_loader("bgl")
                recs = loader.load_file(benchmarks["bgl_2k"], max_lines=int(max_lines_eval))

            train_res = pipeline.train_on_records(
                records=recs,
                log_type=eval_dataset,
                train_ratio=0.60,
                val_ratio=0.20,
                lstm_epochs=8,
                ae_epochs=10,
            )

        test_m = train_res.get("test_metrics", {})
        st.markdown("<div style='font-size: 16px; font-weight: 700; color: #F1F5F9; margin-top: 14px; margin-bottom: 10px;'>Held-Out Test Split Metrics (Newest 20% Unseen Logs)</div>", unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        prec_disp = format_percent(test_m.get("precision"), default="N/A", digits=1)
        rec_disp = format_percent(test_m.get("recall"), default="N/A", digits=1)
        
        f1_val = test_m.get("f1_score")
        f1_disp = format_percent(f1_val, default="N/A", digits=1)
        f1_color = "#10B981" if (f1_val is not None and f1_val >= 0.90) else "#3B82F6" if f1_disp != "N/A" else "#94A3B8"

        fpr_val = test_m.get("false_positive_rate")
        fpr_disp = format_percent(fpr_val, default="N/A", digits=1)
        fpr_color = "#10B981" if (fpr_val is not None and fpr_val <= 0.10) else "#F59E0B" if fpr_disp != "N/A" else "#94A3B8"

        with r1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Precision</div>
                    <div class='kpi-val'>{prec_disp}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='kpi-title'>Recall</div>
                    <div class='kpi-val'>{rec_disp}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r3:
            st.markdown(
                f"""
                <div class='metric-card' style='border-top: 3px solid {f1_color};'>
                    <div class='kpi-title'>F1-Score</div>
                    <div class='kpi-val' style='color: {f1_color};'>{f1_disp}</div>
                    <div class='kpi-sub'>Target: >= 90.0%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r4:
            st.markdown(
                f"""
                <div class='metric-card' style='border-top: 3px solid {fpr_color};'>
                    <div class='kpi-title'>False Positive Rate</div>
                    <div class='kpi-val' style='color: {fpr_color};'>{fpr_disp}</div>
                    <div class='kpi-sub'>Target: <= 10.0%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Confusion Matrix Display
        cm = test_m["confusion_matrix"]
        cm_df = pd.DataFrame(
            [
                [cm["tn"], cm["fp"]],
                [cm["fn"], cm["tp"]]
            ],
            index=["Actual Normal (0)", "Actual Anomaly (1)"],
            columns=["Predicted Normal (0)", "Predicted Anomaly (1)"]
        )
        st.markdown("<div style='font-size: 14px; font-weight: 700; color: #F1F5F9; margin-top: 14px; margin-bottom: 6px;'>Confusion Matrix</div>", unsafe_allow_html=True)
        render_table(cm_df, use_container_width=True)

        st.markdown(
            f"""
            <div style='background-color: #141E2D; border: 1px solid #243347; border-radius: 6px; padding: 10px 14px; margin-top: 10px; font-size: 12px; color: #94A3B8;'>
                <strong>Dataset Partition Telemetry:</strong> Total Sessions: <code>{train_res['total_sessions']}</code> | 
                Train (60%): <code>{train_res['train_sessions']}</code> | 
                Validation (20%): <code>{train_res['val_sessions']}</code> | 
                Test (20%): <code>{train_res['test_sessions']}</code>
            </div>
            """,
            unsafe_allow_html=True
        )
