"""
Streamlit Monitoring Dashboard for Drift-Aware, Explainable AIOps Log Anomaly Detector.
Provides interactive visualization, log analysis, searchable alert audit trail,
concept drift metrics, model retraining, and Loghub benchmark evaluation.
"""

from datetime import datetime
import io
import json
from pathlib import Path
import sys
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
# Streamlit Page Config & Custom Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="AIOps Log Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-critical { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-high { background-color: #fd7e14; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-medium { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-low { background-color: #17a2b8; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-normal { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .explanation-box {
        background-color: #eef2f7;
        border-left: 4px solid #0d6efd;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 15px;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# Initialize DB & Model on Startup
# -------------------------------------------------------------
@st.cache_resource
def initialize_system():
    db.init_db()
    create_synthetic_dev_dataset()
    # Check if baseline model exists or train initial baseline
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

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.title("AIOps Anomaly Detector")
st.sidebar.caption("Drift-Aware, Explainable CPU Log Monitoring")

active_model = db.get_active_model_metadata()
active_ver = active_model["version"] if active_model else "v1.0.0-uninitialized"
st.sidebar.markdown(f"**Model:** `{active_ver}`")
st.sidebar.markdown(f"**Templates:** `{len(pipeline.drain_parser.get_all_templates())}`")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "System Overview",
        "Live Log Analyzer",
        "Alert History & Audit",
        "Concept Drift Monitor",
        "Model Retraining & Refresh",
        "Benchmark Evaluation (Loghub)",
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "**Key Principles**\n"
    "- CPU-Optimized PyTorch\n"
    "- Dual-Perspective (LSTM + AE)\n"
    "- Plain-English Explanations\n"
    "- Zero Random Shuffling Split\n"
    "- Persistent SQLite Storage"
)

# -------------------------------------------------------------
# View 1: System Overview
# -------------------------------------------------------------
if nav_choice == "System Overview":
    st.header("AIOps System Overview & Statistics")
    st.markdown("Real-time operational health, alert volume, severity breakdown, and system posture.")

    stats = db.get_alert_statistics()
    drift_events = db.get_drift_events(limit=1)
    cur_drift = drift_events[0]["drift_status"] if drift_events else "STABLE"

    # KPI Top Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Alerts", stats.get("total_alerts", 0))
    with kpi2:
        sev_counts = stats.get("severity_counts", {})
        crit_high = sev_counts.get("CRITICAL", 0) + sev_counts.get("HIGH", 0)
        st.metric("Critical / High", crit_high)
    with kpi3:
        st.metric("Average Risk Score", f"{stats.get('average_risk_score', 0):.1f} / 100")
    with kpi4:
        st.metric("Active Model Version", active_ver)
    with kpi5:
        st.metric("Drift Status", cur_drift)

    st.divider()

    col_charts1, col_charts2 = st.columns(2)

    with col_charts1:
        st.subheader("Anomaly Severity Distribution")
        if sev_counts:
            df_sev = pd.DataFrame(list(sev_counts.items()), columns=["Severity", "Count"])
            color_map = {
                "CRITICAL": "#dc3545",
                "HIGH": "#fd7e14",
                "MEDIUM": "#ffc107",
                "LOW": "#17a2b8",
            }
            fig_pie = px.pie(
                df_sev,
                values="Count",
                names="Severity",
                color="Severity",
                color_discrete_map=color_map,
                hole=0.45,
            )
            fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No alert records detected yet.")

    with col_charts2:
        st.subheader("Alert Workflow Status")
        status_counts = stats.get("status_counts", {})
        if status_counts:
            df_status = pd.DataFrame(list(status_counts.items()), columns=["Status", "Count"])
            fig_bar = px.bar(
                df_status,
                x="Status",
                y="Count",
                color="Status",
                color_discrete_sequence=["#0d6efd", "#ffc107", "#198754"],
            )
            fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No alert statuses recorded yet.")

    st.subheader("Recent Operational Alerts")
    recent_alerts = db.get_alerts(limit=5)
    if recent_alerts:
        for a in recent_alerts:
            sev_class = a["severity"].lower()
            with st.expander(f"[{a['severity']}] Session {a['session_id']} | Risk: {a['risk_score']:.1f}/100 | {a['timestamp']}"):
                st.markdown(f"<div class='explanation-box'>{a['explanation']}</div>", unsafe_allow_html=True)
                st.caption(f"Source: {a['log_source']} | Sequence Score: {a.get('sequence_score')} | AE Score: {a.get('ae_score')} | Status: {a.get('status')}")
    else:
        st.info("No recent alerts logged. The system is operating normally.")

# -------------------------------------------------------------
# View 2: Live Log Analyzer & Upload
# -------------------------------------------------------------
elif nav_choice == "Live Log Analyzer":
    st.header("Live Log Analyzer & Real-Time Diagnostics")
    st.markdown("Analyze raw log messages, test subtle wording variations, and inspect plain-English explanations.")

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

    # Initialize state so text area and format are never empty on first load
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
            "1. Select Demo Scenario (Auto-fills log lines below):",
            list(scenarios.keys()),
            key="live_scenario",
            on_change=on_scenario_selected,
        )
    with col_cfg2:
        format_options = ["hdfs", "bgl", "custom"]
        fmt_idx = format_options.index(st.session_state.live_format) if st.session_state.live_format in format_options else 0
        log_type = st.selectbox(
            "2. Log Parser Format:",
            format_options,
            index=fmt_idx,
            key="live_format",
        )

    uploaded_file = st.file_uploader("Or Upload Custom Log File (.log, .txt, .csv):", type=["log", "txt", "csv"])
    if uploaded_file is not None:
        st.session_state.live_log_text = uploaded_file.read().decode("utf-8", errors="ignore")

    log_input = st.text_area(
        "Log Text Stream (Inspect or edit before analyzing):",
        key="live_log_text",
        height=180,
    )

    if st.button("Analyze Logs Now", type="primary"):
        if not log_input.strip():
            st.error("Please provide log lines to analyze. Select a scenario above or paste your log lines.")
        else:
            with st.spinner("Processing logs through Drain3, LSTM and Autoencoder..."):
                loader = get_log_loader(log_type)
                lines = log_input.strip().splitlines()
                records = [loader.parse_line(l, line_num=i + 1) for i, l in enumerate(lines) if l.strip()]

                alerts = pipeline.detect_stream(records, log_type=log_type, persist_alerts=True)

            if alerts:
                st.subheader(f"Detected {len(alerts)} Anomaly Alert(s)")
                for a in alerts:
                    sev = a["severity"]
                    risk = a["risk_score"]
                    color = "#dc3545" if sev in ["CRITICAL", "HIGH"] else "#fd7e14" if sev == "MEDIUM" else "#17a2b8"

                    st.markdown(
                        f"""
                        <div style='border: 1px solid {color}; border-radius: 6px; padding: 14px; margin-bottom: 14px;'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <h4>Alert: Session <code>{a['session_id']}</code></h4>
                                <span style='background-color: {color}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;'>
                                    {sev} (Risk: {risk:.1f}/100)
                                </span>
                            </div>
                            <div class='explanation-box' style='margin-top: 10px;'>
                                <strong>Plain-English Diagnostic:</strong><br/>
                                {a['explanation']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if a["suspicious_events"]:
                        st.markdown("**Suspicious Log Lines:**")
                        df_susp = pd.DataFrame(a["suspicious_events"])
                        cols_to_show = [c for c in ["step", "level", "raw_log", "reason"] if c in df_susp.columns]
                        render_table(df_susp[cols_to_show], use_container_width=True)
            else:
                st.success("Normal System Behavior: No anomalies detected across sequential or volumetric models.")

# -------------------------------------------------------------
# View 3: Alert History & Audit Trail
# -------------------------------------------------------------
elif nav_choice == "Alert History & Audit":
    st.header("Persistent Alert History & Audit Trail")
    st.markdown("Query, search, filter, and inspect previously detected alerts persisted in SQLite.")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        sev_filter = st.selectbox("Filter Severity:", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col_f2:
        status_filter = st.selectbox("Filter Status:", ["ALL", "NEW", "ACKNOWLEDGED", "RESOLVED"])
    with col_f3:
        source_filter = st.selectbox("Filter Source:", ["ALL", "hdfs", "bgl", "custom"])
    with col_f4:
        search_query = st.text_input("Search Keyword:", placeholder="e.g. timeout, blk_, checksum")

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

    st.caption(f"Showing {len(alerts_list)} matching alert record(s).")

    if alerts_list:
        table_data = []
        for r in alerts_list:
            table_data.append({
                "ID": r["id"],
                "Timestamp": r["timestamp"],
                "Session ID": r["session_id"],
                "Source": r["log_source"],
                "Severity": r["severity"],
                "Risk": f"{r['risk_score']:.1f}",
                "Status": r.get("status", "NEW"),
            })
        render_table(pd.DataFrame(table_data), use_container_width=True)

        st.subheader("Alert Deep-Dive Inspection")
        alert_ids = [r["id"] for r in alerts_list]
        selected_id = st.selectbox("Select Alert ID to Inspect:", alert_ids)

        chosen = next((r for r in alerts_list if r["id"] == selected_id), None)
        if chosen:
            st.markdown(f"### Alert #{chosen['id']} - Session `{chosen['session_id']}`")
            st.markdown(f"<div class='explanation-box'>{chosen['explanation']}</div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk Score", f"{chosen['risk_score']:.1f} / 100")
            c2.metric("Severity", chosen["severity"])
            c3.metric("Sequence Score", f"{chosen.get('sequence_score', 0):.3f}")
            c4.metric("Autoencoder Score", f"{chosen.get('ae_score', 0):.3f}")

            st.markdown("**Suspicious Events in Trace:**")
            if chosen.get("suspicious_events"):
                render_table(pd.DataFrame(chosen["suspicious_events"]), use_container_width=True)
            else:
                st.info("No specific single-line transition violations logged.")

            # Workflow action buttons
            st.markdown("**Operational Workflow Action:**")
            col_b1, col_b2, _ = st.columns([1, 1, 3])
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
        st.info("No alerts match the specified filter criteria.")

# -------------------------------------------------------------
# View 4: Concept Drift Monitor
# -------------------------------------------------------------
elif nav_choice == "Concept Drift Monitor":
    st.header("Concept Drift Monitoring & Distribution Shift")
    st.markdown("Monitors Population Stability Index (PSI) and novel template emergence over time.")

    drift_records = db.get_drift_events(limit=50)

    if drift_records:
        latest = drift_records[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Status", latest["drift_status"])
        c2.metric("Latest PSI", f"{latest['psi_score']:.4f}")
        c3.metric("Novel Template Ratio", f"{latest['novel_template_ratio']*100:.1f}%")
        c4.metric("Evaluations Count", len(drift_records))

        # Plotly PSI Trend Chart
        df_drift = pd.DataFrame(drift_records)
        df_drift = df_drift.iloc[::-1]  # Chronological order

        fig_psi = go.Figure()
        fig_psi.add_trace(go.Scatter(
            x=df_drift["timestamp"],
            y=df_drift["psi_score"],
            mode="lines+markers",
            name="PSI Score",
            line=dict(color="#0d6efd", width=2)
        ))
        # Add PSI Threshold Lines
        fig_psi.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Moderate Drift (0.10)")
        fig_psi.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="Significant Drift (0.20)")
        fig_psi.update_layout(title="Population Stability Index (PSI) Trend Over Observation Windows", height=350)
        st.plotly_chart(fig_psi, use_container_width=True)

        st.subheader("Historical Drift Evaluations Log")
        render_table(
            df_drift[["id", "timestamp", "log_source", "psi_score", "novel_template_ratio", "drift_status"]],
            use_container_width=True
        )
    else:
        st.info("No drift observations recorded yet. Ingesting live streams will populate drift history.")

# -------------------------------------------------------------
# View 5: Model Retraining & Refresh
# -------------------------------------------------------------
elif nav_choice == "Model Retraining & Refresh":
    st.header("Model Management & Unsupervised Refresh")
    st.markdown("Autonomously adapt models to concept drift using high-confidence pseudo-normal sessions.")

    active_meta = db.get_active_model_metadata()
    if active_meta:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Model Version", active_meta["version"])
        m2.metric("Model Type", active_meta["model_type"].upper())
        m3.metric("Vocabulary Size", active_meta["total_templates"])
        m4.metric("Calibrated Threshold", f"{active_meta.get('threshold_sequence', 0.5):.3f}")

        st.markdown("**Performance Benchmarks:**")
        mb1, mb2, mb3, mb4 = st.columns(4)
        mb1.metric("Validation F1", f"{active_meta.get('val_f1', 0)*100:.1f}%")
        mb2.metric("Validation Precision", f"{active_meta.get('val_precision', 0)*100:.1f}%")
        mb3.metric("Validation Recall", f"{active_meta.get('val_recall', 0)*100:.1f}%")
        mb4.metric("Validation FPR", f"{active_meta.get('val_fpr', 0)*100:.1f}%")

    st.divider()

    st.subheader("Trigger Unsupervised Model Refresh")
    st.markdown(
        "Upload newly arrived operational logs. The system filters out anomalies (risk >= 30), "
        "retains pseudo-normal sessions, updates templates, and fine-tunes models without manual labeling."
    )

    refresh_text = st.text_area(
        "Paste Operational Log Stream for Adaptation:",
        height=140,
        placeholder="Paste log stream here..."
    )
    conf_th = st.slider("Pseudo-Normal Confidence Threshold (Risk < X):", min_value=10.0, max_value=50.0, value=30.0)

    if st.button("Execute Unsupervised Model Refresh", type="primary"):
        if not refresh_text.strip():
            st.error("Please provide operational log lines for model adaptation.")
        else:
            with st.spinner("Filtering pseudo-normal windows and refreshing neural models..."):
                loader = get_log_loader("custom")
                lines = refresh_text.strip().splitlines()
                recs = [loader.parse_line(l, line_num=i+1) for i, l in enumerate(lines) if l.strip()]
                refresh_res = pipeline.refresh_model(recs, log_type="custom")

            if refresh_res.get("status") == "SUCCESS":
                st.success(
                    f"Model Refreshed Successfully! New version: `{refresh_res['model_version']}`. "
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
    st.header("Authentic Dataset Benchmark Evaluation")
    st.markdown(
        "Rigorous evaluation on **Loghub HDFS** and **BGL** benchmarks using a **strict chronological split** "
        "(60% Train, 20% Val, 20% Test) without random shuffling."
    )

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        eval_dataset = st.selectbox("Select Benchmark Dataset:", ["hdfs", "bgl"])
    with col_b2:
        max_lines_eval = st.number_input("Evaluation Sample Lines:", min_value=500, max_value=2000, value=1200, step=200)
    with col_b3:
        st.write("")
        st.write("")
        run_bench_btn = st.button("Run Chronological Benchmark", type="primary")

    if run_bench_btn:
        with st.spinner(f"Running authentic evaluation on {eval_dataset.upper()}..."):
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

        test_m = train_res["test_metrics"]
        st.subheader("Test Split Evaluation Results (Unseen 20% Future Data)")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Precision", f"{test_m['precision']*100:.1f}%")
        r2.metric("Recall", f"{test_m['recall']*100:.1f}%")
        r3.metric("F1-Score", f"{test_m['f1_score']*100:.1f}%", help="Target: >= 90%")
        r4.metric("False Positive Rate", f"{test_m['false_positive_rate']*100:.1f}%", help="Target: < 10%")

        # Target verification alerts
        if test_m["f1_score"] >= 0.90:
            st.success(f"Target F1 Achieved: {test_m['f1_score']*100:.1f}% >= 90.0%")
        else:
            st.info(f"F1-Score: {test_m['f1_score']*100:.1f}% (Target: 90%)")

        if test_m["false_positive_rate"] <= 0.10:
            st.success(f"Target FPR Achieved: {test_m['false_positive_rate']*100:.1f}% <= 10.0%")
        else:
            st.warning(f"FPR: {test_m['false_positive_rate']*100:.1f}%")

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
        st.markdown("**Confusion Matrix:**")
        render_table(cm_df, use_container_width=True)

        st.markdown(
            f"**Split Statistics:** Total Sessions: `{train_res['total_sessions']}` | "
            f"Train (60%): `{train_res['train_sessions']}` | "
            f"Val (20%): `{train_res['val_sessions']}` | "
            f"Test (20%): `{train_res['test_sessions']}`"
        )
