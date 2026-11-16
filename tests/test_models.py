"""Unit tests for LSTM sequence predictor, Autoencoder, and Calibrator."""

import pytest
from src.detection.autoencoder import AutoencoderDetector
from src.detection.calibrator import Calibrator
from src.detection.lstm_detector import LSTMAnomalyDetector
from src.preprocessing.sessionizer import LogSession


def test_lstm_fit_and_score():
    """Verify LSTM trains on normal sequences and flags anomalous sequences."""
    normal_sequences = [
        ["E1", "E2", "E3", "E4"],
        ["E1", "E2", "E3", "E4"],
        ["E1", "E2", "E3", "E4"],
        ["E1", "E2", "E3", "E4"],
    ]

    lstm = LSTMAnomalyDetector(window_size=3, top_k=2, model_path=None)
    fit_res = lstm.fit(normal_sequences, epochs=5, lr=0.01)
    assert fit_res["vocab_size"] >= 4

    # Normal sequence should have low sequence anomaly score
    normal_score, normal_diag = lstm.score_session(["E1", "E2", "E3", "E4"])
    assert normal_score <= 0.50

    # Highly anomalous sequence with unexpected transition
    anom_score, anom_diag = lstm.score_session(["E4", "E1", "E4", "E1"])
    assert len(anom_diag) > 0


def test_autoencoder_fit_and_score():
    """Verify Autoencoder reconstructs normal count vectors and flags high residuals."""
    normal_sessions = [
        ["E1", "E1", "E2"],
        ["E1", "E1", "E2"],
        ["E1", "E1", "E2"],
    ]

    ae = AutoencoderDetector(model_path=None, hidden_dims=[16, 8])
    fit_res = ae.fit(normal_sessions, epochs=5, lr=0.01)
    assert fit_res["vocab_dim"] >= 2

    # Normal session count
    normal_mse, normal_devs = ae.score_session(["E1", "E1", "E2"])
    assert normal_mse < 0.2

    # Severe frequency burst (20 times of E2)
    burst_events = ["E2"] * 20
    burst_mse, burst_devs = ae.score_session(burst_events)
    assert burst_mse > normal_mse
    assert len(burst_devs) > 0


def test_calibrator():
    """Verify threshold calibration and performance evaluation."""
    calibrator = Calibrator()

    lstm = LSTMAnomalyDetector(window_size=3, top_k=2, model_path=None)
    normal_seqs = [["E1", "E2", "E3"]] * 10
    lstm.fit(normal_seqs, epochs=3)

    ae = AutoencoderDetector(model_path=None, hidden_dims=[8, 4])
    ae.fit(normal_seqs, epochs=3)

    val_sessions = [
        LogSession("s1", ["E1", "E2", "E3"], [], is_anomaly=False),
        LogSession("s2", ["E1", "E2", "E3"], [], is_anomaly=False),
        LogSession("s3", ["E3", "E3", "E3", "E3"], [], is_anomaly=True),
    ]

    metrics = calibrator.calibrate(val_sessions, lstm, ae)
    assert calibrator.is_calibrated is True
    assert "calibrated_threshold" in metrics

    eval_res = calibrator.evaluate(val_sessions, lstm, ae)
    assert "f1_score" in eval_res
    assert "confusion_matrix" in eval_res
    assert eval_res["total_evaluated"] == 3
