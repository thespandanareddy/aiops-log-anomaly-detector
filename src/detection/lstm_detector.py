"""
Sequence-Based PyTorch LSTM Anomaly Detector (DeepLog Architecture).
Predicts expected next-event templates and scores sequence transition anomalies.
CPU-optimized for fast training and sub-millisecond inference.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.config import (
    LSTM_BATCH_SIZE,
    LSTM_EMBED_DIM,
    LSTM_EPOCHS,
    LSTM_HIDDEN_DIM,
    LSTM_LR,
    LSTM_NUM_LAYERS,
    LSTM_TOP_K,
    LSTM_WINDOW_SIZE,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class LSTMPredictorNet(nn.Module):
    """PyTorch LSTM Neural Network for Next-Event Prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = LSTM_EMBED_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.vocab_size = max(vocab_size, 2)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, window_size)
        emb = self.embedding(x)  # (batch_size, window_size, embed_dim)
        out, _ = self.lstm(emb)  # (batch_size, window_size, hidden_dim)
        logits = self.fc(out[:, -1, :])  # Take representation of last step
        return logits


class LSTMAnomalyDetector:
    """Manages vocabulary, dataset creation, training, and transition scoring."""

    def __init__(
        self,
        window_size: int = LSTM_WINDOW_SIZE,
        top_k: int = LSTM_TOP_K,
        model_path: Optional[Path] = MODELS_DIR / "lstm_detector.pt",
    ):
        self.window_size = window_size
        self.top_k = top_k
        self.model_path = Path(model_path) if model_path else None
        self.event2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2event: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.model: Optional[LSTMPredictorNet] = None
        self.device = torch.device("cpu")

    def build_vocab(self, all_events: List[str]) -> None:
        """Construct integer vocabulary mappings from event templates."""
        unique_events = sorted(set(e for e in all_events if e not in {"<PAD>", "<UNK>"}))
        for idx, event in enumerate(unique_events, start=2):
            self.event2idx[event] = idx
            self.idx2event[idx] = event

    @property
    def vocab_size(self) -> int:
        return len(self.event2idx)

    def _event_to_idx(self, event: str) -> int:
        return self.event2idx.get(event, self.event2idx["<UNK>"])

    def create_sequences(
        self,
        event_sequences: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform variable-length event lists into fixed window input-target pairs."""
        inputs: List[List[int]] = []
        targets: List[int] = []

        for seq in event_sequences:
            if not seq:
                continue
            int_seq = [self._event_to_idx(e) for e in seq]
            # Pad short sequences
            padded_seq = [self.event2idx["<PAD>"]] * self.window_size + int_seq
            for i in range(len(int_seq)):
                window = padded_seq[i: i + self.window_size]
                target = int_seq[i]
                inputs.append(window)
                targets.append(target)

        if not inputs:
            # Fallback for empty sequences
            return torch.zeros((1, self.window_size), dtype=torch.long), torch.zeros(1, dtype=torch.long)

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def fit(
        self,
        normal_sequences: List[List[str]],
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        lr: float = LSTM_LR,
    ) -> Dict[str, float]:
        """Train the LSTM predictor exclusively on normal log sequences."""
        # Ensure vocab includes all events
        all_events = [e for seq in normal_sequences for e in seq]
        self.build_vocab(all_events)

        x_train, y_train = self.create_sequences(normal_sequences)
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = LSTMPredictorNet(vocab_size=self.vocab_size).to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        epoch_losses = []
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            steps = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            avg_loss = total_loss / max(steps, 1)
            epoch_losses.append(avg_loss)

        if self.model_path:
            self.save()

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "vocab_size": self.vocab_size,
            "total_windows": len(x_train),
        }

    def score_session(
        self,
        sequence: List[str]
    ) -> Tuple[float, List[Dict[str, object]]]:
        """
        Evaluate a sequence of log events.
        Returns:
            (sequence_anomaly_score, anomalous_transitions_diagnostic_info)
        """
        if not sequence or self.model is None:
            return 0.0, []

        self.model.eval()
        int_seq = [self._event_to_idx(e) for e in sequence]
        padded_seq = [self.event2idx["<PAD>"]] * self.window_size + int_seq

        anomalous_transitions = []
        unmatched_count = 0
        total_transitions = len(sequence)

        with torch.no_grad():
            for i in range(total_transitions):
                window = padded_seq[i: i + self.window_size]
                actual_idx = int_seq[i]
                actual_event = sequence[i]

                tensor_x = torch.tensor([window], dtype=torch.long, device=self.device)
                logits = self.model(tensor_x)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

                top_k = min(self.top_k, self.vocab_size)
                topk_probs, topk_indices = torch.topk(probs, k=top_k)

                topk_event_ids = [self.idx2event.get(int(idx), "<UNK>") for idx in topk_indices]
                actual_prob = float(probs[actual_idx])

                is_abnormal = actual_idx not in topk_indices

                if is_abnormal:
                    unmatched_count += 1
                    anomalous_transitions.append({
                        "step_index": i,
                        "actual_event": actual_event,
                        "actual_prob": round(actual_prob, 4),
                        "expected_topk": [
                            {"event_id": eid, "prob": round(float(p), 4)}
                            for eid, p in zip(topk_event_ids, topk_probs)
                        ],
                    })

        # Calculate normalized sequence anomaly score [0.0 - 1.0]
        score = unmatched_count / max(total_transitions, 1)
        return float(score), anomalous_transitions

    def save(self, path: Optional[Path] = None) -> None:
        """Save model weights and vocabulary."""
        target = path or self.model_path
        if not target or self.model is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "vocab": self.event2idx,
                "window_size": self.window_size,
                "top_k": self.top_k,
            },
            target,
        )

    def load(self, path: Optional[Path] = None) -> None:
        """Load model weights and vocabulary."""
        target = path or self.model_path
        if not target or not target.exists():
            return
        checkpoint = torch.load(target, map_location=self.device)
        self.event2idx = checkpoint["vocab"]
        self.idx2event = {v: k for k, v in self.event2idx.items()}
        self.window_size = checkpoint.get("window_size", self.window_size)
        self.top_k = checkpoint.get("top_k", self.top_k)

        self.model = LSTMPredictorNet(vocab_size=self.vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
