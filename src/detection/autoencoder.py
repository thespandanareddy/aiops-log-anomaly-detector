"""
Autoencoder-Based Volumetric Anomaly Detector (LogAnomaly Count Reconstruction).
Identifies abnormal event frequency surges, missing operations, and count deviations.
CPU-optimized PyTorch architecture with template-level residual attribution.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_HIDDEN_DIMS,
    AE_LR,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class AutoencoderNet(nn.Module):
    """Symmetric Feedforward Autoencoder for Count Vector Reconstruction."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = AE_HIDDEN_DIMS):
        super().__init__()
        dims = [input_dim] + hidden_dims

        # Encoder
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        rev_dims = list(reversed(dims))
        for i in range(len(rev_dims) - 1):
            decoder_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1]))
            if i < len(rev_dims) - 2:
                decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(nn.Softplus())  # Non-negative count reconstruction
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class AutoencoderDetector:
    """Manages count vector transformation, autoencoder training, and reconstruction scoring."""

    def __init__(
        self,
        model_path: Optional[Path] = MODELS_DIR / "autoencoder_detector.pt",
        hidden_dims: List[int] = AE_HIDDEN_DIMS,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.hidden_dims = hidden_dims
        self.event_vocab: List[str] = []
        self.event2idx: Dict[str, int] = {}
        self.model: Optional[AutoencoderNet] = None
        self.device = torch.device("cpu")

    def build_vocab(self, all_events: List[str]) -> None:
        """Create ordered list of template IDs for count vector indexing."""
        unique_events = sorted(set(e for e in all_events if e not in {"<PAD>", "<UNK>"}))
        self.event_vocab = unique_events
        self.event2idx = {e: i for i, e in enumerate(self.event_vocab)}

    @property
    def input_dim(self) -> int:
        return max(len(self.event_vocab), 1)

    def events_to_vector(self, events: List[str]) -> np.ndarray:
        """Convert a sequence of event IDs into a log1p-transformed count vector."""
        vec = np.zeros(self.input_dim, dtype=np.float32)
        for e in events:
            if e in self.event2idx:
                vec[self.event2idx[e]] += 1.0
        # Apply log1p for stability
        return np.log1p(vec)

    def fit(
        self,
        normal_sessions_events: List[List[str]],
        epochs: int = AE_EPOCHS,
        batch_size: int = AE_BATCH_SIZE,
        lr: float = AE_LR,
    ) -> Dict[str, float]:
        """Train Autoencoder on normal session count vectors."""
        all_events = [e for seq in normal_sessions_events for e in seq]
        self.build_vocab(all_events)

        vectors = [self.events_to_vector(seq) for seq in normal_sessions_events]
        if not vectors:
            vectors = [np.zeros(self.input_dim, dtype=np.float32)]

        x_train = torch.tensor(np.array(vectors), dtype=torch.float32)
        dataset = TensorDataset(x_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = AutoencoderNet(
            input_dim=self.input_dim, hidden_dims=self.hidden_dims
        ).to(self.device)
        self.model.train()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        epoch_losses = []
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            steps = 0
            for (batch_x,) in dataloader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
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
            "vocab_dim": self.input_dim,
            "total_sessions": len(vectors),
        }

    def score_session(
        self,
        events: List[str]
    ) -> Tuple[float, List[Dict[str, object]]]:
        """
        Compute reconstruction error (MSE) and per-template residuals.
        Returns:
            (reconstruction_error, top_deviating_templates)
        """
        if self.model is None or not events:
            return 0.0, []

        self.model.eval()
        raw_vec = self.events_to_vector(events)
        tensor_x = torch.from_numpy(raw_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            recon = self.model(tensor_x)
            residuals = torch.abs(recon - tensor_x).squeeze(0).numpy()
            mse = float(torch.mean((recon - tensor_x) ** 2).item())

        # Identify top templates with largest reconstruction discrepancies
        top_indices = np.argsort(residuals)[::-1][:3]
        deviations = []
        for idx in top_indices:
            if idx < len(self.event_vocab) and residuals[idx] > 0.05:
                eid = self.event_vocab[idx]
                actual_count = float(np.expm1(raw_vec[idx]))
                expected_count = float(np.expm1(recon.squeeze(0).numpy()[idx]))
                deviations.append({
                    "event_id": eid,
                    "actual_count": round(actual_count, 1),
                    "expected_count": round(expected_count, 1),
                    "residual": round(float(residuals[idx]), 3),
                })

        return mse, deviations

    def save(self, path: Optional[Path] = None) -> None:
        """Save autoencoder weights and vocabulary."""
        target = path or self.model_path
        if not target or self.model is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "vocab": self.event_vocab,
                "hidden_dims": self.hidden_dims,
            },
            target,
        )

    def load(self, path: Optional[Path] = None) -> None:
        """Load autoencoder weights and vocabulary."""
        target = path or self.model_path
        if not target or not target.exists():
            return
        checkpoint = torch.load(target, map_location=self.device)
        self.event_vocab = checkpoint["vocab"]
        self.event2idx = {e: i for i, e in enumerate(self.event_vocab)}
        self.hidden_dims = checkpoint.get("hidden_dims", self.hidden_dims)

        self.model = AutoencoderNet(
            input_dim=self.input_dim, hidden_dims=self.hidden_dims
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
