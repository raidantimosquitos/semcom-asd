"""
Density models for the VQ-VAE latent (code indices) for NLL-based anomaly detection.
Fit on normal training data; at inference, NLL = -log p(sequence) is the anomaly score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch


class HistogramCodeDensity:
    """
    A1: Code usage histogram with Laplace smoothing.
    p(k) = (count(k) + smoothing) / (total + K * smoothing).
    Per-sample NLL = -sum_t log p(k_t). No sequence structure; i.i.d. baseline.
    """

    def __init__(self, num_codes: int, smoothing: float = 1.0):
        self.num_codes = num_codes
        self.smoothing = smoothing
        self._probs: np.ndarray | None = None  # (num_codes,) float

    def fit(self, indices: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Estimate p(k) from code indices (flattened or any shape).
        indices: integer array, values in [0, num_codes).
        """
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64).ravel()
        counts = np.bincount(indices, minlength=self.num_codes)
        counts = counts.astype(np.float64) + self.smoothing
        self._probs = counts / counts.sum()

    def score_nll(self, indices: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Return NLL = -sum_t log p(k_t) for the given code indices.
        Lower NLL = more likely under the model (normal); higher NLL = anomaly.
        """
        if self._probs is None:
            raise RuntimeError("HistogramCodeDensity not fitted; call fit() first.")
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64).ravel()
        probs = self._probs[np.clip(indices, 0, self.num_codes - 1)]
        nll = -np.log(probs + 1e-12).sum()
        return float(nll)

    def score_nll_per_position(self, indices: Union[np.ndarray, torch.Tensor]) -> float:
        """Average NLL per position (so score is comparable across different sequence lengths)."""
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64).ravel()
        n = indices.size
        if n == 0:
            return 0.0
        nll_total = self.score_nll(indices)
        return nll_total / n

    def save(self, path: Union[str, Path]) -> None:
        """Save probs and config to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "probs": torch.from_numpy(self._probs) if self._probs is not None else None,
            "num_codes": self.num_codes,
            "smoothing": self.smoothing,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HistogramCodeDensity":
        """Load from a .pt file."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(num_codes=state["num_codes"], smoothing=state["smoothing"])
        if state.get("probs") is not None:
            obj._probs = state["probs"].numpy()
        return obj
