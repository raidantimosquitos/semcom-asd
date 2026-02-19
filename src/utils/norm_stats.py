"""
Compute per–machine-type mean and std for spectrograms (64 mel bins).
Precomputed data is assumed already in log amplitude. Use once to generate
a stats file, then pass --norm_stats_path to training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from src.data.dataset import MACHINE_TYPES, _build_file_list


def _load_spec(path: Path) -> np.ndarray:
    """Load spectrogram as-is (precomputed data is already log amplitude)."""
    data = np.load(path, mmap_mode="r")
    spec = np.asarray(data, dtype=np.float32) if hasattr(data, "shape") else np.array(data, dtype=np.float32)
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D array, got {spec.shape} at {path}")
    if spec.shape[0] != 64 and spec.shape[1] == 64:
        spec = spec.T
    if spec.shape[0] != 64:
        raise ValueError(f"Expected 64 mel bins, got {spec.shape[0]} at {path}")
    return spec


def compute_norm_stats(
    data_root: str | Path,
    split: str = "train",
    machine_type: str | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Compute per-mel-bin mean and std per machine_type_id on the training set.
    Returns dict: machine_type_id -> {"mean": (64,), "std": (64,)}.
    """
    root = Path(data_root)
    samples = _build_file_list(root, split, machine_type)
    # Accumulate sum and sum_sq per mel bin per machine_type_id
    by_type: dict[int, dict[str, np.ndarray | float]] = {}
    for filepath, _type_id, _machine_id in samples:
        spec = _load_spec(filepath)  # (64, T)
        if _type_id not in by_type:
            by_type[_type_id] = {"sum": np.zeros(64, dtype=np.float64), "sum_sq": np.zeros(64, dtype=np.float64), "count": 0}
        by_type[_type_id]["sum"] += spec.sum(axis=1)
        by_type[_type_id]["sum_sq"] += (spec.astype(np.float64) ** 2).sum(axis=1)
        by_type[_type_id]["count"] += spec.shape[1]

    out: dict[int, dict[str, np.ndarray]] = {}
    for type_id, acc in by_type.items():
        n = acc["count"]
        mean = (acc["sum"] / n).astype(np.float32)
        var = np.maximum(acc["sum_sq"] / n - mean.astype(np.float64) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        std = np.maximum(std, 1e-6)
        out[type_id] = {"mean": mean, "std": std}
    return out


def save_norm_stats(stats: dict[int, dict[str, np.ndarray]], path: str | Path) -> None:
    """Save stats to a .pt file (torch.save)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to tensors for consistent load (dataset can load and use as numpy or tensor)
    out = {k: {"mean": torch.from_numpy(v["mean"]), "std": torch.from_numpy(v["std"])} for k, v in stats.items()}
    torch.save(out, path)


def load_norm_stats(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """Load stats from a .pt file. Returns mean/std as numpy (64,) for compatibility."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        k: {"mean": v["mean"].numpy(), "std": v["std"].numpy()}
        for k, v in data.items()
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute per-machine-type spectrogram norm stats (mean, std) for training.")
    p.add_argument("--data_root", type=str, required=True, help="Root path to DCASE mel-spectrogram dataset")
    p.add_argument("--split", type=str, default="train", help="Split to compute stats on")
    p.add_argument("--machine_type", type=str, default=None, help="Single machine type (e.g. fan) or all if unset")
    p.add_argument("--out", type=str, default="norm_stats.pt", help="Output .pt file path")
    args = p.parse_args()
    stats = compute_norm_stats(
        args.data_root,
        split=args.split,
        machine_type=args.machine_type or None,
    )
    save_norm_stats(stats, args.out)
    print(stats)
    print(f"Saved norm stats for {len(stats)} machine type(s) to {args.out}")
