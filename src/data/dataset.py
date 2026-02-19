"""
DCASE 2020 Task 2 dev dataset: precomputed 64-bin mel spectrograms with patching
and labels for VQ-VAE reconstruction, machine ID, and transformation classification.

Each batch has B samples (spectrograms); each sample yields N_PATCHES patches. So
patches are (B, N_PATCHES, 64, 64): the 16 patches of index 0..15 belong to sample 0,
16..31 to sample 1, etc. (after flattening for the encoder).

- Machine ID: one label per spectrogram (batch["machine_id"] shape (B,)). Aggregate
  the N_PATCHES latents per sample (e.g. mean) then one prediction per sample; CE loss.
- Transformation: one label per spectrogram (batch["transformation_id"] shape (B,)); same
  transform applied to all N_PATCHES of that sample. You can either (a) aggregate and
  one prediction per sample, or (b) predict per patch and take CE with labels repeated
  for the 16 patches: labels[b*16:(b+1)*16] = transformation_id[b].
"""

from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Any, Literal, cast

# Norm stats: dict[machine_type_id, {"mean": (64,), "std": (64,)}]; applied to spectrograms as-loaded
NormStatsT = dict[int, dict[str, Any]]

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.transformations import apply_transformation

MACHINE_TYPES = ("ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve")
NUM_TRANSFORMS = 6

PATCH_LEN = 64
STRIDE = 16
N_PATCHES = 16  # Fixed patches per sample; if file has more, randomly sample N_PATCHES


def _parse_machine_id(filename: str) -> int:
    """Parse id_XX to 0-based index: id_00 -> 0, id_01 -> 1, id_02 -> 2, etc.
    Uses the number as the 0-based index (no minus one) so id_00 and id_01 map to distinct classes.
    """
    match = re.search(r"id_(\d+)", filename, re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(1), 10)


def _build_file_list(
    root: Path,
    split: str,
    machine_type: str | None,
) -> list[tuple[Path, int, int]]:
    """List (filepath, machine_type_id, machine_id). machine_id is contiguous 0..K-1 per type (K = number of distinct machines for that type)."""
    root = Path(root)
    types_to_scan = [machine_type] if machine_type else list(MACHINE_TYPES)
    samples: list[tuple[Path, int, int]] = []
    for type_name in types_to_scan:
        if type_name not in MACHINE_TYPES:
            raise ValueError(f"machine_type must be one of {MACHINE_TYPES}, got {machine_type!r}")
        i = MACHINE_TYPES.index(type_name)
        split_dir = root / type_name / split
        if not split_dir.is_dir():
            continue
        type_samples: list[tuple[Path, int, int]] = []
        for p in sorted(split_dir.iterdir()):
            if p.suffix.lower() != ".npy":
                continue
            type_samples.append((p, i, _parse_machine_id(p.name)))
        if type_samples:
            unique_raw = sorted(set(raw for _, _, raw in type_samples))
            raw_to_idx = {r: k for k, r in enumerate(unique_raw)}
            for p, type_id, raw in type_samples:
                samples.append((p, type_id, raw_to_idx[raw]))
    return samples


def get_machine_id_summary(
    root: str | Path,
    split: str = "train",
) -> dict[str, dict[int, int]]:
    """
    Sanity check: for each machine_type, return a dict mapping machine_id -> file count.
    Keys are machine_type names; value is {machine_id: count}.
    """
    root = Path(root)
    out: dict[str, dict[int, int]] = {}
    for type_name in MACHINE_TYPES:
        samples = _build_file_list(root, split, type_name)
        counts: dict[int, int] = {}
        for _path, _type_id, machine_id in samples:
            counts[machine_id] = counts.get(machine_id, 0) + 1
        if counts:
            out[type_name] = dict(sorted(counts.items()))
    return out


def _n_patches(time_len: int, patch_len: int, stride: int) -> int:
    if time_len < patch_len:
        return 0
    return (time_len - patch_len) // stride + 1


def _extract_patches(spec: np.ndarray, patch_len: int, stride: int) -> np.ndarray:
    """Extract overlapping patches (n_mels, time) -> (n_patches, n_mels, patch_len)."""
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got {spec.shape}")
    n_mels, T = spec.shape
    if T < patch_len:
        spec = np.pad(spec, ((0, 0), (0, patch_len - T)), mode="edge")
        T = spec.shape[1]
    n = _n_patches(T, patch_len, stride)
    if n == 0:
        n = 1
        start_indices = [0]
    else:
        start_indices = [i * stride for i in range(n)]
    return np.stack(
        [spec[:, s : s + patch_len].copy() for s in start_indices],
        axis=0,
        dtype=np.float32,
    )


def collate_dcase_patches(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate to batch; all samples have N_PATCHES patches, so stack directly."""
    device = batch[0]["patches"].device
    out: dict[str, torch.Tensor] = {
        "patches": torch.stack([b["patches"] for b in batch], dim=0),
        "machine_type_id": torch.tensor([b["machine_type_id"] for b in batch], dtype=torch.long, device=device),
        "machine_id": torch.tensor([b["machine_id"] for b in batch], dtype=torch.long, device=device),
        "transformation_id": torch.tensor([b["transformation_id"] for b in batch], dtype=torch.long, device=device),
    }
    if "patches_original" in batch[0]:
        out["patches_original"] = torch.stack([b["patches_original"] for b in batch], dim=0)
    if "is_anomaly" in batch[0]:
        out["is_anomaly"] = torch.tensor([b["is_anomaly"] for b in batch], dtype=torch.long, device=device)
    return out


class DCASEMelSpectrogramDataset(Dataset):
    """
    Precomputed 64-bin mel spectrograms from DCASE 2020 Task 2 dev. Each sample
    returns exactly N_PATCHES patches (16 by default). Files with more patches
    (e.g. ToyCar with 18) randomly sample N_PATCHES; files with fewer repeat.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        machine_type: str | None = None,
        patch_len: int = PATCH_LEN,
        stride: int = STRIDE,
        n_patches: int = N_PATCHES,
        norm_stats: NormStatsT | None = None,
        seed: int | None = None,
        mmap_mode: Literal["r", "r+", "w+", "c"] = "r",
        return_original_patches: bool = False,
        normal_machine_ids: tuple[int, ...] | None = None,
    ):
        """
        Args:
            root: Root directory (e.g. .../64mel-spectr-dcase2020-task2-dev-dataset).
            split: "train" or "test".
            machine_type: If set, only load this type (e.g. "fan", "valve"). If None, load all types.
            patch_len: Patch time length (default 64).
            stride: Stride between patches (default 16).
            n_patches: Patches per sample (default 16); sample or repeat to match.
            norm_stats: Per-machine-type mean/std (64,) to normalize; from load_norm_stats().
            seed: RNG seed for reproducibility.
            mmap_mode: np.load mmap mode.
            return_original_patches: If True, also return "patches_original" (before transform) for recon_target="original".
            normal_machine_ids: If set, each sample gets "is_anomaly": 0 if machine_id in this set else 1 (for AUC/pAUC evaluation).
        """
        root = Path(root)
        self.root = root
        self.split = split
        self.machine_type = machine_type
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = n_patches
        self.norm_stats = norm_stats
        self.seed = seed
        self.mmap_mode = mmap_mode
        self.return_original_patches = return_original_patches
        self.normal_machine_ids = set(normal_machine_ids) if normal_machine_ids is not None else None
        self.samples = _build_file_list(root, split, machine_type)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_spectrogram(self, path: Path) -> np.ndarray:
        data = np.load(path, mmap_mode=cast(Literal["r", "r+", "w+", "c"] | None, self.mmap_mode))
        spec = np.asarray(data, dtype=np.float32) if hasattr(data, "shape") else np.array(data, dtype=np.float32)
        if spec.ndim != 2:
            raise ValueError(f"Expected 2D array, got {spec.shape} at {path}")
        if spec.shape[0] != 64 and spec.shape[1] == 64:
            spec = spec.T
        if spec.shape[0] != 64:
            raise ValueError(f"Expected 64 mel bins, got {spec.shape[0]} at {path}")
        return spec

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed + idx) if self.seed is not None else random.Random()
        if self.seed is not None:
            np.random.seed(self.seed + idx)
        filepath, machine_type_id, machine_id = self.samples[idx]

        spec = self._load_spectrogram(filepath)
        if self.norm_stats is not None and machine_type_id in self.norm_stats:
            mean = self.norm_stats[machine_type_id]["mean"]  # (64,)
            std = self.norm_stats[machine_type_id]["std"]   # (64,)
            spec = (spec - mean[:, np.newaxis]) / (std[:, np.newaxis] + 1e-6)
        patches_np = _extract_patches(spec, self.patch_len, self.stride)
        k = patches_np.shape[0]
        if k > self.n_patches:
            chosen = rng.sample(range(k), self.n_patches)
            patches_np = patches_np[chosen]
        elif k < self.n_patches:
            extra = rng.choices(range(k), k=self.n_patches - k)
            patches_np = np.concatenate([patches_np, patches_np[extra]], axis=0)
        patches = torch.from_numpy(patches_np).float()

        transform_id = rng.randint(0, NUM_TRANSFORMS - 1)
        patches_transformed = apply_transformation(patches, transform_id, rng=rng)

        out: dict[str, Any] = {
            "patches": patches_transformed,
            "machine_type_id": machine_type_id,
            "machine_id": machine_id,
            "transformation_id": transform_id,
        }
        if self.normal_machine_ids is not None:
            out["is_anomaly"] = 0 if machine_id in self.normal_machine_ids else 1
        if self.return_original_patches:
            out["patches_original"] = patches
        return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data.dataset import DCASEMelSpectrogramDataset, collate_dcase_patches

    root = "/mnt/ssd/LaCie/64mel-spectr-dcase2020-task2-dev-dataset"
    # Use machine_type="fan" (or "valve", "pump", etc.) to train on one machine type only
    ds = DCASEMelSpectrogramDataset(root, split="train", machine_type="fan", seed=42)
    if len(ds) == 0:
        raise ValueError(f"No .npy files at {root} under <machine_type>/train")
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_dcase_patches)
    batch = next(iter(loader))
    print("patches", batch["patches"].shape)
    print("machine_type_id", batch["machine_type_id"].shape, "(all same when machine_type is set)")
    print("machine_id", batch["machine_id"].shape)
    print("transformation_id", batch["transformation_id"].shape)

    print("Exploring a batch:")
    for i in range(batch["patches"].shape[0]):
        print(f"sample {i}")
        print("patches", batch["patches"][i].shape)
        print("machine_type_id", batch["machine_type_id"][i])
        print("machine_id", batch["machine_id"][i])
        print("transformation_id", batch["transformation_id"][i])
        print()
