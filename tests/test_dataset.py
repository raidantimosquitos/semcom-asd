"""
Unit tests for DCASEMelSpectrogramDataset: patch count, shape, and collate.
"""

import numpy as np
import pytest
import torch

from src.data.dataset import (
    DCASEMelSpectrogramDataset,
    N_PATCHES,
    NUM_TRANSFORMS,
    PATCH_LEN,
    STRIDE,
    _n_patches,
    collate_dcase_patches,
)
from src.utils.transformations import apply_transformation


def test_n_patches():
    assert _n_patches(64, 64, 8) == 1
    assert _n_patches(128, 64, 8) == 9
    assert _n_patches(64 + 16 * 10, 64, 16) == 11
    assert _n_patches(63, 64, 8) == 0


def test_sample_has_fixed_n_patches(tmp_path):
    (tmp_path / "ToyCar" / "train").mkdir(parents=True)
    spec = np.random.randn(64, 200).astype(np.float32)
    np.save(tmp_path / "ToyCar" / "train" / "normal_id_01_00000000.npy", spec)
    np.save(tmp_path / "ToyCar" / "train" / "normal_id_02_00000001.npy", spec)

    ds = DCASEMelSpectrogramDataset(tmp_path, split="train", seed=42)
    assert len(ds) == 2
    out = ds[0]
    assert out["patches"].shape == (N_PATCHES, 64, 64)
    assert out["machine_type_id"] == 0
    assert out["transformation_id"] in range(NUM_TRANSFORMS)


def test_more_than_n_patches_sampled(tmp_path):
    """File with 21 patches yields exactly N_PATCHES by random sampling."""
    (tmp_path / "fan" / "train").mkdir(parents=True)
    T = 64 + 16 * 20
    spec = np.random.randn(64, T).astype(np.float32)
    np.save(tmp_path / "fan" / "train" / "normal_id_01_00000000.npy", spec)

    ds = DCASEMelSpectrogramDataset(tmp_path, split="train", seed=123)
    out = ds[0]
    assert out["patches"].shape == (N_PATCHES, 64, 64)


def test_collate_shapes(tmp_path):
    (tmp_path / "pump" / "train").mkdir(parents=True)
    spec = np.random.randn(64, 300).astype(np.float32)
    np.save(tmp_path / "pump" / "train" / "normal_id_01_00000000.npy", spec)
    np.save(tmp_path / "pump" / "train" / "normal_id_01_00000001.npy", spec)

    ds = DCASEMelSpectrogramDataset(tmp_path, split="train", seed=1)
    batch = collate_dcase_patches([ds[0], ds[1]])
    assert batch["patches"].shape == (2, N_PATCHES, 64, 64)
    assert batch["machine_type_id"].shape == (2,)
    assert batch["machine_id"].shape == (2,)
    assert batch["transformation_id"].shape == (2,)


def test_load_finite(tmp_path):
    """Spectrograms are loaded as-is (precomputed log amplitude); output should be finite."""
    (tmp_path / "valve" / "train").mkdir(parents=True)
    spec = np.random.randn(64, 150).astype(np.float32)
    np.save(tmp_path / "valve" / "train" / "normal_id_01_00000000.npy", spec)
    ds = DCASEMelSpectrogramDataset(tmp_path, split="train", seed=0)
    out = ds[0]
    assert torch.isfinite(out["patches"]).all()
