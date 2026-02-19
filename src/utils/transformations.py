import random
import torch


def _transform_identity(patches: torch.Tensor) -> torch.Tensor:
    return patches


def _transform_time_mask(
    patches: torch.Tensor,
    mask_len_min: int = 4,
    mask_len_max: int = 12,
    rng: random.Random | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = random.Random()
    out = patches.clone()
    _, _, T = patches.shape
    mask_len = rng.randint(mask_len_min, min(mask_len_max, T))
    start = rng.randint(0, max(0, T - mask_len))
    out[:, :, start : start + mask_len] = 0.0
    return out


def _transform_freq_mask(
    patches: torch.Tensor,
    mask_len_min: int = 4,
    mask_len_max: int = 12,
    rng: random.Random | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = random.Random()
    out = patches.clone()
    _, F, _ = patches.shape
    mask_len = rng.randint(mask_len_min, min(mask_len_max, F))
    start = rng.randint(0, max(0, F - mask_len))
    out[:, start : start + mask_len, :] = 0.0
    return out


def _transform_gaussian_noise(
    patches: torch.Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 0.05,
    rng: random.Random | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = random.Random()
    sigma = rng.uniform(sigma_min, sigma_max)
    noise = torch.randn_like(patches) * sigma
    return patches + noise


def _transform_magnitude_scale(
    patches: torch.Tensor,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
    rng: random.Random | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = random.Random()
    scale = rng.uniform(scale_min, scale_max)
    return patches * scale


def _transform_time_roll(
    patches: torch.Tensor,
    max_shift: int = 8,
    rng: random.Random | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = random.Random()
    shift = rng.randint(-max_shift, max_shift)
    if shift == 0:
        return patches
    return torch.roll(patches, shifts=shift, dims=2)


def apply_transformation(
    patches: torch.Tensor,
    transform_id: int,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Apply the transformation indicated by transform_id to patches (k, 64, 64)."""
    if rng is None:
        rng = random.Random()
    if transform_id == 0:
        return _transform_identity(patches)
    if transform_id == 1:
        return _transform_time_mask(patches, rng=rng)
    if transform_id == 2:
        return _transform_freq_mask(patches, rng=rng)
    if transform_id == 3:
        return _transform_gaussian_noise(patches, rng=rng)
    if transform_id == 4:
        return _transform_magnitude_scale(patches, rng=rng)
    if transform_id == 5:
        return _transform_time_roll(patches, rng=rng)
    raise ValueError(f"Unknown transform_id {transform_id}")
