import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution Block
class DSConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DSConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.use_residual else None
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.act(x)
        if identity is not None:
            x = x + identity
        return x

class LightWeightEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(LightWeightEncoder, self).__init__()

        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )

        # Downsample 64 -> 32
        self.block1 = DSConvBlock(32, 64, stride=2)

        # Downsample 32 -> 16
        self.block2 = DSConvBlock(64, 96, stride=2)

        # Downsample 16 -> 8
        self.block3 = DSConvBlock(96, out_channels, stride=2)

        # Refinement block
        self.refinement = DSConvBlock(out_channels, out_channels, stride=1)

        # Latent projection to the desired dimension
        self.latent_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # Bound latent to (-latent_scale, latent_scale) to match quantizer codebook range and prevent VQ loss explosion
        self.latent_scale = 5.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) – batch of single-channel spectrogram patches (1×64×64 each).
        Returns:
            latent: (B, out_channels, h, w) – encoded latent maps in (-latent_scale, latent_scale).
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.refinement(x)
        x = self.latent_proj(x)
        return self.latent_scale * torch.tanh(x)


def flatten_patches_for_encoder(
    patches: torch.Tensor,
    patch_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reshape batched patches (B, N, H, W) for encoder expecting (batch, 1, H, W).
    Each patch is a separate batch element; patches 0..N-1 are from sample 0, etc.

    Returns:
        flat_patches: (B*N, 1, H, W).  flat_mask: (B*N,) or None.
    """
    B, N, H, W = patches.shape
    flat = patches.reshape(B * N, 1, H, W)
    flat_mask = patch_mask.reshape(B * N) if patch_mask is not None else None
    return flat, flat_mask


def aggregate_latents_per_sample(
    latent: torch.Tensor, n_patches: int
) -> torch.Tensor:
    """
    Aggregate patch latents so there is one vector per spectrogram (for machine_id head).

    Batch layout: first n_patches latents are from sample 0, next n_patches from sample 1, etc.
    So latent has shape (B * n_patches, C, h, w). Returns (B, C, h, w) by mean over patches.

    Use this for machine_id classification: one label per spectrogram, so predict once per sample.
    """
    B_n, C, h, w = latent.shape
    B = B_n // n_patches
    latent = latent.reshape(B, n_patches, C, h, w)
    return latent.mean(dim=1)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data.dataset import DCASEMelSpectrogramDataset, collate_dcase_patches

    ds = DCASEMelSpectrogramDataset(
        "/mnt/ssd/LaCie/64mel-spectr-dcase2020-task2-dev-dataset",
        split="train",
        seed=42,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_dcase_patches)
    batch = next(iter(loader))
    patches = batch["patches"]  # (B, N_PATCHES, 64, 64)
    flat_patches, _ = flatten_patches_for_encoder(patches, None)
    print(flat_patches.shape)
    encoder = LightWeightEncoder(in_channels=1, out_channels=64)
    latent = encoder(flat_patches)
    print(latent.shape)