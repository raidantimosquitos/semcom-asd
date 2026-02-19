import torch
import torch.nn as nn


class LightWeightDecoder(nn.Module):
    """
    Shallow decoder that reconstructs a single-channel spectrogram patch (1×H×W)
    from VQ-VAE codewords (latent space). Expects latent shape (B, latent_dim, h, w)
    and outputs (B, 1, H, W) with H, W determined by num_upsamples (default 64×64).
    """

    def __init__(
        self,
        latent_dim: int,
        out_height: int = 64,
        out_width: int = 64,
        hidden_dims: tuple[int, ...] = (96, 64, 32),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_height = out_height
        self.out_width = out_width

        # Infer upsample factors from target size; assume latent is 8×8 (encoder 64→8)
        # 8 -> 16 -> 32 -> 64 => 3 upsamples
        self.num_upsamples = 3
        scale = 2**self.num_upsamples  # 8
        assert out_height % scale == 0 and out_width % scale == 0, (
            f"out_height/out_width must be divisible by {scale} (e.g. 64)"
        )
        self.latent_h, self.latent_w = out_height // scale, out_width // scale

        dims = [latent_dim] + list(hidden_dims)
        blocks = []
        for i in range(self.num_upsamples):
            in_c = dims[min(i, len(dims) - 1)]
            out_c = dims[min(i + 1, len(dims) - 1)]
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.SiLU(),
                )
            )
        self.upsample_blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(dims[min(self.num_upsamples, len(dims) - 1)], 1, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, h, w) – quantized latent (e.g. from VQ-VAE), typically (B, C, 8, 8).
        Returns:
            (B, 1, out_height, out_width) – reconstructed spectrogram patch.
        """
        assert z.shape[-2:] == (self.latent_h, self.latent_w), (
            f"Decoder expects latent spatial size ({self.latent_h}, {self.latent_w}) "
            f"(from out_height//{2**self.num_upsamples}, out_width//{2**self.num_upsamples}); got {z.shape[-2:]}"
        )
        x = z
        for block in self.upsample_blocks:
            x = block(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # Shape check: latent (B, 64, 8, 8) -> recon (B, 1, 64, 64)
    B, latent_dim, h, w = 4, 64, 8, 8
    decoder = LightWeightDecoder(latent_dim=latent_dim, out_height=64, out_width=64)
    z = torch.randn(B, latent_dim, h, w)
    recon = decoder(z)
    print(f"z: {z.shape} -> recon: {recon.shape}")
    assert recon.shape == (B, 1, 64, 64)
