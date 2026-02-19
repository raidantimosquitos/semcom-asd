import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    Vector quantizer with codebook updated by exponential moving average (no codebook gradient).
    Same interface as VectorQuantizer: (quantized, loss, perplexity). Loss is commitment only.
    Expects encoder output in (-5, 5) (encoder uses tanh scaling); codebook is clamped to [-5, 5].
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float = 0.99, epsilon: float = 1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        # EMA: cluster size (K,) and sum of vectors per cluster (K, D); codebook = sum / (size + eps)
        self.register_buffer("_ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer(
            "_ema_embedding_sum",
            self._embedding.weight.data.clone(),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Either (B, T, D) or (B, C, H, W) with D or C == embedding_dim.
        Returns:
            quantized: Same shape as inputs.
            loss: scalar VQ loss (commitment only).
            perplexity: scalar.
        """
        if inputs.dim() == 4:
            B, C, H, W = inputs.shape
            assert C == self._embedding_dim
            inputs_flat = inputs.permute(0, 2, 3, 1).reshape(B, H * W, C)
            quantized_flat, loss, perplexity = self._forward_3d(inputs_flat)
            quantized = quantized_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return quantized, loss, perplexity
        if inputs.dim() == 3:
            return self._forward_3d(inputs)
        raise ValueError(
            f"VectorQuantizerEMA expects 3D (B, T, D) or 4D (B, C, H, W), got dim={inputs.dim()}"
        )

    def _forward_3d(self, inputs: torch.Tensor):
        """Core logic: inputs (B, T, D), D == embedding_dim."""
        batch_size, time_size, latent_size = inputs.shape
        assert latent_size == self._embedding_dim

        flat_z = inputs.reshape(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self._embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(
            batch_size, time_size, latent_size
        )

        if self.training:
            # EMA update: cluster counts and sum of vectors per cluster
            n_batch = encodings.sum(dim=0)  # (K,)
            embed_sum_batch = torch.matmul(encodings.t(), flat_z)  # (K, D)
            one_minus_decay = 1.0 - self._decay
            self._ema_cluster_size.mul_(self._decay).add_(n_batch, alpha=one_minus_decay)
            self._ema_embedding_sum.mul_(self._decay).add_(embed_sum_batch, alpha=one_minus_decay)
            # Codebook = Laplace-smoothed mean; use min cluster size 1.0 to avoid explosion when few codes used
            n = self._ema_cluster_size.unsqueeze(1).clamp(min=1.0) + self._epsilon
            self._embedding.weight.data.copy_(self._ema_embedding_sum / n)
            # Clamp codebook to prevent runaway growth and NaN (common when perplexity collapses)
            self._embedding.weight.data.clamp_(-5.0, 5.0)

        # Commitment loss: pull encoder output (inputs) toward the chosen codebook vector.
        # Must be MSE(inputs, quantized.detach()) so gradients flow to inputs; MSE(quantized_st, inputs)
        # would have d(quantized_st)/d(inputs)=1 and give zero gradient to the encoder.
        quantized_st = inputs + (quantized - inputs).detach()
        loss = self._commitment_cost * F.mse_loss(inputs, quantized.detach())

        avg_probs = torch.mean(encodings, dim=0)
        # entropy_raw = Σ p·log(p) (negative); standard entropy H = -Σ p·log(p) = -entropy_raw; perplexity = exp(H)
        entropy_raw = torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(-entropy_raw).clamp(max=float(self._num_embeddings))
        if torch.isnan(perplexity) or torch.isinf(perplexity):
            perplexity = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)

        return quantized_st, loss, perplexity

    @torch.no_grad()
    def encode_to_indices(self, inputs: torch.Tensor, return_distances: bool = False):
        """
        Map latent to code indices (and optionally min distances) for density modeling / inference.
        No EMA update; use in eval mode.
        Args:
            inputs: (B, C, H, W) or (B, T, D), C or D == embedding_dim.
            return_distances: If True, also return minimum distance per position.
        Returns:
            indices: (B, H, W) for 4D or (B, T) for 3D, dtype long.
            distances (optional): same shape as indices, dtype float.
        """
        if inputs.dim() == 4:
            B, C, H, W = inputs.shape
            assert C == self._embedding_dim
            inputs_flat = inputs.permute(0, 2, 3, 1).reshape(B, H * W, C)
            indices_flat, dist_flat = self._indices_3d(inputs_flat, return_distances=return_distances)
            indices = indices_flat.view(B, H, W)
            distances = (dist_flat.view(B, H, W) if dist_flat is not None else None) if return_distances else None
            return (indices, distances) if return_distances else indices
        if inputs.dim() == 3:
            indices_flat, dist_flat = self._indices_3d(inputs, return_distances=return_distances)
            if return_distances:
                return indices_flat, dist_flat
            return indices_flat
        raise ValueError(f"encode_to_indices expects 3D or 4D, got dim={inputs.dim()}")

    def _indices_3d(self, inputs: torch.Tensor, return_distances: bool = False):
        """inputs (B, T, D). Returns (indices (B, T), distances (B, T) or None)."""
        flat_z = inputs.reshape(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self._embedding.weight.t())
        )
        min_dist, encoding_indices = distances.min(dim=1)
        encoding_indices = encoding_indices.view(inputs.shape[0], inputs.shape[1])
        min_dist = min_dist.view(inputs.shape[0], inputs.shape[1])
        return encoding_indices, min_dist if return_distances else None