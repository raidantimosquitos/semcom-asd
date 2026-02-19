import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticHeads(nn.Module):
    """
    Classification heads on (aggregated or per-patch) latent. At training time we predict
    machine_id and the augmentation (transform) applied; at inference, inputs are usually
    un-augmented so transform is effectively identity. See ARCHITECTURE.md.
    """
    def __init__(self, latent_dim: int, num_machine_ids: int = 4, num_transforms: int = 6):
        super(SemanticHeads, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.machine_id_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_machine_ids),
        )

        self.transform_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_transforms),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pool(x).flatten(1)
        machine_logits = self.machine_id_head(h)
        transform_logits = self.transform_head(h)

        return machine_logits, transform_logits