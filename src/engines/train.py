"""
Trainer engine for VQ-VAE + semantic heads.

Semantic heads are applied on the continuous (pre-quantization) latent to enforce
task-oriented representations. At inference only encoder, decoder, and a density
model on codewords are used for NLL-based normal/abnormal detection.

See TRAINING_STRATEGIES.md for strategies to avoid codebook collapse and
exploding gradients (e.g. VQ-only warmup, staged training, gradient clipping).
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader, Subset

from src.data.dataset import (
    N_PATCHES,
    NUM_TRANSFORMS,
    PATCH_LEN,
    STRIDE,
    DCASEMelSpectrogramDataset,
    collate_dcase_patches,
)
from src.utils.norm_stats import load_norm_stats
from src.models.decoder import LightWeightDecoder
from src.models.encoder import (
    LightWeightEncoder,
    aggregate_latents_per_sample,
    flatten_patches_for_encoder,
)
from src.models.quantizer import VectorQuantizerEMA
from src.models.semantic_heads import SemanticHeads


@dataclasses.dataclass
class TrainConfig:
    """Training and loss-weight configuration."""

    # Loss weights (tunable)
    lambda_recon: float = 1.0
    lambda_vq: float = 1.0
    lambda_machine_id: float = 0.5
    lambda_transform: float = 0.5

    # Model dimensions (must match dataset / heads)
    latent_dim: int = 64
    num_embeddings: int = 128
    commitment_cost: float = 0.25
    vq_decay: float = 0.99
    num_machine_ids: int = 4
    num_transforms: int = NUM_TRANSFORMS
    n_patches: int = N_PATCHES
    patch_len: int = PATCH_LEN
    stride: int = STRIDE

    # Training
    lr: float = 1e-3
    epochs: int = 50
    grad_clip: float | None = 1.0
    gradient_accumulation_steps: int = 1  # effective batch = batch_size * this
    use_amp: bool = False  # mixed precision; use True for low-memory GPUs
    use_gradient_checkpointing: bool = False  # trade compute for ~40% less activation memory
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    data_root: str | Path = ""
    machine_type: str | None = None
    subset_fraction: float | None = None  # e.g. 0.2 = use 20% of each machine_id (for quick sanity runs)
    norm_stats_path: str | Path | None = None  # path to .pt from compute_norm_stats; apply per-machine-type normalization
    recon_target: str = "transformed"  # "transformed" or "original" (see ARCHITECTURE.md)
    transform_head_per_patch: bool = False  # if True, predict transform per-patch (stronger gradient) instead of per-sample after mean-pool
    batch_size: int = 32
    num_workers: int = 4
    seed: int | None = 42

    # Staged training (TRAINING_STRATEGIES.md): 20% VQ-only, 60% full, 20% codebook frozen
    use_staged_training: bool = True
    phase1_frac: float = 0.2  # first fraction: VQ-VAE only (semantic lambdas 0)
    phase2_frac: float = 0.6  # middle: full loss, all trainable
    phase3_frac: float = 0.2   # last: codebook frozen, encoder/decoder/heads trainable

    # Commitment warmup: scale VQ loss by a factor that ramps from commitment_warmup_start to 1.0 over the first N epochs to avoid early explosion (0 = disabled)
    commitment_warmup_epochs: int = 5
    commitment_warmup_start: float = 0.25  # first epoch VQ term scaled by this

    # LR warmup: linear ramp from 0 to lr over the first N epochs to stabilize early training (0 = disabled)
    lr_warmup_epochs: int = 3

    # Validation and checkpointing
    val_fraction: float = 0.1
    checkpoint_dir: str | Path = "checkpoints"
    save_best_only: bool = True  # by total validation loss


def compute_combined_loss(
    recon_loss: torch.Tensor,
    vq_loss: torch.Tensor,
    ce_machine: torch.Tensor,
    ce_transform: torch.Tensor,
    lambda_recon: float,
    lambda_vq: float,
    lambda_machine_id: float,
    lambda_transform: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of losses and a dict of scalar components for logging."""
    total = (
        lambda_recon * recon_loss
        + lambda_vq * vq_loss
        + lambda_machine_id * ce_machine
        + lambda_transform * ce_transform
    )
    loss_dict = {
        "recon": recon_loss.item(),
        "vq": vq_loss.item(),
        "machine_id": ce_machine.item(),
        "transform": ce_transform.item(),
        "total": total.item(),
    }
    return total, loss_dict


def build_models(config: TrainConfig) -> dict[str, nn.Module]:
    """Build encoder, quantizer, decoder, semantic heads."""
    encoder = LightWeightEncoder(in_channels=1, out_channels=config.latent_dim)
    quantizer = VectorQuantizerEMA(
        num_embeddings=config.num_embeddings,
        embedding_dim=config.latent_dim,
        commitment_cost=config.commitment_cost,
        decay=config.vq_decay,
    )
    decoder = LightWeightDecoder(
        latent_dim=config.latent_dim,
        out_height=64,
        out_width=64,
    )
    semantic_heads = SemanticHeads(
        latent_dim=config.latent_dim,
        num_machine_ids=config.num_machine_ids,
        num_transforms=config.num_transforms,
    )
    return {
        "encoder": encoder,
        "quantizer": quantizer,
        "decoder": decoder,
        "semantic_heads": semantic_heads,
    }


def _commitment_warmup_factor(epoch: int, total_epochs: int, config: TrainConfig) -> float:
    """Scale for VQ loss: ramps from commitment_warmup_start to 1.0 over the first commitment_warmup_epochs."""
    if config.commitment_warmup_epochs <= 0 or total_epochs <= 0:
        return 1.0
    progress = min(1.0, (epoch + 1) / config.commitment_warmup_epochs)
    return config.commitment_warmup_start + (1.0 - config.commitment_warmup_start) * progress


def train_step(
    batch: dict[str, torch.Tensor],
    models: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    *,
    scaler: Any = None,  # GradScaler when use_amp
    accum_scale: float = 1.0,
    lambda_machine_id: float | None = None,
    lambda_transform: float | None = None,
    epoch: int = 0,
    total_epochs: int = 1,
) -> tuple[dict[str, float], float | None]:
    """
    One training step. Returns loss_dict (scalars) and perplexity (or None).
    When using gradient accumulation, call with accum_scale=1/n_steps and only
    step optimizer after n_steps. When using AMP, pass scaler from run_train.
    """
    encoder = models["encoder"]
    quantizer = models["quantizer"]
    decoder = models["decoder"]
    semantic_heads = models["semantic_heads"]
    device = next(encoder.parameters()).device
    use_amp = scaler is not None

    patches = batch["patches"].to(device, non_blocking=True)
    machine_id = batch["machine_id"].to(device, non_blocking=True)
    transformation_id = batch["transformation_id"].to(device, non_blocking=True)

    flat_patches, _ = flatten_patches_for_encoder(patches, None)
    # Encoder input is always transformed patches; target depends on recon_target (see ARCHITECTURE.md)
    if config.recon_target == "original" and "patches_original" in batch:
        target, _ = flatten_patches_for_encoder(
            batch["patches_original"].to(device, non_blocking=True), None
        )
    else:
        target = flat_patches
    # target: (B*N, 1, 64, 64)

    use_ckpt = config.use_gradient_checkpointing
    lam_mid = config.lambda_machine_id if lambda_machine_id is None else lambda_machine_id
    lam_tr = config.lambda_transform if lambda_transform is None else lambda_transform
    semantic_heads_active = (lam_mid > 0 or lam_tr > 0)
    # Fix 1: Always checkpoint encoder when semantic heads are active (highest memory impact)
    use_encoder_ckpt = use_ckpt or semantic_heads_active

    with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore[attr-defined]
        # Encode -> continuous latent (checkpointing saves activation memory)
        if use_encoder_ckpt:
            latent_continuous = cast(
                torch.Tensor,
                grad_checkpoint(encoder, flat_patches, use_reentrant=False),
            )
        else:
            latent_continuous = encoder(flat_patches)

        # Fix 2: Decoder path detached so recon_loss does not backprop to encoder; commitment + semantic heads keep full graph.
        # quantizer(latent_continuous) keeps commitment loss gradient to encoder; decoder(quantized.detach()) trains decoder only.
        quantized, vq_loss, perplexity = quantizer(latent_continuous)
        vq_warmup = _commitment_warmup_factor(epoch, total_epochs, config)
        vq_loss = vq_loss * vq_warmup

        # Decode from detached quantized: no decoder→encoder grad; saves memory and trains decoder only for recon
        if use_ckpt:
            recon = cast(
                torch.Tensor,
                grad_checkpoint(decoder, quantized.detach(), use_reentrant=False),
            )
        else:
            recon = decoder(quantized.detach())
        recon_loss = F.mse_loss(recon, target)

        # Semantic heads: full graph encoder -> latent_continuous -> aggregate -> heads (gradients to encoder)
        latent_agg = aggregate_latents_per_sample(latent_continuous, config.n_patches)
        machine_logits, _ = semantic_heads(latent_agg)
        ce_machine = F.cross_entropy(machine_logits, machine_id, label_smoothing=0.1)
        # Transform head: per-sample (mean-pool first) or per-patch (stronger gradient signal)
        if config.transform_head_per_patch:
            _, transform_logits = semantic_heads(latent_continuous)
            transformation_id_expanded = transformation_id.repeat_interleave(config.n_patches, dim=0)
            ce_transform = F.cross_entropy(transform_logits, transformation_id_expanded, label_smoothing=0.1)
        else:
            _, transform_logits = semantic_heads(latent_agg)
            ce_transform = F.cross_entropy(transform_logits, transformation_id, label_smoothing=0.1)

        total, loss_dict = compute_combined_loss(
            recon_loss,
            vq_loss,
            ce_machine,
            ce_transform,
            config.lambda_recon,
            config.lambda_vq,
            lam_mid,
            lam_tr,
        )
        total = total * accum_scale

    if not (torch.isnan(total) or torch.isinf(total)):
        if scaler is not None:
            scaler.scale(total).backward()
        else:
            total.backward()

    pp_val = None
    if perplexity is not None and perplexity.numel() == 1:
        p = perplexity.item()
        if p == p and abs(p) != float("inf"):
            pp_val = p
    return loss_dict, pp_val


@torch.no_grad()
def validate(
    val_loader: DataLoader[dict[str, torch.Tensor]],
    models: dict[str, nn.Module],
    config: TrainConfig,
) -> tuple[dict[str, float], float]:
    """
    Run validation: same losses as training (recon, vq, machine_id, transform).
    Returns loss_dict and total validation loss (sum of the four components, unweighted
    for comparability) for checkpointing.
    """
    encoder = models["encoder"]
    quantizer = models["quantizer"]
    decoder = models["decoder"]
    semantic_heads = models["semantic_heads"]
    device = next(encoder.parameters()).device

    encoder.eval()
    quantizer.eval()
    decoder.eval()
    semantic_heads.eval()

    running = {"recon": 0.0, "vq": 0.0, "machine_id": 0.0, "transform": 0.0}
    n_batches = 0

    for batch in val_loader:
        patches = batch["patches"].to(device)
        machine_id = batch["machine_id"].to(device)
        transformation_id = batch["transformation_id"].to(device)

        flat_patches, _ = flatten_patches_for_encoder(patches, None)
        if config.recon_target == "original" and "patches_original" in batch:
            target, _ = flatten_patches_for_encoder(batch["patches_original"].to(device), None)
        else:
            target = flat_patches

        # Use AMP for validation forward when enabled and on CUDA (no GradScaler; no backward)
        use_amp_enabled = config.use_amp and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp_enabled):  # type: ignore[attr-defined]
            latent_continuous = encoder(flat_patches)
            quantized, vq_loss, _ = quantizer(latent_continuous)
            recon = decoder(quantized)
            recon_loss = F.mse_loss(recon, target)

            latent_agg = aggregate_latents_per_sample(latent_continuous, config.n_patches)
            machine_logits, _ = semantic_heads(latent_agg)
            ce_machine = F.cross_entropy(machine_logits, machine_id, label_smoothing=0.1)
            if config.transform_head_per_patch:
                _, transform_logits = semantic_heads(latent_continuous)
                transformation_id_expanded = transformation_id.repeat_interleave(config.n_patches, dim=0)
                ce_transform = F.cross_entropy(transform_logits, transformation_id_expanded, label_smoothing=0.1)
            else:
                _, transform_logits = semantic_heads(latent_agg)
                ce_transform = F.cross_entropy(transform_logits, transformation_id, label_smoothing=0.1)

        running["recon"] += recon_loss.item()
        running["vq"] += vq_loss.item()
        running["machine_id"] += ce_machine.item()
        running["transform"] += ce_transform.item()
        n_batches += 1

    for k in running:
        running[k] /= max(n_batches, 1)

    total_val_loss = (
        running["recon"]
        + running["vq"]
        + running["machine_id"]
        + running["transform"]
    )
    running["total"] = total_val_loss

    encoder.train()
    quantizer.train()
    decoder.train()
    semantic_heads.train()

    return running, total_val_loss


def save_checkpoint(
    path: Path,
    models: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: TrainConfig,
) -> None:
    """Save full checkpoint for resuming and deployment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "encoder": models["encoder"].state_dict(),
        "quantizer": models["quantizer"].state_dict(),
        "decoder": models["decoder"].state_dict(),
        "semantic_heads": models["semantic_heads"].state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": dataclasses.asdict(config),
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    models: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    """Load checkpoint; returns (epoch, best_val_loss). Optionally load optimizer."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    models["encoder"].load_state_dict(state["encoder"])
    models["quantizer"].load_state_dict(state["quantizer"])
    models["decoder"].load_state_dict(state["decoder"])
    models["semantic_heads"].load_state_dict(state["semantic_heads"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("epoch", 0), state.get("best_val_loss", float("inf"))


def _staged_phase_and_lambdas(
    epoch: int,
    total_epochs: int,
    config: TrainConfig,
) -> tuple[int, float, float]:
    """
    Return (phase, effective_lambda_machine_id, effective_lambda_transform).
    Phase 1: VQ only (semantic weights 0). Phase 2: full. Phase 3: codebook frozen.
    """
    if not config.use_staged_training or total_epochs <= 0:
        return (2, config.lambda_machine_id, config.lambda_transform)
    p1_end = int(config.phase1_frac * total_epochs)
    p2_end = int((config.phase1_frac + config.phase2_frac) * total_epochs)
    if epoch < p1_end:
        return (1, 0.0, 0.0)
    if epoch < p2_end:
        return (2, config.lambda_machine_id, config.lambda_transform)
    return (3, config.lambda_machine_id, config.lambda_transform)


def stratified_subset_indices(
    dataset: DCASEMelSpectrogramDataset,
    fraction: float,
    seed: int | None = None,
) -> list[int]:
    """
    Return dataset indices that form a stratified subset: approximately
    `fraction` of each machine_id. Useful for quick sanity runs on a small subset.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1), got {fraction}")
    rng = random.Random(seed)
    # Group indices by machine_id (samples[i] = (path, machine_type_id, machine_id))
    by_mid: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        mid = dataset.samples[idx][2]
        by_mid.setdefault(mid, []).append(idx)
    out: list[int] = []
    for mid, indices in by_mid.items():
        n = len(indices)
        k = max(1, int(round(n * fraction)))  # at least 1 per machine_id
        k = min(k, n)
        out.extend(rng.sample(indices, k=k))
    rng.shuffle(out)
    return out


def run_train(config: TrainConfig) -> dict[str, Any]:
    """
    Full training loop: train for config.epochs with validation and checkpointing.
    Best checkpoint is saved by total validation loss (recon + vq + machine_id + transform).
    """
    device = torch.device(config.device)
    data_root = Path(config.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    norm_stats = None
    if config.norm_stats_path and Path(config.norm_stats_path).is_file():
        norm_stats = load_norm_stats(config.norm_stats_path)
        print(f"Loaded norm stats from {config.norm_stats_path} for {len(norm_stats)} machine type(s)")

    if config.recon_target == "original":
        print("Reconstruction target: original patches (encoder input remains transformed)")

    # Data (patch_len, stride, n_patches control patches per spectrogram; stride 32 → fewer patches, less memory)
    train_ds = DCASEMelSpectrogramDataset(
        data_root,
        split="train",
        machine_type=config.machine_type,
        patch_len=config.patch_len,
        stride=config.stride,
        n_patches=config.n_patches,
        seed=config.seed,
        norm_stats=norm_stats,
        return_original_patches=(config.recon_target == "original"),
    )
    # machine_id is already contiguous 0..K-1 in the dataset (per machine_type). Size the head to K.
    if train_ds.samples:
        config.num_machine_ids = max(s[2] for s in train_ds.samples) + 1
        print(f"Machine ID classes: {config.num_machine_ids} (contiguous 0..{config.num_machine_ids - 1} from dataset)")

    # Optionally use a stratified subset (e.g. 0.2 of each machine_id) for quick runs
    if config.subset_fraction is not None:
        subset_idx = stratified_subset_indices(
            train_ds, config.subset_fraction, seed=config.seed
        )
        train_ds = Subset(train_ds, subset_idx)
        print(f"Using stratified subset: {config.subset_fraction*100:.0f}% of each machine_id -> {len(train_ds)} samples")

    n_val = max(1, int(len(train_ds) * config.val_fraction))
    n_train = len(train_ds) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(config.seed or 0)
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_dcase_patches,
        pin_memory=(device.type == "cuda" and config.batch_size > 1),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_dcase_patches,
    )

    # Models and optimizer
    models = build_models(config)
    for m in models.values():
        m.to(device)
    all_params = []
    for m in models.values():
        all_params += list(m.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.lr)
    scaler = torch.amp.GradScaler("cuda") if (config.use_amp and device.type == "cuda") else None  # type: ignore[attr-defined]
    accum_steps = max(1, config.gradient_accumulation_steps)

    checkpoint_dir = Path(config.checkpoint_dir)
    best_val_loss = float("inf")
    start_epoch = 0

    for epoch in range(start_epoch, config.epochs):
        # LR warmup: scale lr linearly over the first lr_warmup_epochs
        if config.lr_warmup_epochs > 0:
            warmup = min(1.0, (epoch + 1) / config.lr_warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = config.lr * warmup

        # Staged training: phase and effective loss weights
        phase, eff_lam_mid, eff_lam_tr = _staged_phase_and_lambdas(epoch, config.epochs, config)
        if config.use_staged_training:
            if phase == 3:
                for p in models["quantizer"].parameters():
                    p.requires_grad = False
            else:
                for p in models["quantizer"].parameters():
                    p.requires_grad = True

        # Train
        for m in models.values():
            m.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: dict[str, list[float]] = {"recon": [], "vq": [], "machine_id": [], "transform": [], "total": []}
        perplexities: list[float] = []
        for step_idx, batch in enumerate(train_loader):
            loss_dict, pp = train_step(
                batch,
                models,
                optimizer,
                config,
                scaler=scaler,
                accum_scale=1.0 / accum_steps,
                lambda_machine_id=eff_lam_mid,
                lambda_transform=eff_lam_tr,
                epoch=epoch,
                total_epochs=config.epochs,
            )
            step_has_nan = any(
                v != v or v == float("inf")  # nan or inf
                for k, v in loss_dict.items()
                if isinstance(v, float)
            )
            if step_has_nan:
                optimizer.zero_grad(set_to_none=True)
                if step_idx == 0:
                    print(f"  [WARNING] NaN/Inf loss at epoch {epoch + 1}, step {step_idx + 1}; skipping step.")
                continue

            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k].append(v)
            if pp is not None:
                perplexities.append(pp)

            if (step_idx + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(all_params, config.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        # Handle leftover gradients when len(loader) not divisible by accum_steps
        if len(train_loader) % accum_steps != 0 and len(train_loader) > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(all_params, config.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        n_steps = max(len(epoch_losses["total"]), 1)
        train_recon = sum(epoch_losses["recon"]) / n_steps
        train_vq = sum(epoch_losses["vq"]) / n_steps
        train_machine_id = sum(epoch_losses["machine_id"]) / n_steps
        train_transform = sum(epoch_losses["transform"]) / n_steps
        train_total = sum(epoch_losses["total"]) / n_steps
        avg_pp = sum(perplexities) / max(len(perplexities), 1) if perplexities else 0.0

        # Validate
        val_loss_dict, total_val_loss = validate(val_loader, models, config)

        # Checkpoint by best total validation loss
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            if config.save_best_only:
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    models,
                    optimizer,
                    epoch + 1,
                    best_val_loss,
                    config,
                )

        # Log: detailed train and val metrics (Phase 1 only logs recon/vq — semantic losses are not trained)
        phase_tag = ""
        if config.use_staged_training:
            if phase == 1:
                phase_tag = " [Phase 1: VQ only]"
            elif phase == 2:
                phase_tag = " [Phase 2: full]"
            else:
                phase_tag = " [Phase 3: codebook frozen]"
        print(f"Epoch {epoch + 1}/{config.epochs}{phase_tag}")
        if phase == 1 and config.use_staged_training:
            val_vq_only_total = val_loss_dict["recon"] + val_loss_dict["vq"]
            print(
                f"  train  recon: {train_recon:.4f}  vq: {train_vq:.4f}  total: {train_total:.4f}  perplexity: {avg_pp:.2f}"
            )
            print(
                f"  val    recon: {val_loss_dict['recon']:.4f}  vq: {val_loss_dict['vq']:.4f}  total: {val_vq_only_total:.4f}"
            )
        else:
            print(
                f"  train  recon: {train_recon:.4f}  vq: {train_vq:.4f}  machine_id: {train_machine_id:.4f}  transform: {train_transform:.4f}  total: {train_total:.4f}  perplexity: {avg_pp:.2f}"
            )
            print(
                f"  val    recon: {val_loss_dict['recon']:.4f}  vq: {val_loss_dict['vq']:.4f}  machine_id: {val_loss_dict['machine_id']:.4f}  transform: {val_loss_dict['transform']:.4f}  total: {total_val_loss:.4f}"
            )
        if device.type == "cuda" and config.use_amp:
            torch.cuda.empty_cache()

    return {
        "best_val_loss": best_val_loss,
        "checkpoint_dir": str(checkpoint_dir),
    }
