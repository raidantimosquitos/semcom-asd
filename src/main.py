"""
Entry point for training the VQ-VAE + semantic heads pipeline.

Example:
  python -m src.main --data_root /path/to/64mel-spectr-dcase2020-task2-dev-dataset
"""

from __future__ import annotations

import argparse

import torch

from src.engines.train import TrainConfig, run_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VQ-VAE + semantic heads")
    p.add_argument("--data_root", type=str, required=True, help="Root path to DCASE mel-spectrogram dataset")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save best checkpoint")
    p.add_argument("--machine_type", type=str, default=None, help="Train on single machine type (e.g. fan, valve) or all if unset")
    p.add_argument("--patch_len", type=int, default=64, help="Patch time length (samples)")
    p.add_argument("--stride", type=int, default=16, help="Stride between patches (32 → fewer patches, less GPU memory)")
    p.add_argument("--n_patches", type=int, default=16, help="Patches per spectrogram; sample or repeat to match (e.g. 8 with stride 32)")
    p.add_argument("--subset_fraction", type=float, default=None, metavar="F", help="Use F fraction of each machine_id (e.g. 0.2 for quick sanity run)")
    p.add_argument("--norm_stats_path", type=str, default=None, help="Path to .pt from python -m src.utils.norm_stats (per-machine-type normalization)")
    p.add_argument("--recon_target", type=str, default="transformed", choices=("transformed", "original"),
        help="Reconstruction target: transformed (default) or original; encoder always sees transformed")
    p.add_argument("--transform_head_per_patch", action="store_true",
        help="Predict transform per-patch for stronger gradient; default is per-sample after mean-pool")
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--lambda_recon", type=float, default=1.0, help="Reconstruction loss weight")
    p.add_argument("--lambda_vq", type=float, default=1.0, help="VQ loss weight")
    p.add_argument("--lambda_machine_id", type=float, default=0.5, help="Machine ID CE loss weight")
    p.add_argument("--lambda_transform", type=float, default=0.5, help="Transformation CE loss weight")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients this many steps (effective batch = batch_size * this)")
    p.add_argument("--use_amp", action="store_true", help="Use mixed precision (AMP) to reduce GPU memory")
    p.add_argument("--use_gradient_checkpointing", action="store_true", help="Recompute activations in backward to save GPU memory (slower)")
    p.add_argument("--memory_saver", action="store_true", help="Enable AMP + gradient checkpointing (recommended for 24GB GPUs with batch_size 16)")
    p.add_argument("--no_staged_training", action="store_true", help="Disable staged training (train full loss all epochs)")
    p.add_argument("--phase1_frac", type=float, default=0.2, help="Fraction of epochs for Phase 1 (VQ only)")
    p.add_argument("--phase2_frac", type=float, default=0.6, help="Fraction of epochs for Phase 2 (full)")
    p.add_argument("--phase3_frac", type=float, default=0.2, help="Fraction of epochs for Phase 3 (codebook frozen)")
    p.add_argument("--commitment_warmup_epochs", type=int, default=5, help="Scale VQ loss up from commitment_warmup_start to 1.0 over this many epochs (0 = off)")
    p.add_argument("--commitment_warmup_start", type=float, default=0.25, help="VQ loss scale in first epoch when warmup enabled")
    p.add_argument("--lr_warmup_epochs", type=int, default=3, help="Linear LR warmup over this many epochs (0 = off)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default="", help="Device (cuda/cpu); default auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        machine_type=args.machine_type or None,
        patch_len=args.patch_len,
        stride=args.stride,
        n_patches=args.n_patches,
        subset_fraction=args.subset_fraction,
        norm_stats_path=args.norm_stats_path,
        recon_target=args.recon_target,
        transform_head_per_patch=args.transform_head_per_patch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_recon=args.lambda_recon,
        lambda_vq=args.lambda_vq,
        lambda_machine_id=args.lambda_machine_id,
        lambda_transform=args.lambda_transform,
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp or args.memory_saver,
        use_gradient_checkpointing=args.use_gradient_checkpointing or args.memory_saver,
        use_staged_training=not args.no_staged_training,
        phase1_frac=args.phase1_frac,
        phase2_frac=args.phase2_frac,
        phase3_frac=args.phase3_frac,
        commitment_warmup_epochs=args.commitment_warmup_epochs,
        commitment_warmup_start=args.commitment_warmup_start,
        lr_warmup_epochs=args.lr_warmup_epochs,
        seed=args.seed,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    run_train(config)


if __name__ == "__main__":
    main()
