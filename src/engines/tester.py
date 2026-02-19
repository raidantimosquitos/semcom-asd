"""
Inference and density-based anomaly scoring for the trained VQ-VAE.
Load encoder, quantizer, decoder; optionally load or fit a histogram density on code indices;
compute NLL per sample as the anomaly score (higher NLL = more anomalous).
Evaluation: AUC and pAUC (partial AUC, FPR in [0, max_fpr]) computed per spectrogram.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import DCASEMelSpectrogramDataset, collate_dcase_patches
from src.models.decoder import LightWeightDecoder
from src.models.density import HistogramCodeDensity
from src.models.encoder import LightWeightEncoder, flatten_patches_for_encoder
from src.models.quantizer import VectorQuantizerEMA


# Default config values when loading from checkpoint (in case keys missing)
_DEFAULT_CONFIG = {
    "latent_dim": 64,
    "num_embeddings": 512,
    "commitment_cost": 0.25,
    "vq_decay": 0.99,
    "num_machine_ids": 4,
    "num_transforms": 6,
    "n_patches": 16,
    "patch_len": 64,
    "stride": 16,
}


def _build_encoder_quantizer_decoder(config: dict[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    """Build encoder, quantizer, decoder from config dict (e.g. from checkpoint)."""
    c = {**_DEFAULT_CONFIG, **config}
    encoder = LightWeightEncoder(in_channels=1, out_channels=c["latent_dim"])
    quantizer = VectorQuantizerEMA(
        num_embeddings=c["num_embeddings"],
        embedding_dim=c["latent_dim"],
        commitment_cost=c["commitment_cost"],
        decay=c["vq_decay"],
    )
    decoder = LightWeightDecoder(
        latent_dim=c["latent_dim"],
        out_height=64,
        out_width=64,
    )
    for m in (encoder, quantizer, decoder):
        m.to(device)
        m.eval()
    return {"encoder": encoder, "quantizer": quantizer, "decoder": decoder}


def load_checkpoint_models(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    """
    Load encoder, quantizer, decoder from a training checkpoint.
    Returns (models_dict, config_dict). Semantic heads are not loaded (not needed for NLL scoring).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(
            f"{checkpoint_path} does not look like a VQ-VAE checkpoint (expected a dict). "
            "If you previously saved the density to this path, use a different file for --fit_density (e.g. checkpoints/density_fan.pt) and restore the real checkpoint (re-train or restore from backup)."
        )
    required = ("encoder", "quantizer", "decoder")
    missing = [k for k in required if k not in state]
    if missing:
        if "probs" in state and "num_codes" in state:
            raise ValueError(
                f"{checkpoint_path} looks like a density file (has 'probs', 'num_codes'), not a VQ-VAE checkpoint. "
                "Do not use the checkpoint path for --fit_density. Use e.g. --fit_density checkpoints/density_fan.pt. "
                "You need a real training checkpoint (checkpoints/best.pt from training); if it was overwritten, re-train or restore from backup."
            )
        raise ValueError(
            f"{checkpoint_path} is missing keys: {missing}. Not a valid VQ-VAE checkpoint. "
            "Re-train to produce checkpoints/best.pt or restore the checkpoint from backup."
        )
    config = state.get("config", {})
    models = _build_encoder_quantizer_decoder(config, device)
    models["encoder"].load_state_dict(state["encoder"])
    models["quantizer"].load_state_dict(state["quantizer"])
    models["decoder"].load_state_dict(state["decoder"])
    return models, config


def fit_histogram_density(
    models: dict[str, torch.nn.Module],
    data_loader: DataLoader,
    num_codes: int,
    n_patches: int,
    device: torch.device,
    smoothing: float = 1.0,
) -> HistogramCodeDensity:
    """
    Collect code indices from the data loader (encoder -> quantizer.encode_to_indices)
    and fit HistogramCodeDensity. Use training (normal) data only.
    """
    density = HistogramCodeDensity(num_codes=num_codes, smoothing=smoothing)
    all_indices = []
    with torch.no_grad():
        for batch in data_loader:
            patches = batch["patches"].to(device)
            flat_patches, _ = flatten_patches_for_encoder(patches, None)
            latent = models["encoder"](flat_patches)
            indices = models["quantizer"].encode_to_indices(latent, return_distances=False)
            # indices (B*N, 8, 8) -> flatten to (B*N*64,)
            all_indices.append(indices.cpu().reshape(-1))
    indices_concat = torch.cat(all_indices, dim=0).numpy()
    density.fit(indices_concat)
    return density


def score_batch_nll(
    models: dict[str, torch.nn.Module],
    density: HistogramCodeDensity,
    batch: dict[str, torch.Tensor],
    n_patches: int,
    device: torch.device,
    per_position: bool = True,
) -> list[float]:
    """
    For a batch of samples, return NLL per sample (list of length B).
    Each sample has n_patches patches; each patch 8x8 codes. NLL is summed (or averaged per position) over all codes of that sample.
    """
    patches = batch["patches"].to(device)
    B = patches.shape[0]
    flat_patches, _ = flatten_patches_for_encoder(patches, None)
    with torch.no_grad():
        latent = models["encoder"](flat_patches)
        indices = models["quantizer"].encode_to_indices(latent, return_distances=False)
    # indices (B*n_patches, 8, 8)
    indices = indices.cpu().numpy()
    nll_list = []
    for b in range(B):
        sample_indices = indices[b * n_patches : (b + 1) * n_patches].ravel()
        if per_position:
            nll_list.append(density.score_nll_per_position(sample_indices))
        else:
            nll_list.append(density.score_nll(sample_indices))
    return nll_list


def compute_auc_pauc(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    max_fpr: float = 0.1,
) -> dict[str, float]:
    """
    Compute AUC and partial AUC (pAUC) from per-spectrogram anomaly scores and binary labels.
    Higher score = more anomalous. labels: 0 = normal, 1 = anomaly.
    pAUC is the area under the ROC curve for FPR in [0, max_fpr] (default 0.1).
    Returns dict with "auc" and "pauc".
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or np.unique(y).size < 2:
        return {"auc": float("nan"), "pauc": float("nan")}
    auc = roc_auc_score(y, s)
    pauc = roc_auc_score(y, s, max_fpr=max_fpr)
    return {"auc": float(auc), "pauc": float(pauc)}


def run_anomaly_evaluation(
    checkpoint_path: str | Path,
    density_path: str | Path,
    data_root: str | Path,
    split: str = "test",
    machine_type: str | None = None,
    normal_machine_ids: tuple[int, ...] = (0,),
    batch_size: int = 32,
    n_patches: int | None = None,
    norm_stats_path: str | Path | None = None,
    device: str | None = None,
    per_position_nll: bool = True,
    max_fpr: float = 0.1,
) -> dict[str, Any]:
    """
    Run anomaly inference and compute AUC and pAUC (per spectrogram).
    Requires normal_machine_ids so that each sample gets a binary label:
    is_anomaly = 0 if machine_id in normal_machine_ids else 1.
    Returns dict with "auc", "pauc", "scores", "labels" (scores and labels are per-spectrogram lists).
    """
    from src.utils.norm_stats import load_norm_stats

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    models, config = load_checkpoint_models(checkpoint_path, dev)
    npatch = n_patches if n_patches is not None else config.get("n_patches", _DEFAULT_CONFIG["n_patches"])

    norm_stats = None
    if norm_stats_path and Path(norm_stats_path).is_file():
        norm_stats = load_norm_stats(norm_stats_path)

    dataset = DCASEMelSpectrogramDataset(
        data_root,
        split=split,
        machine_type=machine_type,
        patch_len=config.get("patch_len", _DEFAULT_CONFIG["patch_len"]),
        stride=config.get("stride", _DEFAULT_CONFIG["stride"]),
        n_patches=npatch,
        norm_stats=norm_stats,
        normal_machine_ids=normal_machine_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dcase_patches,
    )

    density = HistogramCodeDensity.load(density_path)
    all_scores: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            nll_batch = score_batch_nll(
                models,
                density,
                batch,
                n_patches=npatch,
                device=dev,
                per_position=per_position_nll,
            )
            all_scores.extend(nll_batch)
            all_labels.extend(batch["is_anomaly"].cpu().tolist())

    metrics = compute_auc_pauc(all_scores, all_labels, max_fpr=max_fpr)
    return {
        "auc": metrics["auc"],
        "pauc": metrics["pauc"],
        "scores": all_scores,
        "labels": all_labels,
    }


def run_anomaly_inference(
    checkpoint_path: str | Path,
    density_path: str | Path,
    data_root: str | Path,
    split: str = "test",
    machine_type: str | None = None,
    batch_size: int = 32,
    n_patches: int | None = None,
    norm_stats_path: str | Path | None = None,
    device: str | None = None,
    per_position_nll: bool = True,
) -> list[float]:
    """
    Load VQ-VAE and density; run on dataset and return NLL (anomaly score) per sample.
    Higher NLL = more anomalous. Uses test split by default.
    """
    from src.utils.norm_stats import load_norm_stats

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    models, config = load_checkpoint_models(checkpoint_path, dev)
    npatch = n_patches if n_patches is not None else config.get("n_patches", _DEFAULT_CONFIG["n_patches"])
    num_codes = config.get("num_embeddings", _DEFAULT_CONFIG["num_embeddings"])

    norm_stats = None
    if norm_stats_path and Path(norm_stats_path).is_file():
        norm_stats = load_norm_stats(norm_stats_path)

    dataset = DCASEMelSpectrogramDataset(
        data_root,
        split=split,
        machine_type=machine_type,
        patch_len=config.get("patch_len", _DEFAULT_CONFIG["patch_len"]),
        stride=config.get("stride", _DEFAULT_CONFIG["stride"]),
        n_patches=npatch,
        norm_stats=norm_stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dcase_patches,
    )

    density = HistogramCodeDensity.load(density_path)
    all_nll = []
    for batch in loader:
        nll_batch = score_batch_nll(
            models,
            density,
            batch,
            n_patches=npatch,
            device=dev,
            per_position=per_position_nll,
        )
        all_nll.extend(nll_batch)
    return all_nll


def fit_and_save_density(
    checkpoint_path: str | Path,
    data_root: str | Path,
    density_save_path: str | Path,
    split: str = "train",
    machine_type: str | None = None,
    batch_size: int = 32,
    smoothing: float = 1.0,
    device: str | None = None,
    norm_stats_path: str | Path | None = None,
) -> HistogramCodeDensity:
    """
    Load VQ-VAE from checkpoint, run on training data to collect code indices,
    fit HistogramCodeDensity, and save to density_save_path. Use this once on normal data before run_anomaly_inference.
    """
    from src.utils.norm_stats import load_norm_stats

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    models, config = load_checkpoint_models(checkpoint_path, dev)
    npatch = config.get("n_patches", _DEFAULT_CONFIG["n_patches"])
    num_codes = config.get("num_embeddings", _DEFAULT_CONFIG["num_embeddings"])

    norm_stats = None
    if norm_stats_path and Path(norm_stats_path).is_file():
        norm_stats = load_norm_stats(norm_stats_path)

    dataset = DCASEMelSpectrogramDataset(
        data_root,
        split=split,
        machine_type=machine_type,
        patch_len=config.get("patch_len", _DEFAULT_CONFIG["patch_len"]),
        stride=config.get("stride", _DEFAULT_CONFIG["stride"]),
        n_patches=npatch,
        norm_stats=norm_stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dcase_patches,
    )

    density = fit_histogram_density(
        models,
        loader,
        num_codes=num_codes,
        n_patches=npatch,
        device=dev,
        smoothing=smoothing,
    )
    density.save(density_save_path)
    return density


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run anomaly detection: fit density and/or evaluate (AUC/pAUC) with checkpoints/best.pt")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to VQ-VAE checkpoint")
    p.add_argument("--data_root", type=str, required=True, help="Root path to DCASE mel-spectrogram dataset")
    p.add_argument("--machine_type", type=str, default=None, help="Machine type (e.g. fan, valve); must match training")
    p.add_argument("--fit_density", type=str, default=None, metavar="PATH", help="Fit histogram density on train split and save to PATH (use a different file than the checkpoint, e.g. checkpoints/density_fan.pt)")
    p.add_argument("--density_path", type=str, default=None, help="Path to saved density .pt (required for --evaluate or inference)")
    p.add_argument("--evaluate", action="store_true", help="Run evaluation on test split and print AUC / pAUC")
    p.add_argument("--normal_machine_ids", type=int, nargs="+", default=[0], help="Machine IDs treated as normal (default: 0)")
    p.add_argument("--norm_stats_path", type=str, default=None, help="Path to norm stats .pt (optional)")
    p.add_argument("--device", type=str, default="", help="Device (cuda/cpu); default auto")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for data loader")
    args = p.parse_args()

    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.fit_density:
        density_save_path = Path(args.fit_density).resolve()
        if density_save_path == checkpoint_path.resolve():
            raise SystemExit(
                "Refusing to save density to the same path as the checkpoint (would overwrite it). "
                "Use a different file, e.g. --fit_density checkpoints/density_fan.pt"
            )
        print("Fitting histogram density on train split...")
        fit_and_save_density(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            density_save_path=density_save_path,
            split="train",
            machine_type=args.machine_type,
            batch_size=args.batch_size,
            device=device,
            norm_stats_path=Path(args.norm_stats_path) if args.norm_stats_path else None,
        )
        print("Saved density to", args.fit_density)
        density_path = args.fit_density
    else:
        density_path = args.density_path

    if args.evaluate:
        if not density_path:
            raise SystemExit("For --evaluate you must provide --density_path or --fit_density")
        print("Running evaluation on test split...")
        result = run_anomaly_evaluation(
            checkpoint_path=checkpoint_path,
            density_path=Path(density_path),
            data_root=data_root,
            split="test",
            machine_type=args.machine_type,
            normal_machine_ids=tuple(args.normal_machine_ids),
            batch_size=args.batch_size,
            device=device,
            norm_stats_path=Path(args.norm_stats_path) if args.norm_stats_path else None,
        )
        print("AUC:", result["auc"])
        print("pAUC (FPR≤0.1):", result["pauc"])
    elif not args.fit_density:
        if density_path:
            scores = run_anomaly_inference(
                checkpoint_path=checkpoint_path,
                density_path=Path(density_path),
                data_root=data_root,
                split="test",
                machine_type=args.machine_type,
                batch_size=args.batch_size,
                device=device,
                norm_stats_path=Path(args.norm_stats_path) if args.norm_stats_path else None,
            )
            print("Inference: NLL per spectrogram (first 5):", scores[:5])
        else:
            print("Load test: use --fit_density PATH to fit and save density, then --evaluate with --density_path PATH")
            models, config = load_checkpoint_models(checkpoint_path, torch.device(device))
            print("Loaded encoder, quantizer, decoder from", checkpoint_path)
            print("Config (n_patches, num_embeddings):", config.get("n_patches"), config.get("num_embeddings"))


if __name__ == "__main__":
    _main()
