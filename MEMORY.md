# GPU memory and OOM options

Training runs with **batch_size × n_patches** effective “batch” through the encoder/decoder (e.g. 16×16 = 256 patches). Activation memory in the encoder and decoder dominates; the VQ quantizer adds smaller but non-trivial allocations (distance and one-hot matrices over all patch positions).

## Recommended: one-flag memory saver (24GB GPUs)

Use `--memory_saver` to enable mixed precision (AMP) and gradient checkpointing. This usually lets you run `batch_size=16` on an RTX 4090 (24GB) without OOM:

```bash
python -m src.main --data_root /path/to/64mel-spectr-dcase2020-task2-dev-dataset/ \
  --device cuda --machine_type fan --batch_size 16 --epochs 100 \
  --norm_stats_path checkpoints/stats/stats_fan.pt --memory_saver
```

## Other options (if you still OOM or want to tune)

1. **Smaller batch + gradient accumulation**  
   Keep effective batch size by reducing per-step batch and accumulating gradients:
   - `--batch_size 8 --gradient_accumulation_steps 2` (effective batch 16).
   - Combine with `--memory_saver` if needed.

2. **Fewer patches per spectrogram**  
   Less activation memory and smaller quantizer buffers:
   - `--n_patches 8` (half the “batch” through the model; dataset will sample or repeat to 8 patches per sample).

3. **Larger stride (fewer patches per file)**  
   Fewer patches per spectrogram from the dataset:
   - `--stride 32` (and optionally `--n_patches 8`) so each sample contributes fewer patches.

4. **PyTorch allocator (fragmentation)**  
   If the error says “reserved but unallocated memory is large”, try:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   python -m src.main ...
   ```

5. **Manual flags instead of `--memory_saver`**  
   - `--use_amp` — mixed precision (FP16) forward/backward.
   - `--use_gradient_checkpointing` — recompute encoder/decoder activations in backward (saves ~40% activation memory, more compute).

## Discretization and anomaly detection

The latent space is discretized by the VQ codebook; at inference only code indices (and optionally a density model on them) are used for anomaly scoring. If AD performance is poor, possible directions (separate from OOM) include:

- **Codebook size / capacity**: `num_embeddings` (e.g. 128) and `latent_dim` (e.g. 64) in the config; larger codebooks can represent more modes but may need more data and tuning.
- **Commitment and warmup**: `commitment_cost`, `commitment_warmup_epochs`, and `commitment_warmup_start` control how strongly the encoder is pulled toward the codebook; affects code usage and reconstruction.
- **Staged training**: Phase 1 (VQ-only), Phase 2 (full loss), Phase 3 (codebook frozen) in `TRAINING_STRATEGIES.md`; helps avoid collapse and can affect how well the discrete codes separate normal vs abnormal.

Fixing OOM first (e.g. with `--memory_saver` and batch/accumulation) lets you run full training and then tune these for AD.
