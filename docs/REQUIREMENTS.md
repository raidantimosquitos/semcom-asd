# Training requirements and low-memory options

## Why is training so demanding on GPU?

Memory is dominated by **activations** (intermediate feature maps) kept for the backward pass, not by model parameters.

- **Effective batch size**: Each sample is one spectrogram yielding **N patches** (e.g. 8 or 16). The model processes **batch_size × n_patches** patches per step (e.g. 1×8 = 8). Every patch goes through:
  - **Encoder**: 1×64×64 → 32×32 → 64×16×16 → 96×8×8 → 64×8×8 (several conv stages with increasing channels).
  - **Quantizer**: no extra activations.
  - **Decoder**: 64×8×8 → 96×16×16 → 64×32×32 → 32×64×64 → 1×64×64.
- All intermediate tensors for those **B×N** patches are stored for backprop. So memory scales roughly with **batch_size × n_patches**. With 8 patches and batch 1 you already have 8 full encoder+decoder activation sets.
- **Optimizer**: AdamW keeps two buffers per parameter (~2× model size). **Model** (encoder + VQ + decoder + heads) is on the order of a few million parameters (~tens of MB), so parameters + optimizer are small compared to activations.

**Rough ballpark**: Comfortable training (batch_size 8–16, n_patches 16) typically needs **≥ 6–8 GiB** GPU memory. With batch 1, n_patches 8, AMP, and gradient checkpointing, you can often get down to **~4 GiB** or a bit less.

---

## Recommended hardware

| Setup | GPU memory | Notes |
|-------|------------|--------|
| **Comfortable** | ≥ 6–8 GiB | Default or moderate batch_size, n_patches=16. |
| **Tight** | ~4 GiB | batch_size=1, n_patches=8, stride=32, AMP, gradient checkpointing. |
| **Minimal / CPU** | — | Train on CPU (omit `--device cuda` or use `--device cpu`). Slow but no GPU limit. |

---

## Options to reduce GPU memory

Use these in combination; they are supported from the main script.

1. **Fewer patches per spectrogram**
   - `--stride 32` (fewer, non-overlapping patches).
   - `--n_patches 8` or `--n_patches 4`.
   - Cuts activation memory roughly in proportion to the number of patches.

2. **Smaller batch size and gradient accumulation**
   - `--batch_size 1`.
   - `--gradient_accumulation_steps 8` (or 4) to keep effective batch size.

3. **Mixed precision (AMP)**
   - `--use_amp`.
   - Lowers activation and parameter memory use.

4. **Gradient checkpointing**
   - `--use_gradient_checkpointing`.
   - Recomputes encoder and decoder activations in the backward pass instead of storing them. **Reduces activation memory by a large fraction** (often ~30–50%) at the cost of ~20–30% more compute.

5. **Train on CPU**
   - `--device cpu` (or do not pass `--device cuda` if you want to force CPU when CUDA is available, you’d need to pass `--device cpu` explicitly).
   - No GPU memory limit; training is slower.

6. **Clear cache after each epoch**
   - The trainer already runs `torch.cuda.empty_cache()` after validation to limit fragmentation.

---

## Example: low-memory command (~4 GiB or less)

```bash
python -m src.main \
  --data_root /path/to/64mel-spectr-dcase2020-task2-dev-dataset \
  --device cuda \
  --machine_type fan \
  --batch_size 1 \
  --stride 32 \
  --n_patches 4 \
  --gradient_accumulation_steps 8 \
  --use_amp \
  --use_gradient_checkpointing
```

If that still hits OOM, try `--n_patches 4` with `--device cpu` to confirm training runs (slowly), then consider a machine with more RAM/GPU or a smaller model (e.g. reduced `latent_dim` / channels in the code).
