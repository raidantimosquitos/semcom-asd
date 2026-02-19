# Training Strategies: Codebook Collapse and Exploding Gradients

This document outlines strategies to stabilize VQ-VAE + semantic-heads training and avoid codebook collapse and exploding gradients.

---

## 1. Codebook collapse

**What it is:** Only a small fraction of codebook entries are used; the rest become "dead" and never get updated. Perplexity (effective number of used codes) stays low.

**Causes:** Bad initialization, encoder outputs clustering in a small region of latent space, commitment loss dominating so encoder always maps to the same codes, or semantic losses pulling representations away before the codebook is well used.

### Strategies

- **VQ-only warmup (recommended for this codebase)**  
Train only reconstruction + VQ for the first few epochs (e.g. 5–15). Set `lambda_machine_id` and `lambda_transform` to 0. Then turn on semantic losses so the encoder already produces diverse latents that use more of the codebook before task losses reshape the space.
- **Staged training (encoder → full → freeze encoder)**  
  - **Phase 1:** VQ-VAE only (recon + VQ) for N₁ epochs.  
  - **Phase 2:** Add semantic heads (recon + VQ + machine_id + transform) for N₂ epochs.  
  - **Phase 3 (optional):** Freeze encoder (and optionally quantizer); train only decoder + semantic heads for a few epochs to refine reconstruction and classification without moving the latent space.
- **Commitment cost schedule**  
Start with a smaller commitment cost (e.g. 0.1) and increase it over epochs (e.g. to 0.25). This lets the encoder explore more before being strongly pulled toward codebook entries.
- **Codebook reseeding**  
Periodically copy encoder outputs to unused codebook entries (e.g. when perplexity is below a threshold). Our quantizer uses EMA, which already updates used codes; reseeding targets unused ones.
- **Entropy / perplexity auxiliary loss**  
Add a small term that encourages higher perplexity (e.g. negative entropy of code usage) so more codes are used. Use a small weight to avoid hurting reconstruction.

---

## 2. Exploding gradients

**What it is:** Gradients grow very large, leading to NaNs or unstable updates.

### Strategies

- **Encoder output bounding (implemented)**  
The encoder outputs `5 * tanh(…))` so latents stay in **(-5, 5)**. The quantizer codebook is clamped to **[-5, 5]**. This keeps commitment loss and its gradients bounded and prevents VQ loss explosion; without it, unconstrained encoder outputs can grow and cause NaNs.
- **Gradient clipping (already in the trainer)**  
`TrainConfig.grad_clip` (e.g. 1.0) is used after `scaler.unscale_` (when AMP is on) via `clip_grad_norm_` over all parameters. Keep this on; 1.0 is a safe default.
- **Learning rate**  
Use a moderate LR (e.g. 1e-3). If loss spikes, try 3e-4 or add warmup.
- **LR warmup (implemented)**  
`lr_warmup_epochs` (e.g. 3) linearly ramps LR from 0 to the target over the first N epochs. Reduces early instability when the codebook and encoder are still adapting.
- **Separate LRs (optional)**  
Use a lower LR for the quantizer (or only for the codebook) so codebook updates are gentler and the encoder is not pulled too aggressively.
- **Batch norm**  
The encoder/decoder use BatchNorm, which helps gradient scale. Ensure batch size is not too small (e.g. ≥ 16) so BatchNorm statistics are stable.

---

## 2b. Transform head semantics

**Training:** The transform head predicts which of the 6 augmentations was applied at training time (one label per spectrogram; all N patches share that label). By default we mean-pool the N patch latents and predict once per sample; use `--transform_head_per_patch` to predict per-patch (labels repeated) for a stronger gradient signal.

**Inference:** At test time, inputs are usually **not** augmented (effective transform is identity). So the transform head will typically see “identity” inputs; it is an auxiliary for training (e.g. to encourage useful invariances). For deployment, use reconstruction and/or machine_id; the transform logits need not be used unless you explicitly feed augmented views.

---

## 3. Suggested schedule for this project

A simple schedule that matches the “VQ first, then semantics, then optional freeze” idea:


| Phase               | Epochs (e.g. 50) | Recon | VQ  | Machine ID | Transform | Codebook   |
| ------------------- | ----------------- | ----- | --- | ---------- | --------- | ---------- |
| 1 – VQ warmup       | first 20%         | ✓     | ✓   | 0          | 0         | train      |
| 2 – Full            | next 60%          | ✓     | ✓   | ✓          | ✓         | train      |
| 3 – Codebook frozen | last 20%          | ✓     | ✓   | ✓          | ✓         | **frozen** |


- **Phase 1:** Reduces codebook collapse risk by letting the codebook and encoder settle with only recon + VQ.  
- **Phase 2:** Adds semantic losses so the latent space becomes task-oriented.  
- **Phase 3:** Codebook (quantizer) frozen; encoder, decoder, and heads keep training. Refines reconstruction and classification without changing the codebook.

**Implemented in the trainer:** Staged training is on by default. Use `--no_staged_training` to disable. Set `--phase1_frac`, `--phase2_frac`, `--phase3_frac` (defaults 0.2, 0.6, 0.2) to change the schedule. Phase 3 freezes the **codebook** (quantizer), not the encoder.

---

## 4. Classification loss explosion (e.g. validation machine_id)

**What you might see:** Train recon/VQ improve; train machine_id CE goes down; **validation machine_id CE explodes** (e.g. 1.5 → 5 → 18) while val recon/VQ stay reasonable.

**Interpretation:**

- **Overfitting:** The model fits training machine IDs and generalizes poorly. With a small subset (e.g. `--subset_fraction 0.2`) this is very likely: few samples per machine_id, so the head memorizes them.
- **Tiny validation set:** With 0.2 subset and 10% val fraction, val can be a handful of samples. One or two wrong predictions make CE loss spike; the metric is noisy.
- **Latent shift:** As the encoder and codebook change (e.g. perplexity rising), the continuous latent distribution shifts. The semantic head was tuned for an earlier distribution and can “break” on the new one, especially if the head is trained hard early (high lambda_machine_id) before the latent stabilizes.

**Suggestions:**

- **Input normalization:** Use per–machine-type mean/std on log-mel spectrograms (compute once, save, apply in the dataset). Stabilizes scale and often helps convergence. See “Normalization” below.
- **VQ warmup then semantics:** Train with only recon + VQ for several epochs, then add machine_id/transform losses (see §3). Reduces head overfitting to a still-changing latent.
- **Lower semantic weights or LR for heads:** Use smaller `lambda_machine_id` / `lambda_transform`, or a separate (lower) learning rate for the semantic heads so they don’t overfit while the encoder is still moving.
- **Less aggressive head training:** Increase dropout on the semantic heads (e.g. 0.4–0.5) or add weight decay (AdamW) for the head parameters.
- **Don’t over-read val loss with a tiny subset:** With very small val sets, prefer train metrics and full-dataset validation for final decisions. Optionally save best by recon+vq only when using a small subset.

---

## 5. Normalization

**Current:** Only `log1p` is applied; no per-channel or per–machine-type standardization.

**Recommendation:** Normalize spectrograms (after log) with per–mel-bin mean and std, **one set of stats per machine type** (fan, slider, etc.), computed on the training set. That keeps different machine types on a similar scale and often stabilizes training.

**Usage:**

1. Compute stats once (e.g. `python -m src.utils.norm_stats --data_root ... --machine_type fan --out stats_fan.pt`).
2. Save the stats file (e.g. next to checkpoints or in the project root).
3. Pass `--norm_stats_path stats_fan.pt` when training; the dataset loads the file and applies `(x - mean) / (std + eps)` per machine type before patching.

Stats can be saved alongside checkpoints or in a dedicated path; reuse the same file for future runs and for inference.

---

## 6. References

- Original VQ-VAE: van den Oord et al., “Neural Discrete Representation Learning” (NeurIPS 2017).  
- Codebook collapse and remedies: e.g. “Representation Collapsing Problems in Vector Quantization” (arXiv), “Dimensional Collapse in VQVAEs” (OpenReview), and “Online Clustered Codebook” (ICCV 2023).

