# Semantic tasks: machine_id and transform

This document explains how the two classification tasks are computed, how their losses work, what config affects them, and why you see aggressive overfitting (train loss down, val loss up).

---

## 1. Logic of the two tasks

### Task 1: Machine ID

- **Label:** One per spectrogram. `batch["machine_id"]` has shape `(B,)`, values in `{0, 2, 4, 6}` for fan (from `id_00`, `id_02`, etc. in filenames).
- **Representation:** We do **not** predict per-patch. We aggregate the N patch latents per sample into one vector per spectrogram:
  - `latent_continuous`: `(B*N, C, h, w)` (all patches from the batch).
  - `aggregate_latents_per_sample(latent_continuous, n_patches)` → `(B, C, h, w)` by **mean** over the N patches of each sample.
- **Head:** `(B, C, h, w)` → AdaptiveAvgPool2d(1) → `(B, C)` → Linear(C, 128) → SiLU → Dropout(0.3) → Linear(128, num_machine_ids) → **machine_logits** `(B, num_machine_ids)`.
- **Loss:** `F.cross_entropy(machine_logits, machine_id)` where `machine_id` is `(B,)` with integer class indices in `[0, num_machine_ids - 1]`.

So: one aggregated latent vector per spectrogram → one vector of logits → one CE loss per sample, averaged over the batch.

### Task 2: Transform (augmentation)

- **Label:** One per spectrogram. `batch["transformation_id"]` has shape `(B,)`, values in `{0, 1, ..., 5}` (which of the 6 augmentations was applied).
- **Representation (default):** Same as machine_id: aggregated latent `(B, C, h, w)` → head → **transform_logits** `(B, 6)`.
- **Representation (optional, `--transform_head_per_patch`):** Per-patch latent `(B*N, C, h, w)` → head → `(B*N, 6)`; labels expanded to `(B*N,)` (each of the N patches gets the same transform label); CE over all B*N predictions. Stronger gradient signal, same number of classes.
- **Loss:** `F.cross_entropy(transform_logits, transformation_id)` (or with expanded labels when per-patch).

---

## 2. How Cross-Entropy is used (and softmax)

- **Inputs:**  
  - `logits`: `(B, num_classes)` (raw scores, no softmax).  
  - `targets`: `(B,)` integer class indices in `[0, num_classes-1]`.
- **Definition:**  
`F.cross_entropy(logits, targets)` = **negative log-likelihood of the correct class after softmax**:
  - `p = softmax(logits)` → `p[b, k] = exp(logits[b,k]) / sum_j exp(logits[b,j])`.
  - For each sample `b`, the loss is `-log(p[b, targets[b]])`.
  - So we **do** use softmax: it’s inside CE. The head outputs **logits**; CE turns them into probabilities with softmax and then takes `-log(correct_class_prob)`.
- **“Softmax over machine ids”:**  
We already have “softmax over the machine_id classes”: the last layer has `num_machine_ids` outputs, and CE applies softmax over those. If by “softmax over machine ids” you mean **soft labels** (e.g. smoothing the one-hot target so that the model is less overconfident), that’s **label smoothing** and is a good way to reduce overfitting (see below).

---

## 3. Number of classes: from data, not fixed

- **Config default:** `TrainConfig.num_machine_ids = 4` is only a lower bound.
- **What actually happens:** Before building the model, the trainer does:
  ```python
  if train_ds.samples:
      max_machine_id = max(s[2] for s in train_ds.samples)
      config.num_machine_ids = max(config.num_machine_ids, max_machine_id + 1)
  ```
  So **num_machine_ids is inferred from the training set**: it becomes `max(machine_id) + 1` (and at least 4). For **fan** we have machine_ids `{0, 2, 4, 6}` → `max = 6` → `num_machine_ids = 7`. The head therefore has **7** output units; classes 1, 3, 5 are unused in the data but the indices are valid for CE.
- **Transform:** Always **6** classes (fixed by the six augmentations), set by `num_transforms = NUM_TRANSFORMS` (no inference from data).

So we do **not** “enforce the number of classes from scratch” for machine_id: we **infer** it from the data and use it to size the head. For transform we fix 6 from the start.

---

## 4. Configuration that affects semantic overfitting

What influences how much the semantic heads overfit:


| Config                                      | Role                                                                    | Impact on overfitting                                                                    |
| ------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **lambda_machine_id**, **lambda_transform** | Weights of the two CE losses in the total loss.                         | Larger → head is fit more aggressively → more overfitting if data is small.              |
| **subset_fraction**                         | Uses only a fraction of each machine_id (e.g. 0.2 → 735 samples total). | Smaller fraction → fewer train samples → much easier to overfit.                         |
| **val_fraction**                            | Fraction of (sub)set used as validation (e.g. 0.1).                     | With 735 samples, 10% val ≈ 73 samples; val loss is noisy and can spike.                 |
| **batch_size**                              | You use 1.                                                              | Very noisy gradients; BatchNorm statistics unstable; can worsen overfitting.             |
| **Dropout** (in SemanticHeads)              | Currently 0.3 in both heads.                                            | Increase (e.g. 0.4–0.5) to regularize the heads.                                         |
| **Phase 2 start**                           | Semantic losses are 0 in Phase 1; from Phase 2 they’re on.              | If the encoder is still changing a lot, the head may overfit to a shifting latent space. |
| **num_machine_ids**                         | Inferred as above.                                                      | Only affects output size; with 7 classes and only 4 used, the head has extra capacity.   |


So: **small subset + small val + batch_size=1 + moderate lambdas** is a setup where the semantic heads can overfit very quickly (train CE drops, val CE explodes), as in your log.

---

## 5. Why your curves look like that

- **Phase 1:** Only recon + VQ. Train/val recon and VQ improve; no semantic loss.
- **Phase 2:** Semantic losses turn on. The machine_id head has 7 outputs, 4 classes actually present, and is trained on ~662 train samples (735 × 0.9) with batch_size 1.
- The head quickly fits the training set (train machine_id CE goes from ~1.45 to ~0.39).
- Val machine_id CE grows (1.59 → 16.41) because:
  - Very few val samples (≈73) → high variance.
  - The head is overfitting to the train latent distribution; at val time the same encoder can produce slightly different latent statistics, so the head’s decision boundary generalizes poorly.
  - With only 4 machine_ids and small data, the head can memorize train patterns instead of learning robust features.

So the **logic** of the two tasks and their losses is consistent; the **setup** (subset + batch size + lambdas) is what makes overfitting dominant.

---

## 6. Options before changing architecture

- **Label smoothing:** Use `F.cross_entropy(..., label_smoothing=0.1)` so targets are soft (e.g. 0.9 on correct class, 0.1/(K-1) on others). Reduces overconfidence and often helps val.
- **Smaller semantic weights:** Lower `lambda_machine_id` and `lambda_transform` (e.g. 0.2) so the head is updated more gently.
- **Stronger dropout:** Increase dropout in the semantic heads (e.g. 0.4–0.5).
- **Larger batch size:** If possible, use e.g. 8 or 16 so gradients and BatchNorm are more stable.
- **Save best by recon+vq only:** When using a small subset, best checkpoint by “total val loss” is dominated by the exploding machine_id val CE; you can instead save best by recon+val VQ only so that the best checkpoint is not chosen by an overfitting head.
- **Separate LR for heads:** Use a smaller learning rate for the semantic head parameters so they don’t overfit as fast.

We can add any of these (e.g. label smoothing + config for “save best by recon+vq”) as the next step.