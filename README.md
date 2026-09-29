# Experiment 1: Shared vs. Separate SDF Encoder

Modular, top-down implementation of the Experiment 1 plan (Data Processing →
Training Config → Utilities → Training → Translation Test).

## Structure (top-down)

```
config/training_config.yaml   Config schema: one YAML per encoder variant
models/interfaces.py          EncoderDecoderINR protocol + build_model() factory
sdf/coordinates.py            get_3d_coordinates: physical-space coordinate grid
sdf/targets.py                create_mask_sdf_with_clipping, create_multilabel_sdf, sdf_to_channel_masks
training/losses.py            image_reconstruction_loss (Encoder I), masked_eikonal_sdf_loss (Encoder II)
training/train_loop.py        Shared train_encoder_decoder loop, used by both encoders
data/transforms.py            MONAI preprocessing: isometric spacing + z-norm
data/msd.py                    Reads each task's dataset.json directly; seeded, persisted splits (no torch/monai)
data/dataset.py                Thin MONAI dataset wrapper over data/msd.py
utils/metrics.py              psnr_3d, ssim_3d, dice_score, normalized_surface_distance, per_label_metrics
utils/io.py                   save_encoder_weights / save_decoder_weights / load_model_weights
utils/logging_utils.py        wandb wrapper
experiments/run_encoder_I.py       Trains the intensity encoder
experiments/run_encoder_II.py      Trains the SDF encoder
experiments/run_translation_test.py  The actual Experiment 1 comparison
tests/                         Unit tests for sdf/targets.py (single + multi-label) and training/losses.py
```

## The one thing left unimplemented, on purpose

`models/interfaces.py::build_model()` raises `NotImplementedError`. Everything
else (data pipeline, SDF construction, losses, metrics, training loop,
translation test, significance testing) is complete and independently
testable — `tests/` runs today without a model plugged in. This isolates the
one open decision from the discussion: whether to adopt Alpine's `Strainer`
class (same first author as the STRAINER paper) or write a SIREN by hand.
Whichever is chosen just needs to satisfy the `EncoderDecoderINR` protocol —
in particular `reset_decoder()`, which is what implements "decoder replaced
per image, encoder kept across images" from Phase I.

## How the earlier review feedback is reflected in this code

1. **Encoder I and Encoder II use different losses on purpose.**
   `run_encoder_I.py` uses plain MSE (`image_reconstruction_loss`).
   `run_encoder_II.py` uses the full masked-eikonal-MSE
   (`masked_eikonal_sdf_loss`, Equation 3), with `alpha` and
   `eikonal_lambda` as explicit, versioned config fields
   (`config/training_config.yaml`) saved alongside every checkpoint
   (`utils/io.py::save_encoder_weights`) — so it's always traceable which
   settings produced which weights.

2. **The translation test now computes Dice and NSD, not just PSNR/SSIM.**
   `run_translation_test.py::fit_decoder_and_eval` thresholds the predicted
   SDF at zero and computes both, matching the success criterion the
   Experiment 1 write-up actually specifies.

3. **Paired Wilcoxon signed-rank test, not a t-test.**
   `run_translation_test.py::run_significance_tests` — chosen over a paired
   t-test given the small expected held-out sample count and no guarantee of
   normally distributed Dice/PSNR at that scale.

4. **Checkpoints are saved with their config**, so a run using one `alpha`
   is never silently compared against a run using another.

## Data loading: dataset.json is the source of truth

There is no manifest. `data/msd.py` reads each task's own `dataset.json`:

- File lists come from its `training` entries (paths are checked up front, so
  a missing file fails immediately rather than mid-training).
- `image_channels` is `len(modality)` and label names come from `labels`, so
  neither is configured by hand. `sdf.label_groups: "auto"` builds one
  exclusive channel per non-background label, and explicit groups are
  validated against the task's labels, which catches wrong label IDs.
- Set `data.root` (the folder holding `Task*/dataset.json`) and `data.tasks`;
  `--data_root` overrides the root per machine.

One thing `dataset.json` cannot provide is a held-out split. MSD's `test`
images ship without labels, so validation and test cases are carved out of the
`training` list. The split is generated once from `data.split.seed`, saved to
`splits/<task>.json`, and then reused. Loading refuses to proceed if the saved
seed, fractions, or case set no longer match the config, so a changed config
can never silently evaluate on a different split. Reuse these files when
running the nnU-Net and Meta-Seg baselines so all models share identical
splits.

**Test-split size.** A two-sided Wilcoxon signed-rank test cannot reach
p < 0.05 with fewer than 6 pairs. `load_or_create_splits` warns when a task's
test split is that small (and when it is under 10). Check `numTraining` in
`dataset.json` for the smaller MSD tasks such as heart and prostate. At a 20%
test fraction they can fall below 6, so pool tasks or raise the fraction if
those tasks feed a significance test.

**Per-case coordinates.** Coordinates are built per case. An earlier version
built one grid from the first case and reused it, which is wrong whenever
volumes differ in shape after 1 mm resampling.

**Cross-task runs.** A single run needs one channel count across its tasks
(the decoder width is fixed). `[["foreground"]]` gives every task one channel
regardless of its label set. Mixing tasks with different image-channel counts
(e.g. 1-channel CT with 4-channel brain MRI) is rejected. It could be
supported later, since the encoder only sees coordinates and only the
decoder's output width would vary.

## Multi-label datasets (BraTS-style and other MSD tasks)

One SDF channel per entry in `sdf.label_groups` (config). Each entry is a list
of label IDs whose union defines that channel, so one mechanism covers both:

| Case | `label_groups` |
|---|---|
| Single organ | `[[1]]` |
| Mutually exclusive labels (organ + tumour) | `[[1], [2]]` |
| BraTS-style nested regions (whole / core / enhancing) | `[[1, 2, 3], [2, 3], [3]]` |

The label groups you choose should mirror the evaluation regions you plan to
report; the exact IDs depend on each task's `dataset.json`.

What changed to support this:

- `sdf/targets.py`: `create_multilabel_sdf` returns `(D, H, W, K)`. Empty
  channels (label absent in a case) become all `+alpha` plateau and full
  channels all `-alpha`, since a distance transform of an empty or full mask
  has no surface to measure from.
- `training/losses.py`: the Eikonal term now does one coordinate-gradient
  backward pass per channel. Taking one gradient of the summed `(N, K)`
  output gives the gradient of the *sum* of channels, which is not the
  per-channel unit-norm condition. Cost is K backward passes per step for the
  Eikonal term. Subsampling points to reduce that cost would require a
  separate forward pass on a subset, so it is not implemented here.
- Decoding (`sdf_to_channel_masks`): `independent` (threshold each channel at
  0; use for nested or overlapping regions) or `exclusive` (argmin across
  channels; use only when groups are disjoint).
- `utils/metrics.py::per_label_metrics`: per-label Dice and NSD, plus means.
  Labels absent from a case's ground truth are NaN and excluded from the
  means, otherwise a correct all-empty prediction scores Dice = 1 and
  inflates the average.
- Encoder I supports multi-channel images (`data.image_channels`, 4 for MSD
  brain tumour's MRI modalities).
- The SDF decoder's `out_features` is `len(label_groups)`; only the final
  decoder layer grows with K, so the hypernetwork's output dimension barely
  changes.

## Translation test fairness

`run_translation_test.py` takes the task settings (`alpha`,
`eikonal_lambda`, `label_groups`, `decode_mode`) and the decoder-fitting
optimizer and scheduler from Encoder II's config and applies them to *both*
encoders. Reading each checkpoint's own settings would let a mismatch in
alpha or optimizer masquerade as an encoder difference. It asserts that both
checkpoints share a model architecture and voxel spacing.

## Test status

`tests/test_sdf_targets.py`, `tests/test_multilabel_targets.py` and
`tests/test_msd.py` pass (numpy/scipy only). `tests/test_losses.py`, including the new per-channel
Eikonal tests, has not been executed: the sandbox this was written in has a
broken torch install. Run it in your own environment before relying on the
loss.
