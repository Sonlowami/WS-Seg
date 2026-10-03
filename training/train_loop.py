"""
Shared encoder-decoder training loop.

Implements the pattern from the notes: given dataloader, model, optimizer,
scheduler, epoch_count, and coordinates -- train an encoder-decoder for each
image for all epochs, log every log_every_n_epochs, and after all images,
save the encoder and drop the decoders.

This is the same loop for both Encoder I (intensity) and Encoder II (SDF);
only the loss_fn and target extraction differ, injected by the caller
(experiments/run_encoder_I.py, experiments/run_encoder_II.py). This mirrors
Phase I of the architecture: the encoder is kept across images, the decoder
is replaced for each new image (see models/interfaces.py: reset_decoder).

Two ways to learn the shared encoder (training.mode):
  joint       train_encoder_jointly -- STRAINER: one decoder head per training
              item, all trained together, each step mixing several cases, so
              the encoder learns what is shared across the whole set.
  sequential  train_encoder_decoder -- one case at a time with a fresh decoder;
              the encoder drifts towards whichever cases came last.
"""
from typing import Callable
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from models.interfaces import EncoderDecoderINR
from sdf.coordinates import coords_from_indices
from utils.logging_utils import log as wandb_log

# Used when a config has no training.points_per_step. Never default to the
# whole volume: a single 1 mm CT is ~3e7-1e8 voxels, far beyond GPU memory.
DEFAULT_POINTS_PER_STEP = 2 ** 18


def build_optimizer(params, optimizer_cfg: dict):
    if optimizer_cfg["name"] == "adamw":
        return torch.optim.AdamW(params, lr=optimizer_cfg["initial_lr"])
    elif optimizer_cfg["name"] == "adam":
        return torch.optim.Adam(
            params, lr=optimizer_cfg["initial_lr"],
            weight_decay=optimizer_cfg.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unknown optimizer: {optimizer_cfg['name']}")


def build_scheduler(optimizer, scheduler_cfg: dict, total_epochs: int):
    if scheduler_cfg["name"] == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif scheduler_cfg["name"] == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_epochs)
    raise ValueError(f"Unknown scheduler: {scheduler_cfg['name']}")


def resolve_device(name=None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sample_points(coords, target, n_points=None, requires_grad=False):
    """
    Random voxel subset for one optimization step.

    A whole volume at 1 mm isotropic is ~10^8 voxels; pushing it through the
    INR with autograd needs hidden_dim floats per voxel per layer (~90 GB per
    layer at hidden_dim=256), so every step fits a random batch instead.
    Sampling with replacement (randint) avoids a full permutation each step.
    Both losses are per-point means with scalar kwargs, so a subset is an
    unbiased estimate of the full-volume loss.
    """
    if n_points is not None and n_points < coords.shape[0]:
        idx = torch.randint(coords.shape[0], (n_points,), device=coords.device)
        coords, target = coords[idx], target[idx]
    # detach(): the sampled coords must be a leaf for the Eikonal gradient.
    return coords.detach().requires_grad_(requires_grad), target


@torch.no_grad()
def predict_in_chunks(model, coords, chunk_size: int = 2 ** 20) -> torch.Tensor:
    """Full-volume prediction without autograd, returned on CPU for metrics."""
    return torch.cat([model.forward(coords[i:i + chunk_size]).cpu()
                      for i in range(0, coords.shape[0], chunk_size)])




def train_encoder_decoder(
    dataloader,
    model: EncoderDecoderINR,
    coords_fn: Callable,              # (case) -> (coords (N, 3), spatial_shape (D, H, W))
    target_extractor: Callable,       # (case) -> (target_tensor, extra_kwargs)
    loss_fn: Callable,                # (pred, coords, target, **extra) -> dict with "total"
    metric_fn: Callable,              # (pred, target, shape) -> dict of scalar metrics to log
    optimizer_cfg: dict,
    scheduler_cfg: dict,
    epoch_count: int,
    log_every_n_epochs: int,
    wandb_run,
    points_per_step: int = DEFAULT_POINTS_PER_STEP,  # random voxels per step; None = whole volume
    eval_chunk_size: int = 2 ** 20,   # voxels per no-grad forward when logging metrics
    device=None,                      # "auto"/None -> cuda if available
):
    """
    Trains one shared encoder across the full dataloader, fitting (and
    discarding) a fresh decoder per case. Returns the trained encoder's
    state_dict; decoders are intentionally not retained past their own case.

    Each "epoch" is one optimizer step on a random batch of points_per_step
    voxels (see sample_points). Metrics are computed on the full volume.
    """
    device = resolve_device(device)
    model.to(device)
    print(f"train_encoder_decoder: device={device}, points_per_step="
          f"{points_per_step or 'whole volume'}, eval_chunk_size={eval_chunk_size}")
    global_step = 0

    for case in dataloader:
        model.reset_decoder()
        # Coordinates are built per case: after isometric resampling, volumes
        # differ in shape (across cases, and certainly across tasks), so a grid
        # built once from the first case would be wrong for the rest.
        coords, shape = coords_fn(case)
        target, extra = target_extractor(case)
        # MONAI MetaTensor -> plain tensor: metadata would otherwise be carried
        # through every op in the loop (slower) and into the logged losses,
        # which wandb cannot serialize.
        if hasattr(target, "as_tensor"):
            target = target.as_tensor()
        coords, target_dev = coords.to(device), target.to(device)
        needs_coord_grad = extra.get("needs_coord_grad", False)

        params = list(model.encoder_parameters()) + list(model.decoder_parameters())
        optimizer = build_optimizer(params, optimizer_cfg)
        scheduler = build_scheduler(optimizer, scheduler_cfg, epoch_count)

        for epoch in range(epoch_count):
            optimizer.zero_grad()
            coords_batch, target_batch = sample_points(
                coords, target_dev, points_per_step, requires_grad=needs_coord_grad)
            pred = model.forward(coords_batch)
            loss_dict = loss_fn(pred, coords_batch, target_batch, **extra.get("loss_kwargs", {}))
            loss_dict["total"].backward()
            optimizer.step()
            scheduler.step()

            if epoch % log_every_n_epochs == 0:
                full_pred = predict_in_chunks(model, coords, eval_chunk_size)
                metrics = metric_fn(full_pred, target, shape)
                metrics.update({f"loss/{k}": v for k, v in loss_dict.items() if k != "total"})
                metrics["loss/total"] = loss_dict["total"].detach()
                wandb_log(wandb_run, metrics, step=global_step)
            global_step += 1

        del coords, target_dev
    return model.encoder_state_dict()


# ---------------------------------------------------------------- joint (STRAINER) training

def _first(x):
    """DataLoader (batch_size=1) wraps string fields in a list."""
    return x[0] if isinstance(x, (list, tuple)) else x


def prepare_items(dataloader, target_extractor: Callable, shape_fn: Callable,
                  storage_dtype=torch.float16) -> list:
    """
    Load every training item once: {case_id, shape, target (N, C) on CPU,
    extra}. Joint training needs every item's target at every step, so they
    are kept in CPU memory (float16 by default: ample for z-scored
    intensities and for SDFs clipped to +-alpha); coordinates are never
    stored, they are rebuilt from sampled indices (coords_from_indices).
    """
    items = []
    for case in dataloader:
        target, extra = target_extractor(case)
        if hasattr(target, "as_tensor"):                 # MONAI MetaTensor
            target = target.as_tensor()
        items.append({"case_id": _first(case["case_id"]), "shape": tuple(shape_fn(case)),
                      "target": target.to(storage_dtype).contiguous(), "extra": extra})
    total = sum(i["target"].numel() * i["target"].element_size() for i in items)
    print(f"prepared {len(items)} training items; targets use {total / 1e9:.2f} GB of CPU memory")
    return items


@torch.no_grad()
def predict_volume(model, shape: tuple, spacing_mm: tuple, decoder_index: int = 0,
                   chunk_size: int = 2 ** 20, device="cpu") -> torch.Tensor:
    """Full-volume prediction of one head, chunked over voxel indices so no
    full coordinate grid is ever built. Returned on CPU as (N, C)."""
    n = shape[0] * shape[1] * shape[2]
    out = []
    for start in range(0, n, chunk_size):
        idx = torch.arange(start, min(start + chunk_size, n), device=device)
        out.append(model.forward(coords_from_indices(idx, shape, spacing_mm), decoder_index).cpu())
    return torch.cat(out)


def train_encoder_jointly(
    items: list,                      # from prepare_items
    model: EncoderDecoderINR,         # built with num_decoders=len(items)
    loss_fn: Callable,                # (pred, coords, target, **extra) -> dict with "total"
    metric_fn: Callable,              # (pred, target, shape) -> dict of scalar metrics to log
    optimizer_cfg: dict,
    scheduler_cfg: dict,
    steps: int,                       # total optimizer steps
    log_every_n_steps: int,
    wandb_run,
    spacing_mm: tuple,
    cases_per_step: int = 8,
    points_per_step: int = DEFAULT_POINTS_PER_STEP,  # split evenly across the step's cases
    eval_chunk_size: int = 2 ** 20,
    device=None,
):
    """
    STRAINER-style joint training: one shared encoder, decoder head i fits
    item i, and every step mixes cases_per_step random items, so each update
    of the encoder is pulled by several different cases at once. Losses are
    averaged over the step's items. Heads not drawn in a step get no
    gradient (grads are None, so Adam leaves them untouched). Metrics are
    computed on the full volume of one drawn item per log step. Returns the
    encoder's state_dict; the heads are dropped, as in sequential training.
    """
    n_heads = len(getattr(model, "decoder", []))
    if n_heads and n_heads != len(items):
        raise ValueError(f"Model has {n_heads} decoder heads but there are {len(items)} items; "
                         f"build it with num_decoders=len(items).")
    device = resolve_device(device)
    model.to(device)
    per_step = min(cases_per_step, len(items))
    per_case = max(points_per_step // per_step, 1)
    print(f"train_encoder_jointly: device={device}, {len(items)} heads, {steps} steps, "
          f"{per_step} items x {per_case} points per step "
          f"(~{steps * per_step / len(items):.0f} steps per head), eval_chunk_size={eval_chunk_size}")

    optimizer = build_optimizer(model.parameters(), optimizer_cfg)
    scheduler = build_scheduler(optimizer, scheduler_cfg, steps)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        chosen = torch.randperm(len(items))[:per_step].tolist()
        losses = []
        for i in chosen:
            item = items[i]
            idx = torch.randint(item["target"].shape[0], (per_case,))
            target = item["target"][idx].to(device, dtype=torch.float32, non_blocking=True)
            coords = coords_from_indices(idx.to(device), item["shape"], spacing_mm)
            coords.requires_grad_(item["extra"].get("needs_coord_grad", False))
            pred = model.forward(coords, decoder_index=i)
            losses.append(loss_fn(pred, coords, target, **item["extra"].get("loss_kwargs", {})))
        total = torch.stack([l["total"] for l in losses]).mean()
        total.backward()
        optimizer.step()
        scheduler.step()

        if step % log_every_n_steps == 0 or step == steps - 1:
            i = chosen[0]
            item = items[i]
            full = predict_volume(model, item["shape"], spacing_mm, i, eval_chunk_size, device)
            metrics = metric_fn(full, item["target"].float(), item["shape"])
            for key in losses[0]:
                if key != "total":
                    metrics[f"loss/{key}"] = torch.stack([l[key] for l in losses]).mean()
            metrics["loss/total"] = total.detach()
            wandb_log(wandb_run, metrics, step=step)

    return model.encoder_state_dict()
