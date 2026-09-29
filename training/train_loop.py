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
"""
from typing import Callable
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from models.interfaces import EncoderDecoderINR
from utils.logging_utils import log as wandb_log


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
    points_per_step: int = None,      # random voxels per step; None = whole volume
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
    global_step = 0

    for case in dataloader:
        model.reset_decoder()
        # Coordinates are built per case: after isometric resampling, volumes
        # differ in shape (across cases, and certainly across tasks), so a grid
        # built once from the first case would be wrong for the rest.
        coords, shape = coords_fn(case)
        target, extra = target_extractor(case)
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
