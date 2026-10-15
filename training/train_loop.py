"""
Encoder training, STRAINER-style: one shared encoder and one decoder head per
training item (case, case/modality or case/label group), all trained
together, every step mixing several cases so the encoder learns what is
shared across the whole set. After training only the encoder is kept.

The same loop trains Encoder I (intensity) and Encoder II (SDF); only the
loss, metrics and target extraction differ, injected by the caller
(experiments/run_encoder_I.py, experiments/run_encoder_II.py).

fit_inr is the single-signal counterpart: fit one model (typically a fresh
decoder on a fixed encoder) to one case. It backs SDF fitting
(training/sdf_fit.py) and per-image decoder training
(experiments/run_decoder_training.py).
"""
import time
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


def build_scheduler(optimizer, scheduler_cfg: dict, total_steps: int):
    if scheduler_cfg["name"] == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    elif scheduler_cfg["name"] == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    raise ValueError(f"Unknown scheduler: {scheduler_cfg['name']}")


def resolve_device(name=None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def fit_inr(model: EncoderDecoderINR, target: torch.Tensor, shape: tuple, spacing_mm: tuple,
            loss_fn: Callable, steps: int, optimizer_cfg: dict, scheduler_cfg: dict,
            points_per_step: int = DEFAULT_POINTS_PER_STEP, eval_chunk_size: int = 2 ** 20,
            device="cpu", eval_at=(), on_eval: Callable = None, log: bool = False):
    """
    Fit head 0 of `model` to one signal for `steps` optimizer steps, each on a
    random batch of points_per_step voxels (None = every voxel).

    target: (N, C) over the grid `shape` (C-order flattened, as coords_from_indices).
    loss_fn(pred, coords, target_batch) -> dict with "total"; coords require
        grad, so losses may differentiate through them (the Eikonal term).
    Only parameters with requires_grad are optimized: freeze the encoder to
    fit a decoder alone. Callers seed torch beforehand for reproducible batches.
    eval_at / on_eval: after each step count in eval_at, on_eval(step,
        prediction (N, C) on CPU, fit_seconds) is called; fit_seconds excludes
        evaluation time. The lr schedule always spans `steps`.
    Returns (full prediction (N, C) on CPU after `steps`, final loss dict as floats).
    """
    optimizer = build_optimizer([p for p in model.parameters() if p.requires_grad], optimizer_cfg)
    scheduler = build_scheduler(optimizer, scheduler_cfg, steps)
    target = target.to(device)
    n = target.shape[0]
    eval_at = set(eval_at)

    def predict():
        return predict_volume(model, shape, spacing_mm, 0, eval_chunk_size, device)

    fit_seconds, last_pred = 0.0, None
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        idx = (torch.arange(n, device=device) if points_per_step is None
               else torch.randint(n, (points_per_step,), device=device))
        coords = coords_from_indices(idx, shape, spacing_mm).requires_grad_(True)
        loss = loss_fn(model.forward(coords), coords, target[idx])
        loss["total"].backward()
        optimizer.step()
        scheduler.step()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        fit_seconds += time.perf_counter() - t0
        if log and (step % max(steps // 10, 1) == 0 or step in (1, steps)):
            print(f"  step {step:>5}: " + " ".join(f"{k}={v.item():.5f}" for k, v in loss.items()))
        if step in eval_at and on_eval is not None:
            last_pred = predict()
            on_eval(step, last_pred, fit_seconds)

    final = {key: float(v.detach()) for key, v in loss.items()}
    return (last_pred if steps in eval_at and last_pred is not None else predict()), final


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
    encoder's state_dict; the heads are dropped.
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
