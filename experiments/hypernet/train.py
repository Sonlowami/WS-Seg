"""
Training and evaluation shared by the floor decoder and the hypernetwork.

Both fit SDF decoder weights for many cases at once on a frozen encoder; they
differ only in where a case's weights come from: the floor uses one decoder
for every case, the hypernet predicts each case's from its intensity
decoder. train_weights takes that as params_for(chosen items) and handles the
rest: case sampling, point batches, the masked MSE + Eikonal loss and the
optimizer step, as in training/train_loop.train_encoder_jointly.
"""
import time

import torch

from experiments.hypernet.model import encode, run_head, load_head
from sdf.coordinates import coords_from_indices
from training.losses import masked_eikonal_sdf_loss
from training.sdf_fit import evaluate_sdf_fit
from training.train_loop import predict_volume, stratified_indices


def case_loss(model, params: list, item: dict, n_points: int, sdf_cfg: dict, spacing_mm: tuple,
              device) -> dict:
    """Masked MSE + Eikonal of decoder `params` on the frozen encoder of
    `model`, on n_points random voxels of item's SDF, sdf.band_fraction of
    them from its near-boundary band (training/train_loop.stratified_indices)."""
    idx = stratified_indices(item["target"].shape[0], item["band"], n_points,
                             sdf_cfg.get("band_fraction", 0.0))
    target = item["target"][idx].to(device, dtype=torch.float32, non_blocking=True)
    coords = coords_from_indices(idx.to(device), item["shape"], spacing_mm).requires_grad_(True)
    pred = run_head(model.decoder[0], params, encode(model, coords))
    return masked_eikonal_sdf_loss(pred, coords, target, sdf_cfg["alpha"], sdf_cfg["eikonal_lambda"],
                                   mm_per_unit=item["mm_per_unit"])


def _mean_losses(losses: list) -> dict:
    return {k: torch.stack([l[k] for l in losses]).mean() for k in losses[0]}


def train_weights(model, items: list, params_for, optimizer, scheduler, steps: int,
                  cases_per_step: int, points_per_step: int, sdf_cfg: dict, spacing_mm: tuple,
                  device, log_every: int = 100, on_log=None, monitor_items: list = None,
                  monitor_params_for=None) -> dict:
    """
    steps optimizer steps; each draws cases_per_step random items (dicts with
    target / shape / mm_per_unit, see data.load_sdf_targets), gets their
    decoder weights from params_for(chosen indices) -> [param list per index]
    and averages their losses; points_per_step is split across them.
    Every log_every steps on_log(step, {name: float}) gets the training
    losses and, with monitor_items, the same loss on up to cases_per_step of
    those (monitoring only, no gradient step). Returns the last losses.
    """
    per_step = min(cases_per_step, len(items))
    per_case = max(points_per_step // per_step, 1)
    print(f"  {steps} steps, {per_step} of {len(items)} items x {per_case} points per step")
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        chosen = torch.randperm(len(items))[:per_step].tolist()
        params = params_for(chosen)
        losses = _mean_losses([case_loss(model, p, items[i], per_case, sdf_cfg, spacing_mm, device)
                               for i, p in zip(chosen, params)])
        losses["total"].backward()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step in (1, steps):
            logged = {f"train/{k}": float(v.detach()) for k, v in losses.items()}
            if monitor_items:
                watch = torch.randperm(len(monitor_items))[:per_step].tolist()
                with torch.no_grad():
                    watch_params = monitor_params_for(watch)
                watched = [case_loss(model, p, monitor_items[i], per_case, sdf_cfg, spacing_mm, device)
                           for i, p in zip(watch, watch_params)]
                logged.update({f"test/{k}": float(v.detach()) for k, v in _mean_losses(watched).items()})
            logged["seconds"] = time.perf_counter() - t0
            print(f"  step {step:>6}: " + " ".join(f"{k}={v:.5f}" for k, v in logged.items()
                                                   if k != "seconds") + f" ({logged['seconds']:.0f}s)")
            if on_log is not None:
                on_log(step, logged)
    return {k: float(v.detach()) for k, v in losses.items()}


@torch.no_grad()
def predict_and_score(model, part_params: list, case: dict, spacing_mm: tuple, decode_mode: str,
                      chunk_size: int, device) -> tuple:
    """
    Predict case's full SDF volume with one decoder per label part
    (data.label_targets, in channel order: K 1-channel decoders with
    sdf.per_label, else one K-channel decoder), stack the parts back into K
    channels and score them (training/sdf_fit.evaluate_sdf_fit). Returns
    (evaluation, seconds for the prediction).
    """
    t0 = time.perf_counter()
    preds = []
    for params in part_params:
        load_head(model.decoder[0], params)
        preds.append(predict_volume(model, case["shape"], spacing_mm, 0, chunk_size, device))
    pred = torch.cat(preds, dim=-1)
    seconds = time.perf_counter() - t0
    volume = pred.reshape(*case["shape"], pred.shape[-1]).numpy()
    return evaluate_sdf_fit(volume, case["sdf_case"], decode_mode), seconds
