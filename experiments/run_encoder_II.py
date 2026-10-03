"""
Encoder II training entrypoint (mask SDFs, one channel per label group).

Usage: python -m experiments.run_encoder_II --config config/encoder_II_sdf.yaml

Same loop as Encoder I, with the deliberate differences that keep the
Experiment 1 comparison honest:
  1. Targets are clipped SDFs (sdf/targets.py), one channel per label group.
  2. The loss is the full masked-eikonal-MSE (training/losses.py), applied
     per channel, not plain MSE -- otherwise this wouldn't be training a real
     SDF prior and the comparison against Encoder I would be confounded.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.logging_utils import configure_wandb
from utils.metrics import psnr_3d, ssim_3d, per_label_metrics
from utils.io import save_encoder_weights
from data.dataset import build_dataset
import warnings
from data.msd import (
    load_tasks, resolve_shared_label_groups, resolve_label_groups_per_task,
)
from sdf.coordinates import get_3d_coordinates, mm_per_unit
from sdf.targets import create_multilabel_sdf, sdf_to_channel_masks
from models.interfaces import build_model, print_model_summary
from training.losses import masked_eikonal_sdf_loss
from training.train_loop import train_encoder_decoder, DEFAULT_POINTS_PER_STEP


def _first(x):
    """DataLoader (batch_size=1) wraps string fields in a list."""
    return x[0] if isinstance(x, (list, tuple)) else x


def target_extractor(case, groups_by_task, alpha, spacing_mm, eikonal_lambda):
    label_groups = groups_by_task[_first(case["task"])]         # resolved from dataset.json
    if "label_group" in case:                                    # per_label: one group per item
        label_groups = [label_groups[int(case["label_group"])]]
    label_map = case["mask"].squeeze().cpu().numpy()             # (D, H, W)
    sdf_np = create_multilabel_sdf(label_map, label_groups, spacing_mm, alpha)
    target = torch.from_numpy(sdf_np).reshape(-1, sdf_np.shape[-1]).float()
    return target, {
        "needs_coord_grad": True,   # Eikonal term requires d(pred)/d(coords)
        "loss_kwargs": {"alpha": alpha, "eikonal_lambda": eikonal_lambda,
                        "mm_per_unit": mm_per_unit(label_map.shape, spacing_mm)},
    }


def metric_fn(pred, target, shape, spacing_mm, decode_mode):
    k = target.shape[-1]
    pred_sdf = pred.reshape(*shape, k).cpu().numpy()
    target_sdf = target.reshape(*shape, k).cpu().numpy()
    data_range = 2 * abs(target_sdf).max()

    # Ground-truth masks are recovered from the target SDF itself (sign
    # convention), so no second copy of the label map needs to be threaded through.
    gt_masks = target_sdf < 0.0
    pred_masks = sdf_to_channel_masks(pred_sdf, mode=decode_mode)

    metrics = {
        "psnr": psnr_3d(pred_sdf, target_sdf, data_range=data_range),
        "ssim": ssim_3d(pred_sdf, target_sdf, data_range=data_range, channel_axis=-1),
    }
    metrics.update(per_label_metrics(pred_masks, gt_masks, spacing_mm))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None,
                        help="Override data.root (paths differ between machines)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    assert cfg["target_signal"] == "sdf", (
        "This entrypoint is for Encoder II (SDF) only. "
        "Use run_encoder_I.py for the intensity encoder."
    )

    alpha = cfg["sdf"]["alpha"]
    eikonal_lambda = cfg["sdf"]["eikonal_lambda"]
    decode_mode = cfg["sdf"].get("decode_mode", "independent")
    spacing_mm = tuple(cfg["data"]["spacing_mm"])

    # Label groups are resolved per task from dataset.json (and validated
    # against its labels). The encoder sees only coordinates, so the group
    # count only sets the decoder width. per_label fits each group as its own
    # 1-channel SDF, so tasks with different label sets can share one encoder;
    # otherwise every task must yield the same channel count.
    per_label = cfg["sdf"].get("per_label", True)
    tasks = load_tasks(cfg["data"])
    if per_label:
        groups_by_task = resolve_label_groups_per_task(cfg["sdf"]["label_groups"], tasks)
        n_channels = 1
        if decode_mode == "exclusive":
            warnings.warn("decode_mode 'exclusive' needs several channels; "
                          "per_label fits one group at a time, so using 'independent'.")
            decode_mode = "independent"
    else:
        groups_by_task = resolve_shared_label_groups(cfg["sdf"]["label_groups"], tasks)
        n_channels = len(next(iter(groups_by_task.values())))
    cfg["sdf"]["resolved_label_groups"] = groups_by_task     # saved with the checkpoint

    dataset = build_dataset(cfg["data"], split="train",
                            label_groups_by_task=groups_by_task if per_label else None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = build_model(cfg["model"], out_features=n_channels)
    print_model_summary(model, title=f"Encoder II STRAINER (out_features={n_channels})")
    wandb_run = configure_wandb(cfg["wandb"], cfg["experiment_name"], cfg)

    def coords_fn(case):
        shape = tuple(case["mask"].shape[-3:])
        return get_3d_coordinates(shape, spacing_mm).coords, shape

    def bound_target_extractor(case):
        return target_extractor(case, groups_by_task, alpha, spacing_mm, eikonal_lambda)

    def bound_metric_fn(pred, target, shape):
        return metric_fn(pred, target, shape, spacing_mm, decode_mode)

    encoder_state_dict = train_encoder_decoder(
        dataloader=dataloader,
        model=model,
        coords_fn=coords_fn,
        target_extractor=bound_target_extractor,
        loss_fn=masked_eikonal_sdf_loss,
        metric_fn=bound_metric_fn,
        optimizer_cfg=cfg["optimizer"],
        scheduler_cfg=cfg["scheduler"],
        epoch_count=cfg["training"]["epochs"],
        log_every_n_epochs=cfg["training"]["log_every_n_epochs"],
        wandb_run=wandb_run,
        points_per_step=cfg["training"].get("points_per_step", DEFAULT_POINTS_PER_STEP),
        eval_chunk_size=cfg["training"].get("eval_chunk_size", 2 ** 20),
        device=cfg["training"].get("device", "auto"),
    )

    save_encoder_weights(encoder_state_dict, cfg, out_dir=f"checkpoints/{cfg['experiment_name']}")


if __name__ == "__main__":
    main()
