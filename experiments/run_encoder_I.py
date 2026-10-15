"""
Encoder I training entrypoint (image intensities).

Usage: python -m experiments.run_encoder_I --config config/encoder_I_intensity.yaml
"""
import argparse
from torch.utils.data import DataLoader
from utils.metrics import metric_fn
from utils.config import load_config
from utils.logging_utils import configure_wandb

from utils.io import save_encoder_weights
from data.dataset import build_dataset
from data.msd import load_tasks, shared_image_channels
from models.interfaces import build_model, print_model_summary
from training.losses import image_reconstruction_loss
from training.train_loop import train_encoder_jointly, prepare_items, DEFAULT_POINTS_PER_STEP


def target_extractor(case):
    # (1, C, D, H, W) from the DataLoader -> (D*H*W, C), matching the
    # coordinate ordering from sdf.coordinates.
    image = case["image"][0]
    target = image.permute(1, 2, 3, 0).reshape(-1, image.shape[0])
    # Encoder I needs no coordinate gradients: intensities carry no
    # Eikonal constraint, unlike Encoder II's SDF targets.
    return target, {"needs_coord_grad": False, "loss_kwargs": {}}


def train_encoder_I(cfg: dict, wandb_kwargs: dict = None) -> dict:
    """
    Train Encoder I (image intensities), STRAINER-style (training/train_loop.py).
    cfg: a full training config (experiment_name names the wandb run).
    wandb_kwargs: extra wandb.init arguments, e.g. group for a sweep.
    Returns the trained encoder's state_dict; the wandb run is finished.
    """
    assert cfg["target_signal"] == "intensity", (
        "This entrypoint is for Encoder I (intensity) only. "
        "Use run_encoder_II.py for the SDF encoder."
    )

    spacing_mm = tuple(cfg["data"]["spacing_mm"])
    # The encoder sees only coordinates; image channels set only the decoder
    # width. per_modality fits each modality as its own 1-channel image, so
    # tasks with different modalities (CT vs [T2, ADC]) can share one encoder.
    per_modality = cfg["data"].get("per_modality", True)
    if per_modality:
        channels = 1
    else:
        channels = shared_image_channels(load_tasks(cfg["data"]))   # derived from dataset.json

    dataset = build_dataset(cfg["data"], split="train", per_modality=per_modality)
    train_cfg = cfg["training"]
    if "steps" not in train_cfg:
        raise SystemExit("training.steps (total optimizer steps) is required")
    loss_fn = lambda pred, coords, target: {"total": image_reconstruction_loss(pred, target)}
    bound_metric_fn = lambda pred, target, shape: metric_fn(pred, target, shape, channels)
    # STRAINER: one decoder head per training item, all trained together.
    items = prepare_items(DataLoader(dataset, batch_size=1, shuffle=False), target_extractor,
                          shape_fn=lambda case: case["image"].shape[-3:])
    model = build_model(cfg["model"], out_features=channels, num_decoders=len(items))
    print_model_summary(model, title=f"Encoder I STRAINER, {len(items)} heads (out_features={channels})")
    wandb_run = configure_wandb(cfg["wandb"], cfg["experiment_name"], cfg, **(wandb_kwargs or {}))
    encoder_state_dict = train_encoder_jointly(
        items=items, model=model, loss_fn=loss_fn, metric_fn=bound_metric_fn,
        optimizer_cfg=cfg["optimizer"], scheduler_cfg=cfg["scheduler"], steps=train_cfg["steps"],
        log_every_n_steps=train_cfg["log_every_n_epochs"], wandb_run=wandb_run, spacing_mm=spacing_mm,
        cases_per_step=train_cfg.get("cases_per_step", 8),
        points_per_step=train_cfg.get("points_per_step", DEFAULT_POINTS_PER_STEP),
        eval_chunk_size=train_cfg.get("eval_chunk_size", 2 ** 20),
        device=train_cfg.get("device", "auto"),
    )

    if wandb_run is not None:
        wandb_run.finish()
    return encoder_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None,
                        help="Override data.root (paths differ between machines)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    encoder_state_dict = train_encoder_I(cfg)
    save_encoder_weights(encoder_state_dict, cfg, out_dir=f"checkpoints/{cfg['experiment_name']}")


if __name__ == "__main__":
    main()
