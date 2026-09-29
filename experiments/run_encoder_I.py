"""
Encoder I training entrypoint (image intensities).

Usage: python -m experiments.run_encoder_I --config config/encoder_I_intensity.yaml
"""
import argparse
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.logging_utils import configure_wandb
from utils.metrics import psnr_3d, ssim_3d
from utils.io import save_encoder_weights
from data.dataset import build_dataset
from data.msd import load_tasks, shared_image_channels
from sdf.coordinates import get_3d_coordinates
from models.interfaces import build_model
from training.losses import image_reconstruction_loss
from training.train_loop import train_encoder_decoder, DEFAULT_POINTS_PER_STEP


def target_extractor(case):
    # (1, C, D, H, W) from the DataLoader -> (D*H*W, C), matching the
    # coordinate ordering from get_3d_coordinates.
    image = case["image"][0]
    target = image.permute(1, 2, 3, 0).reshape(-1, image.shape[0])
    # Encoder I needs no coordinate gradients: intensities carry no
    # Eikonal constraint, unlike Encoder II's SDF targets.
    return target, {"needs_coord_grad": False, "loss_kwargs": {}}


def metric_fn(pred, target, shape, channels):
    full_shape = (*shape, channels)
    pred_vol = pred.reshape(full_shape).cpu().numpy()
    target_vol = target.reshape(full_shape).cpu().numpy()
    return {
        "psnr": psnr_3d(pred_vol, target_vol),
        "ssim": ssim_3d(pred_vol, target_vol, channel_axis=-1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None,
                        help="Override data.root (paths differ between machines)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = build_model(cfg["model"], out_features=channels)
    wandb_run = configure_wandb(cfg["wandb"], cfg["experiment_name"], cfg)

    def coords_fn(case):
        shape = tuple(case["image"].shape[-3:])
        return get_3d_coordinates(shape, spacing_mm).coords, shape

    encoder_state_dict = train_encoder_decoder(
        dataloader=dataloader,
        model=model,
        coords_fn=coords_fn,
        target_extractor=target_extractor,
        loss_fn=lambda pred, coords, target: {
            "total": image_reconstruction_loss(pred, target)
        },
        metric_fn=lambda pred, target, shape: metric_fn(pred, target, shape, channels),
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
