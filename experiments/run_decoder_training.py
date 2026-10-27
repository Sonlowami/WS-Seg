"""
Per-image decoder training on a pretrained encoder.

For every image of the chosen split (every case/modality pair when
data.per_modality is set) a fresh decoder is fitted to the image intensities
on top of the frozen encoder from --encoder_ckpt, scored on the full volume
(PSNR/SSIM) and saved with utils.io.save_decoder_weights.

Outputs in --out_dir:
  decoder_<case_id>.pt   one decoder state_dict per image ("/" in ids -> "__")
  decoders.csv           case_id, task, modality, decoder file, steps, fit
                         seconds, final loss, PSNR, SSIM -- one row per image
  config.json            the full config used, with the encoder checkpoint
                         path: everything needed to rebuild the model that a
                         decoder belongs to (build_model(config["model"]) +
                         load the encoder + load the decoder)

Settings come from the config's dec_training block (all optional):
  steps            decoder-fitting steps per image (default 4000)
  split            which images (default "train")
  optimizer, scheduler, points_per_step
                   default to the config's optimizer / scheduler / training
The encoder's architecture always comes from its checkpoint; the decoder's
depth and width (model.decoder_layers, model.decoder_hidden_dim -- default the
encoder's hidden_dim, can be lower for a smaller decoder) come from --config
or --decoder_layers / --decoder_hidden_dim. Each decoder size gets its own
default --out_dir (<encoder_ckpt>_decoders_d<width>_L<depth>) and wandb run;
WandB gets every image's metrics plus the decoder size, mean/median PSNR and
SSIM, and a table of all images.

Usage:
  python -m experiments.run_decoder_training --config config/encoder_I_intensity.yaml \\
      --encoder_ckpt checkpoints/encoder_I_intensity [--out_dir ...] [--resume]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.logging_utils import configure_wandb
from utils.metrics import metric_fn
from utils.io import load_model_weights, save_decoder_weights
from utils.results_io import RowWriter, read_rows
from data.dataset import build_dataset
from data.msd import load_tasks, shared_image_channels
from experiments.run_encoder_I import target_extractor
from models.interfaces import build_model, merge_model_config, print_model_summary
from training.losses import image_reconstruction_loss
from training.sdf_fit import case_seed
from training.train_loop import fit_inr, resolve_device, DEFAULT_POINTS_PER_STEP

FIELDS = ["case_id", "task", "modality", "decoder_file", "steps", "fit_seconds", "loss",
          "psnr", "ssim"]


def _first(x):
    """DataLoader (batch_size=1) wraps string fields in a list."""
    return x[0] if isinstance(x, (list, tuple)) else x


def decoder_size(model_cfg: dict) -> tuple:
    """(hidden width, depth) of the decoder a model config builds."""
    return model_cfg.get("decoder_hidden_dim") or model_cfg["hidden_dim"], model_cfg["decoder_layers"]


def log_summary(wandb_run, manifest: Path, decoder_info: dict):
    """Mean/median PSNR and SSIM over every image in decoders.csv (including
    ones from earlier --resume runs), printed and sent to wandb."""
    rows = read_rows([manifest], float_fields=("psnr", "ssim", "loss", "fit_seconds"),
                     fieldnames=FIELDS)
    if not rows:
        return
    summary = {"n_images": len(rows), **decoder_info}
    for metric in ("psnr", "ssim"):
        values = [r[metric] for r in rows]
        summary[f"mean_{metric}"] = float(np.nanmean(values))
        summary[f"median_{metric}"] = float(np.nanmedian(values))
    print("\nsummary: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in summary.items()))
    if wandb_run is not None:
        import wandb
        wandb_run.summary.update(summary)
        wandb_run.log({"decoders": wandb.Table(columns=FIELDS,
                                               data=[[r[c] for c in FIELDS] for r in rows])})


def train_decoders(cfg: dict, encoder_state_dict: dict, out_dir, wandb_kwargs: dict = None,
                   resume: bool = False, seed: int = 0) -> list:
    """
    Fit and save one decoder per image on the frozen encoder_state_dict.
    cfg["model"] must describe that encoder (see merge_model_config).
    resume skips images already listed in <out_dir>/decoders.csv.
    Returns the decoders.csv rows written by this call.
    """
    # Read the config to do data processing
    data_cfg = cfg["data"]
    per_modality = data_cfg.get("per_modality", True)
    if per_modality:
        channels = 1
    else:
        channels = shared_image_channels(load_tasks(cfg["data"]))
    spacing_mm = tuple(data_cfg["spacing_mm"])
    dec_cfg, train_cfg = cfg.get("dec_training") or {}, cfg.get("training") or {}
    steps = dec_cfg.get("steps", 4000)
    optimizer_cfg = dec_cfg.get("optimizer", cfg["optimizer"])
    scheduler_cfg = dec_cfg.get("scheduler", cfg["scheduler"])
    points = dec_cfg.get("points_per_step", train_cfg.get("points_per_step", DEFAULT_POINTS_PER_STEP))
    chunk = train_cfg.get("eval_chunk_size", 2 ** 20)
    device = resolve_device(train_cfg.get("device", "auto"))

    # Build the dataset; images are loaded one at a time, so no volume is
    # held in memory beyond its own fit.
    split = dec_cfg.get("split", "train")
    dataset = build_dataset(data_cfg, split=split, per_modality=per_modality)
    loss_fn = lambda pred, coords, target: {"total": image_reconstruction_loss(pred, target)}

    out_dir = Path(out_dir)
    manifest = out_dir / "decoders.csv"
    done = {r["case_id"] for r in read_rows([manifest], fieldnames=FIELDS)} \
        if resume and manifest.exists() else set()
    writer = RowWriter(manifest, FIELDS, resume)

    # The architecture every decoder belongs to (printed once, before any
    # seeded fit, so the summary's random input never shifts a fit's batches).
    template = build_model(cfg["model"], out_features=channels)
    template.load_encoder_state_dict(encoder_state_dict, freeze=True)
    print_model_summary(template, title=f"STRAINER for decoder training (out_features={channels})")
    width, depth = decoder_size(cfg["model"])
    decoder_info = {"decoder_hidden_dim": width, "decoder_layers": depth,
                    "decoder_params": sum(p.numel() for p in template.decoder_parameters())}
    del template

    print(f"{len(dataset)} {split} images ({len(done)} already done), {steps} steps each, "
          f"frozen encoder, decoder {width} wide x {depth} layers "
          f"({decoder_info['decoder_params']:,} params), device {device}, decoders -> {out_dir}")
    wandb_run = configure_wandb(cfg["wandb"], f"{cfg['experiment_name']}_decoders_d{width}_L{depth}",
                                {**cfg, "decoder": decoder_info}, **(wandb_kwargs or {}))
    rows = []
    for n, case in enumerate(DataLoader(dataset, batch_size=1, shuffle=False), 1):
        case_id = _first(case["case_id"])
        if case_id in done:
            continue
        target, _ = target_extractor(case)
        if hasattr(target, "as_tensor"):                 # MONAI MetaTensor
            target = target.as_tensor()
        shape = tuple(case["image"].shape[-3:])

        # Instantiate a model: the pretrained encoder, frozen, and a fresh decoder.
        torch.manual_seed(case_seed(seed, case_id))
        model = build_model(cfg["model"], out_features=channels).to(device)
        model.load_encoder_state_dict(encoder_state_dict, freeze=True)
        model.reset_decoder()

        # Fit the decoder to this image.
        timing = {}
        pred, final = fit_inr(model, target, shape, spacing_mm, loss_fn, steps, optimizer_cfg,
                              scheduler_cfg, points_per_step=points, eval_chunk_size=chunk,
                              device=device, eval_at={steps},
                              on_eval=lambda step, p, seconds: timing.update(seconds=seconds))
        metrics = metric_fn(pred, target.float(), shape, channels)

        # Save the decoder state dictionary.
        path = save_decoder_weights(model.decoder_state_dict(), case_id, out_dir)
        row = {"case_id": case_id, "task": _first(case["task"]),
               "modality": _first(case.get("modality", "")), "decoder_file": path.name,
               "steps": steps, "fit_seconds": round(timing["seconds"], 2), "loss": final["total"],
               "psnr": metrics["psnr"], "ssim": metrics["ssim"]}
        writer.write([row])
        rows.append(row)
        print(f"[{n}/{len(dataset)}] {case_id}: psnr={row['psnr']:.2f} ssim={row['ssim']:.4f} "
              f"loss={row['loss']:.5f} ({row['fit_seconds']:.0f}s) -> {path.name}")
        if wandb_run is not None:
            wandb_run.log({"case_id": case_id, "psnr": row["psnr"], "ssim": row["ssim"],
                           "loss": row["loss"], "fit_seconds": row["fit_seconds"]})

    log_summary(wandb_run, manifest, decoder_info)
    if wandb_run is not None:
        wandb_run.finish()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Data, dec_training, optimizer, wandb settings")
    parser.add_argument("--encoder_ckpt", required=True, help="Checkpoint dir (encoder.pt + config.json)")
    parser.add_argument("--out_dir", default=None,
                        help="Default: <encoder_ckpt>_decoders_d<width>_L<depth> next to the checkpoint")
    parser.add_argument("--decoder_hidden_dim", type=int, default=None,
                        help="Override model.decoder_hidden_dim (default: the encoder's hidden_dim)")
    parser.add_argument("--decoder_layers", type=int, default=None,
                        help="Override model.decoder_layers")
    parser.add_argument("--split", default=None, help="Override dec_training.split")
    parser.add_argument("--steps", type=int, default=None, help="Override dec_training.steps")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already listed in <out_dir>/decoders.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    ckpt = load_model_weights(args.encoder_ckpt)
    cfg_model = dict(cfg.get("model") or {})
    for key in ("decoder_hidden_dim", "decoder_layers"):
        if getattr(args, key) is not None:
            cfg_model[key] = getattr(args, key)
    cfg["model"] = merge_model_config(ckpt["config"]["model"], cfg_model)
    cfg["dec_training"] = dict(cfg.get("dec_training") or {})
    for key in ("split", "steps"):
        if getattr(args, key) is not None:
            cfg["dec_training"][key] = getattr(args, key)

    ckpt_dir = Path(args.encoder_ckpt)
    width, depth = decoder_size(cfg["model"])
    out_dir = Path(args.out_dir) if args.out_dir else \
        ckpt_dir.with_name(f"{ckpt_dir.name}_decoders_d{width}_L{depth}")
    saved = out_dir / "config.json"
    if args.resume and saved.exists():
        previous = json.loads(saved.read_text())
        if previous.get("model") != cfg["model"] or previous.get("encoder_ckpt") != str(ckpt_dir):
            raise SystemExit(f"{out_dir} holds decoders for encoder {previous.get('encoder_ckpt')} "
                             f"with model {previous.get('model')}, not {ckpt_dir} with {cfg['model']}; "
                             f"resuming would mix decoders. Choose another --out_dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(saved, "w") as f:
        json.dump({"encoder_ckpt": str(ckpt_dir), **cfg}, f, indent=2)
    train_decoders(cfg, ckpt["encoder_state_dict"], out_dir, resume=args.resume, seed=args.seed)


if __name__ == "__main__":
    main()
