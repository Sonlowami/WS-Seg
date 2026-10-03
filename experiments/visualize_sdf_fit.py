"""
Fit an SDF decoder to one case and write the result as NIfTI for viewing.

A fresh decoder is fitted to the case's multi-label clipped SDF (all label
groups jointly, as in the translation test), either on a trained encoder
(--encoder_ckpt, frozen unless --train_encoder) or from scratch. The decoded
mask is written twice:

  <out_dir>/pred_labels.nii.gz             original grid of the MSD label file,
  <out_dir>/pred_<group>.nii.gz            with that file's own header, so it
                                           overlays directly on imagesTr/labelsTr
  <out_dir>/resampled/{image,gt_labels,pred_labels}.nii.gz
                                           the isometric training grid, with
                                           the affine MONAI's Spacingd produced

The prediction lives on the resampled grid (that is what the INR is fitted
on). It is mapped back with the inverse of the two affines: original voxel
index -> world (original affine) -> resampled voxel index (resampled affine),
nearest-neighbour for labels, linear for SDFs. Orientation is never changed,
so no axis flips or transposes are involved.

Label values: a group that is a single label keeps that label's id; any other
group (e.g. [["foreground"]] or nested BraTS regions) gets value k + 1 and is
also written as its own binary pred_<group>.nii.gz, since nested groups
overlap and one label map can only show the innermost.

Usage:
  python -m experiments.visualize_sdf_fit --config config/encoder_II_sdf.yaml \\
      --case_id Task03_Liver/liver_3 --steps 1000
  python -m experiments.visualize_sdf_fit --encoder_ckpt checkpoints/encoder_II_sdf \\
      --split test --index 0 --steps 200 --save_sdf
"""
import argparse
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import affine_transform

from utils.config import load_config
from utils.io import load_model_weights
from utils.metrics import per_label_metrics
from data.dataset import build_dataset
from data.msd import load_tasks, resolve_label_groups
from sdf.coordinates import get_3d_coordinates
from sdf.targets import create_multilabel_sdf, sdf_to_channel_masks
from models.interfaces import build_model
from training.losses import masked_eikonal_sdf_loss
from training.train_loop import (
    build_optimizer, build_scheduler, resolve_device, sample_points, predict_in_chunks,
    DEFAULT_POINTS_PER_STEP,
)


# ---------------------------------------------------------------- config / data

def resolve_config(args) -> tuple:
    """Returns (cfg, encoder_state_dict or None)."""
    if not args.config and not args.encoder_ckpt:
        raise SystemExit("Give --config, --encoder_ckpt, or both.")
    encoder_state_dict = None
    if args.encoder_ckpt:
        ckpt = load_model_weights(args.encoder_ckpt)
        encoder_state_dict = ckpt["encoder_state_dict"]
    cfg = load_config(args.config) if args.config else ckpt["config"]
    if args.encoder_ckpt and args.config:
        # Encoder weights only fit the architecture they were trained with.
        if cfg["model"] != ckpt["config"]["model"]:
            warnings.warn("--config model block differs from the checkpoint's; "
                          "using the checkpoint's architecture.")
        cfg["model"] = ckpt["config"]["model"]
    if "sdf" not in cfg:
        raise SystemExit("Config has no sdf: block (alpha, eikonal_lambda, label_groups); "
                         "pass --config with one.")
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    return cfg, encoder_state_dict


def load_case(data_cfg: dict, split: str, case_id: str, index: int):
    """Transformed case (MetaTensors, so affines are kept) plus its raw entry."""
    dataset = build_dataset(data_cfg, split=split)
    ids = [e["case_id"] for e in dataset.data]
    if case_id is not None:
        if case_id not in ids:
            shown = ", ".join(ids[:5])
            raise SystemExit(f"{case_id!r} is not in the {split} split ({len(ids)} cases, "
                             f"e.g. {shown}).")
        index = ids.index(case_id)
    if not 0 <= index < len(ids):
        raise SystemExit(f"--index {index} is out of range for the {split} split ({len(ids)} cases).")
    return dataset[index], dataset.data[index]


# ---------------------------------------------------------------- fitting

def fit_sdf(cfg, encoder_state_dict, coords, target, steps, train_encoder, device, scale):
    sdf_cfg, train_cfg = cfg["sdf"], cfg.get("training", {})
    model = build_model(cfg["model"], out_features=target.shape[-1]).to(device)
    if encoder_state_dict is not None:
        model.load_encoder_state_dict(encoder_state_dict, freeze=not train_encoder)
    model.reset_decoder()

    params = list(model.decoder_parameters())
    if encoder_state_dict is None or train_encoder:
        params += list(model.encoder_parameters())
    optimizer = build_optimizer(params, cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], steps)
    points = train_cfg.get("points_per_step", DEFAULT_POINTS_PER_STEP)

    coords, target = coords.to(device), target.to(device)
    for step in range(steps):
        optimizer.zero_grad()
        coords_batch, target_batch = sample_points(coords, target, points, requires_grad=True)
        pred = model.forward(coords_batch)
        loss = masked_eikonal_sdf_loss(pred, coords_batch, target_batch,
                                       sdf_cfg["alpha"], sdf_cfg["eikonal_lambda"],
                                       mm_per_unit=scale)
        loss["total"].backward()
        optimizer.step()
        scheduler.step()
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            print(f"  step {step:>5}: loss={loss['total'].item():.5f} "
                  f"mse={loss['mse'].item():.5f} eikonal={loss['eikonal'].item():.5f}")

    return predict_in_chunks(model, coords, train_cfg.get("eval_chunk_size", 2 ** 20))


# ---------------------------------------------------------------- labels / resampling

def group_label_values(groups: list) -> list:
    """Single-label groups keep their label id; any other group gets k + 1."""
    if all(len(g) == 1 for g in groups):
        return [g[0] for g in groups]
    return list(range(1, len(groups) + 1))


def masks_to_label_map(masks: np.ndarray, values: list) -> np.ndarray:
    """(D, H, W, K) bool -> (D, H, W) uint8. Later groups paint over earlier
    ones, so nested groups listed outer-to-inner show their innermost region."""
    label_map = np.zeros(masks.shape[:3], dtype=np.uint8)
    for k, value in enumerate(values):
        label_map[masks[..., k]] = value
    return label_map


def to_original_grid(volume: np.ndarray, resampled_affine: np.ndarray,
                     original_affine: np.ndarray, original_shape: tuple,
                     order: int, cval: float = 0.0) -> np.ndarray:
    """
    Pull `volume` (on the resampled grid) onto the original voxel grid.
    For each original voxel o: world = A_orig @ o, resampled index =
    inv(A_res) @ world. affine_transform samples input at matrix @ o + offset.
    """
    m = np.linalg.inv(resampled_affine) @ original_affine
    return affine_transform(volume, m[:3, :3], offset=m[:3, 3], output_shape=original_shape,
                            order=order, mode="constant", cval=cval)


def _affine(x) -> np.ndarray:
    return np.asarray(x.cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float64)


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path, header=None):
    """With `header` (an original file's), keep its orientation codes and
    sform/qform as they are; otherwise mark the affine as scanner space."""
    if header is not None:
        img = nib.Nifti1Image(data, header.get_best_affine(), header=header.copy())
    else:
        img = nib.Nifti1Image(data, affine)
        img.set_sform(affine, code=1)
        img.set_qform(affine, code=1)
    img.set_data_dtype(data.dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    print(f"  wrote {path}")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="Training config (sdf/optimizer/data settings)")
    parser.add_argument("--encoder_ckpt", default=None,
                        help="Checkpoint dir (encoder.pt + config.json); omit to fit from scratch")
    parser.add_argument("--train_encoder", action="store_true",
                        help="Fine-tune the checkpoint encoder too (default: frozen)")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--case_id", default=None, help="e.g. Task03_Liver/liver_3")
    parser.add_argument("--index", type=int, default=0, help="Case index in the split if no --case_id")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out_dir", default=None, help="Default: visualizations/<case_id>")
    parser.add_argument("--save_sdf", action="store_true", help="Also write predicted SDFs")
    args = parser.parse_args()

    cfg, encoder_state_dict = resolve_config(args)
    sdf_cfg = cfg["sdf"]
    spacing_mm = tuple(cfg["data"]["spacing_mm"])
    device = resolve_device(cfg.get("training", {}).get("device", "auto"))

    case, entry = load_case(cfg["data"], args.split, args.case_id, args.index)
    task = next(t for t in load_tasks(cfg["data"]) if t.name == case["task"])
    groups = resolve_label_groups(sdf_cfg["label_groups"], task)
    names = ["+".join(task.labels[i] for i in g) for g in groups]
    out_dir = Path(args.out_dir or Path("visualizations") / case["case_id"])
    print(f"case {case['case_id']} ({args.split}), groups {dict(zip(names, groups))}, device {device}")

    # Prediction and target live on the resampled (isometric) grid.
    mask = case["mask"]
    shape = tuple(mask.shape[-3:])
    resampled_affine = _affine(mask.affine)
    label_map = mask[0].cpu().numpy()
    sdf_np = create_multilabel_sdf(label_map, groups, spacing_mm, sdf_cfg["alpha"])
    target = torch.from_numpy(sdf_np).reshape(-1, len(groups)).float()
    grid = get_3d_coordinates(shape, spacing_mm)

    source = "frozen encoder" if encoder_state_dict is not None and not args.train_encoder \
        else "fine-tuned encoder" if encoder_state_dict is not None else "scratch"
    print(f"fitting {args.steps} steps ({source}) on grid {shape}")
    pred_sdf = fit_sdf(cfg, encoder_state_dict, grid.coords, target, args.steps,
                       args.train_encoder, device, grid.mm_per_unit
                       ).reshape(*shape, len(groups)).numpy()

    decode_mode = sdf_cfg.get("decode_mode", "independent")
    pred_masks = sdf_to_channel_masks(pred_sdf, mode=decode_mode)
    metrics = per_label_metrics(pred_masks, sdf_np < 0.0, spacing_mm)
    for k, name in enumerate(names):
        print(f"  {name}: dice={metrics[f'dice_label{k}']:.4f} nsd={metrics[f'nsd_label{k}']:.4f}")

    values = group_label_values(groups)
    pred_labels = masks_to_label_map(pred_masks, values)
    gt_labels = masks_to_label_map(sdf_np < 0.0, values)
    per_group = any(len(g) > 1 for g in groups)

    # Resampled grid: everything shares the affine MONAI computed in Spacingd.
    image = case["image"].cpu().numpy()                       # (C, D, H, W)
    image = image[0] if image.shape[0] == 1 else np.moveaxis(image, 0, -1)
    save_nifti(image.astype(np.float32), resampled_affine, out_dir / "resampled" / "image.nii.gz")
    save_nifti(gt_labels, resampled_affine, out_dir / "resampled" / "gt_labels.nii.gz")
    save_nifti(pred_labels, resampled_affine, out_dir / "resampled" / "pred_labels.nii.gz")
    if args.save_sdf:
        for k, name in enumerate(names):
            save_nifti(pred_sdf[..., k].astype(np.float32), resampled_affine,
                       out_dir / "resampled" / f"pred_sdf_{name}.nii.gz")

    # Original grid: map back through the affines and reuse the source label
    # file's header, so the result overlays the raw MSD files exactly.
    source_label = nib.load(entry["mask"])
    original_shape = source_label.shape[:3]
    original_affine = _affine(mask.meta.get("original_affine", source_label.affine))
    if not np.allclose(original_affine, source_label.affine, atol=1e-3):
        warnings.warn("MONAI's original_affine differs from the label file's header affine "
                      "(reader qform/sform choice); outputs follow the header.")

    def back(volume, order, cval=0.0):
        return to_original_grid(volume, resampled_affine, original_affine, original_shape,
                                order=order, cval=cval)

    save_nifti(back(pred_labels, order=0), None, out_dir / "pred_labels.nii.gz",
               header=source_label.header)
    if per_group:
        for k, name in enumerate(names):
            save_nifti(back(pred_masks[..., k].astype(np.uint8), order=0), None,
                       out_dir / f"pred_{name}.nii.gz", header=source_label.header)
    if args.save_sdf:
        for k, name in enumerate(names):
            save_nifti(back(pred_sdf[..., k].astype(np.float32), order=1, cval=sdf_cfg["alpha"]),
                       None, out_dir / f"pred_sdf_{name}.nii.gz", header=source_label.header)

    print(f"\nOverlay {out_dir / 'pred_labels.nii.gz'} on\n  {entry['image']}\n  {entry['mask']}")


if __name__ == "__main__":
    main()
