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

--diagnose separates "corrupt data" from "hard to fit" when masks come out
empty: it checks the labels and SDF targets before fitting, and runs two
control fits on the same grid -- a sphere SDF at the label's centroid, and the
real target cropped to its bounding box. Fit diagnostics (predicted SDF range,
fraction below zero, oracle-threshold Dice) are always printed.
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
from utils.metrics import per_label_metrics, dice_score
from data.dataset import build_dataset
from data.msd import load_tasks, resolve_label_groups
from sdf.coordinates import get_3d_coordinates
from sdf.targets import (
    create_multilabel_sdf, create_mask_sdf_with_clipping, build_channel_masks, sdf_to_channel_masks,
)
from models.interfaces import build_model, print_model_summary
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

def fit_sdf(cfg, encoder_state_dict, coords, target, steps, train_encoder, device, scale,
            summarize=True, log=True):
    """Returns (full-volume prediction on CPU, final loss dict as floats)."""
    sdf_cfg, train_cfg = cfg["sdf"], cfg.get("training", {})
    model = build_model(cfg["model"], out_features=target.shape[-1]).to(device)
    if encoder_state_dict is not None:
        model.load_encoder_state_dict(encoder_state_dict, freeze=not train_encoder)
    model.reset_decoder()
    if summarize:
        print_model_summary(model, title=f"STRAINER for SDF fit (out_features={target.shape[-1]})")

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
        if log and (step % max(steps // 10, 1) == 0 or step == steps - 1):
            print(f"  step {step:>5}: loss={loss['total'].item():.5f} "
                  f"mse={loss['mse'].item():.5f} eikonal={loss['eikonal'].item():.5f}")

    final = {k: float(v.detach()) for k, v in loss.items()}
    return predict_in_chunks(model, coords, train_cfg.get("eval_chunk_size", 2 ** 20)), final


# ---------------------------------------------------------------- diagnostics

def _status(ok: bool) -> str:
    return "PASS" if ok else "WARN"


def oracle_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice of the |gt| lowest-predicted voxels: is the dip in the right place,
    whatever its offset? High here with Dice 0 = right shape, wrong level."""
    n = int(gt.sum())
    if n == 0:
        return float("nan")
    threshold = np.partition(pred.ravel(), n - 1)[n - 1]
    return dice_score(pred <= threshold, gt)


def print_fit_diagnostics(pred_sdf, sdf_np, names, alpha):
    print("fit diagnostics (per channel):")
    for k, name in enumerate(names):
        p, gt = pred_sdf[..., k], sdf_np[..., k] < 0.0
        print(f"  {name}: pred range [{p.min():.3f}, {p.max():.3f}] (target [-{alpha:g}, {alpha:g}]), "
              f"pred<0 {(p < 0).mean():.4%} vs GT {gt.mean():.4%}, "
              f"oracle-threshold dice {oracle_dice(p, gt):.4f}")


def check_data(case, entry, groups, names, sdf_np, spacing_mm, alpha):
    """Checks 1-4: labels survive loading/resampling, grids agree, and the
    target is a valid clipped SDF. Prints PASS/WARN with the numbers."""
    print("\ndata checks:")
    raw_label = nib.load(entry["mask"])
    raw = np.rint(np.asarray(raw_label.dataobj)).astype(np.int64)
    raw_ml = float(np.prod(raw_label.header.get_zooms()[:3])) / 1000
    res_ml = float(np.prod(spacing_mm)) / 1000
    gt = build_channel_masks(case["mask"][0].cpu().numpy(), groups)          # (D, H, W, K)
    for k, (name, g) in enumerate(zip(names, groups)):
        v_raw = np.isin(raw, g).sum() * raw_ml
        v_res = gt[..., k].sum() * res_ml
        ok = v_raw > 0 and v_res > 0 and abs(v_res - v_raw) <= 0.1 * v_raw
        print(f"  [{_status(ok)}] 1. {name} volume: raw {v_raw:.2f} ml, resampled {v_res:.2f} ml")
    print(f"       raw label values {np.unique(raw).tolist()}, raw shape {raw.shape}, "
          f"zooms {tuple(round(float(z), 3) for z in raw_label.header.get_zooms()[:3])}")

    image, mask = case["image"], case["mask"]
    same_shape = tuple(image.shape[-3:]) == tuple(mask.shape[-3:])
    same_affine = np.allclose(_affine(image.affine), _affine(mask.affine), atol=1e-3)
    print(f"  [{_status(same_shape and same_affine)}] 2. resampled image/mask grids: "
          f"shapes {tuple(image.shape[-3:])} vs {tuple(mask.shape[-3:])}, affines equal {same_affine}")
    raw_image = nib.load(entry["image"])
    raw_same = np.allclose(raw_image.affine, raw_label.affine, atol=1e-3) \
        and raw_image.shape[:3] == raw_label.shape[:3]
    print(f"  [{_status(raw_same)}] 2. raw image/label headers agree "
          f"(image {raw_image.shape}, label {raw_label.shape})")

    for k, name in enumerate(names):
        t = sdf_np[..., k]
        signs_ok = int((t < 0).sum()) == int(gt[..., k].sum())
        range_ok = t.min() >= -alpha - 1e-4 and t.max() <= alpha + 1e-4 and t.min() < 0
        band = np.abs(t) < alpha - 1.0                      # away from the clipping kink
        grads = np.linalg.norm(np.stack(np.gradient(t, *spacing_mm)), axis=0)[band]
        med = float(np.median(grads)) if grads.size else float("nan")
        print(f"  [{_status(signs_ok and range_ok and 0.8 < med < 1.2)}] 3. {name} target: "
              f"#(target<0)={int((t < 0).sum())} vs #GT={int(gt[..., k].sum())}, "
              f"range [{t.min():.2f}, {t.max():.2f}], median |grad| in band {med:.3f} /mm (expect ~1)")
        print(f"         4. inside {(t < 0).mean():.4%}, band |t|<alpha {(np.abs(t) < alpha).mean():.4%}, "
              f"constant-predictor MSE {t.var():.4f}")


def sphere_control(gt_masks, spacing_mm, alpha):
    """Per channel: clipped SDF of a sphere at the GT centroid with the GT's
    volume. Same grid, size and imbalance as the real target, trivial shape."""
    shape = gt_masks.shape[:3]
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)
    channels = []
    for k in range(gt_masks.shape[-1]):
        gt = gt_masks[..., k]
        if not gt.any():
            channels.append(np.full(shape, alpha, dtype=np.float32))
            continue
        centroid = np.argwhere(gt).mean(axis=0)
        radius_mm = (3 * gt.sum() * np.prod(spacing_mm) / (4 * np.pi)) ** (1 / 3)
        dist_mm = np.linalg.norm((idx - centroid) * np.asarray(spacing_mm), axis=-1)
        channels.append(create_mask_sdf_with_clipping(dist_mm < radius_mm, spacing_mm, alpha))
    return np.stack(channels, axis=-1)


def crop_to_labels(sdf_np, spacing_mm, alpha):
    """Real target cropped to the union bounding box plus a 2*alpha margin."""
    inside = (sdf_np < 0).any(axis=-1)
    if not inside.any():
        return sdf_np
    lo, hi = np.argwhere(inside).min(axis=0), np.argwhere(inside).max(axis=0) + 1
    margin = np.ceil(2 * alpha / np.asarray(spacing_mm)).astype(int)
    lo, hi = np.maximum(lo - margin, 0), np.minimum(hi + margin, inside.shape)
    return sdf_np[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def run_control(label, target_np, cfg, encoder_state_dict, steps, train_encoder, device,
                spacing_mm, names, decode_mode):
    shape, k = target_np.shape[:3], target_np.shape[-1]
    grid = get_3d_coordinates(shape, spacing_mm)
    target = torch.from_numpy(np.ascontiguousarray(target_np)).reshape(-1, k).float()
    pred, final = fit_sdf(cfg, encoder_state_dict, grid.coords, target, steps, train_encoder,
                          device, grid.mm_per_unit, summarize=False, log=False)
    pred = pred.reshape(*shape, k).numpy()
    gt = target_np < 0.0
    metrics = per_label_metrics(sdf_to_channel_masks(pred, mode=decode_mode), gt, spacing_mm)
    print(f"  {label} on grid {shape}, {steps} steps: final mse {final['mse']:.4f} "
          f"(constant-predictor {target_np.reshape(-1, k).var(axis=0).mean():.4f}), "
          f"eikonal {final['eikonal']:.4f}")
    for i, name in enumerate(names):
        print(f"    {name}: dice {metrics[f'dice_label{i}']:.4f}, "
              f"oracle-threshold dice {oracle_dice(pred[..., i], gt[..., i]):.4f}")


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
    parser.add_argument("--diagnose", action="store_true",
                        help="Check labels/targets and run sphere + crop control fits")
    parser.add_argument("--control_steps", type=int, default=None,
                        help="Steps for each control fit (default: --steps)")
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
    if args.diagnose:
        check_data(case, entry, groups, names, sdf_np, spacing_mm, sdf_cfg["alpha"])

    source = "frozen encoder" if encoder_state_dict is not None and not args.train_encoder \
        else "fine-tuned encoder" if encoder_state_dict is not None else "scratch"
    print(f"fitting {args.steps} steps ({source}) on grid {shape}")
    pred_sdf, _ = fit_sdf(cfg, encoder_state_dict, grid.coords, target, args.steps,
                          args.train_encoder, device, grid.mm_per_unit)
    pred_sdf = pred_sdf.reshape(*shape, len(groups)).numpy()

    decode_mode = sdf_cfg.get("decode_mode", "independent")
    pred_masks = sdf_to_channel_masks(pred_sdf, mode=decode_mode)
    metrics = per_label_metrics(pred_masks, sdf_np < 0.0, spacing_mm)
    for k, name in enumerate(names):
        print(f"  {name}: dice={metrics[f'dice_label{k}']:.4f} nsd={metrics[f'nsd_label{k}']:.4f}")
    print_fit_diagnostics(pred_sdf, sdf_np, names, sdf_cfg["alpha"])

    if args.diagnose:
        steps = args.control_steps or args.steps
        print("\ncontrol fits (same config, grid spacing and device):")
        run_control("5. sphere control", sphere_control(sdf_np < 0.0, spacing_mm, sdf_cfg["alpha"]),
                    cfg, encoder_state_dict, steps, args.train_encoder, device, spacing_mm, names,
                    decode_mode)
        run_control("6. crop control", crop_to_labels(sdf_np, spacing_mm, sdf_cfg["alpha"]),
                    cfg, encoder_state_dict, steps, args.train_encoder, device, spacing_mm, names,
                    decode_mode)
        print("  Reading: sphere fits but real target doesn't -> label shape/size; crop fits but "
              "full volume doesn't -> imbalance/scale; data checks WARN or nothing fits -> data.\n")

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
