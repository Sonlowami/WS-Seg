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
index -> world (original affine) -> resampled voxel index (resampled affine).
The predicted SDF is interpolated linearly and thresholded on the original
grid, so the written masks are exactly the ones the native-grid Dice scores.
Orientation is never changed, so no axis flips or transposes are involved.

Fitting, seeding and scoring are shared with run_translation_test
(training/sdf_fit.py): the same case, settings and encoder give the same
numbers in both scripts.

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

from utils.config import load_config
from utils.io import load_model_weights
from utils.metrics import per_label_metrics
from data.dataset import build_dataset
from data.msd import load_tasks, resolve_label_groups
from sdf.targets import create_mask_sdf_with_clipping, build_channel_masks, sdf_to_channel_masks
from training.train_loop import resolve_device
from training.sdf_fit import (
    as_affine, to_original_grid, group_names, group_label_values, masks_to_label_map, oracle_dice,
    prepare_sdf_case, fit_sdf, evaluate_sdf_fit, case_seed,
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
        # Encoder weights only fit the encoder they were trained with, but the
        # decoder is always freshly initialized, so its depth may come from
        # --config (e.g. a non-linear decoder on a frozen prior).
        ckpt_model, cfg_model = ckpt["config"]["model"], cfg["model"]
        encoder_keys = [k for k in ckpt_model if k != "decoder_layers"]
        differing = [k for k in encoder_keys if cfg_model.get(k, ckpt_model[k]) != ckpt_model[k]]
        if differing:
            warnings.warn(f"--config model keys {differing} differ from the checkpoint's; "
                          f"using the checkpoint's values (they define the encoder).")
        cfg["model"] = {**ckpt_model,
                        "decoder_layers": cfg_model.get("decoder_layers", ckpt_model["decoder_layers"])}
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


# ---------------------------------------------------------------- diagnostics

def _status(ok: bool) -> str:
    return "PASS" if ok else "WARN"


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
    same_affine = np.allclose(as_affine(image.affine), as_affine(mask.affine), atol=1e-3)
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


def run_control(label, target_np, cfg, encoder_state_dict, steps, freeze_encoder, device,
                spacing_mm, names, decode_mode, seed):
    shape, k = target_np.shape[:3], target_np.shape[-1]
    control = {"shape": shape, "sdf": target_np, "spacing_mm": tuple(spacing_mm)}
    pred, final = fit_sdf(cfg, control, steps, encoder_state_dict=encoder_state_dict,
                          freeze_encoder=freeze_encoder, device=device, seed=seed)
    gt = target_np < 0.0
    metrics = per_label_metrics(sdf_to_channel_masks(pred, mode=decode_mode), gt, spacing_mm)
    print(f"  {label} on grid {shape}, {steps} steps: final mse {final['mse']:.4f} "
          f"(constant-predictor {target_np.reshape(-1, k).var(axis=0).mean():.4f}), "
          f"eikonal {final['eikonal']:.4f}")
    for i, name in enumerate(names):
        print(f"    {name}: dice {metrics[f'dice_label{i}']:.4f}, "
              f"oracle-threshold dice {oracle_dice(pred[..., i], gt[..., i]):.4f}")


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
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; per-case seeds match run_translation_test's")
    args = parser.parse_args()

    cfg, encoder_state_dict = resolve_config(args)
    sdf_cfg = cfg["sdf"]
    spacing_mm = tuple(cfg["data"]["spacing_mm"])
    device = resolve_device(cfg.get("training", {}).get("device", "auto"))

    case, entry = load_case(cfg["data"], args.split, args.case_id, args.index)
    task = next(t for t in load_tasks(cfg["data"]) if t.name == case["task"])
    groups = resolve_label_groups(sdf_cfg["label_groups"], task)
    names = group_names(groups, task.labels)
    out_dir = Path(args.out_dir or Path("visualizations") / case["case_id"])
    print(f"case {case['case_id']} ({args.split}), groups {dict(zip(names, groups))}, device {device}")

    # Prediction and target live on the resampled (isometric) grid.
    sdf_case = prepare_sdf_case(case, entry["mask"], groups, names, spacing_mm, sdf_cfg["alpha"])
    sdf_np, shape = sdf_case["sdf"], sdf_case["shape"]
    resampled_affine = sdf_case["resampled_affine"]
    if args.diagnose:
        check_data(case, entry, groups, names, sdf_np, spacing_mm, sdf_cfg["alpha"])

    freeze = encoder_state_dict is not None and not args.train_encoder
    source = "frozen encoder" if freeze else \
        "fine-tuned encoder" if encoder_state_dict is not None else "scratch"
    seed = case_seed(args.seed, case["case_id"])
    print(f"fitting {args.steps} steps ({source}) on grid {shape}, seed {seed}")
    pred_sdf, _ = fit_sdf(cfg, sdf_case, args.steps, encoder_state_dict=encoder_state_dict,
                          freeze_encoder=freeze, device=device, seed=seed, summarize=True, log=True)

    decode_mode = sdf_cfg.get("decode_mode", "independent")
    ev = evaluate_sdf_fit(pred_sdf, sdf_case, decode_mode, keep_masks=True)
    for lab in ev["labels"]:
        print(f"  {lab['label']}: native dice={lab['native_dice']:.4f} nsd={lab['native_nsd']:.4f} | "
              f"resampled dice={lab['dice']:.4f} nsd={lab['nsd']:.4f}")
    print(f"  mean: native dice={ev['native_mean_dice']:.4f} | resampled dice={ev['mean_dice']:.4f}")
    print_fit_diagnostics(pred_sdf, sdf_np, names, sdf_cfg["alpha"])

    if args.diagnose:
        steps = args.control_steps or args.steps
        print("\ncontrol fits (same config, grid spacing and device):")
        run_control("5. sphere control", sphere_control(sdf_np < 0.0, spacing_mm, sdf_cfg["alpha"]),
                    cfg, encoder_state_dict, steps, freeze, device, spacing_mm, names,
                    decode_mode, seed)
        run_control("6. crop control", crop_to_labels(sdf_np, spacing_mm, sdf_cfg["alpha"]),
                    cfg, encoder_state_dict, steps, freeze, device, spacing_mm, names,
                    decode_mode, seed)
        print("  Reading: sphere fits but real target doesn't -> label shape/size; crop fits but "
              "full volume doesn't -> imbalance/scale; data checks WARN or nothing fits -> data.\n")

    values = group_label_values(groups)
    per_group = any(len(g) > 1 for g in groups)

    # Resampled grid: everything shares the affine MONAI computed in Spacingd.
    image = case["image"].cpu().numpy()                       # (C, D, H, W)
    image = image[0] if image.shape[0] == 1 else np.moveaxis(image, 0, -1)
    save_nifti(image.astype(np.float32), resampled_affine, out_dir / "resampled" / "image.nii.gz")
    save_nifti(masks_to_label_map(sdf_np < 0.0, values), resampled_affine,
               out_dir / "resampled" / "gt_labels.nii.gz")
    save_nifti(masks_to_label_map(ev["pred_masks"], values), resampled_affine,
               out_dir / "resampled" / "pred_labels.nii.gz")
    if args.save_sdf:
        for k, name in enumerate(names):
            save_nifti(pred_sdf[..., k].astype(np.float32), resampled_affine,
                       out_dir / "resampled" / f"pred_sdf_{name}.nii.gz")

    # Original grid: the masks the native Dice scored, written with the source
    # label file's header so they overlay the raw MSD files exactly.
    native = sdf_case["native"]
    save_nifti(masks_to_label_map(ev["native_masks"], values), None, out_dir / "pred_labels.nii.gz",
               header=native["header"])
    if per_group:
        for k, name in enumerate(names):
            save_nifti(ev["native_masks"][..., k].astype(np.uint8), None,
                       out_dir / f"pred_{name}.nii.gz", header=native["header"])
    if args.save_sdf:
        for k, name in enumerate(names):
            native_sdf = to_original_grid(pred_sdf[..., k].astype(np.float32), resampled_affine,
                                          native["affine"], native["shape"], order=1,
                                          cval=sdf_cfg["alpha"])
            save_nifti(native_sdf, None, out_dir / f"pred_sdf_{name}.nii.gz", header=native["header"])

    print(f"\nOverlay {out_dir / 'pred_labels.nii.gz'} on\n  {entry['image']}\n  {entry['mask']}")


if __name__ == "__main__":
    main()
