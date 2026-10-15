"""
SDF decoder fitting and evaluation shared by experiments/visualize_sdf_fit.py
and experiments/run_translation_test.py, so a fit is set up, run and scored
the same way in both: same target, point sampling, loss, decoding and Dice.

Two grids are scored:
  resampled  the isometric grid the INR is fitted on, against the resampled
             (nearest-neighbour) ground truth -- what the fit itself achieves
  native     the original label file's grid, against the original annotation
             -- the clinically standard number. The predicted SDF is mapped
             back with linear interpolation and thresholded there, which is
             more faithful than nearest-neighbour on an already-binary mask.
"""
import warnings
import zlib

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import affine_transform

from utils.metrics import per_label_metrics, dice_score, psnr_3d
from sdf.coordinates import mm_per_unit
from sdf.targets import create_multilabel_sdf, build_channel_masks, sdf_to_channel_masks
from models.interfaces import build_model, print_model_summary
from training.losses import masked_eikonal_sdf_loss
from training.train_loop import fit_inr, DEFAULT_POINTS_PER_STEP


# ---------------------------------------------------------------- geometry

def as_affine(x) -> np.ndarray:
    return np.asarray(x.cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float64)


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


# ---------------------------------------------------------------- labels

def group_names(groups: list, labels: dict) -> list:
    return ["+".join(labels[i] for i in g) for g in groups]


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


def oracle_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice of the |gt| lowest-predicted voxels: is the dip in the right place,
    whatever its offset? High here with Dice 0 = right shape, wrong level."""
    n = int(gt.sum())
    if n == 0:
        return float("nan")
    threshold = np.partition(pred.ravel(), n - 1)[n - 1]
    return dice_score(pred <= threshold, gt)


# ---------------------------------------------------------------- case

def case_seed(base: int, case_id: str) -> int:
    """Seed for every fit of a case: the same for all arms of the translation
    test and for visualize_sdf_fit, stable across shards and reruns."""
    return base + zlib.crc32(case_id.encode()) % 1_000_000


def prepare_sdf_case(case: dict, mask_path: str, groups: list, names: list,
                     spacing_mm: tuple, alpha: float, native: bool = True) -> dict:
    """
    Everything a fit and its evaluation need for one transformed case (from
    data.dataset.build_dataset, so case["mask"] is a MetaTensor with affines).
    mask_path is the raw label file (the dataset entry's "mask").
    """
    mask = case["mask"]
    sdf = create_multilabel_sdf(mask[0].cpu().numpy(), groups, spacing_mm, alpha)
    out = {
        "case_id": case["case_id"], "task": case["task"], "mask_path": mask_path,
        "groups": groups, "names": names, "spacing_mm": tuple(spacing_mm), "alpha": alpha,
        "shape": tuple(mask.shape[-3:]), "sdf": sdf, "resampled_affine": as_affine(mask.affine),
        "native": None,
    }
    if native:
        raw = nib.load(mask_path)
        original_affine = as_affine(getattr(mask, "meta", {}).get("original_affine", raw.affine))
        if not np.allclose(original_affine, raw.affine, atol=1e-3):
            warnings.warn(f"{case['case_id']}: MONAI's original_affine differs from the label "
                          f"file's header affine (reader qform/sform choice).")
        out["native"] = {
            "affine": original_affine, "header": raw.header, "shape": raw.shape[:3],
            "spacing_mm": tuple(float(z) for z in raw.header.get_zooms()[:3]),
            "gt": build_channel_masks(np.asarray(raw.dataobj), groups),
        }
    return out


# ---------------------------------------------------------------- fitting

def fit_sdf(cfg: dict, sdf_case: dict, steps: int, encoder_state_dict: dict = None,
            freeze_encoder: bool = False, device="cpu", eval_at=(), on_eval=None,
            seed: int = None, summarize: bool = False, log: bool = False):
    """
    Fit a fresh decoder -- and the encoder too unless freeze_encoder -- to
    sdf_case's clipped SDF for `steps` optimizer steps.

    encoder_state_dict: prior to start from; None = randomly initialized encoder.
    eval_at / on_eval: after each step count in eval_at, on_eval(step,
        pred_sdf (D, H, W, K) numpy, fit_seconds) is called; fit_seconds
        excludes evaluation time. The lr schedule always spans `steps`.
    seed: seeds the decoder init and the point sampling, so fits that differ
        only in encoder_state_dict see identical decoders and batches.
    Returns (pred_sdf after `steps`, final loss dict as floats).
    """
    if seed is not None:
        torch.manual_seed(seed)
    sdf_cfg, train_cfg = cfg["sdf"], cfg.get("training", {})
    shape, spacing = sdf_case["shape"], sdf_case["spacing_mm"]
    k = sdf_case["sdf"].shape[-1]

    model = build_model(cfg["model"], out_features=k).to(device)
    if encoder_state_dict is not None:
        model.load_encoder_state_dict(encoder_state_dict, freeze=freeze_encoder)
    elif freeze_encoder:
        for p in model.encoder_parameters():
            p.requires_grad_(False)
    model.reset_decoder()
    if summarize:
        # torchinfo draws a random input; keep it from shifting the seeded RNG,
        # or the summarized arm would sample different batches from the others.
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        print_model_summary(model, title=f"STRAINER for SDF fit (out_features={k})")
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    target = torch.from_numpy(np.ascontiguousarray(sdf_case["sdf"])).reshape(-1, k).float()
    scale = mm_per_unit(shape, spacing)

    def loss_fn(pred, coords, target_batch):
        return masked_eikonal_sdf_loss(pred, coords, target_batch, sdf_cfg["alpha"],
                                       sdf_cfg["eikonal_lambda"], mm_per_unit=scale)

    def as_volume(pred):
        return pred.reshape(*shape, k).numpy()

    pred, final = fit_inr(
        model, target, shape, spacing, loss_fn, steps, cfg["optimizer"], cfg["scheduler"],
        points_per_step=train_cfg.get("points_per_step", DEFAULT_POINTS_PER_STEP),
        eval_chunk_size=train_cfg.get("eval_chunk_size", 2 ** 20), device=device, eval_at=eval_at,
        on_eval=None if on_eval is None else (lambda step, p, t: on_eval(step, as_volume(p), t)),
        log=log)
    return as_volume(pred), final


# ---------------------------------------------------------------- evaluation

def evaluate_sdf_fit(pred_sdf: np.ndarray, sdf_case: dict, decode_mode: str,
                     keep_masks: bool = False) -> dict:
    """
    Score a predicted (D, H, W, K) SDF. Per label group: Dice, NSD and
    oracle-threshold Dice on the resampled grid, plus Dice/NSD on the native
    grid when sdf_case has native data. Labels absent from the ground truth
    are NaN and excluded from the means (see utils.metrics.per_label_metrics).
    """
    sdf, alpha, spacing = sdf_case["sdf"], sdf_case["alpha"], sdf_case["spacing_mm"]
    gt = sdf < 0.0
    pred_masks = sdf_to_channel_masks(pred_sdf, mode=decode_mode)
    res = per_label_metrics(pred_masks, gt, spacing)

    nat, native_masks = None, None
    native = sdf_case["native"]
    if native is not None:
        native_sdf = np.stack([
            to_original_grid(pred_sdf[..., k].astype(np.float32), sdf_case["resampled_affine"],
                             native["affine"], native["shape"], order=1, cval=alpha)
            for k in range(pred_sdf.shape[-1])
        ], axis=-1)
        native_masks = sdf_to_channel_masks(native_sdf, mode=decode_mode)
        nat = per_label_metrics(native_masks, native["gt"], native["spacing_mm"])

    nan = float("nan")
    labels = []
    for k, name in enumerate(sdf_case["names"]):
        labels.append({
            "label": name,
            "gt_voxels": int(gt[..., k].sum()),
            "dice": res[f"dice_label{k}"], "nsd": res[f"nsd_label{k}"],
            "oracle_dice": oracle_dice(pred_sdf[..., k], gt[..., k]),
            "native_dice": nat[f"dice_label{k}"] if nat else nan,
            "native_nsd": nat[f"nsd_label{k}"] if nat else nan,
        })
    out = {
        "psnr": psnr_3d(pred_sdf, sdf, data_range=2 * alpha),
        "mean_dice": res["mean_dice"], "mean_nsd": res["mean_nsd"],
        "native_mean_dice": nat["mean_dice"] if nat else nan,
        "native_mean_nsd": nat["mean_nsd"] if nat else nan,
        "labels": labels,
    }
    if keep_masks:
        out["pred_masks"], out["native_masks"] = pred_masks, native_masks
    return out
