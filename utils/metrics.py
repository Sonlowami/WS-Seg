"""
Metrics for Experiment 1.

3D_PSNR / 3D_SSIM are used for both Encoder I and Encoder II (signal
fidelity). Dice/NSD are computed only for the SDF encoder, after
thresholding at zero -- per the methodology's evaluation protocol, and per
the gap flagged in the translation-test plan (PSNR/SSIM alone don't tell you
whether the boundary comes out right).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as sk_ssim


def psnr_3d(pred: np.ndarray, target: np.ndarray, data_range: float = None) -> float:
    if data_range is None:
        data_range = target.max() - target.min()
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def ssim_3d(pred: np.ndarray, target: np.ndarray, data_range: float = None,
            channel_axis: int = None) -> float:
    """channel_axis=-1 for (D, H, W, K) multi-channel volumes; None for (D, H, W)."""
    if data_range is None:
        data_range = target.max() - target.min()
    return float(sk_ssim(pred, target, data_range=data_range, channel_axis=channel_axis))


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return float((2.0 * intersection + eps) / (pred_mask.sum() + gt_mask.sum() + eps))


def _surface_points_mask(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: inside voxels with at least one outside 6-neighbor."""
    mask = mask.astype(bool)
    eroded = np.zeros_like(mask)
    eroded[1:-1, 1:-1, 1:-1] = (
        mask[1:-1, 1:-1, 1:-1]
        & mask[:-2, 1:-1, 1:-1] & mask[2:, 1:-1, 1:-1]
        & mask[1:-1, :-2, 1:-1] & mask[1:-1, 2:, 1:-1]
        & mask[1:-1, 1:-1, :-2] & mask[1:-1, 1:-1, 2:]
    )
    return mask & ~eroded


def normalized_surface_distance(pred_mask: np.ndarray, gt_mask: np.ndarray,
                                 spacing_mm: tuple, tau_mm: float = 1.0) -> float:
    """
    NSD with tolerance tau, as defined in the methodology (Section 4.5).
    Distances are computed in physical mm via `spacing_mm`.
    """
    pred_surface = _surface_points_mask(pred_mask)
    gt_surface = _surface_points_mask(gt_mask)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0

    dist_to_gt_surface = distance_transform_edt(~gt_surface, sampling=spacing_mm)
    dist_to_pred_surface = distance_transform_edt(~pred_surface, sampling=spacing_mm)

    pred_within_tau = (dist_to_gt_surface[pred_surface] <= tau_mm).sum()
    gt_within_tau = (dist_to_pred_surface[gt_surface] <= tau_mm).sum()

    return float((pred_within_tau + gt_within_tau) / (pred_surface.sum() + gt_surface.sum()))


def per_label_metrics(pred_masks: np.ndarray, gt_masks: np.ndarray,
                      spacing_mm: tuple, tau_mm: float = 1.0) -> dict:
    """
    Per-channel Dice and NSD for (D, H, W, K) boolean masks, plus means.

    Channels whose ground truth is empty in this case (e.g. an absent tumour
    sub-region) are reported as NaN and excluded from the means. Otherwise a
    correct all-empty prediction scores Dice = 1 through the epsilon term and
    inflates the average.
    """
    k_channels = gt_masks.shape[-1]
    out, dices, nsds = {}, [], []
    for k in range(k_channels):
        gt_k, pred_k = gt_masks[..., k], pred_masks[..., k]
        if not gt_k.any():
            dice_k, nsd_k = float("nan"), float("nan")
        else:
            dice_k = dice_score(pred_k, gt_k)
            nsd_k = normalized_surface_distance(pred_k, gt_k, spacing_mm, tau_mm)
        out[f"dice_label{k}"] = dice_k
        out[f"nsd_label{k}"] = nsd_k
        dices.append(dice_k)
        nsds.append(nsd_k)
    out["mean_dice"] = float(np.nanmean(dices)) if not np.all(np.isnan(dices)) else float("nan")
    out["mean_nsd"] = float(np.nanmean(nsds)) if not np.all(np.isnan(nsds)) else float("nan")
    return out
