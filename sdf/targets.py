"""
Signed distance function target construction from label maps.

Sign convention: negative inside the region, positive outside, zero at the
boundary. Distances are computed in physical units (mm) via `sampling` in
scipy's distance transform, so the Eikonal property (|grad S| = 1) refers to
true physical distance.

Multi-label handling
--------------------
Each output channel k is defined by a *label group*: a list of label IDs whose
union forms that channel's region. This one mechanism covers both cases:

  * Mutually exclusive labels (e.g. liver + tumour as separate classes):
        label_groups = [[1], [2]]
  * BraTS-style nested regions (whole tumour / tumour core / enhancing):
        label_groups = [[1, 2, 3], [1, 3], [3]]

Every channel is an independent scalar SDF, so channels may overlap or nest
freely -- nothing constrains them to partition the volume.
"""
from typing import Sequence
import numpy as np
from scipy.ndimage import distance_transform_edt


def create_mask_sdf_with_clipping(mask: np.ndarray, spacing_mm: tuple,
                                   alpha: float) -> np.ndarray:
    """
    mask: binary array (D, H, W), 1 = inside region, 0 = outside.
    Returns float32 (D, H, W) with values in [-alpha, alpha].

    Degenerate masks are handled explicitly, because the distance transform
    of an empty (or full) mask has no meaningful surface to measure from:
      * empty mask -> whole channel is plateau at +alpha
      * full mask  -> whole channel is plateau at -alpha
    """
    mask = mask.astype(bool)
    if not mask.any():
        return np.full(mask.shape, alpha, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -alpha, dtype=np.float32)

    dist_outside = distance_transform_edt(~mask, sampling=spacing_mm)
    dist_inside = distance_transform_edt(mask, sampling=spacing_mm)

    sdf = np.where(mask, -dist_inside, dist_outside).astype(np.float32)
    return np.clip(sdf, -alpha, alpha)


def build_channel_masks(label_map: np.ndarray,
                        label_groups: Sequence[Sequence[int]]) -> np.ndarray:
    """
    label_map: integer label volume (D, H, W). Float dtypes (as MONAI often
               returns) are rounded to the nearest integer first.
    Returns a bool array (D, H, W, K), one binary mask per label group.
    """
    label_map = np.rint(label_map).astype(np.int64)
    channels = [np.isin(label_map, list(group)) for group in label_groups]
    return np.stack(channels, axis=-1)


def create_multilabel_sdf(label_map: np.ndarray,
                          label_groups: Sequence[Sequence[int]],
                          spacing_mm: tuple, alpha: float) -> np.ndarray:
    """Returns float32 (D, H, W, K): one clipped SDF per label group."""
    masks = build_channel_masks(label_map, label_groups)
    channels = [
        create_mask_sdf_with_clipping(masks[..., k], spacing_mm, alpha)
        for k in range(masks.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def sdf_to_binary_mask(sdf: np.ndarray) -> np.ndarray:
    """Threshold at zero to recover a binary mask (single-channel or per-channel)."""
    return (sdf < 0.0).astype(np.uint8)


def sdf_to_channel_masks(sdf: np.ndarray, mode: str = "independent") -> np.ndarray:
    """
    Decode a multi-channel SDF (D, H, W, K) into bool masks (D, H, W, K).

    mode="independent": threshold each channel at zero. Use for nested or
        overlapping regions (BraTS whole tumour / core / enhancing). A voxel
        may belong to several channels or none.
    mode="exclusive": each voxel goes to the channel with the most negative
        SDF, and to none if every channel is >= 0 there. Use only when the
        label groups are disjoint; guarantees no voxel is claimed twice.
    """
    if mode == "independent":
        return sdf < 0.0
    if mode == "exclusive":
        winner = np.argmin(sdf, axis=-1)
        inside_any = sdf.min(axis=-1) < 0.0
        onehot = np.zeros(sdf.shape, dtype=bool)
        np.put_along_axis(onehot, winner[..., None], True, axis=-1)
        return onehot & inside_any[..., None]
    raise ValueError(f"Unknown decode mode: {mode!r}")
