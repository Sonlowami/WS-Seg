"""
3D coordinate generation for coordinate-based INR fitting.

Coordinates are normalized to [-1, 1] per axis, which is what the encoder
consumes. Physical spacing is applied *before* normalization so that the
Eikonal property (computed in losses.py) refers to true physical distance,
not raw voxel index distance -- this matters for anisotropic volumes.
"""
from dataclasses import dataclass
import torch


@dataclass
class CoordinateGrid:
    coords: torch.Tensor        # (N, 3), normalized to [-1, 1]
    shape: tuple                # (D, H, W) original voxel grid shape
    spacing_mm: tuple           # physical spacing used for normalization
    mm_per_unit: float          # mm per normalized coordinate unit (see below)


def mm_per_unit(shape: tuple, spacing_mm: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    Scale between normalized coordinates and mm: the half-extent of the
    longest physical axis, which get_3d_coordinates maps to 1. A gradient
    taken w.r.t. normalized coordinates is this many times its per-mm value,
    so the Eikonal loss divides by it (training/losses.py).
    """
    return max(n * s for n, s in zip(shape, spacing_mm)) / 2


def get_3d_coordinates(shape: tuple, spacing_mm: tuple = (1.0, 1.0, 1.0),
                        device: str = "cpu") -> CoordinateGrid:
    """
    Build a flattened (N, 3) coordinate grid for a volume of `shape`
    (D, H, W), normalized to [-1, 1] in physical space.

    Since preprocessing already resamples every volume to isometric spacing
    (Section: Data Processing), spacing_mm is typically (1.0, 1.0, 1.0) here
    and mainly documents the assumption rather than doing real work -- but
    the physical-space normalization is kept explicit so this still behaves
    correctly if that assumption is ever relaxed.
    """
    d, h, w = shape
    sd, sh, sw = spacing_mm

    zs = torch.linspace(-1, 1, d, device=device) * (d * sd) / 2
    ys = torch.linspace(-1, 1, h, device=device) * (h * sh) / 2
    xs = torch.linspace(-1, 1, w, device=device) * (w * sw) / 2

    grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3)

    # Re-normalize to [-1, 1] after physical scaling, since the encoder
    # expects a bounded input domain regardless of the volume's physical size.
    scale = mm_per_unit(shape, spacing_mm)
    coords = coords / scale

    return CoordinateGrid(coords=coords, shape=shape, spacing_mm=spacing_mm, mm_per_unit=scale)
