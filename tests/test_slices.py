import numpy as np
import torch

from experiments.slices.data import put_slice, slice_index
from sdf.coordinates import coords_from_indices


def test_slice_index_keeps_the_axis():
    volume = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    assert volume[slice_index(2, 3)].shape == (4, 5, 1)
    assert np.array_equal(volume[slice_index(0, 1)][0], volume[1])


def test_put_slice_round_trips():
    volume = np.zeros((4, 5, 6, 2), dtype=np.float32)
    values = np.random.rand(4, 5, 1, 1).astype(np.float32)
    put_slice(volume, 2, 3, slice(1, 2), values)
    assert np.array_equal(volume[slice_index(2, 3) + (slice(1, 2),)], values)
    assert volume[..., 0].sum() == 0 and volume[:, :, [0, 1, 2, 4, 5], 1].sum() == 0


def test_one_voxel_thick_slices_have_a_constant_sliced_coordinate():
    coords = coords_from_indices(torch.arange(4 * 5), (4, 5, 1))
    assert torch.unique(coords[:, 2]).numel() == 1
    assert torch.unique(coords[:, 0]).numel() == 4 and torch.unique(coords[:, 1]).numel() == 5
