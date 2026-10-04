import torch
from sdf.coordinates import get_3d_coordinates, coords_from_indices


def test_coords_from_indices_matches_full_grid():
    """Joint training samples coordinates by flat voxel index; they must be
    exactly the rows of the full grid, including anisotropic and size-1 axes."""
    for shape, spacing in [((7, 5, 9), (1.0, 1.0, 1.0)),
                           ((12, 4, 6), (1.5, 0.8, 3.0)),
                           ((1, 6, 3), (2.0, 1.0, 1.0))]:
        full = get_3d_coordinates(shape, spacing).coords
        idx = torch.randint(full.shape[0], (500,))
        assert torch.equal(coords_from_indices(idx, shape, spacing), full[idx]), shape
        every = torch.arange(full.shape[0])
        assert torch.equal(coords_from_indices(every, shape, spacing), full), shape


if __name__ == "__main__":
    test_coords_from_indices_matches_full_grid()
    print("Coordinate tests passed.")
