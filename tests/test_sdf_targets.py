import numpy as np
from sdf.targets import create_mask_sdf_with_clipping, sdf_to_binary_mask


def test_sign_convention_and_clipping():
    mask = np.zeros((11, 11, 11), dtype=np.uint8)
    mask[4:7, 4:7, 4:7] = 1  # a 3x3x3 cube of "organ" in the center

    sdf = create_mask_sdf_with_clipping(mask, spacing_mm=(1.0, 1.0, 1.0), alpha=2.0)

    # Center of the cube should be negative (inside).
    assert sdf[5, 5, 5] < 0
    # A far corner should be clipped to exactly +alpha, not the raw distance.
    assert sdf[0, 0, 0] == 2.0
    # Nothing should exceed the clip range in either direction.
    assert sdf.min() >= -2.0 and sdf.max() <= 2.0


def test_threshold_recovers_mask():
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    mask[3:6, 3:6, 3:6] = 1
    sdf = create_mask_sdf_with_clipping(mask, spacing_mm=(1.0, 1.0, 1.0), alpha=3.0)
    recovered = sdf_to_binary_mask(sdf)
    assert np.array_equal(recovered, mask)


if __name__ == "__main__":
    test_sign_convention_and_clipping()
    test_threshold_recovers_mask()
    print("All SDF target tests passed.")
