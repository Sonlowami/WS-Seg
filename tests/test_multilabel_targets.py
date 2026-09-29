import numpy as np
from sdf.targets import (
    create_multilabel_sdf, build_channel_masks, sdf_to_channel_masks,
    create_mask_sdf_with_clipping,
)
from utils.metrics import per_label_metrics

SPACING = (1.0, 1.0, 1.0)
ALPHA = 3.0


def _brats_like_volume():
    """Label 1 = edema shell, 2 = non-enhancing core, 3 = enhancing center."""
    vol = np.zeros((21, 21, 21), dtype=np.int64)
    vol[4:17, 4:17, 4:17] = 1
    vol[7:14, 7:14, 7:14] = 2
    vol[9:12, 9:12, 9:12] = 3
    return vol


def test_nested_groups_are_nested():
    vol = _brats_like_volume()
    groups = [[1, 2, 3], [2, 3], [3]]          # whole, core, enhancing
    masks = build_channel_masks(vol, groups)
    assert masks.shape == vol.shape + (3,)
    # Each smaller region must sit entirely inside the larger one.
    assert not (masks[..., 1] & ~masks[..., 0]).any()
    assert not (masks[..., 2] & ~masks[..., 1]).any()


def test_multilabel_sdf_shape_range_and_sign():
    vol = _brats_like_volume()
    sdf = create_multilabel_sdf(vol, [[1, 2, 3], [2, 3], [3]], SPACING, ALPHA)
    assert sdf.shape == vol.shape + (3,)
    assert sdf.min() >= -ALPHA and sdf.max() <= ALPHA
    # Center voxel is inside all three regions.
    assert (sdf[10, 10, 10] < 0).all()
    # A voxel in the outer shell is inside the whole-region channel only.
    assert sdf[5, 10, 10, 0] < 0 and sdf[5, 10, 10, 1] > 0 and sdf[5, 10, 10, 2] > 0


def test_absent_label_is_all_plateau():
    vol = np.zeros((9, 9, 9), dtype=np.int64)
    vol[3:6, 3:6, 3:6] = 1                      # label 3 never appears
    sdf = create_multilabel_sdf(vol, [[1], [3]], SPACING, ALPHA)
    assert np.all(sdf[..., 1] == ALPHA)
    assert (sdf[..., 0] < 0).any()


def test_full_mask_is_all_negative_plateau():
    full = np.ones((5, 5, 5), dtype=bool)
    sdf = create_mask_sdf_with_clipping(full, SPACING, ALPHA)
    assert np.all(sdf == -ALPHA)


def test_float_label_maps_are_rounded():
    vol = _brats_like_volume().astype(np.float32) + 1e-4   # MONAI-style float labels
    masks = build_channel_masks(vol, [[1, 2, 3]])
    assert masks[..., 0].sum() == (_brats_like_volume() > 0).sum()


def test_decode_modes():
    vol = np.zeros((11, 11, 11), dtype=np.int64)
    vol[1:5, 1:5, 1:5] = 1
    vol[6:10, 6:10, 6:10] = 2
    sdf = create_multilabel_sdf(vol, [[1], [2]], SPACING, ALPHA)

    excl = sdf_to_channel_masks(sdf, mode="exclusive")
    assert not (excl[..., 0] & excl[..., 1]).any()          # no double-claimed voxel
    assert excl[2, 2, 2, 0] and excl[7, 7, 7, 1]
    assert not excl[0, 10, 0].any()                          # background claimed by none

    indep = sdf_to_channel_masks(sdf, mode="independent")
    assert np.array_equal(indep, sdf < 0)


def test_per_label_metrics_skip_absent_labels():
    vol = np.zeros((9, 9, 9), dtype=np.int64)
    vol[3:6, 3:6, 3:6] = 1
    gt = build_channel_masks(vol, [[1], [3]])               # channel 1 empty in GT
    pred = gt.copy()                                         # perfect prediction
    m = per_label_metrics(pred, gt, SPACING)
    assert m["dice_label0"] > 0.99
    assert np.isnan(m["dice_label1"])                        # absent label not scored
    assert m["mean_dice"] > 0.99                             # not dragged or inflated by it


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} passed")
    print("All multi-label target tests passed.")
