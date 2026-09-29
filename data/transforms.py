"""
Preprocessing transforms: isometric spacing + z-score intensity normalization,
implemented as MONAI dict-transforms so they compose directly into a
monai.data.Dataset / CacheDataset pipeline.
"""
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, ToTensord,
)


def build_transforms(spacing_mm: tuple, keys=("image", "mask")):
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=spacing_mm, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # mask stays binary/unnormalized -- SDF conversion happens downstream
        # in sdf/targets.py, not here, since clipping alpha is a training
        # hyperparameter rather than a fixed preprocessing step.
        ToTensord(keys=keys),
    ])
