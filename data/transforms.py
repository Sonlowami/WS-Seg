"""
Preprocessing transforms: isometric spacing + z-score intensity normalization,
implemented as MONAI dict-transforms so they compose directly into a
monai.data.Dataset / CacheDataset pipeline.
"""
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, ToTensord, MapTransform,
)


class SelectChanneld(MapTransform):
    """Keep only channel d["channel"] of each key (set by expand_by_modality)."""

    def __call__(self, data):
        d = dict(data)
        c = d["channel"]
        for key in self.key_iterator(d):
            d[key] = d[key][c:c + 1]
        return d


def build_transforms(spacing_mm: tuple, keys=("image", "mask"), select_channel=False):
    # Selecting before resampling avoids resampling channels that are dropped;
    # normalization is channel-wise, so the order does not change the result.
    select = [SelectChanneld(keys=["image"])] if select_channel else []
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        *select,
        Spacingd(keys=keys, pixdim=spacing_mm, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # mask stays binary/unnormalized -- SDF conversion happens downstream
        # in sdf/targets.py, not here, since clipping alpha is a training
        # hyperparameter rather than a fixed preprocessing step.
        ToTensord(keys=keys),
    ])
