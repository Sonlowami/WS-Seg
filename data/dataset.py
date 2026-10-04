"""
MONAI dataset built directly from MSD dataset.json files (see data/msd.py).
No manifest CSV: file lists, labels, and modalities come from each task's own
dataset.json, and splits come from the persisted, seeded split files.
"""
from monai.data import Dataset as MonaiDataset

from data.msd import (
    load_tasks, load_or_create_splits, expand_by_modality, expand_by_label_group,
)
from data.transforms import build_transforms


def build_dataset(data_cfg: dict, split: str, per_modality: bool = False,
                  label_groups_by_task: dict = None):
    """
    data_cfg: the `data:` block of the training config.
    split: "train" | "val" | "test" | "all" (every case, no split files; for
        external tasks the encoders never saw)
    per_modality: yield one single-channel item per (case, modality) instead of
        one multi-channel item per case (see data.msd.expand_by_modality).
    label_groups_by_task: if given, yield one item per (case, label group) instead
        (see data.msd.expand_by_label_group).
    Returns a MONAI Dataset yielding dicts with image, mask, case_id, task.
    """
    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"Unknown split: {split!r}")

    entries = []
    for task in load_tasks(data_cfg):
        if split == "all":
            selected = list(task.cases)
        else:
            splits = load_or_create_splits(task, data_cfg["split"], data_cfg["splits_dir"])
            wanted = set(splits[split])
            selected = [c for c in task.cases if c["case_id"] in wanted]
        if per_modality:
            selected = expand_by_modality(selected, task.modalities)
        if label_groups_by_task is not None:
            selected = expand_by_label_group(
                selected, label_groups_by_task[task.name], task.labels)
        entries += selected

    if not entries:
        raise ValueError(f"No cases found for split={split!r} in tasks {data_cfg['tasks']}")

    transforms = build_transforms(tuple(data_cfg["spacing_mm"]), select_channel=per_modality)
    return MonaiDataset(data=entries, transform=transforms)
