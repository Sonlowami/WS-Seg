"""
Direct reader for Medical Segmentation Decathlon tasks.

Everything about a task -- file lists, label names, modalities -- comes from
that task's own dataset.json. Nothing is duplicated into a hand-maintained
manifest, and image_channels / label_groups no longer need to be configured
separately from the data.

The one thing dataset.json does NOT provide is a held-out split: MSD's "test"
entries ship without labels, so validation and test cases must be carved out
of the "training" list. That split is generated once from a seed and saved to
splits/<task>.json so that it is reproducible and can be reused verbatim by the
nnU-Net and Meta-Seg baselines. This module deliberately imports neither torch
nor monai, so it can be tested anywhere.
"""
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

FOREGROUND = "foreground"   # label-group token: every non-background label of a task


@dataclass
class TaskInfo:
    name: str
    labels: dict                # {label_id: label_name}, includes 0 = background
    modalities: list            # e.g. ["CT"] or ["FLAIR", "T1w", "t1gd", "T2w"]
    cases: list = field(default_factory=list)   # [{"case_id", "task", "image", "mask"}]

    @property
    def image_channels(self) -> int:
        return len(self.modalities)

    @property
    def foreground_ids(self) -> list:
        return sorted(k for k in self.labels if k != 0)


def _stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def read_task_info(task_dir) -> TaskInfo:
    task_dir = Path(task_dir)
    meta_path = task_dir / "dataset.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No dataset.json found in {task_dir}")
    meta = json.loads(meta_path.read_text())

    labels = {int(k): v for k, v in meta["labels"].items()}
    modalities = [meta["modality"][k] for k in sorted(meta["modality"], key=int)]

    cases, missing = [], []
    for entry in meta["training"]:
        image = (task_dir / entry["image"]).resolve()
        mask = (task_dir / entry["label"]).resolve()
        for p in (image, mask):
            if not p.is_file():
                missing.append(str(p))
        cases.append({
            "case_id": f"{task_dir.name}/{_stem(image)}",
            "task": task_dir.name,
            "image": str(image),
            "mask": str(mask),
        })
    if missing:
        shown = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} file(s) listed in {meta_path} are missing, e.g. {shown}"
        )

    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate case ids in {meta_path}")
    cases.sort(key=lambda c: c["case_id"])
    return TaskInfo(name=task_dir.name, labels=labels, modalities=modalities, cases=cases)


def load_tasks(data_cfg: dict) -> list:
    root = Path(data_cfg["root"])
    return [read_task_info(root / name) for name in data_cfg["tasks"]]


# ---------------------------------------------------------------- splits

def make_splits(case_ids: list, fractions: dict, seed: int) -> dict:
    total = fractions["train"] + fractions["val"] + fractions["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1, got {total}")
    ids = sorted(case_ids)
    order = np.random.RandomState(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]

    n = len(ids)
    n_test = int(round(fractions["test"] * n))
    n_val = int(round(fractions["val"] * n))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Split leaves no training cases (n={n})")

    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }


def _warn_if_underpowered(task_name: str, splits: dict):
    n = len(splits["test"])
    if n < 6:
        warnings.warn(
            f"{task_name}: test split has {n} case(s). A two-sided Wilcoxon "
            f"signed-rank test cannot reach p < 0.05 with fewer than 6 pairs "
            f"(minimum possible p = {2 / 2 ** max(n, 1):.4f}). Pool tasks or "
            f"raise the test fraction if this task feeds a significance test."
        )
    elif n < 10:
        warnings.warn(f"{task_name}: test split has only {n} cases; "
                      f"significance testing will be low-powered.")


def load_or_create_splits(task: TaskInfo, split_cfg: dict, splits_dir) -> dict:
    """
    Returns {"train": [...], "val": [...], "test": [...]} of case ids.

    The first call writes splits/<task>.json. Later calls reuse it, and refuse
    to proceed if the saved seed/fractions or the set of cases no longer match,
    so a changed config can never silently evaluate on a different split.
    """
    fractions = {k: split_cfg[k] for k in ("train", "val", "test")}
    seed = split_cfg["seed"]
    ids = [c["case_id"] for c in task.cases]
    path = Path(splits_dir) / f"{task.name}.json"

    if path.exists():
        saved = json.loads(path.read_text())
        if saved["seed"] != seed or saved["fractions"] != fractions:
            raise ValueError(
                f"{path} was created with seed={saved['seed']}, "
                f"fractions={saved['fractions']}, but the config now says "
                f"seed={seed}, fractions={fractions}. Delete {path} to "
                f"regenerate, or point splits_dir elsewhere."
            )
        saved_ids = sorted(sum(saved["splits"].values(), []))
        if saved_ids != sorted(ids):
            raise ValueError(
                f"{path} covers a different set of cases than {task.name}/"
                f"dataset.json lists now. Delete it to regenerate."
            )
        splits = saved["splits"]
    else:
        splits = make_splits(ids, fractions, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"task": task.name, "seed": seed, "fractions": fractions, "splits": splits},
            indent=2,
        ))

    _warn_if_underpowered(task.name, splits)
    return splits


# ---------------------------------------------------------------- labels

def resolve_label_groups(spec, task: TaskInfo) -> list:
    """
    spec: "auto"  -> one exclusive channel per non-background label
          list of groups, each a list of label ids and/or "foreground"
              [[1]]                     single channel from label 1
              [[1, 2, 3], [2, 3], [3]]  BraTS-style nested regions
              [["foreground"]]          one channel: any non-background label
    Label ids are validated against this task's dataset.json.
    """
    if spec == "auto":
        return [[k] for k in task.foreground_ids]

    groups = []
    for group in spec:
        ids = []
        for item in group:
            if item == FOREGROUND:
                ids.extend(task.foreground_ids)
            elif item == 0:
                raise ValueError("Label 0 is background and cannot be part of a label group")
            elif int(item) not in task.labels:
                raise ValueError(
                    f"Label {item} is not in {task.name}/dataset.json labels {task.labels}"
                )
            else:
                ids.append(int(item))
        groups.append(sorted(set(ids)))
    return groups


def resolve_label_groups_per_task(spec, tasks: list) -> dict:
    """{task name: resolved groups}. Channel counts may differ between tasks."""
    return {t.name: resolve_label_groups(spec, t) for t in tasks}


def resolve_shared_label_groups(spec, tasks: list) -> dict:
    """
    Resolve per task, and require every task to yield the same number of
    channels, since one model (fixed decoder width) trains across all of them.
    Use [["foreground"]], or sdf.per_label: true (see expand_by_label_group),
    to train a shared encoder across tasks with different label sets.
    """
    by_task = resolve_label_groups_per_task(spec, tasks)
    counts = {name: len(g) for name, g in by_task.items()}
    if len(set(counts.values())) > 1:
        raise ValueError(
            f"Tasks resolve to different channel counts {counts}. Use an "
            f"explicit spec such as [[\"foreground\"]] that gives every task "
            f"the same number of channels, or set sdf.per_label: true."
        )
    return by_task


def shared_image_channels(tasks: list) -> int:
    counts = {t.name: t.image_channels for t in tasks}
    if len(set(counts.values())) > 1:
        raise ValueError(
            f"Tasks have different numbers of image channels {counts}; "
            f"set data.per_modality: true to train on them together."
        )
    return next(iter(counts.values()))


# ---------------------------------------------------------------- modalities

def expand_by_modality(cases: list, modalities: list) -> list:
    """
    One entry per (case, modality), each carrying the `channel` index to keep.

    The encoder sees only coordinates, so image channels only set the width of
    the per-case decoder. Splitting every modality into its own single-channel
    target lets tasks with different modalities (e.g. CT vs [T2, ADC]) share
    one encoder with out_features=1. Expand AFTER split lookup so that all
    modalities of a case stay in the same split.
    """
    return [
        {**c, "case_id": f"{c['case_id']}/{m}", "channel": i, "modality": m}
        for c in cases
        for i, m in enumerate(modalities)
    ]


def expand_by_label_group(cases: list, groups: list, labels: dict) -> list:
    """
    One entry per (case, label group), each carrying the `label_group` index.

    The Encoder II counterpart of expand_by_modality: every group becomes its
    own single-channel SDF target, so tasks with different numbers of label
    groups (e.g. liver+tumour vs spleen) share one encoder with
    out_features=1. The masked-Eikonal loss is already per channel, so this
    only drops the decoder layers shared between channels. Expand AFTER split
    lookup so that all groups of a case stay in the same split.
    """
    names = ["+".join(labels[i] for i in g) for g in groups]
    return [
        {**c, "case_id": f"{c['case_id']}/{n}", "label_group": i, "label_group_name": n}
        for c in cases
        for i, n in enumerate(names)
    ]
