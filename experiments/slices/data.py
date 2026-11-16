"""
Slices of the resampled volumes, for the slice-decoder pilot.

A slice is the volume at one index along `axis`, kept as a volume one voxel
thick: shape (X, Y, 1) for axis 2. Coordinates, encoders, decoders, losses
and metrics then apply unchanged; the sliced coordinate is simply constant.
Slice SDFs are taken from the case's 3D clipped SDF: thresholded at 0 they
give exactly the 3D mask's slice, so stacking predicted slices back into a
volume (put_slice) is scored against the same ground truth as the 3D
experiment. Every slice is used, empty or not: nothing about which slices
to keep comes from the ground truth.
"""
import numpy as np
import torch

from data.msd import load_tasks
from sdf.coordinates import mm_per_unit
from training.encoder_eval import load_case_refs
from training.sdf_fit import prepare_sdf_case, group_names


def slice_index(axis: int, k: int) -> tuple:
    """Spatial index of slice k along axis, keeping the axis (size 1)."""
    index = [slice(None)] * 3
    index[axis] = slice(k, k + 1)
    return tuple(index)


def put_slice(volume: np.ndarray, axis: int, k: int, channels: slice, values: np.ndarray):
    """Write values (slice shape x channels) into volume (X, Y, Z, K) at slice k."""
    volume[slice_index(axis, k) + (channels,)] = values


def load_slice_cases(data_cfg: dict, label_spec, split: str, case_ids: set, eval_ids: set,
                     spacing_mm: tuple, alpha: float, axis: int, stride_for) -> dict:
    """
    {case_id: case} for every case in case_ids (of the data split), with case = {case_id, task,
    names (label groups), modalities, n_slices, sdf_case (eval_ids only,
    with native-grid data; None otherwise), slices}. slices maps every slice
    index k with k % stride_for(case_id) == 0 to
    {"k", "position" (k / (n_slices - 1)), "shape", "mm_per_unit",
    "image" (N, C) float16, "sdf" (N, K) float16}, flattened in C order over
    the slice shape. The volumes themselves are not kept.
    """
    refs = [r for r in load_case_refs(data_cfg, label_spec, split) if r[0] in case_ids]
    missing = sorted(case_ids - {r[0] for r in refs})
    if missing:
        raise SystemExit(f"{len(missing)} cases not found in the data, e.g. {missing[:3]}")
    modalities = {t.name: t.modalities for t in load_tasks(data_cfg)}
    cases = {}
    for n, (case_id, dataset, i, groups_by_task, labels_by_task) in enumerate(refs, 1):
        case, entry = dataset[i], dataset.data[i]
        groups = groups_by_task[case["task"]]
        sdf_case = prepare_sdf_case(case, entry["mask"], groups, group_names(groups, labels_by_task[case["task"]]),
                                    spacing_mm, alpha, native=case_id in eval_ids)
        image = case["image"]
        image = (image.as_tensor() if hasattr(image, "as_tensor") else image).numpy()   # (C, X, Y, Z)
        sdf = sdf_case["sdf"]                                                       # (X, Y, Z, K)
        n_slices = sdf.shape[axis]
        kept = [k for k in range(n_slices) if k % stride_for(case_id) == 0]
        slices = {}
        for k in kept:
            index = slice_index(axis, k)
            shape = sdf[index].shape[:3]
            img = image[(slice(None),) + index].reshape(image.shape[0], -1).T
            slices[k] = {"k": k, "position": k / max(n_slices - 1, 1), "shape": shape,
                         "mm_per_unit": mm_per_unit(shape, spacing_mm),
                         "image": torch.from_numpy(np.ascontiguousarray(img)).half(),
                         "sdf": torch.from_numpy(np.ascontiguousarray(sdf[index].reshape(-1, sdf.shape[-1]))).half()}
        cases[case_id] = {"case_id": case_id, "task": case["task"], "names": sdf_case["names"],
                          "modalities": modalities[case["task"]], "n_slices": n_slices, "slices": slices,
                          "sdf_case": sdf_case if case_id in eval_ids else None}
        print(f"  loaded {n}/{len(refs)} {case_id}: {len(slices)} of {n_slices} slices")
    return cases
