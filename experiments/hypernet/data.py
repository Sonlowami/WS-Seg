"""
Data for the hypernetwork: the saved intensity decoders (inputs), each
case's clipped SDF (targets) and a deterministic case-level train/test split.

One input per decoder in run_decoder_training's output folder; with
data.per_modality each modality of a case has its own decoder, so a case
contributes one input per modality, all mapped to the same SDF. Splits are
by case, so no case's modalities end up on both sides.
"""
import json
from pathlib import Path

import torch

from data.msd import make_splits
from experiments.hypernet.model import flatten_state_dict
from experiments.run_decoder_training import FIELDS as DECODER_FIELDS
from sdf.coordinates import mm_per_unit
from training.encoder_eval import load_case_refs, prepare_ref
from training.train_loop import band_indices
from utils.results_io import read_rows


def base_case_id(case_id: str, modality: str) -> str:
    """'Task05_Prostate/prostate_00/T2' (modality 'T2') -> 'Task05_Prostate/prostate_00'."""
    suffix = f"/{modality}" if modality else ""
    return case_id[:-len(suffix)] if suffix and case_id.endswith(suffix) else case_id


def load_decoder_bank(decoders_dir) -> tuple:
    """
    (the folder's config.json, inputs): one input per row of decoders.csv,
    {"input": decoder case_id, "case_id": its case, "task", "modality",
    "x": flat decoder weights}, sorted by input id.
    """
    decoders_dir = Path(decoders_dir)
    config = json.loads((decoders_dir / "config.json").read_text())
    rows = read_rows([decoders_dir / "decoders.csv"], fieldnames=DECODER_FIELDS)
    if not rows:
        raise SystemExit(f"No decoders listed in {decoders_dir / 'decoders.csv'}")
    inputs = []
    for r in sorted(rows, key=lambda r: r["case_id"]):
        state_dict = torch.load(decoders_dir / r["decoder_file"], map_location="cpu")
        inputs.append({"input": r["case_id"], "case_id": base_case_id(r["case_id"], r["modality"]),
                       "task": r["task"], "modality": r["modality"], "x": flatten_state_dict(state_dict)})
    sizes = {i["x"].numel() for i in inputs}
    if len(sizes) > 1:
        raise SystemExit(f"Decoders in {decoders_dir} have different sizes {sorted(sizes)}; "
                         f"one hypernet needs one input size.")
    return config, inputs


def split_cases(case_ids: list, test_fraction: float, seed: int) -> dict:
    """Deterministic {"train": [...], "test": [...]} of the distinct case ids
    (the same seeded permutation as the data splits, data.msd.make_splits)."""
    splits = make_splits(sorted(set(case_ids)), {"train": 1.0 - test_fraction, "val": 0.0,
                                                 "test": test_fraction}, seed)
    if not splits["test"]:
        raise SystemExit(f"hypernet.test_fraction={test_fraction} leaves no test case "
                         f"among {len(set(case_ids))} cases.")
    return {"train": splits["train"], "test": splits["test"]}


def load_sdf_targets(data_cfg: dict, label_spec, case_ids: set, native_ids: set,
                     spacing_mm: tuple, alpha: float) -> dict:
    """
    {case_id: {"case_id", "target": (N, K) float16 CPU tensor, "band": its
    near-boundary voxel indices (for stratified sampling), "shape",
    "mm_per_unit", "sdf_case"}} for every case in case_ids. sdf_case
    (training/sdf_fit.prepare_sdf_case, for evaluation) is kept only for
    native_ids, with native-grid data; None for the others.
    """
    refs = [r for r in load_case_refs(data_cfg, label_spec, "all") if r[0] in case_ids]
    missing = sorted(case_ids - {r[0] for r in refs})
    if missing:
        raise SystemExit(f"{len(missing)} decoder cases not found in the data, e.g. {missing[:3]}; "
                         f"is --data_root the folder the decoders were trained from?")
    counts = {task: len(groups) for task, groups in refs[0][3].items()
              if any(r[1].data[r[2]]["task"] == task for r in refs)}
    if len(set(counts.values())) > 1:
        raise SystemExit(f"Tasks resolve to different label-group counts {counts}, but one hypernet "
                         f"predicts one decoder size; use an sdf.label_groups spec such as "
                         f'[["foreground"]] that gives every task the same count.')
    targets = {}
    for n, ref in enumerate(refs, 1):
        sdf_case = prepare_ref(ref, spacing_mm, alpha, native=ref[0] in native_ids)
        k = sdf_case["sdf"].shape[-1]
        target = torch.from_numpy(sdf_case["sdf"]).reshape(-1, k)
        targets[ref[0]] = {
            "case_id": ref[0], "shape": sdf_case["shape"], "target": target.half(),
            "band": band_indices(target, alpha),
            "mm_per_unit": mm_per_unit(sdf_case["shape"], spacing_mm),
            "sdf_case": sdf_case if ref[0] in native_ids else None,
        }
        print(f"  loaded SDF {n}/{len(refs)} {ref[0]} grid {sdf_case['shape']} x {k} groups, "
              f"{len(targets[ref[0]]['band']) / len(target):.1%} of voxels in the band")
    return targets
