import json
import tempfile
import warnings
from pathlib import Path

from data.msd import (
    read_task_info, make_splits, load_or_create_splits,
    resolve_label_groups, resolve_shared_label_groups, shared_image_channels,
)

SPLIT_CFG = {"train": 0.7, "val": 0.1, "test": 0.2, "seed": 42}


def _make_task(root: Path, name: str, n_cases: int, labels: dict, modalities: dict):
    task_dir = root / name
    (task_dir / "imagesTr").mkdir(parents=True)
    (task_dir / "labelsTr").mkdir(parents=True)
    training = []
    for i in range(n_cases):
        img, lab = f"./imagesTr/case_{i:03d}.nii.gz", f"./labelsTr/case_{i:03d}.nii.gz"
        (task_dir / img).touch()
        (task_dir / lab).touch()
        training.append({"image": img, "label": lab})
    meta = {
        "name": name, "labels": labels, "modality": modalities,
        "numTraining": n_cases, "training": training,
        "test": [f"./imagesTs/case_{i:03d}.nii.gz" for i in range(3)],
    }
    (task_dir / "dataset.json").write_text(json.dumps(meta))
    return task_dir


def test_read_task_info_derives_everything_from_dataset_json():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_task(Path(tmp), "Task03_Liver", 10,
                       {"0": "background", "1": "liver", "2": "cancer"}, {"0": "CT"})
        info = read_task_info(d)
        assert info.labels == {0: "background", 1: "liver", 2: "cancer"}
        assert info.modalities == ["CT"] and info.image_channels == 1
        assert info.foreground_ids == [1, 2]
        assert len(info.cases) == 10
        assert info.cases[0]["case_id"] == "Task03_Liver/case_000"
        assert info.cases[0]["image"].endswith("case_000.nii.gz")
        assert Path(info.cases[0]["mask"]).is_file()


def test_multimodal_channel_count():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_task(Path(tmp), "Task01_BrainTumour", 4,
                       {"0": "background", "1": "edema", "2": "non-enh", "3": "enh"},
                       {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"})
        assert read_task_info(d).image_channels == 4


def test_missing_files_fail_early():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_task(Path(tmp), "T", 3, {"0": "bg", "1": "x"}, {"0": "CT"})
        (d / "labelsTr" / "case_001.nii.gz").unlink()
        try:
            read_task_info(d)
        except FileNotFoundError as e:
            assert "missing" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")


def test_splits_are_disjoint_complete_and_deterministic():
    ids = [f"t/c{i}" for i in range(50)]
    a = make_splits(ids, SPLIT_CFG, seed=42)
    b = make_splits(list(reversed(ids)), SPLIT_CFG, seed=42)   # input order must not matter
    assert a == b
    all_ids = a["train"] + a["val"] + a["test"]
    assert sorted(all_ids) == sorted(ids) and len(set(all_ids)) == len(ids)
    assert len(a["test"]) == 10 and len(a["val"]) == 5 and len(a["train"]) == 35
    assert make_splits(ids, SPLIT_CFG, seed=7) != a


def test_split_is_persisted_and_reused_and_guarded():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_task(Path(tmp), "Task03_Liver", 40, {"0": "bg", "1": "x"}, {"0": "CT"})
        info = read_task_info(d)
        splits_dir = Path(tmp) / "splits"

        first = load_or_create_splits(info, SPLIT_CFG, splits_dir)
        assert (splits_dir / "Task03_Liver.json").exists()
        assert load_or_create_splits(info, SPLIT_CFG, splits_dir) == first

        changed = dict(SPLIT_CFG, seed=1)
        try:
            load_or_create_splits(info, changed, splits_dir)
        except ValueError as e:
            assert "seed" in str(e)
        else:
            raise AssertionError("changed seed must not silently reuse an old split")


def test_underpowered_test_split_warns():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_task(Path(tmp), "Task02_Heart", 20, {"0": "bg", "1": "la"}, {"0": "MRI"})
        info = read_task_info(d)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_or_create_splits(info, SPLIT_CFG, Path(tmp) / "splits")   # 20 * 0.2 = 4 test cases
        assert any("Wilcoxon" in str(x.message) for x in w)


def test_label_group_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_task(Path(tmp), "T", 3,
                       {"0": "bg", "1": "a", "2": "b", "3": "c"}, {"0": "MRI"})
        info = read_task_info(d)
        assert resolve_label_groups("auto", info) == [[1], [2], [3]]
        assert resolve_label_groups([[1, 2, 3], [2, 3], [3]], info) == [[1, 2, 3], [2, 3], [3]]
        assert resolve_label_groups([["foreground"]], info) == [[1, 2, 3]]
        for bad in ([[9]], [[0]]):
            try:
                resolve_label_groups(bad, info)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for {bad}")


def test_shared_resolution_across_tasks():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        liver = read_task_info(_make_task(root, "Liver", 3,
                               {"0": "bg", "1": "liver", "2": "tumour"}, {"0": "CT"}))
        spleen = read_task_info(_make_task(root, "Spleen", 3,
                                {"0": "bg", "1": "spleen"}, {"0": "CT"}))

        # "auto" gives 2 channels for liver and 1 for spleen: must be rejected.
        try:
            resolve_shared_label_groups("auto", [liver, spleen])
        except ValueError as e:
            assert "channel counts" in str(e)
        else:
            raise AssertionError("mismatched channel counts must be rejected")

        shared = resolve_shared_label_groups([["foreground"]], [liver, spleen])
        assert shared == {"Liver": [[1, 2]], "Spleen": [[1]]}


def test_mixed_image_channels_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = read_task_info(_make_task(root, "A", 2, {"0": "bg", "1": "x"}, {"0": "CT"}))
        b = read_task_info(_make_task(root, "B", 2, {"0": "bg", "1": "x"},
                                      {"0": "T2", "1": "ADC"}))
        assert shared_image_channels([a]) == 1
        try:
            shared_image_channels([a, b])
        except ValueError:
            pass
        else:
            raise AssertionError("mixed channel counts must be rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} passed")
    print("All dataset.json loader tests passed.")
