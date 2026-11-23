import math
import subprocess
import sys

from experiments.slices.compare_2d_3d import case_values, compare, gains
from utils.results_io import RowWriter, read_rows

METRIC_FIELDS = ("dice", "nsd", "native_dice", "native_nsd")


def _row(method, case, label, value, modality="", **extra):
    return {"method": method, "case_id": case, "input": f"{case}/{modality}" if modality else "",
            "task": "T", "modality": modality, "label": label,
            **{m: value for m in METRIC_FIELDS}, **extra}


def test_case_values_average_modalities_then_labels():
    rows = [_row("hypernet", "c0", "A", 0.6, "T2"), _row("hypernet", "c0", "A", 0.8, "ADC"),
            _row("hypernet", "c0", "B", float("nan"), "T2"), _row("hypernet", "c0", "B", 0.4, "ADC")]
    v = case_values(rows, "dice")
    assert math.isclose(v[("c0", "A")], 0.7) and math.isclose(v[("c0", "B")], 0.4)
    assert math.isclose(v[("c0", "all")], 0.55)                          # NaN excluded


def test_compare_pairs_by_case_and_signs_the_difference():
    rows_2d = [_row("hypernet", f"c{i}", lab, 0.5 + 0.01 * i + 0.1) for i in range(8) for lab in "AB"]
    rows_3d = [_row("hypernet", f"c{i}", lab, 0.5 + 0.01 * i) for i in range(8) for lab in "AB"]
    stats, cases = compare(rows_2d, rows_3d, ("dice",))
    overall = next(r for r in stats if r["label"] == "all")
    assert overall["n"] == 8 and overall["wins_2d"] == 8 and overall["wins_3d"] == 0
    assert math.isclose(overall["mean_diff"], 0.1)
    assert overall["ci_low"] - 1e-9 <= 0.1 <= overall["ci_high"] + 1e-9
    assert overall["p"] < 0.05 and len(cases) == 3 * 8                    # all, A, B


def test_cli_reports_on_shared_cases_only(tmp_path):
    fields_2d = ["point", "case_fraction", "stride", "n_train_cases", "n_train_slices", "method", "case_id",
                 "input", "task", "modality", "psnr", "label", "gt_voxels", *METRIC_FIELDS, "oracle_dice"]
    fields_3d = ["method", "case_id", "input", "task", "modality", "steps", "fit_seconds", "psnr", "label",
                 "gt_voxels", *METRIC_FIELDS, "oracle_dice"]
    meta = {"point": "cases1_stride2", "case_fraction": 1.0, "stride": 2, "n_train_cases": 10,
            "n_train_slices": 100, "psnr": 0, "gt_voxels": 10, "oracle_dice": 0}
    RowWriter(tmp_path / "2d.csv", fields_2d, False).write(
        [_row("hypernet", f"c{i}", "A", 0.8, **meta) for i in range(6)])
    RowWriter(tmp_path / "3d.csv", fields_3d, False).write(
        [_row("hypernet", f"c{i}", "A", 0.7, steps=1, fit_seconds=0, psnr=0, gt_voxels=10, oracle_dice=0)
         for i in range(5)])                                             # c5 missing in 3D
    out = subprocess.run([sys.executable, "-m", "experiments.slices.compare_2d_3d", "--slices",
                          str(tmp_path / "2d.csv"), "--volume", str(tmp_path / "3d.csv"),
                          "--metrics", "dice", "--out", str(tmp_path / "cmp")],
                         capture_output=True, text=True, check=True).stdout
    assert "5 shared" in out and "WARNING: the test sets differ" in out
    stats = read_rows([tmp_path / "cmp_stats.csv"], ("n",), ("mean_diff",))
    assert stats[0]["n"] == 5 and math.isclose(stats[0]["mean_diff"], 0.1)


def test_gains_subtract_the_baseline_per_case_and_label():
    rows = [_row("hypernet", f"c{i}", "A", 0.8) for i in range(3)]
    base = [_row("position_only", f"c{i}", "A", 0.5 + 0.1 * i) for i in range(3)]
    g = gains(rows, base, ("dice",))["dice"]
    assert math.isclose(g[("c0", "A")], 0.3) and math.isclose(g[("c2", "all")], 0.1)


def test_cli_compares_what_the_image_adds(tmp_path):
    fields_2d = ["point", "n_train_cases", "n_train_slices", "method", "case_id", "input", "task",
                 "modality", "label", *METRIC_FIELDS]
    fields_3d = ["method", "case_id", "input", "task", "modality", "label", *METRIC_FIELDS]
    meta = {"point": "cases1_stride2", "n_train_cases": 10, "n_train_slices": 100}
    rows_2d = [_row(m, f"c{i}", "A", v + 0.001 * i, **meta)
               for i in range(8) for m, v in (("hypernet", 0.8), ("position_only", 0.7))]
    rows_3d = [_row(m, f"c{i}", "A", v + 0.001 * i)
               for i in range(8) for m, v in (("hypernet", 0.7), ("floor", 0.3))]
    RowWriter(tmp_path / "2d.csv", fields_2d, False).write(rows_2d)
    RowWriter(tmp_path / "3d.csv", fields_3d, False).write(rows_3d)
    out = subprocess.run([sys.executable, "-m", "experiments.slices.compare_2d_3d", "--slices",
                          str(tmp_path / "2d.csv"), "--volume", str(tmp_path / "3d.csv"), "--metrics", "dice",
                          "--baseline2d", "position_only", "--baseline3d", "floor", "--out", str(tmp_path / "cmp")],
                         capture_output=True, text=True, check=True).stdout
    assert "what the image adds in 2D" in out and "gain 2D - gain 3D" in out
    gain = read_rows([tmp_path / "cmp_gain_stats.csv"], ("n",), ("mean_diff",))
    assert gain[0]["n"] == 8 and math.isclose(gain[0]["mean_diff"], 0.1 - 0.4)   # 3D gains more here
