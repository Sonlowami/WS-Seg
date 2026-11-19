"""
2D (slice decoders) vs 3D (volume decoders): paired comparison on the same
test cases.

Reads the slice pilot's results.csv (experiments/slices/run_slices.py) and the
3D experiment's results.csv (experiments/hypernet/run_hypernet.py). Both score
each test case's full 3D volume -- the slice pilot restacks its predicted
slices before scoring -- with the same metrics on the same grids, so the rows
are directly comparable.

For each (test case, label) a method's value is the mean over the case's
inputs (modalities); "all" is each case's mean over its labels. The two
methods are paired by case: per label and metric this reports the means,
the mean difference (2D - 3D) with a 95% bootstrap interval over cases, the
median difference, wins / losses, and a paired Wilcoxon signed-rank test,
Holm-corrected over every (label, metric) tested. Ceilings from either file
are printed alongside for context.

The 2D point defaults to the one trained on the most slices.

Usage:
  python -m experiments.slices.compare_2d_3d \\
      --slices results/slices/results.csv --volume checkpoints/<decoders>_hypernet/results.csv \\
      [--point cases1_stride2] [--method2d hypernet] [--method3d hypernet] [--out results/compare_2d_3d]
"""
import argparse
import math
from pathlib import Path

import numpy as np

from utils.results_io import read_rows, write_csv
from utils.stats import paired_wilcoxon, holm_adjust

METRICS = ("native_dice", "native_nsd", "dice", "nsd")


def case_values(rows: list, metric: str) -> dict:
    """{(case_id, label): mean over the case's inputs} plus {(case_id, "all"):
    mean over its labels}, NaN (label absent from the ground truth) excluded."""
    per_label = {}
    for r in rows:
        if not math.isnan(r[metric]):
            per_label.setdefault((r["case_id"], r["label"]), []).append(r[metric])
    values = {k: float(np.mean(v)) for k, v in per_label.items()}
    per_case = {}
    for (case_id, _), v in values.items():
        per_case.setdefault(case_id, []).append(v)
    values.update({(c, "all"): float(np.mean(v)) for c, v in per_case.items()})
    return values


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple:
    """95% percentile interval of the mean of diffs, resampling cases."""
    if len(diffs) < 2:
        return float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(diffs, (n_boot, len(diffs))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(rows_2d: list, rows_3d: list, metrics=METRICS) -> tuple:
    """(stats rows, per-case rows) for every (label, metric)."""
    stats, cases = [], []
    for metric in metrics:
        a, b = case_values(rows_2d, metric), case_values(rows_3d, metric)
        for label in sorted({k[1] for k in a} & {k[1] for k in b}, key=lambda l: (l != "all", l)):
            keys = sorted(k for k in set(a) & set(b) if k[1] == label)
            x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
            d = x - y
            lo, hi = bootstrap_ci(d)
            stats.append({"metric": metric, "label": label, "n": len(keys),
                          "mean_2d": float(x.mean()), "mean_3d": float(y.mean()),
                          "mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi,
                          "wins_2d": int((d > 0).sum()), "wins_3d": int((d < 0).sum()),
                          "ties": int((d == 0).sum()), **paired_wilcoxon(list(x), list(y))})
            cases += [{"metric": metric, "label": label, "case_id": k[0], "value_2d": a[k], "value_3d": b[k],
                       "diff": a[k] - b[k]} for k in keys]
    for row, p in zip(stats, holm_adjust([r["p"] for r in stats])):
        row["p_holm"] = p
    return stats, cases


def verdict(row: dict, alpha: float = 0.05) -> str:
    if math.isnan(row["p_holm"]) or row["p_holm"] >= alpha:
        return "no significant difference"
    return "2D better" if row["mean_diff"] > 0 else "3D better"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slices", required=True, help="experiments/slices results.csv (2D)")
    parser.add_argument("--volume", required=True, help="experiments/hypernet results.csv (3D)")
    parser.add_argument("--point", default=None, help="2D learning-curve point (default: most training slices)")
    parser.add_argument("--method2d", default="hypernet")
    parser.add_argument("--method3d", default="hypernet")
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default=None, help="Prefix for <out>_stats.csv and <out>_cases.csv")
    args = parser.parse_args()

    floats = ("dice", "nsd", "native_dice", "native_nsd")
    all_2d = read_rows([args.slices], ("n_train_slices", "n_train_cases"), floats)
    all_3d = read_rows([args.volume], (), floats)

    learned = [r for r in all_2d if r["method"] == args.method2d]
    if not learned:
        raise SystemExit(f"No {args.method2d!r} rows in {args.slices}.")
    point = args.point or max(learned, key=lambda r: (r["n_train_slices"], r["n_train_cases"]))["point"]
    rows_2d = [r for r in learned if r["point"] == point]
    rows_3d = [r for r in all_3d if r["method"] == args.method3d]
    if not rows_2d:
        raise SystemExit(f"No {args.method2d!r} rows at point {point!r}; points: "
                         f"{sorted({r['point'] for r in learned})}")
    if not rows_3d:
        raise SystemExit(f"No {args.method3d!r} rows in {args.volume}; methods: "
                         f"{sorted({r['method'] for r in all_3d})}")

    # The test cases must be the same, or the comparison is not paired.
    cases_2d, cases_3d = {r["case_id"] for r in rows_2d}, {r["case_id"] for r in rows_3d}
    shared = cases_2d & cases_3d
    print(f"2D: {args.method2d} at {point} ({rows_2d[0]['n_train_cases']} training cases, "
          f"{rows_2d[0]['n_train_slices']} slices), {len(cases_2d)} test cases")
    print(f"3D: {args.method3d}, {len(cases_3d)} test cases; {len(shared)} shared")
    if cases_2d != cases_3d:
        print(f"WARNING: the test sets differ ({len(cases_2d - cases_3d)} only in 2D, "
              f"{len(cases_3d - cases_2d)} only in 3D); comparing the {len(shared)} shared cases. "
              f"Same data split and hypernet.test_fraction / split_seed give identical sets.")
    if not shared:
        raise SystemExit("No shared test cases.")
    rows_2d = [r for r in rows_2d if r["case_id"] in shared]
    rows_3d = [r for r in rows_3d if r["case_id"] in shared]

    stats, cases = compare(rows_2d, rows_3d, args.metrics)
    print(f"\n==== 2D - 3D, paired over test cases (Holm over all {len(stats)} tests) ====")
    print(f"  {'metric':<12}{'label':<12}{'n':>4}{'2D':>8}{'3D':>8}{'diff':>9}{'95% CI':>20}"
          f"{'2D/3D wins':>12}{'p_holm':>10}  verdict")
    for r in stats:
        print(f"  {r['metric']:<12}{r['label']:<12}{r['n']:>4}{r['mean_2d']:>8.4f}{r['mean_3d']:>8.4f}"
              f"{r['mean_diff']:>+9.4f}   [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
              f"{r['wins_2d']:>6}/{r['wins_3d']:<5}{r['p_holm']:>10.3g}  {verdict(r, args.alpha)}")

    for name, rows, method in (("2D", all_2d, "ceiling"), ("3D", all_3d, "ceiling")):
        ceiling = [r for r in rows if r["method"] == method and r["case_id"] in shared]
        if ceiling:
            values = case_values(ceiling, args.metrics[0])
            print(f"  context: {name} ceiling, mean {args.metrics[0]} over cases "
                  f"{np.mean([v for k, v in values.items() if k[1] == 'all']):.4f}")

    if args.out:
        write_csv(Path(f"{args.out}_stats.csv"), [{**r, "verdict": verdict(r, args.alpha)} for r in stats])
        write_csv(Path(f"{args.out}_cases.csv"), cases)
        print(f"\nWrote {args.out}_stats.csv and {args.out}_cases.csv")


if __name__ == "__main__":
    main()
