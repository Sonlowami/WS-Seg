"""
Statistics for the translation test: per-case aggregation, paired Wilcoxon
tests with Holm correction, convergence checks and case sharding. Plain
numpy/scipy on lists of row dicts, so results merged from several shards'
CSVs are summarized exactly like a single run's.
"""
import math

import numpy as np
from scipy.stats import wilcoxon

ARMS = ("enc_I", "enc_II", "random")
# Each pair is reported as first - second.
PAIRS = (("enc_II", "enc_I"), ("enc_I", "random"), ("enc_II", "random"))
CASE_METRICS = ("native_dice", "dice", "native_nsd", "nsd", "oracle_dice")


def parse_shard(spec: str) -> tuple:
    """'i/n' -> (i, n) with 0 <= i < n."""
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"Shard {spec!r}: need 0 <= i < n")
    return i, n


def shard(items: list, spec: str, key=lambda x: x) -> list:
    """Deterministic round-robin partition of items sorted by key; the n
    shards are disjoint and together cover every item."""
    if spec is None:
        return list(items)
    i, n = parse_shard(spec)
    return sorted(items, key=key)[i::n]


def _nanmean(values) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def case_means(rows: list) -> dict:
    """
    Per-label rows -> {(group, case_id, arm, steps): {metric: mean over the
    case's labels, NaN labels (absent from the GT) excluded}}.
    """
    by_key = {}
    for r in rows:
        by_key.setdefault((r["group"], r["case_id"], r["arm"], int(r["steps"])), []).append(r)
    return {key: {m: _nanmean(float(r[m]) for r in rs) for m in CASE_METRICS}
            for key, rs in by_key.items()}


def paired_wilcoxon(x: list, y: list) -> dict:
    """Two-sided Wilcoxon signed-rank on paired values, NaN pairs dropped.
    All-zero differences give p = 1 (scipy raises there)."""
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    n = len(pairs)
    out = {"n": n, "median_diff": float("nan"), "stat": float("nan"), "p": float("nan")}
    if n == 0:
        return out
    diffs = np.array([a - b for a, b in pairs])
    out["median_diff"] = float(np.median(diffs))
    if n < 2:
        return out
    if np.all(diffs == 0):
        out.update(stat=0.0, p=1.0)
        return out
    stat, p = wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
    out.update(stat=float(stat), p=float(p))
    return out


def holm_adjust(pvalues: list) -> list:
    """Holm-Bonferroni adjusted p-values (NaN entries pass through)."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvalues[i])
    m, running, adjusted = len(order), 0.0, list(pvalues)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def summarize(rows: list, metrics=("native_dice", "dice"), tol: float = 0.01) -> dict:
    """
    rows: per-label rows (fields group, case_id, arm, steps, label and the
    CASE_METRICS). Returns {"table", "tests", "convergence"}:
      table        per (group, steps, arm): n cases, mean and median of case means
      tests        per (group, metric, steps, pair): paired Wilcoxon over cases,
                   Holm-adjusted within each (group, metric) across steps x pairs
      convergence  per (group, arm, metric): median case value at the last two
                   budgets and whether it moved by less than tol
    """
    means = case_means(rows)
    groups = sorted({k[0] for k in means})
    arms = [a for a in ARMS if any(k[2] == a for k in means)]
    steps_list = sorted({k[3] for k in means})

    def values(group, arm, steps, metric):
        return {k[1]: v[metric] for k, v in means.items()
                if k[0] == group and k[2] == arm and k[3] == steps}

    table = []
    for group in groups:
        for steps in steps_list:
            for arm in arms:
                row = {"group": group, "steps": steps, "arm": arm}
                for metric in metrics:
                    vals = [v for v in values(group, arm, steps, metric).values() if not math.isnan(v)]
                    row[f"n_{metric}"] = len(vals)
                    row[f"mean_{metric}"] = float(np.mean(vals)) if vals else float("nan")
                    row[f"median_{metric}"] = float(np.median(vals)) if vals else float("nan")
                table.append(row)

    tests = []
    for group in groups:
        for metric in metrics:
            family = []
            for steps in steps_list:
                for a, b in PAIRS:
                    if a not in arms or b not in arms:
                        continue
                    va, vb = values(group, a, steps, metric), values(group, b, steps, metric)
                    cases = sorted(set(va) & set(vb))
                    res = paired_wilcoxon([va[c] for c in cases], [vb[c] for c in cases])
                    family.append({"group": group, "metric": metric, "steps": steps,
                                   "pair": f"{a} - {b}", **res})
            for row, p_holm in zip(family, holm_adjust([r["p"] for r in family])):
                row["p_holm"] = p_holm
            tests += family

    convergence = []
    if len(steps_list) >= 2:
        prev, last = steps_list[-2], steps_list[-1]
        for group in groups:
            for arm in arms:
                for metric in metrics:
                    med = []
                    for s in (prev, last):
                        vals = [v for v in values(group, arm, s, metric).values() if not math.isnan(v)]
                        med.append(float(np.median(vals)) if vals else float("nan"))
                    change = med[1] - med[0]
                    convergence.append({"group": group, "arm": arm, "metric": metric,
                                        "steps_prev": prev, "steps_last": last,
                                        "median_prev": med[0], "median_last": med[1],
                                        "change": change,
                                        "converged": (not math.isnan(change)) and abs(change) < tol})
    return {"table": table, "tests": tests, "convergence": convergence}


def aggregate_cases(rows: list, by: tuple, metrics=CASE_METRICS) -> list:
    """
    Summaries over cases for every combination of the `by` fields: each case
    is first averaged over its label rows (NaN labels excluded, as in
    case_means), then n / mean / median are taken across cases. Pass a
    `by` that includes "label" for per-label summaries.
    """
    per_case = {}
    for r in rows:
        per_case.setdefault(tuple(r[k] for k in by) + (r["case_id"],), []).append(r)
    grouped = {}
    for key, rs in per_case.items():
        grouped.setdefault(key[:-1], []).append(
            {m: _nanmean(float(r[m]) for r in rs) for m in metrics})
    out = []
    for key in sorted(grouped, key=lambda k: tuple(str(x) for x in k)):
        row = dict(zip(by, key))
        for m in metrics:
            vals = [c[m] for c in grouped[key] if not math.isnan(c[m])]
            row[f"n_{m}"] = len(vals)
            row[f"mean_{m}"] = float(np.mean(vals)) if vals else float("nan")
            row[f"median_{m}"] = float(np.median(vals)) if vals else float("nan")
        out.append(row)
    return out
