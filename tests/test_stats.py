import math
from utils.stats import shard, case_means, paired_wilcoxon, holm_adjust, summarize


def test_shards_partition_cases():
    ids = [f"T/case_{i}" for i in range(11)]
    parts = [shard(ids, f"{i}/3") for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(ids)
    assert all(not set(a) & set(b) for i, a in enumerate(parts) for b in parts[i + 1:])
    assert shard(ids, None) == ids


def test_case_means_exclude_absent_labels():
    rows = [{"group": "internal", "case_id": "c", "arm": "enc_I", "steps": "100", "label": l,
             "native_dice": d, "dice": d, "native_nsd": 0.5, "nsd": 0.5, "oracle_dice": 0.9}
            for l, d in (("a", 0.8), ("b", float("nan")))]
    m = case_means(rows)[("internal", "c", "enc_I", 100)]
    assert m["dice"] == 0.8 and m["nsd"] == 0.5


def test_wilcoxon_edge_cases():
    assert paired_wilcoxon([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])["p"] == 1.0       # all ties
    assert math.isnan(paired_wilcoxon([0.5], [0.4])["p"])                        # n < 2
    r = paired_wilcoxon([0.9, 0.8, float("nan"), 0.7], [0.5, 0.4, 0.3, 0.2])
    assert r["n"] == 3 and r["median_diff"] == 0.4


def test_holm():
    adj = holm_adjust([0.01, 0.04, 0.03, float("nan")])
    assert [round(a, 4) for a in adj[:3]] == [0.03, 0.06, 0.06] and math.isnan(adj[3])


def test_summarize_tests_pairs_and_convergence():
    rows = []
    for i in range(8):
        for steps, base in ((500, 0.5), (1000, 0.6)):
            for arm, bonus in (("enc_I", 0.0), ("enc_II", 0.1), ("random", -0.1)):
                d = base + bonus + 0.01 * i
                rows.append({"group": "internal", "case_id": f"c{i}", "arm": arm, "steps": steps,
                             "label": "x", "native_dice": d, "dice": d, "native_nsd": d,
                             "nsd": d, "oracle_dice": d})
    s = summarize(rows, metrics=("native_dice",), tol=0.01)
    t = {(r["steps"], r["pair"]): r for r in s["tests"]}
    assert t[(1000, "enc_II - enc_I")]["median_diff"] > 0 and t[(1000, "enc_II - enc_I")]["n"] == 8
    assert all(r["p_holm"] >= r["p"] for r in s["tests"])
    assert len(s["tests"]) == 6                                   # 2 budgets x 3 pairs
    conv = {r["arm"]: r for r in s["convergence"]}
    assert not conv["enc_I"]["converged"] and abs(conv["enc_I"]["change"] - 0.1) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("Stats tests passed.")
