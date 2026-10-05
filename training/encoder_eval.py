"""
Score an encoder as a prior for SDF fitting, case by case.

The building blocks shared by experiments/run_translation_test.py (which
compares encoders) and experiments/run_encoder_sweep.py (which compares
encoder sizes): load evaluation cases, then for one encoder and one case fit
a fresh decoder (training/sdf_fit.py) and score it at several step budgets.
Rows are plain dicts, one per label group per budget, so callers add their
own identifying fields (arm, configuration, ...) and write them anywhere.
"""
from data.dataset import build_dataset
from data.msd import load_tasks, resolve_label_groups_per_task
from training.sdf_fit import prepare_sdf_case, fit_sdf, evaluate_sdf_fit, group_names


def load_case_refs(data_cfg: dict, label_spec, split: str, max_cases: int = None) -> list:
    """
    Lightweight references to evaluation cases, sorted by case_id:
    [(case_id, dataset, index, groups_by_task, labels_by_task)]. Nothing is
    loaded from disk until prepare_ref is called, so the list can be sharded
    or filtered first. split: "train" | "val" | "test" | "all".
    """
    tasks = load_tasks(data_cfg)
    groups_by_task = resolve_label_groups_per_task(label_spec, tasks)
    labels_by_task = {t.name: t.labels for t in tasks}
    dataset = build_dataset(data_cfg, split=split)
    ids = sorted((e["case_id"], i) for i, e in enumerate(dataset.data))
    if max_cases:
        ids = ids[:max_cases]
    return [(case_id, dataset, i, groups_by_task, labels_by_task) for case_id, i in ids]


def prepare_ref(ref: tuple, spacing_mm: tuple, alpha: float, native: bool = True) -> dict:
    """Load and resample one referenced case; returns its sdf_case."""
    case_id, dataset, i, groups_by_task, labels_by_task = ref
    case, entry = dataset[i], dataset.data[i]
    groups = groups_by_task[case["task"]]
    return prepare_sdf_case(case, entry["mask"], groups,
                            group_names(groups, labels_by_task[case["task"]]),
                            spacing_mm, alpha, native=native)


def score_encoder(cfg: dict, sdf_case: dict, encoder_state_dict, budgets, freeze_encoder: bool,
                  device, seed: int, separate_fits: bool = False, summarize: bool = False,
                  on_rows=None) -> list:
    """
    Fit a fresh decoder on top of encoder_state_dict (None = random encoder),
    frozen or not, and score it at every step count in budgets.

    By default one fit runs to max(budgets) and is scored at each budget;
    separate_fits gives each budget its own fit with its own lr schedule.
    on_rows(new_rows, evaluation) is called after each budget, e.g. to
    print or log progress. Returns rows with case_id, task, steps,
    fit_seconds, psnr and the per-label metrics of evaluate_sdf_fit.
    """
    decode_mode = cfg["sdf"].get("decode_mode", "independent")
    budgets = sorted(set(budgets))
    rows = []

    def on_eval(step, pred_sdf, fit_seconds):
        ev = evaluate_sdf_fit(pred_sdf, sdf_case, decode_mode)
        new = [{"case_id": sdf_case["case_id"], "task": sdf_case["task"], "steps": step,
                "fit_seconds": round(fit_seconds, 2), "psnr": ev["psnr"], **lab}
               for lab in ev["labels"]]
        rows.extend(new)
        if on_rows is not None:
            on_rows(new, ev)

    fits = [(b, [b]) for b in budgets] if separate_fits else [(budgets[-1], budgets)]
    for steps, eval_at in fits:
        fit_sdf(cfg, sdf_case, steps, encoder_state_dict=encoder_state_dict,
                freeze_encoder=freeze_encoder, device=device, eval_at=eval_at,
                on_eval=on_eval, seed=seed, summarize=summarize)
        summarize = False
    return rows


def format_progress(label: str, rows: list, ev: dict) -> str:
    """One console line for a scored budget: native/resampled Dice per label."""
    per_label = " ".join(f"{r['label']}={r['native_dice']:.3f}/{r['dice']:.3f}" for r in rows)
    return (f"  {label:<10} steps={rows[0]['steps']:>5} native/resampled dice: {per_label} "
            f"(mean {ev['native_mean_dice']:.4f}/{ev['mean_dice']:.4f}) {rows[0]['fit_seconds']:.0f}s")
