"""
Translation test: is an image-pretrained (Encoder I) or SDF-pretrained
(Encoder II) STRAINER encoder a better prior for fitting an unseen case's
mask SDF -- and is either better than no prior at all?

Arms, fitted to every case with identical settings:
  enc_I    encoder initialized from the Encoder I checkpoint
  enc_II   encoder initialized from the Encoder II checkpoint
  random   randomly initialized encoder (the no-prior baseline)
Protocol: the encoder is frozen and only a fresh decoder is fitted (default,
as in visualize_sdf_fit), or with --train_encoder the whole network is fitted
from the prior (STRAINER's test-time protocol). The random arm follows the
same protocol, so "frozen" compares fixed random features with fixed priors.

Fit length: every case/arm is fitted once for max(--steps) and scored at each
budget in --steps (an "anytime" curve; the lr schedule spans the longest
budget). --separate_fits instead fits each budget separately with its own
full schedule. The summary reports whether the curves have flattened at the
last budget, which is what backs a claim at "adequate" fitting length.

Cases: the held-out --split of the tasks the encoders were trained on
("internal"), and/or every case of tasks they never saw (--external_root /
--external_tasks, "external"), which tests whether the prior generalizes.
--shard i/n partitions the case list across jobs; --merge combines the
shards' per-label CSVs into one summary.

Results go to CSV and, unless --no_wandb, to Weights & Biases: every scored
fit as it happens (keyed <group>/<arm>/...), and at the end the per-label
rows and summary tables as wandb Tables, the headline medians and Holm
p-values in the run summary, and the CSV files as an artifact. Use the same
--wandb_group for all shards and their --merge run.

Fairness: task and fit settings (alpha, eikonal_lambda, label_groups,
decode_mode, optimizer, scheduler, points) come from ONE config -- Encoder
II's checkpoint, or --config -- and apply to every arm. Each case uses one
seed for all arms, so decoder initialization and point batches are identical
and arms differ only by encoder. Fitting and scoring are the same code as
visualize_sdf_fit (training/sdf_fit.py).

Dice: per label group, on the resampled grid ("dice", what the INR fits) and
on the original label file's grid ("native_dice", against the original
annotation; the headline number). Labels absent from a case's ground truth
are NaN and excluded. Paired Wilcoxon signed-rank tests over cases compare
enc_II - enc_I, enc_I - random and enc_II - random at every budget, with Holm
correction within each (case group, metric).

Usage:
  python -m experiments.run_translation_test \\
      --encoder_I_ckpt checkpoints/encoder_I_intensity \\
      --encoder_II_ckpt checkpoints/encoder_II_sdf \\
      --split test --steps 250 500 1000 2000 4000 \\
      --external_root /data/Decathlon --external_tasks Task09_Spleen \\
      --shard 0/4 --out results/tt_shard0 --wandb_group tt_frozen
  python -m experiments.run_translation_test --merge results/tt_shard*_per_label.csv \\
      --out results/tt_all --wandb_group tt_frozen --config config/encoder_II_sdf.yaml
"""
import argparse
import copy
import json
import math
import re
from pathlib import Path

from utils.config import load_config
from utils.io import load_model_weights
from utils.logging_utils import configure_wandb
from utils.results_io import RowWriter, read_rows as _read_rows, write_csv
from utils.stats import ARMS, shard, summarize
from training.train_loop import resolve_device
from training.sdf_fit import case_seed
from training.encoder_eval import load_case_refs, prepare_ref, score_encoder, format_progress

FIELDS = ["group", "task", "case_id", "arm", "protocol", "schedule", "steps", "fit_seconds",
          "psnr", "label", "gt_voxels", "dice", "nsd", "oracle_dice", "native_dice", "native_nsd"]
INT_FIELDS = {"steps", "gt_voxels"}
FLOAT_FIELDS = {"fit_seconds", "psnr", "dice", "nsd", "oracle_dice", "native_dice", "native_nsd"}


def read_rows(paths: list) -> list:
    """Per-label rows with numeric fields parsed (NaN stays NaN)."""
    return _read_rows(paths, INT_FIELDS, FLOAT_FIELDS)


# ---------------------------------------------------------------- config

def _encoder_keys(model_cfg: dict) -> dict:
    return {k: v for k, v in model_cfg.items() if k != "decoder_layers"}


def resolve_fit_config(ckpt_I: dict, ckpt_II: dict, config_path: str, data_root: str) -> dict:
    cfg_I, cfg_II = ckpt_I["config"], ckpt_II["config"]
    if _encoder_keys(cfg_I["model"]) != _encoder_keys(cfg_II["model"]):
        raise SystemExit(f"Encoders have different architectures ({cfg_I['model']} vs "
                         f"{cfg_II['model']}); their weights are not comparable.")
    if cfg_I["data"]["spacing_mm"] != cfg_II["data"]["spacing_mm"]:
        raise SystemExit("Encoders were trained at different voxel spacings.")
    # Same tasks and split, or a "held-out" case for one may be a training case for the other.
    for key in ("tasks", "split"):
        if cfg_I["data"][key] != cfg_II["data"][key]:
            raise SystemExit(f"Encoders were trained with different data.{key}: "
                             f"{cfg_I['data'][key]} vs {cfg_II['data'][key]}")

    cfg = copy.deepcopy(cfg_II)
    if config_path:
        override = load_config(config_path)
        for block in ("sdf", "optimizer", "scheduler", "training"):
            if block in override:
                cfg[block] = override[block]
        # The decoder is always fresh, so its depth may differ from training.
        if "decoder_layers" in override.get("model", {}):
            cfg["model"]["decoder_layers"] = override["model"]["decoder_layers"]
    if data_root:
        cfg["data"]["root"] = data_root
    return cfg


# ---------------------------------------------------------------- cases

def collect_cases(cfg: dict, args, trained_tasks: set) -> list:
    """[(group, ref)] with refs from training/encoder_eval.load_case_refs,
    sorted, capped per group by --max_cases, then sharded."""
    if bool(args.external_root) != bool(args.external_tasks):
        if not args.external_root:
            raise SystemExit("--external_tasks needs --external_root (the folder holding them).")
        root = Path(args.external_root)
        found = sorted(d.name for d in root.iterdir() if (d / "dataset.json").is_file()) \
            if root.is_dir() else []
        listing = ", ".join(f"{t} (trained on, not allowed)" if t in trained_tasks else t
                            for t in found) or "none -- is the path right?"
        raise SystemExit(f"--external_root is set but --external_tasks is not, so no external cases "
                         f"would be evaluated. Tasks with a dataset.json under {root}: {listing}")

    refs, group_names_seen = [], []
    if not args.skip_internal:
        refs += [("internal", r) for r in load_case_refs(
            cfg["data"], cfg["sdf"]["label_groups"], args.split, args.max_cases)]
        group_names_seen.append("internal")
    if args.external_tasks:
        overlap = set(args.external_tasks) & trained_tasks
        if overlap:
            raise SystemExit(f"External tasks {sorted(overlap)} were used to train the encoders; "
                             f"they cannot test generalization.")
        ext_cfg = {"root": args.external_root, "tasks": args.external_tasks,
                   "spacing_mm": cfg["data"]["spacing_mm"]}
        spec = "auto" if args.external_label_groups == "auto" else json.loads(args.external_label_groups)
        refs += [("external", r) for r in load_case_refs(ext_cfg, spec, "all", args.max_cases)]
        group_names_seen.append("external")
    if not group_names_seen:
        raise SystemExit("Nothing to evaluate: --skip_internal without --external_tasks.")

    selected = shard(refs, args.shard, key=lambda r: (r[0], r[1][0]))
    counts = {g: sum(r[0] == g for r in selected) for g in group_names_seen}
    print("cases to evaluate" + (f" (shard {args.shard})" if args.shard else "") + ": "
          + ", ".join(f"{g} {n}" for g, n in counts.items()))
    return selected


# ---------------------------------------------------------------- csv

# ---------------------------------------------------------------- wandb

def _nanmean(values) -> float:
    values = [float(v) for v in values if not math.isnan(float(v))]
    return sum(values) / len(values) if values else float("nan")


def _wandb_value(v):
    """wandb tables and summaries take None, not NaN."""
    return None if isinstance(v, float) and math.isnan(v) else v


class WandbLogger:
    """
    Mirrors the CSVs in Weights & Biases. log_fit: one entry per scored fit
    (case x arm x budget) as it happens. finish: the per-label rows and the
    summary tables as wandb Tables, headline medians and Holm p-values in the
    run summary, and the CSV files as an artifact. A no-op without a run.
    """

    def __init__(self, wandb_cfg: dict, name: str, config: dict, group: str, job_type: str):
        self.run = None
        if wandb_cfg is None:
            return
        # The training run_name would otherwise name this run too.
        cfg = {k: v for k, v in wandb_cfg.items() if k != "run_name"}
        self.run = configure_wandb(cfg, name, config, group=group, job_type=job_type)

    def log_fit(self, rows: list):
        if self.run is None or not rows:
            return
        r0 = rows[0]
        prefix = f"{r0['group']}/{r0['arm']}"
        data = {"case_id": r0["case_id"], "task": r0["task"], "budget": r0["steps"],
                f"{prefix}/fit_seconds": r0["fit_seconds"], f"{prefix}/psnr": r0["psnr"]}
        for metric in ("native_dice", "dice", "native_nsd", "nsd", "oracle_dice"):
            data[f"{prefix}/mean_{metric}"] = _nanmean(r[metric] for r in rows)
            for r in rows:
                data[f"{prefix}/{metric}/{r['label']}"] = r[metric]
        self.run.log({k: _wandb_value(v) for k, v in data.items()})

    def finish(self, rows: list, summary: dict, files: list, artifact_name: str):
        if self.run is None:
            return
        import wandb

        def table(rs):
            columns = list(rs[0].keys()) if rs else []
            return wandb.Table(columns=columns,
                               data=[[_wandb_value(r[c]) for c in columns] for r in rs])

        self.run.log({"per_label": table(rows),
                      **{f"summary/{name}": table(summary[name])
                         for name in ("table", "tests", "convergence")}})
        for r in summary["table"]:
            for key, v in r.items():
                if key.startswith("median_"):
                    self.run.summary[f"{r['group']}/{r['arm']}/{key}@{r['steps']}"] = _wandb_value(v)
        for t in summary["tests"]:
            pair = t["pair"].replace(" ", "")
            self.run.summary[f"{t['group']}/p_holm/{t['metric']}/{pair}@{t['steps']}"] = \
                _wandb_value(t["p_holm"])
        for c in summary["convergence"]:
            self.run.summary[f"{c['group']}/{c['arm']}/converged/{c['metric']}"] = c["converged"]
        artifact = wandb.Artifact(re.sub(r"[^A-Za-z0-9_.-]", "-", artifact_name),
                                  type="translation_results")
        for f in files:
            if Path(f).exists():
                artifact.add_file(str(f))
        self.run.log_artifact(artifact)
        self.run.finish()


def wandb_settings(args, cfg: dict):
    """The wandb block to log with (None = don't): the fit config's, else
    --config's (for --merge), with --wandb_entity/--wandb_project on top."""
    if args.no_wandb:
        return None
    block = dict((cfg or {}).get("wandb") or {})
    if not block and args.config:
        block = dict(load_config(args.config).get("wandb") or {})
    if args.wandb_entity:
        block["entity"] = args.wandb_entity
    if args.wandb_project:
        block["project"] = args.wandb_project
    if not (block.get("entity") and block.get("project")):
        print("No wandb entity/project (give --config with a wandb block or --wandb_entity "
              "and --wandb_project); results go to CSV only.")
        return None
    return block


# ---------------------------------------------------------------- summary

def report(rows: list, out: str, tol: float):
    has_native = any(not math.isnan(float(r["native_dice"])) for r in rows)
    metrics = ("native_dice", "dice") if has_native else ("dice",)
    headline = metrics[0]
    s = summarize(rows, metrics=metrics, tol=tol)
    files = [Path(f"{out}_{name}.csv") for name in ("table", "tests", "convergence")]
    for name, path in zip(("table", "tests", "convergence"), files):
        write_csv(path, s[name])

    print(f"\n==== {headline} (median of per-case means over label groups) ====")
    for group in sorted({r["group"] for r in s["table"]}):
        print(f"[{group}]")
        arms = [a for a in ARMS if any(r["arm"] == a and r["group"] == group for r in s["table"])]
        print("  steps  " + "".join(f"{a:>16}" for a in arms))
        for steps in sorted({r["steps"] for r in s["table"] if r["group"] == group}):
            cells = {r["arm"]: r for r in s["table"]
                     if r["group"] == group and r["steps"] == steps}
            print(f"  {steps:>5}  " + "".join(
                f"{cells[a][f'median_{headline}']:>10.4f} (n={cells[a][f'n_{headline}']:<2})"
                for a in arms))
        print("  paired Wilcoxon (median difference, Holm-adjusted p):")
        for t in s["tests"]:
            if t["group"] == group and t["metric"] == headline:
                print(f"    steps={t['steps']:>5} {t['pair']:<17} n={t['n']:<3} "
                      f"diff={t['median_diff']:+.4f} p={t['p']:.4g} p_holm={t['p_holm']:.4g}")
        print("  convergence (median change over the last two budgets):")
        for c in s["convergence"]:
            if c["group"] == group and c["metric"] == headline:
                flag = "converged" if c["converged"] else "STILL CHANGING"
                print(f"    {c['arm']:<7} {c['steps_prev']}->{c['steps_last']}: "
                      f"{c['median_prev']:.4f} -> {c['median_last']:.4f} "
                      f"({c['change']:+.4f}, {flag} at tol {tol})")
    print(f"\nSummary CSVs: {out}_table.csv, {out}_tests.csv, {out}_convergence.csv")
    return s, files


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoder_I_ckpt")
    parser.add_argument("--encoder_II_ckpt")
    parser.add_argument("--config", default=None,
                        help="Override fit settings (sdf/optimizer/scheduler/training blocks, "
                             "model.decoder_layers); default: Encoder II's checkpoint config")
    parser.add_argument("--data_root", default=None, help="Override data.root of the trained tasks")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--skip_internal", action="store_true", help="Only evaluate external tasks")
    parser.add_argument("--external_root", default=None, help="Folder with external Task*/dataset.json")
    parser.add_argument("--external_tasks", nargs="+", default=None)
    parser.add_argument("--external_label_groups", default="auto",
                        help='"auto" or a JSON spec such as \'[["foreground"]]\'')
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--steps", nargs="+", type=int, default=[250, 500, 1000, 2000, 4000])
    parser.add_argument("--separate_fits", action="store_true",
                        help="Fit each budget separately with its own schedule")
    parser.add_argument("--train_encoder", action="store_true",
                        help="Fit the whole network from the prior (default: frozen encoder)")
    parser.add_argument("--no_native", action="store_true",
                        help="Skip native-grid metrics (saves time/memory on very large CTs)")
    parser.add_argument("--max_cases", type=int, default=None, help="Cap per case group")
    parser.add_argument("--shard", default=None, help="i/n: evaluate the i-th of n case partitions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--converged_tol", type=float, default=0.01)
    parser.add_argument("--out", default="results/translation_test",
                        help="Output prefix: <out>_per_label.csv and summary CSVs")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted --out")
    parser.add_argument("--merge", nargs="+", default=None,
                        help="Summarize existing per-label CSVs (e.g. from shards) without fitting")
    parser.add_argument("--no_wandb", action="store_true", help="CSV only")
    parser.add_argument("--wandb_entity", default=None, help="Override the config's wandb entity")
    parser.add_argument("--wandb_project", default=None, help="Override the config's wandb project")
    parser.add_argument("--wandb_name", default=None, help="Run name (default: basename of --out)")
    parser.add_argument("--wandb_group", default=None,
                        help="Group shards and their merge run together")
    args = parser.parse_args()

    run_name = args.wandb_name or Path(args.out).name
    if args.merge:
        rows = read_rows(args.merge)
        merged = Path(f"{args.out}_per_label.csv")
        if merged.resolve() in {Path(m).resolve() for m in args.merge}:
            raise SystemExit(f"--out would overwrite input {merged}; choose another --out.")
        merged.parent.mkdir(parents=True, exist_ok=True)
        write_csv(merged, rows)
        logger = WandbLogger(wandb_settings(args, None), run_name,
                             {"translation_test": vars(args)}, args.wandb_group,
                             job_type="translation_test_merge")
        summary, files = report(rows, args.out, args.converged_tol)
        logger.finish(rows, summary, [merged, *files], artifact_name=f"translation-{run_name}")
        return
    if not (args.encoder_I_ckpt and args.encoder_II_ckpt):
        parser.error("--encoder_I_ckpt and --encoder_II_ckpt are required unless --merge")

    ckpt_I = load_model_weights(args.encoder_I_ckpt)
    ckpt_II = load_model_weights(args.encoder_II_ckpt)
    cfg = resolve_fit_config(ckpt_I, ckpt_II, args.config, args.data_root)
    trained_tasks = set(ckpt_I["config"]["data"]["tasks"]) | set(ckpt_II["config"]["data"]["tasks"])
    spacing_mm = tuple(cfg["data"]["spacing_mm"])
    alpha = cfg["sdf"]["alpha"]
    device = resolve_device(cfg.get("training", {}).get("device", "auto"))
    budgets = sorted(set(args.steps))
    protocol = "finetune" if args.train_encoder else "frozen"
    schedule = "separate_fits" if args.separate_fits else "single_fit"
    encoders = {"enc_I": ckpt_I["encoder_state_dict"], "enc_II": ckpt_II["encoder_state_dict"],
                "random": None}

    refs = collect_cases(cfg, args, trained_tasks)
    out_path = Path(f"{args.out}_per_label.csv")
    done = set()
    if args.resume and out_path.exists():
        seen = {}
        for r in read_rows([out_path]):
            seen.setdefault((r["case_id"], r["arm"]), set()).add(int(r["steps"]))
        done = {key for key, s in seen.items() if set(budgets) <= s}
    writer = RowWriter(out_path, FIELDS, args.resume)
    logger = WandbLogger(wandb_settings(args, cfg), run_name,
                         {"translation_test": vars(args), "fit_config": cfg}, args.wandb_group,
                         job_type="translation_test")
    print(f"{len(refs)} cases x {len(args.arms)} arms, budgets {budgets} ({schedule}, {protocol} "
          f"encoder), device {device}; {len(done)} case/arm fits already done")

    first = True
    for n, (group, ref) in enumerate(refs, 1):
        case_id = ref[0]
        todo = [a for a in args.arms if (case_id, a) not in done]
        if not todo:
            continue
        sdf_case = prepare_ref(ref, spacing_mm, alpha, native=not args.no_native)
        seed = case_seed(args.seed, case_id)
        print(f"\n[{n}/{len(refs)}] {group} {case_id} grid {sdf_case['shape']}, "
              f"groups {dict(zip(sdf_case['names'], sdf_case['groups']))}")

        for arm in todo:
            meta = {"group": group, "arm": arm, "protocol": protocol, "schedule": schedule}

            def on_rows(new, ev):
                logger.log_fit([{**meta, **r} for r in new])
                print(format_progress(arm, new, ev))

            rows = score_encoder(cfg, sdf_case, encoders[arm], budgets,
                                 freeze_encoder=not args.train_encoder, device=device, seed=seed,
                                 separate_fits=args.separate_fits, summarize=first, on_rows=on_rows)
            first = False
            writer.write([{**meta, **r} for r in rows])

    print(f"\nPer-label results: {out_path}")
    rows = read_rows([out_path])
    summary, files = report(rows, args.out, args.converged_tol)
    logger.finish(rows, summary, [out_path, *files], artifact_name=f"translation-{run_name}")


if __name__ == "__main__":
    main()
