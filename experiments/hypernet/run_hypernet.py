"""
Hypernetwork: predict a case's SDF decoder from its intensity decoder.

Inputs are the per-image intensity decoders from run_decoder_training (one
per case, or per case/modality), each flattened to a vector. An MLP maps that
vector to the weights of an SDF decoder that sits on a frozen encoder; it is
trained through the decoder's output with the masked MSE + Eikonal SDF loss,
so it never needs target weights, only each case's clipped mask SDF. The
predicted decoder is thresholded for Dice and NSD as in visualize_sdf_fit.

Labels: with sdf.per_label (the output-side counterpart of
data.per_modality) every label group of a case is its own 1-channel SDF
decoder, so tasks with different label sets (spleen; prostate PZ, TZ) share
one decoder size; the MLP is told which label to predict by a one-hot of
"<task>/<group>". Otherwise one decoder covers all K groups and every task
must have the same K. For scoring, a case's per-label predictions are
stacked back into its K channels.

Methods, all scored on the same held-out test cases:
  floor     per label, one SDF decoder fitted to every training case at once
            (the average shape): what a decoder knows without seeing the case
  ridge     the highest-bias weight-space map (hypernet.ridge; null skips it):
            per label, PCA of the training intensity decoders, then ridge
            regression onto the training cases' ceiling decoders, the penalty
            chosen by leave-one-case-out error (experiments/hypernet/ridge.py)
  hypernet  the MLP's prediction from the case's intensity decoder (and label)
  ceiling   per label, a decoder fitted to the test case's own SDF (frozen
            encoder, training/sdf_fit.fit_sdf): the best this decoder can do
Expected: floor < ridge, hypernet < ceiling. Paired Wilcoxon tests over cases
(Holm corrected per metric) test the gaps.

Ceiling fits, of training cases (ridge's targets) and test cases alike, all
start from one shared initialization (--seed), as the intensity decoders do
(run_decoder_training), so decoders share one weight-space chart and can be
averaged and regressed on.

Cases are split deterministically, by case (all modalities of a case on one
side), from hypernet.test_fraction and hypernet.split_seed; the split is
saved and checked on --resume.

The SDF decoder sits on the same frozen encoder as the intensity decoders
(Encoder I, the decoders folder's encoder_ckpt) and has their width and depth
unless hypernet.decoder_hidden_dim / decoder_layers say otherwise, so the
output vector is the size of the input when the label-group count equals the
image channel count.

Config: the hypernet block of --config (see config/training_config.yaml),
plus sdf (alpha, eikonal_lambda, label_groups, decode_mode), optimizer and
scheduler (the floor's and the ceiling's decoder fits), training
(points_per_step for the ceiling, eval_chunk_size, device) and wandb. The
data settings come from the decoders folder's config.json.

Outputs in --out_dir (default <decoders_dir>_hypernet):
  split.json             train/test case ids
  floor_decoders.pt      {label: the floor's decoder state_dict}
  hypernet.pt            the MLP (with its normalization and base), sizes, config
  predicted_decoders/    each test input's (and label's) predicted SDF decoder
  predicted_decoders_ridge/   the same for ridge
  ceiling_decoders/      every ceiling fit (training and test cases), reused on --resume
  ridge.json             per label: chosen k and penalty, explained variance and
                         the leave-one-case-out search, against the mean decoder
  results.csv            method x test input x label group: Dice/NSD (resampled
                         and native), PSNR, fit or prediction seconds
  summary.csv            per method: n / mean / median over cases of the
                         per-case mean (hypernet: averaged over modalities first)
  summary_per_label.csv, summary_per_modality.csv   the same, split further
  tests.csv              paired Wilcoxon of each pair in PAIRS
WandB gets the training curves (floor/*, hypernet/*, with test-case loss as a
monitor), every evaluation, the summaries as tables and the headline means
and ordering in the run summary.

Usage:
  python -m experiments.hypernet.run_hypernet --config config/training_config.yaml \\
      --decoders_dir checkpoints/encoder_I_intensity_decoders_d64_L2 [--resume]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.hypernet.data import load_decoder_bank, split_cases, load_sdf_targets, label_targets
from experiments.hypernet.model import HeadLayout, HyperMLP, flatten_state_dict
from experiments.hypernet.ridge import PCARidge
from experiments.hypernet.train import train_weights, predict_and_score
from models.interfaces import DECODER_KEYS, build_model
from training.sdf_fit import fit_sdf
from training.train_loop import build_optimizer, build_scheduler, resolve_device
from utils.config import load_config
from utils.io import load_model_weights, save_decoder_weights
from utils.logging_utils import configure_wandb
from utils.results_io import RowWriter, read_rows, write_csv
from utils.stats import aggregate_cases, paired_wilcoxon, holm_adjust

FIELDS = ["method", "case_id", "input", "task", "modality", "steps", "fit_seconds", "psnr", "label",
          "gt_voxels", "dice", "nsd", "oracle_dice", "native_dice", "native_nsd"]
INT_FIELDS = ("steps", "gt_voxels")
FLOAT_FIELDS = ("fit_seconds", "psnr", "dice", "nsd", "oracle_dice", "native_dice", "native_nsd")
METRICS = ("native_dice", "dice", "native_nsd", "nsd")
# Each pair is reported as first - second; all are expected to be positive.
PAIRS = (("hypernet", "floor"), ("ridge", "floor"), ("hypernet", "ridge"),
         ("ceiling", "hypernet"), ("ceiling", "ridge"), ("ceiling", "floor"))
METHODS = ("floor", "ridge", "hypernet", "ceiling")
RIDGE_DEFAULTS = {"components": None, "alphas": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]}

DEFAULTS = {
    "decoders_dir": None, "test_fraction": 0.2, "split_seed": 0,
    "hidden_dim": 1024, "layers": 4, "activation": "gelu", "lr": 1e-4, "weight_decay": 0.01,
    "scheduler": "cosine", "steps": 20000, "cases_per_step": 8, "points_per_step": 2 ** 16,
    "log_every": 100, "init_from_floor": True, "floor_steps": 4000, "ceiling_steps": 2000,
    "ridge": None,
}


# ---------------------------------------------------------------- setup

def sdf_model_config(decoder_model: dict, hcfg: dict) -> dict:
    """The intensity decoders' model (their encoder, depth and width), with
    hypernet.decoder_layers / decoder_hidden_dim overriding the SDF decoder's."""
    return {**decoder_model, **{k: hcfg[k] for k in DECODER_KEYS if hcfg.get(k) is not None}}


# ---------------------------------------------------------------- training

def train_floor(model, layout, train_items, test_items, cfg, hcfg, spacing_mm, device, on_log):
    """One fresh decoder fitted to all of train_items (every training case's
    target for one label); returns its flat weights."""
    model.reset_decoder()
    head = model.decoder[0]
    for p in head.parameters():
        p.requires_grad_(True)
    optimizer = build_optimizer(list(head.parameters()), cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], hcfg["floor_steps"])
    shared = lambda chosen: [list(head.parameters())] * len(chosen)
    train_weights(model, train_items, shared, optimizer, scheduler, hcfg["floor_steps"],
                  hcfg["cases_per_step"], hcfg["points_per_step"], cfg["sdf"], spacing_mm, device,
                  hcfg["log_every"], on_log, test_items, shared)
    for p in head.parameters():
        p.requires_grad_(False)
    return layout.flatten(head)


def hypernet_params_for(hyper, layout, items, device):
    """params_for (experiments/hypernet/train.train_weights) of the hypernet:
    a batched MLP forward on the chosen items' input vectors and labels."""
    inputs = torch.stack([i["x"] for i in items]).to(device)
    labels = torch.tensor([i["label_index"] for i in items], device=device)
    return lambda chosen: [layout.unflatten(v) for v in hyper(inputs[chosen], labels[chosen])]


def train_hypernet(hyper, model, layout, train_items, test_items, cfg, hcfg, spacing_mm, device,
                   on_log):
    optimizer = torch.optim.AdamW(hyper.parameters(), lr=hcfg["lr"], weight_decay=hcfg["weight_decay"])
    scheduler = build_scheduler(optimizer, {"name": hcfg["scheduler"]}, hcfg["steps"])
    hyper.train()
    train_weights(model, train_items, hypernet_params_for(hyper, layout, train_items, device),
                  optimizer, scheduler, hcfg["steps"], hcfg["cases_per_step"], hcfg["points_per_step"],
                  cfg["sdf"], spacing_mm, device, hcfg["log_every"], on_log, test_items,
                  hypernet_params_for(hyper, layout, test_items, device))
    hyper.eval()


# ---------------------------------------------------------------- report

def case_values(rows: list, method: str, metric: str) -> dict:
    """{case_id: mean over the case's label rows (and modalities)}, NaN excluded."""
    by_case = {}
    for r in rows:
        if r["method"] == method and not math.isnan(r[metric]):
            by_case.setdefault(r["case_id"], []).append(r[metric])
    return {c: float(np.mean(v)) for c, v in by_case.items()}


def paired_tests(rows: list) -> list:
    tests = []
    for metric in METRICS:
        family = []
        for a, b in PAIRS:
            va, vb = case_values(rows, a, metric), case_values(rows, b, metric)
            cases = sorted(set(va) & set(vb))
            if cases:
                family.append({"metric": metric, "pair": f"{a} - {b}",
                               **paired_wilcoxon([va[c] for c in cases], [vb[c] for c in cases])})
        for row, p in zip(family, holm_adjust([r["p"] for r in family])):
            row["p_holm"] = p
        tests += family
    return tests


def report(rows: list, out_dir: Path, wandb_run):
    """Write the summaries and tests, print the headline, mirror to wandb."""
    summary = aggregate_cases(rows, ("method", "steps"), METRICS)
    tables = {"summary": summary,
              "summary_per_label": aggregate_cases(rows, ("method", "task", "label"), METRICS),
              "summary_per_modality": aggregate_cases(rows, ("method", "modality"), METRICS),
              "tests": paired_tests(rows)}
    for name, table in tables.items():
        write_csv(out_dir / f"{name}.csv", table)

    metric = "native_dice" if any(not math.isnan(r["native_dice"]) for r in rows) else "dice"
    by_method = {r["method"]: r for r in summary}
    print(f"\n==== {metric} over test cases: mean / median (n) ====")
    for method in METHODS:
        if method in by_method:
            r = by_method[method]
            print(f"  {method:<9} {r[f'mean_{metric}']:.4f} / {r[f'median_{metric}']:.4f} "
                  f"({r[f'n_{metric}']})")
    for t in tables["tests"]:
        if t["metric"] == metric:
            print(f"  {t['pair']:<20} median diff {t['median_diff']:+.4f}, p_holm={t['p_holm']:.3g} "
                  f"(n={t['n']})")
    mean = {m: by_method[m][f"mean_{metric}"] for m in by_method}
    ordering = {}
    for method in ("ridge", "hypernet"):
        if "floor" in mean and method in mean:
            ordering[f"{method}_beats_floor"] = bool(mean[method] > mean["floor"])
        if "ceiling" in mean and method in mean:
            ordering[f"{method}_below_ceiling"] = bool(mean[method] < mean["ceiling"])
    print("  " + ", ".join(f"{k}: {v}" for k, v in ordering.items()))
    print(f"\nResults: {out_dir / 'results.csv'}; summaries: " +
          ", ".join(f"{out_dir / name}.csv" for name in tables))

    if wandb_run is None:
        return
    import wandb
    clean = lambda v: None if isinstance(v, float) and math.isnan(v) else v
    wandb_run.summary.update({**ordering, **{f"{r['method']}/{stat}_{m}": clean(r[f"{stat}_{m}"])
                                             for r in summary for m in METRICS
                                             for stat in ("mean", "median")}})
    for name, table in [("results", rows)] + list(tables.items()):
        if table:
            columns = list(table[0])
            wandb_run.log({name: wandb.Table(columns=columns,
                                             data=[[clean(r[c]) for c in columns] for r in table])})


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="hypernet, sdf, optimizer, scheduler, training, wandb")
    parser.add_argument("--decoders_dir", default=None, help="Override hypernet.decoders_dir")
    parser.add_argument("--data_root", default=None, help="Default: the decoders' data root")
    parser.add_argument("--out_dir", default=None, help="Default: <decoders_dir>_hypernet")
    parser.add_argument("--skip_ceiling", action="store_true",
                        help="Skip the ceiling's test-case fits (ridge still fits training cases)")
    parser.add_argument("--skip_hypernet", action="store_true", help="Skip training and scoring the MLP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse saved floor/hypernet and skip evaluations already in results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if "sdf" not in cfg:
        raise SystemExit("The config needs an sdf: block (alpha, eikonal_lambda, label_groups, decode_mode).")
    hcfg = {**DEFAULTS, **(cfg.get("hypernet") or {})}
    if hcfg["ridge"] is not None:
        hcfg["ridge"] = {**RIDGE_DEFAULTS, **hcfg["ridge"]}
    if args.decoders_dir:
        hcfg["decoders_dir"] = args.decoders_dir
    if not hcfg["decoders_dir"]:
        raise SystemExit("Set hypernet.decoders_dir or --decoders_dir (a run_decoder_training output folder).")
    decoders_dir = Path(hcfg["decoders_dir"])
    dec_config, inputs = load_decoder_bank(decoders_dir)
    encoder = load_model_weights(dec_config["encoder_ckpt"])["encoder_state_dict"]
    data_cfg = dict(dec_config["data"])
    if args.data_root:
        data_cfg["root"] = args.data_root
    spacing_mm = tuple(data_cfg["spacing_mm"])
    train_cfg = cfg.get("training") or {}
    device = resolve_device(train_cfg.get("device", "auto"))
    chunk = train_cfg.get("eval_chunk_size", 2 ** 20)
    decode_mode = cfg["sdf"].get("decode_mode", "independent")
    sdf_model = sdf_model_config(dec_config["model"], hcfg)

    # Split by case and refuse to resume into a different experiment.
    split = split_cases([i["case_id"] for i in inputs], hcfg["test_fraction"], hcfg["split_seed"])
    out_dir = Path(args.out_dir or f"{decoders_dir}_hypernet")
    setup = {"decoders_dir": str(decoders_dir), "encoder_ckpt": dec_config["encoder_ckpt"],
             "sdf_model": sdf_model, "sdf": cfg["sdf"], "hypernet": hcfg, "split": split,
             "seed": args.seed}
    saved = out_dir / "config.json"
    if args.resume and saved.exists() and json.loads(saved.read_text()) != json.loads(json.dumps(setup)):
        raise SystemExit(f"{saved} was written with different settings; resuming would mix "
                         f"experiments. Choose another --out_dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(setup, indent=2))
    (out_dir / "split.json").write_text(json.dumps(split, indent=2))

    test_ids = set(split["test"])
    print(f"{len(inputs)} decoders from {len(split['train']) + len(split['test'])} cases: "
          f"{len(split['train'])} train / {len(split['test'])} test cases; loading SDFs")
    alpha, per_label = cfg["sdf"]["alpha"], cfg["sdf"].get("per_label", False)
    targets = load_sdf_targets(data_cfg, cfg["sdf"]["label_groups"], {i["case_id"] for i in inputs},
                               test_ids, spacing_mm, alpha)
    # Decoder targets per case: one per label group with sdf.per_label, else
    # one of all groups (then every task needs the same group count).
    parts = {c: label_targets(t, per_label, alpha) for c, t in targets.items()}
    widths = {t["task"]: parts[c][0]["target"].shape[1] for c, t in targets.items()}
    if len(set(widths.values())) > 1:
        raise SystemExit(f"Tasks resolve to different label-group counts {widths}, but one hypernet "
                         f"predicts one decoder size; set sdf.per_label: true (one 1-channel decoder "
                         f"per label group) or use an sdf.label_groups spec that gives every task "
                         f"the same count.")
    k = next(iter(widths.values()))
    labels = sorted({p["label"] for c in split["train"] for p in parts[c]})
    unseen = sorted({p["label"] for c in split["test"] for p in parts[c]} - set(labels))
    if unseen:
        raise SystemExit(f"Test labels {unseen} have no training case; change hypernet.split_seed "
                         f"or test_fraction.")
    label_index = {label: j for j, label in enumerate(labels)}

    def label_parts(case_ids, label=None):
        return [p for c in sorted(case_ids) for p in parts[c] if label in (None, p["label"])]

    def input_items(case_ids):
        """Hypernet items: one per (intensity decoder, label part)."""
        return [{**p, **i, "label_index": label_index[p["label"]]}
                for i in inputs if i["case_id"] in case_ids for p in parts[i["case_id"]]]

    train_items, test_items = input_items(set(split["train"])), input_items(test_ids)
    test_cases = [targets[c] for c in split["test"]]

    # The SDF model: frozen encoder, one decoder head whose weights are
    # swapped per case and label. Its fresh init sets the hypernet's per-tensor scale.
    torch.manual_seed(args.seed)
    model = build_model(sdf_model, out_features=k).to(device)
    model.load_encoder_state_dict(encoder, freeze=True)
    model.reset_decoder()
    for p in model.decoder_parameters():
        p.requires_grad_(False)
    head = model.decoder[0]
    layout = HeadLayout(head)
    fresh, scale = layout.flatten(head), layout.tensor_stds(head)
    in_dim = inputs[0]["x"].numel()
    sizes = {"input_size": in_dim, "output_size": layout.numel, "sdf_decoder": sdf_model,
             "out_features": k, "per_label": per_label, "labels": labels,
             "n_train_items": len(train_items), "n_test_items": len(test_items)}
    print(f"hypernet: {in_dim:,} intensity-decoder weights + label -> {layout.numel:,} SDF-decoder "
          f"weights ({k} channel{'s' * (k > 1)}) on {dec_config['encoder_ckpt']}; labels {labels}; "
          f"{len(train_items)} train / {len(test_items)} test items; device {device}")

    wandb_run = configure_wandb(cfg["wandb"], f"{cfg['experiment_name']}_hypernet",
                                {**setup, "sizes": sizes})
    if wandb_run is not None:
        for prefix in ("floor", "hypernet"):
            wandb_run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")

    def on_log(prefix):
        def log(step, values):
            if wandb_run is not None:
                wandb_run.log({f"{prefix.split('/')[0]}/step": step,
                               **{f"{prefix}/{k}": v for k, v in values.items()}})
        return log

    # Floor: one decoder per label, fitted to all training cases.
    floor_path = out_dir / "floor_decoders.pt"
    if args.resume and floor_path.exists():
        floors = {label: flatten_state_dict(sd).to(device)
                  for label, sd in torch.load(floor_path, map_location="cpu").items()}
        print(f"floor: reusing {floor_path}")
    else:
        floors, saved_floors = {}, {}
        for label in labels:
            print(f"floor {label}: one decoder fitted to all training cases")
            torch.manual_seed(args.seed)
            floors[label] = train_floor(model, layout, label_parts(split["train"], label),
                                        label_parts(test_ids, label), cfg, hcfg, spacing_mm, device,
                                        on_log(f"floor/{label}" if per_label else "floor"))
            saved_floors[label] = {n: v.detach().cpu().clone() for n, v in model.decoder_state_dict().items()}
        torch.save(saved_floors, floor_path)

    # Hypernet.
    hyper_path = out_dir / "hypernet.pt"
    hyper = None
    if args.skip_hypernet:
        print("hypernet: skipped (--skip_hypernet)")
    else:
        hyper = HyperMLP(in_dim, layout.numel, len(labels), hcfg["hidden_dim"], hcfg["layers"],
                         hcfg["activation"]).to(device)
    if hyper is not None and args.resume and hyper_path.exists():
        hyper.load_state_dict(torch.load(hyper_path, map_location=device)["state_dict"])
        hyper.eval()
        print(f"hypernet: reusing {hyper_path}")
    elif hyper is not None:
        n_params = sum(p.numel() for p in hyper.parameters())
        print(f"hypernet: {hcfg['layers']}-layer MLP, {hcfg['hidden_dim']} wide, {n_params:,} params, "
              f"starting from the {'floor' if hcfg['init_from_floor'] else 'fresh SIREN init'}")
        torch.manual_seed(args.seed)
        base = (torch.stack([floors[label] for label in labels]) if hcfg["init_from_floor"]
                else fresh.expand(len(labels), -1))
        hyper.set_reference(torch.stack([i["x"] for i in inputs if i["case_id"] not in test_ids]).to(device),
                            base, scale.to(device))
        train_hypernet(hyper, model, layout, train_items, test_items, cfg, hcfg, spacing_mm, device,
                       on_log("hypernet"))
        torch.save({"state_dict": hyper.state_dict(), **sizes, "hidden_dim": hcfg["hidden_dim"],
                    "layers": hcfg["layers"], "activation": hcfg["activation"]}, hyper_path)

    # Ceiling decoders: a direct fit per (case, label part), cached on disk.
    # Every fit uses the same seed, hence the same initial decoder.
    fit_cfg = {**cfg, "model": sdf_model}
    keys = list(model.decoder_state_dict())
    ceiling_dir = out_dir / "ceiling_decoders"

    def part_name(case_id, part):
        return f"{case_id}/{part['label'].split('/', 1)[1]}" if per_label else case_id

    def ceiling_decoder(case_id, part) -> tuple:
        """(flat decoder weights on CPU, fit seconds; NaN when reused)."""
        path = ceiling_dir / f"decoder_{part_name(case_id, part).replace('/', '__')}.pt"
        if path.exists():
            return flatten_state_dict(torch.load(path, map_location="cpu")), float("nan")
        shape = part["shape"]
        sdf_case = {"sdf": part["target"].float().reshape(*shape, -1).numpy(), "shape": shape,
                    "spacing_mm": spacing_mm}
        t0 = time.perf_counter()
        _, _, fitted = fit_sdf(fit_cfg, sdf_case, hcfg["ceiling_steps"], encoder_state_dict=encoder,
                               freeze_encoder=True, device=device, seed=args.seed, return_model=True)
        seconds = time.perf_counter() - t0
        state = {k: v.detach().cpu() for k, v in fitted.decoder_state_dict().items()}
        save_decoder_weights(state, part_name(case_id, part), ceiling_dir)
        return flatten_state_dict(state), seconds

    # Ridge: per label, PCA of the training intensity decoders -> ceiling decoders.
    ridges = {}
    if hcfg["ridge"] is not None:
        n_fits = sum(len(parts[c]) for c in split["train"])
        print(f"ridge: {n_fits} ceiling fits of training cases ({hcfg['ceiling_steps']} steps each, "
              f"cached in {ceiling_dir})")
        ceilings = {}
        for n, c in enumerate(split["train"], 1):
            for part in parts[c]:
                ceilings[(c, part["label"])] = ceiling_decoder(c, part)[0]
            print(f"  [{n}/{len(split['train'])}] {c}")
        report_ridge = {}
        for label in labels:
            rows = [i for i in train_items if i["label"] == label]
            X = torch.stack([i["x"] for i in rows])
            Y = torch.stack([ceilings[(i["case_id"], label)] for i in rows])
            ridge = PCARidge(hcfg["ridge"]["components"], hcfg["ridge"]["alphas"]).fit(
                X, Y, [i["case_id"] for i in rows], y_scale=scale)
            ridges[label] = ridge
            report_ridge[label] = {"n_rows": len(rows), "n_cases": len({i["case_id"] for i in rows}),
                                   "k": ridge.k, "alpha": ridge.alpha, "explained_variance": ridge.explained,
                                   "loco_mse": min(r["loco_mse"] for r in ridge.search),
                                   "mean_decoder_loco_mse": ridge.mean_loco_mse, "search": ridge.search}
            print(f"  ridge {label}: {len(rows)} rows, k={ridge.k} ({ridge.explained:.1%} of X variance), "
                  f"alpha={ridge.alpha:g}; leave-one-case-out error {report_ridge[label]['loco_mse']:.4f} "
                  f"vs {ridge.mean_loco_mse:.4f} for the mean decoder")
        (out_dir / "ridge.json").write_text(json.dumps(report_ridge, indent=2))
        if wandb_run is not None:
            wandb_run.summary.update({f"ridge/{label}/{k}": v for label, r in report_ridge.items()
                                      for k, v in r.items() if k != "search"})

    # Evaluate every method on the test cases: each predicts one decoder per
    # label part, stacked back into the case's K channels for scoring.
    results = out_dir / "results.csv"
    done = {(r["method"], r["input"] or r["case_id"])
            for r in read_rows([results], fieldnames=FIELDS)} if args.resume and results.exists() else set()
    writer = RowWriter(results, FIELDS, args.resume)

    def record(method, item, steps, seconds, ev):
        rows = [{"method": method, "case_id": item["case_id"], "input": item.get("input", ""),
                 "task": item["task"], "modality": item.get("modality", ""), "steps": steps,
                 "fit_seconds": round(seconds, 3), "psnr": ev["psnr"],
                 **{f: lab[f] for f in FIELDS[8:]}} for lab in ev["labels"]]
        writer.write(rows)
        print(f"  {method:<9} {rows[0]['input'] or rows[0]['case_id']}: native/resampled dice " +
              " ".join(f"{r['label']}={r['native_dice']:.3f}/{r['dice']:.3f}" for r in rows))
        if wandb_run is not None:
            wandb_run.log({"case_id": item["case_id"],
                           f"eval/{method}/mean_native_dice": float(np.nanmean([r["native_dice"] for r in rows])),
                           f"eval/{method}/mean_dice": float(np.nanmean([r["dice"] for r in rows]))})

    def score_inputs(method, predict, steps, save_dir):
        """predict(x (m, d), parts) -> (m, p) flat decoders, one per label part."""
        for i in inputs:
            if i["case_id"] not in test_ids or (method, i["input"]) in done:
                continue
            case_parts = parts[i["case_id"]]
            t0 = time.perf_counter()
            with torch.no_grad():
                predicted = predict(i["x"].to(device).expand(len(case_parts), -1), case_parts)
            part_params = [layout.unflatten(v.to(device)) for v in predicted]
            seconds = time.perf_counter() - t0
            for part, params in zip(case_parts, part_params):
                save_decoder_weights(dict(zip(keys, (t.cpu() for t in params))),
                                     part_name(i["input"], part), save_dir)
            ev, render_seconds = predict_and_score(model, part_params, targets[i["case_id"]], spacing_mm,
                                                   decode_mode, chunk, device)
            record(method, {**targets[i["case_id"]], **i}, steps, seconds + render_seconds, ev)

    print(f"\nevaluating on {len(test_cases)} test cases")
    for case in test_cases:
        if ("floor", case["case_id"]) not in done:
            ev, _ = predict_and_score(model, [layout.unflatten(floors[p["label"]]) for p in parts[case["case_id"]]],
                                      case, spacing_mm, decode_mode, chunk, device)
            record("floor", case, hcfg["floor_steps"], 0.0, ev)
    if ridges:
        score_inputs("ridge", lambda x, ps: torch.cat([ridges[p["label"]].predict(x[j:j + 1])
                                                       for j, p in enumerate(ps)]),
                     0, out_dir / "predicted_decoders_ridge")
    if hyper is not None:
        score_inputs("hypernet", lambda x, ps: hyper(x, torch.tensor([label_index[p["label"]] for p in ps],
                                                                     device=device)),
                     hcfg["steps"], out_dir / "predicted_decoders")
    if not args.skip_ceiling:
        for case in test_cases:
            if ("ceiling", case["case_id"]) in done:
                continue
            fitted = [ceiling_decoder(case["case_id"], p) for p in parts[case["case_id"]]]
            ev, _ = predict_and_score(model, [layout.unflatten(v.to(device)) for v, _ in fitted], case,
                                      spacing_mm, decode_mode, chunk, device)
            record("ceiling", case, hcfg["ceiling_steps"], sum(t for _, t in fitted), ev)

    report(read_rows([results], INT_FIELDS, FLOAT_FIELDS, FIELDS), out_dir, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
