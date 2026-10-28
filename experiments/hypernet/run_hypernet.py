"""
Hypernetwork: predict a case's SDF decoder from its intensity decoder.

Inputs are the per-image intensity decoders from run_decoder_training (one
per case, or per case/modality), each flattened to a vector. An MLP maps that
vector to the weights of an SDF decoder that sits on a frozen encoder; it is
trained through the decoder's output with the masked MSE + Eikonal SDF loss,
so it never needs target weights, only each case's clipped mask SDF. The
predicted decoder is thresholded for Dice and NSD as in visualize_sdf_fit.

Methods, all scored on the same held-out test cases:
  floor     one SDF decoder fitted to every training case at once (the
            average shape): what a decoder knows without seeing the case
  hypernet  the MLP's prediction from the case's intensity decoder
  ceiling   a decoder fitted to the test case's own SDF (as in the
            translation test, frozen encoder): the best this decoder can do
Expected: floor < hypernet < ceiling. Paired Wilcoxon tests over cases (Holm
corrected per metric) test both gaps.

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
  floor_decoder.pt       the floor's decoder state_dict
  hypernet.pt            the MLP (with its normalization and base), sizes, config
  predicted_decoders/    each test input's predicted SDF decoder
  results.csv            method x test input x label group: Dice/NSD (resampled
                         and native), PSNR, fit or prediction seconds
  summary.csv            per method: n / mean / median over cases of the
                         per-case mean (hypernet: averaged over modalities first)
  summary_per_label.csv, summary_per_modality.csv   the same, split further
  tests.csv              paired Wilcoxon: hypernet - floor, ceiling - hypernet,
                         ceiling - floor
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
from pathlib import Path

import numpy as np
import torch

from experiments.hypernet.data import load_decoder_bank, split_cases, load_sdf_targets
from experiments.hypernet.model import HeadLayout, HyperMLP, flatten_state_dict
from experiments.hypernet.train import train_weights, predict_and_score
from models.interfaces import DECODER_KEYS, build_model
from training.encoder_eval import score_encoder
from training.sdf_fit import case_seed
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
PAIRS = (("hypernet", "floor"), ("ceiling", "hypernet"), ("ceiling", "floor"))

DEFAULTS = {
    "decoders_dir": None, "test_fraction": 0.2, "split_seed": 0,
    "hidden_dim": 1024, "layers": 4, "activation": "gelu", "lr": 1e-4, "weight_decay": 0.01,
    "scheduler": "cosine", "steps": 20000, "cases_per_step": 8, "points_per_step": 2 ** 16,
    "log_every": 100, "init_from_floor": True, "floor_steps": 4000, "ceiling_steps": 2000,
}


# ---------------------------------------------------------------- setup

def sdf_model_config(decoder_model: dict, hcfg: dict) -> dict:
    """The intensity decoders' model (their encoder, depth and width), with
    hypernet.decoder_layers / decoder_hidden_dim overriding the SDF decoder's."""
    return {**decoder_model, **{k: hcfg[k] for k in DECODER_KEYS if hcfg.get(k) is not None}}


# ---------------------------------------------------------------- training

def train_floor(model, layout, train_items, test_items, cfg, hcfg, spacing_mm, device, on_log):
    """One decoder fitted to all training cases; returns its flat weights."""
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
    a batched MLP forward on the chosen items' input vectors."""
    inputs = torch.stack([i["x"] for i in items]).to(device)
    return lambda chosen: [layout.unflatten(v) for v in hyper(inputs[chosen])]


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
    for method in ("floor", "hypernet", "ceiling"):
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
    if "floor" in mean and "hypernet" in mean:
        ordering["hypernet_beats_floor"] = bool(mean["hypernet"] > mean["floor"])
    if "ceiling" in mean and "hypernet" in mean:
        ordering["hypernet_below_ceiling"] = bool(mean["hypernet"] < mean["ceiling"])
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
    parser.add_argument("--skip_ceiling", action="store_true", help="Skip the per-case ceiling fits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse saved floor/hypernet and skip evaluations already in results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if "sdf" not in cfg:
        raise SystemExit("The config needs an sdf: block (alpha, eikonal_lambda, label_groups, decode_mode).")
    hcfg = {**DEFAULTS, **(cfg.get("hypernet") or {})}
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
    setup = {"decoders_dir": str(decoders_dir), "encoder_ckpt": dec_config["encoder_ckpt"], "sdf_model": sdf_model, "sdf": cfg["sdf"],
             "hypernet": hcfg, "split": split, "seed": args.seed}
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
    targets = load_sdf_targets(data_cfg, cfg["sdf"]["label_groups"], {i["case_id"] for i in inputs},
                               test_ids, spacing_mm, cfg["sdf"]["alpha"])
    k = next(iter(targets.values()))["target"].shape[-1]
    train_inputs = [{**targets[i["case_id"]], **i} for i in inputs if i["case_id"] not in test_ids]
    test_inputs = [{**targets[i["case_id"]], **i} for i in inputs if i["case_id"] in test_ids]
    train_cases = [targets[c] for c in split["train"]]
    test_cases = [targets[c] for c in split["test"]]

    # The SDF model: frozen encoder, one decoder head whose weights are
    # swapped per case. Its fresh init sets the hypernet's per-tensor scale.
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
             "label_groups": k, "n_train_inputs": len(train_inputs), "n_test_inputs": len(test_inputs)}
    print(f"hypernet: {in_dim:,} intensity-decoder weights -> {layout.numel:,} SDF-decoder weights "
          f"({k} label groups) on {dec_config['encoder_ckpt']}; device {device}")

    wandb_run = configure_wandb(cfg["wandb"], f"{cfg['experiment_name']}_hypernet",
                                {**setup, "sizes": sizes})
    if wandb_run is not None:
        for prefix in ("floor", "hypernet"):
            wandb_run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")

    def on_log(prefix):
        def log(step, values):
            if wandb_run is not None:
                wandb_run.log({f"{prefix}/step": step, **{f"{prefix}/{k}": v for k, v in values.items()}})
        return log

    # Floor.
    floor_path = out_dir / "floor_decoder.pt"
    if args.resume and floor_path.exists():
        floor = flatten_state_dict(torch.load(floor_path, map_location="cpu")).to(device)
        print(f"floor: reusing {floor_path}")
    else:
        print("floor: one decoder fitted to all training cases")
        torch.manual_seed(args.seed)
        floor = train_floor(model, layout, train_cases, test_cases, cfg, hcfg, spacing_mm, device,
                            on_log("floor"))
        torch.save(model.decoder_state_dict(), floor_path)

    # Hypernet.
    hyper_path = out_dir / "hypernet.pt"
    hyper = HyperMLP(in_dim, layout.numel, hcfg["hidden_dim"], hcfg["layers"], hcfg["activation"]).to(device)
    if args.resume and hyper_path.exists():
        hyper.load_state_dict(torch.load(hyper_path, map_location=device)["state_dict"])
        hyper.eval()
        print(f"hypernet: reusing {hyper_path}")
    else:
        n_params = sum(p.numel() for p in hyper.parameters())
        print(f"hypernet: {hcfg['layers']}-layer MLP, {hcfg['hidden_dim']} wide, {n_params:,} params, "
              f"starting from the {'floor' if hcfg['init_from_floor'] else 'fresh SIREN init'}")
        torch.manual_seed(args.seed)
        hyper.set_reference(torch.stack([i["x"] for i in train_inputs]).to(device),
                            floor if hcfg["init_from_floor"] else fresh, scale.to(device))
        train_hypernet(hyper, model, layout, train_inputs, test_inputs, cfg, hcfg, spacing_mm, device,
                       on_log("hypernet"))
        torch.save({"state_dict": hyper.state_dict(), **sizes, "hidden_dim": hcfg["hidden_dim"],
                    "layers": hcfg["layers"], "activation": hcfg["activation"]}, hyper_path)

    # Evaluate every method on the test cases.
    results = out_dir / "results.csv"
    done = {(r["method"], r["input"] or r["case_id"])
            for r in read_rows([results], fieldnames=FIELDS)} if args.resume and results.exists() else set()
    writer = RowWriter(results, FIELDS, args.resume)
    keys = list(model.decoder_state_dict())

    def record(method, item, steps, seconds, ev_rows, psnr):
        rows = [{"method": method, "case_id": item["case_id"], "input": item.get("input", ""),
                 "task": item["sdf_case"]["task"], "modality": item.get("modality", ""), "steps": steps,
                 "fit_seconds": round(seconds, 3), "psnr": psnr,
                 **{f: lab[f] for f in FIELDS[8:]}} for lab in ev_rows]
        writer.write(rows)
        mean = float(np.nanmean([r["native_dice"] for r in rows]))
        print(f"  {method:<9} {rows[0]['input'] or rows[0]['case_id']}: native/resampled dice " +
              " ".join(f"{r['label']}={r['native_dice']:.3f}/{r['dice']:.3f}" for r in rows))
        if wandb_run is not None:
            wandb_run.log({"case_id": item["case_id"], f"eval/{method}/mean_native_dice": mean,
                           f"eval/{method}/mean_dice": float(np.nanmean([r["dice"] for r in rows]))})

    print(f"\nevaluating on {len(test_cases)} test cases")
    floor_params = layout.unflatten(floor)
    for item in test_cases:
        if ("floor", item["case_id"]) not in done:
            ev, seconds = predict_and_score(model, floor_params, item, spacing_mm, decode_mode, chunk, device)
            record("floor", item, hcfg["floor_steps"], 0.0, ev["labels"], ev["psnr"])
    with torch.no_grad():
        for item in test_inputs:
            if ("hypernet", item["input"]) in done:
                continue
            params = layout.unflatten(hyper(item["x"].to(device)[None])[0])
            save_decoder_weights(dict(zip(keys, (p.cpu() for p in params))), item["input"],
                                 out_dir / "predicted_decoders")
            ev, seconds = predict_and_score(model, params, item, spacing_mm, decode_mode, chunk, device)
            record("hypernet", item, hcfg["steps"], seconds, ev["labels"], ev["psnr"])
    if not args.skip_ceiling:
        fit_cfg = {**cfg, "model": sdf_model}
        for item in test_cases:
            if ("ceiling", item["case_id"]) in done:
                continue
            rows = score_encoder(fit_cfg, item["sdf_case"], encoder,
                                 [hcfg["ceiling_steps"]], freeze_encoder=True, device=device,
                                 seed=case_seed(args.seed, item["case_id"]))
            record("ceiling", item, hcfg["ceiling_steps"], rows[0]["fit_seconds"], rows, rows[0]["psnr"])

    report(read_rows([results], INT_FIELDS, FLOAT_FIELDS, FIELDS), out_dir, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
