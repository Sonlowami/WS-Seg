"""
Encoder size sweep: how do encoder width and depth change the prior?

For every combination of --hidden_dims x --encoder_layers:
  1. train an encoder with the config's own training (Encoder I or II,
     chosen by target_signal; joint/sequential as configured) and save it
     to <checkpoint_dir>/<sweep>/h<width>_L<depth>/
  2. use it as the prior for fitting a fresh SDF decoder to every evaluation
     case (the same fitting and scoring code as run_translation_test) and
     record Dice/NSD at each --steps budget, on the resampled grid ("dice")
     and against the original annotation ("native_dice")
  3. unless --no_random, score an equally sized random-init encoder too, so a
     size effect can be told apart from "bigger networks fit better anyway"

Width sweep:  --hidden_dims 256 384 512 --encoder_layers 5
Depth sweep:  --hidden_dims 256 --encoder_layers 4 5 6 7
(any combination of the two lists is a grid)

Outputs (prefix --out):
  <out>_summary.csv            one row per configuration x arm x budget: size,
                               parameter count, training time, and n / mean /
                               median over cases of the per-case mean Dice/NSD
                               (resampled and native). The file to read.
  <out>_summary_per_label.csv  the same, per task and label group
  <out>_per_label.csv          raw rows: configuration x arm x case x budget x label
Training runs log to wandb as usual (group = basename of --out). --resume
reuses saved checkpoints and skips fits already in <out>_per_label.csv.

Usage:
  python -m experiments.run_encoder_sweep --config config/encoder_I_intensity.yaml \\
      --hidden_dims 256 384 512 --encoder_layers 5 --steps 500 2000 --out results/sweep_width
"""
import argparse
import copy
import math
import time
from pathlib import Path

from utils.config import load_config
from utils.io import load_model_weights, save_encoder_weights
from utils.results_io import RowWriter, read_rows, write_csv
from utils.stats import aggregate_cases
from training.train_loop import resolve_device
from training.sdf_fit import case_seed
from training.encoder_eval import load_case_refs, prepare_ref, score_encoder, format_progress

CONFIG_FIELDS = ["config", "hidden_dim", "encoder_layers", "decoder_layers", "encoder_params"]
FIELDS = CONFIG_FIELDS + ["arm", "protocol", "case_id", "task", "steps", "fit_seconds", "psnr",
                          "label", "gt_voxels", "dice", "nsd", "oracle_dice", "native_dice",
                          "native_nsd"]
INT_FIELDS = ("hidden_dim", "encoder_layers", "decoder_layers", "encoder_params", "steps",
              "gt_voxels")
FLOAT_FIELDS = ("fit_seconds", "psnr", "dice", "nsd", "oracle_dice", "native_dice", "native_nsd")


def trainer_for(signal: str):
    """The training function for a config's target_signal (imported lazily:
    each pulls in its own losses and metrics)."""
    if signal == "intensity":
        from experiments.run_encoder_I import train_encoder_I
        return train_encoder_I
    if signal == "sdf":
        from experiments.run_encoder_II import train_encoder_II
        return train_encoder_II
    raise SystemExit(f"Unknown target_signal {signal!r}")


def variant_config(base: dict, hidden_dim: int, encoder_layers: int, decoder_layers: int,
                   sweep: str) -> tuple:
    """(tag, config) for one sweep point; everything except the size is base's."""
    tag = f"h{hidden_dim}_L{encoder_layers}"
    cfg = copy.deepcopy(base)
    cfg["model"].update(hidden_dim=hidden_dim, encoder_layers=encoder_layers)
    if decoder_layers is not None:
        cfg["model"]["decoder_layers"] = decoder_layers
    cfg["experiment_name"] = f"{sweep}_{tag}"
    return tag, cfg


def train_or_load(tag: str, cfg: dict, ckpt_dir: Path, resume: bool, sweep: str) -> dict:
    """Encoder state_dict and training info for one sweep point."""
    if resume and (ckpt_dir / "encoder.pt").exists():
        ckpt = load_model_weights(ckpt_dir)
        print(f"[{tag}] reusing checkpoint {ckpt_dir}")
        return {"state_dict": ckpt["encoder_state_dict"],
                "train_seconds": ckpt["config"].get("sweep", {}).get("train_seconds", float("nan"))}
    print(f"\n[{tag}] training {cfg['target_signal']} encoder {cfg['model']}")
    t0 = time.time()
    state_dict = trainer_for(cfg["target_signal"])(
        cfg, wandb_kwargs={"group": sweep, "job_type": "encoder_sweep"})
    seconds = time.time() - t0
    cfg["sweep"] = {"name": sweep, "tag": tag, "train_seconds": round(seconds, 1)}
    save_encoder_weights(state_dict, cfg, out_dir=ckpt_dir)
    return {"state_dict": state_dict, "train_seconds": round(seconds, 1)}


def report(rows: list, points: dict, out: str):
    """Write the two summary CSVs and print the headline table."""
    by = ("config", "hidden_dim", "encoder_layers", "decoder_layers", "encoder_params",
          "arm", "protocol", "steps")
    summary = aggregate_cases(rows, by)
    for r in summary:
        r["train_seconds"] = points.get(r["config"], {}).get("train_seconds", float("nan")) \
            if r["arm"] == "trained" else 0.0
    write_csv(Path(f"{out}_summary.csv"), summary)
    write_csv(Path(f"{out}_summary_per_label.csv"), aggregate_cases(rows, by + ("task", "label")))

    metric = "native_dice" if any(not math.isnan(r["native_dice"]) for r in rows) else "dice"
    arms = sorted({r["arm"] for r in summary}, key=lambda a: a != "trained")
    print(f"\n==== mean / median {metric} over cases (n cases) ====")
    print(f"  {'config':<12}{'params':>10}{'steps':>7}" + "".join(f"{a:>26}" for a in arms))
    for key in sorted({(r["encoder_layers"], r["hidden_dim"], r["config"], r["encoder_params"],
                        r["steps"]) for r in summary}):
        _, _, config, params, steps = key
        cells = {r["arm"]: r for r in summary if r["config"] == config and r["steps"] == steps}
        print(f"  {config:<12}{params:>10,}{steps:>7}" + "".join(
            f"{cells[a][f'mean_{metric}']:>12.4f} /{cells[a][f'median_{metric}']:>7.4f} "
            f"({cells[a][f'n_{metric}']:>2})" if a in cells else f"{'-':>26}" for a in arms))
    print(f"\nSummaries: {out}_summary.csv, {out}_summary_per_label.csv; raw: {out}_per_label.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Encoder training config (Encoder I or II)")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=None,
                        help="Widths to sweep (default: the config's)")
    parser.add_argument("--encoder_layers", nargs="+", type=int, default=None,
                        help="Encoder depths to sweep (default: the config's)")
    parser.add_argument("--decoder_layers", type=int, default=None,
                        help="Decoder depth for training and evaluation (default: the config's)")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Evaluation cases (held out from the encoder's training split)")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--steps", nargs="+", type=int, default=[500, 2000],
                        help="Decoder-fitting budgets at which Dice is recorded")
    parser.add_argument("--train_encoder", action="store_true",
                        help="Evaluation fits the whole network from the prior (default: frozen)")
    parser.add_argument("--no_random", action="store_true",
                        help="Skip the equally sized random-init control")
    parser.add_argument("--no_native", action="store_true", help="Skip native-grid metrics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out", default="results/encoder_sweep")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse saved checkpoints and skip fits already recorded")
    args = parser.parse_args()

    base = load_config(args.config)
    if args.data_root:
        base["data"]["root"] = args.data_root
    if "sdf" not in base:
        raise SystemExit("The config needs an sdf: block (alpha, eikonal_lambda, label_groups, "
                         "decode_mode): evaluation fits SDF decoders.")
    sweep = Path(args.out).name
    hidden_dims = args.hidden_dims or [base["model"]["hidden_dim"]]
    depths = args.encoder_layers or [base["model"]["encoder_layers"]]
    protocol = "finetune" if args.train_encoder else "frozen"
    spacing_mm = tuple(base["data"]["spacing_mm"])
    device = resolve_device(base.get("training", {}).get("device", "auto"))

    # Open the results file first, so a clash with an existing --out fails
    # before hours of training rather than after.
    out_path = Path(f"{args.out}_per_label.csv")
    done = set()
    if args.resume and out_path.exists():
        seen = {}
        for r in read_rows([out_path], INT_FIELDS, FLOAT_FIELDS):
            seen.setdefault((r["config"], r["arm"], r["case_id"]), set()).add(r["steps"])
        done = {k for k, s in seen.items() if set(args.steps) <= s}
    writer = RowWriter(out_path, FIELDS, args.resume)

    # 1. Train (or reuse) one encoder per sweep point.
    points = {}
    for h in hidden_dims:
        for depth in depths:
            tag, cfg = variant_config(base, h, depth, args.decoder_layers, sweep)
            info = train_or_load(tag, cfg, Path(args.checkpoint_dir) / sweep / tag, args.resume, sweep)
            info.update(cfg=cfg, params=sum(v.numel() for v in info["state_dict"].values()))
            points[tag] = info
    print("\nsweep points: " + ", ".join(f"{t} ({p['params']:,} encoder params)"
                                         for t, p in points.items()))

    # 2. Score each encoder (and its random control) as a prior on every case.
    arms = ["trained"] + ([] if args.no_random else ["random"])
    refs = load_case_refs(base["data"], base["sdf"]["label_groups"], args.split, args.max_cases)
    print(f"{len(refs)} {args.split} cases x {len(points)} sizes x {len(arms)} arms, budgets "
          f"{sorted(set(args.steps))}, {protocol} encoder, device {device}")

    for n, ref in enumerate(refs, 1):
        case_id = ref[0]
        todo = [(t, a) for t in points for a in arms if (t, a, case_id) not in done]
        if not todo:
            continue
        sdf_case = prepare_ref(ref, spacing_mm, base["sdf"]["alpha"], native=not args.no_native)
        print(f"\n[{n}/{len(refs)}] {case_id} grid {sdf_case['shape']}")
        for tag, arm in todo:
            point = points[tag]
            model = point["cfg"]["model"]
            meta = {"config": tag, "hidden_dim": model["hidden_dim"],
                    "encoder_layers": model["encoder_layers"],
                    "decoder_layers": model["decoder_layers"], "encoder_params": point["params"],
                    "arm": arm, "protocol": protocol}
            rows = score_encoder(
                point["cfg"], sdf_case, point["state_dict"] if arm == "trained" else None,
                args.steps, freeze_encoder=not args.train_encoder, device=device,
                seed=case_seed(args.seed, case_id),
                on_rows=lambda new, ev, label=f"{tag}/{arm}": print(format_progress(label, new, ev)))
            writer.write([{**meta, **r} for r in rows])

    report(read_rows([out_path], INT_FIELDS, FLOAT_FIELDS), points, args.out)


if __name__ == "__main__":
    main()
