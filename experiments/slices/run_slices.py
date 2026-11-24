"""
Slice-decoder pilot: does the weight-space learner do better when each
volume yields many slice decoders instead of one?

The 3D experiment (experiments/hypernet) has one intensity decoder per case,
so a few dozen training examples. Here every slice of a volume gets its own
intensity decoder, multiplying the examples -- enough, if they carry new
information, for data-hungry learners such as transformers. Methods:
  floor     per label, one slice decoder fitted to every training slice
  hypernet  HyperMLP from a slice's intensity decoder to its SDF decoder, with
            the hypernet block's MLP, regularization and training (trained
            through the SDF loss, so it needs no target decoders), plus the
            slice's position in its volume, k / (n_slices - 1)
            (slices.use_position): relative to the whole volume, so it
            carries no ground-truth information
  position_only  the hypernet ablation: the same model, seed, init and
            training with the intensity-decoder input zeroed, so it sees
            only the label and the slice position -- a positional atlas, the
            2D counterpart of the 3D floor (which knows z through its
            coordinates). hypernet - position_only is what the image adds.
            Needs slices.use_position.
  transformer  SliceTransformer (experiments/slices/transformer.py): every
            slice's SDF decoder from the whole stack of its volume's slice
            intensity decoders (raw tokens + Fourier position + label
            embedding, slices.transformer_dim / _layers / _heads), with the
            hypernet block's optimizer, steps, items and points per step,
            floor start, and its dropout and weight decay unless
            slices.transformer_dropout / transformer_weight_decay are set. Trained and scored on whole stacks, so only
            at stride-1 points (slices.base_stride: 1). No augmentation.
  transformer_nocontext  the same model with each token attending only to
            itself: transformer - transformer_nocontext is what context adds.
  ceiling   each test slice's own fitted SDF decoder (test slices only)
For scoring, each test case's predicted slices are stacked back into its
volume and scored exactly like the 3D experiment -- resampled and native
Dice/NSD per label.

Training uses every base_stride-th slice and every slice of each test
volume is predicted, empty or not: nothing comes from the test ground
truth, and the numbers are directly comparable with experiments/hypernet's
summary.csv on the same test cases (same split settings, see below).

The test is a learning curve over slices.points, (case fraction, slice
stride) pairs: denser slices from the same training cases (does adding
decoders help?) and more training cases (the reference). If test Dice rises
with slice density, slices add usable examples; if it only rises with cases,
neighbouring slices are too redundant to count as data.

Slices are volumes one voxel thick (experiments/slices/data.py), so every
decoder sits on a frozen 3D-coordinate encoder: by default a STRAINER
encoder trained here on a sample of training slices (slices.encoder_ckpt
null), or any saved encoder, e.g. Encoder I. All decoders start from one
shared init (--seed).

Cases: data.split (slices.split, default the decoder-training split
"train"), divided by case with slices.test_fraction / split_seed (default the
hypernet block's), as experiments/hypernet does -- the same test cases when
the 3D decoders came from that split.

Cost: one intensity decoder per (training slice at base_stride, modality) and
per (test slice, modality), and with the ceiling one SDF decoder per (test
slice, label), each slices.decoder_steps steps; all cached in --out_dir, so
further points or reruns only retrain floor and hypernet.

Outputs in --out_dir: encoder/, slice_decoders/, ceiling_decoders/,
split.json, results.csv (point x method x test input x label),
summary.csv (point x method: n / mean / median over cases),
summary_per_label.csv. WandB: training curves, per-point summaries, tables.

Usage:
  python -m experiments.slices.run_slices --config config/training_config.yaml \\
      --out_dir results/slices [--methods floor hypernet position_only ceiling] [--resume]

  Adding a method to an existing run: the same --out_dir with --resume reuses
  the encoder, the slice decoders and every result already written; --points
  limits the run to some learning-curve points, e.g.
    ... --methods position_only --points cases1_stride2 --resume
  base_stride and points may change between resumed runs: they pick which
  cached slices are used, not how any decoder is fitted. A method whose test
  inputs are all scored at a point is not retrained, and a resumed run stops
  if a method's settings differ from those its rows were scored with
  (learner_settings.json), so other settings need another --out_dir.
  The transformer, e.g. (slices.base_stride: 1, a point [1.0, 1]):
    ... --methods transformer transformer_nocontext --points cases1_stride1 --resume
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.hypernet.data import split_cases, label_targets
from experiments.hypernet.model import HeadLayout, HyperMLP, flatten_state_dict, load_head, print_hypernet_summary
from experiments.hypernet.run_hypernet import DEFAULTS as HYPERNET_DEFAULTS, train_floor, train_hypernet
from experiments.hypernet.train import train_weights
from experiments.slices.data import load_slice_cases, put_slice
from experiments.slices.resume import (SEQUENCE_LEARNERS, SEQUENCE_OVERRIDES, SEQUENCE_SIZES, check_settings,
                                       record_settings, sequence_config)
from experiments.slices.transformer import SliceTransformer, print_parameter_summary, sequence_predictor
from models.interfaces import build_model, merge_model_config
from training.encoder_eval import load_case_refs
from training.losses import image_reconstruction_loss
from training.sdf_fit import fit_sdf, evaluate_sdf_fit
from training.train_loop import build_scheduler, fit_inr, predict_volume, resolve_device, train_encoder_jointly
from utils.config import load_config
from utils.io import load_model_weights, save_decoder_weights, save_encoder_weights
from utils.logging_utils import configure_wandb
from utils.metrics import metric_fn
from utils.results_io import RowWriter, read_rows, write_csv
from utils.stats import aggregate_cases

FIELDS = ["point", "case_fraction", "stride", "n_train_cases", "n_train_slices", "method", "case_id",
          "input", "task", "modality", "psnr", "label", "gt_voxels", "dice", "nsd", "oracle_dice",
          "native_dice", "native_nsd"]
INT_FIELDS = ("stride", "n_train_cases", "n_train_slices", "gt_voxels")
FLOAT_FIELDS = ("case_fraction", "psnr", "dice", "nsd", "oracle_dice", "native_dice", "native_nsd")
METRICS = ("native_dice", "dice", "native_nsd", "nsd")
METHODS = ("floor", "hypernet", "position_only", "transformer", "transformer_nocontext", "ceiling")
LEARNERS = ("hypernet", "position_only") + SEQUENCE_LEARNERS
# No cached encoder or decoder depends on these, so --resume lets them differ
# from config.json. The learners' own settings are checked per method instead
# (experiments/slices/resume.py).
RESUME_FREE = ("base_stride", "points") + SEQUENCE_SIZES + tuple(SEQUENCE_OVERRIDES)
DEFAULTS = {
    "split": "train", "test_fraction": None, "split_seed": None,     # None = the hypernet block's
    "axis": 2, "base_stride": 2,
    "encoder_ckpt": None, "encoder_slices": 512, "encoder_steps": 5000,
    "decoder_steps": 500, "points_per_step": 16384, "use_position": True,
    "transformer_dim": 512, "transformer_layers": 6, "transformer_heads": 4,
    "transformer_dropout": None, "transformer_weight_decay": None,   # None = the hypernet block's
    "points": [[1.0, 16], [1.0, 8], [1.0, 4], [1.0, 2], [0.25, 2], [0.5, 2]],
}


def initial_decoder(model_cfg: dict, encoder: dict, out_features: int, seed: int, device) -> tuple:
    """
    (model with the frozen encoder, flat initial decoder): the one starting
    decoder of every decoder in this experiment -- intensity slices, ceilings,
    floors and the hypernet's base. Seed, build, reset: exactly the sequence
    of training/sdf_fit.fit_sdf (the ceilings) and run_decoder_training.
    """
    torch.manual_seed(seed)
    model = build_model(model_cfg, out_features=out_features).to(device)
    model.load_encoder_state_dict(encoder, freeze=True)
    model.reset_decoder()
    return model, HeadLayout(model.decoder[0]).flatten(model.decoder[0])


def decoder_path(directory: Path, key: str) -> Path:
    return directory / f"decoder_{key.replace('/', '__')}.pt"


def cached_decoder(directory: Path, key: str, fit) -> torch.Tensor:
    """The flat decoder saved under key, or fit() -> decoder state_dict, saved."""
    path = decoder_path(directory, key)
    if path.exists():
        return flatten_state_dict(torch.load(path, map_location="cpu"))
    state = {k: v.detach().cpu() for k, v in fit().items()}
    save_decoder_weights(state, key, directory)
    return flatten_state_dict(state)


def slice_encoder(cfg, scfg, cases, train_ids, out_dir, spacing_mm, device, seed, wandb_run):
    """(encoder state_dict, model config): slices.encoder_ckpt, or a STRAINER
    encoder trained on up to encoder_slices training (slice, modality) images."""
    if scfg["encoder_ckpt"]:
        ckpt = load_model_weights(scfg["encoder_ckpt"])
        return ckpt["encoder_state_dict"], merge_model_config(ckpt["config"]["model"], cfg["model"])
    path = out_dir / "encoder"
    if (path / "encoder.pt").exists():
        print(f"slice encoder: reusing {path}")
        return load_model_weights(path)["encoder_state_dict"], cfg["model"]
    images = [(c, m, s) for c in sorted(train_ids) for s in cases[c]["slices"].values()
              for m in range(len(cases[c]["modalities"]))]
    pick = np.random.RandomState(seed).permutation(len(images))[:scfg["encoder_slices"]]
    items = [{"case_id": f"{images[j][0]}/{images[j][2]['k']}", "shape": images[j][2]["shape"],
              "target": images[j][2]["image"][:, images[j][1]:images[j][1] + 1],
              "extra": {"needs_coord_grad": False, "loss_kwargs": {}}} for j in sorted(pick)]
    print(f"slice encoder: STRAINER on {len(items)} training slice images, {scfg['encoder_steps']} steps")
    torch.manual_seed(seed)
    model = build_model(cfg["model"], out_features=1, num_decoders=len(items))
    train_cfg = cfg.get("training") or {}
    state = train_encoder_jointly(
        items, model, lambda pred, coords, target: {"total": image_reconstruction_loss(pred, target)},
        lambda pred, target, shape: metric_fn(pred, target, shape, 1),    # PSNR + SSIM, as Encoder I
        cfg["optimizer"], cfg["scheduler"], scfg["encoder_steps"], max(scfg["encoder_steps"] // 10, 1),
        wandb_run, spacing_mm, cases_per_step=train_cfg.get("cases_per_step", 8),
        points_per_step=scfg["points_per_step"], eval_chunk_size=train_cfg.get("eval_chunk_size", 2 ** 20),
        device=device)
    save_encoder_weights(state, {**cfg, "slices": scfg}, path)
    return state, cfg["model"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="data, model, sdf, optimizer, hypernet, slices, wandb")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--points", nargs="+", default=None,
                        help="Run only these learning-curve points, e.g. cases1_stride2 (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (point, method, input) results already in results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    scfg = {**DEFAULTS, **(cfg.get("slices") or {})}
    lcfg = {**HYPERNET_DEFAULTS, **(cfg.get("hypernet") or {})}          # the learners' settings
    for key in ("test_fraction", "split_seed"):
        scfg[key] = lcfg[key] if scfg[key] is None else scfg[key]
    sdf_cfg, train_cfg = cfg["sdf"], cfg.get("training") or {}
    alpha, per_label = sdf_cfg["alpha"], sdf_cfg.get("per_label", False)
    decode_mode = sdf_cfg.get("decode_mode", "independent")
    spacing_mm = tuple(cfg["data"]["spacing_mm"])
    device = resolve_device(train_cfg.get("device", "auto"))
    chunk = train_cfg.get("eval_chunk_size", 2 ** 20)
    axis, base_stride, methods = scfg["axis"], scfg["base_stride"], set(args.methods)
    points = [(float(f), int(s)) for f, s in scfg["points"]]
    bad = [s for _, s in points if s % base_stride]
    if bad:
        raise SystemExit(f"slices.points strides {bad} are not multiples of slices.base_stride={base_stride}.")
    if "position_only" in methods and not scfg["use_position"]:
        raise SystemExit("position_only needs slices.use_position: true (it would see no input at all).")
    sequence_methods = sorted(methods & set(SEQUENCE_LEARNERS))
    if sequence_methods and not scfg["use_position"]:
        raise SystemExit(f"{sequence_methods} need slices.use_position: true (the slice order).")
    if sequence_methods and not any(s == 1 for _, s in points):
        raise SystemExit(f"{sequence_methods} train on whole stacks: set slices.base_stride: 1 and add a "
                         f"stride-1 point to slices.points, e.g. [1.0, 1].")
    out_dir = Path(args.out_dir)

    # Cases and split (by case, as experiments/hypernet).
    case_ids = sorted(r[0] for r in load_case_refs(cfg["data"], sdf_cfg["label_groups"], scfg["split"]))
    split = split_cases(case_ids, scfg["test_fraction"], scfg["split_seed"])
    test_ids, train_ids = set(split["test"]), set(split["train"])
    setup = {"data": cfg["data"], "model": cfg["model"], "sdf": sdf_cfg, "slices": scfg, "split": split,
             "seed": args.seed}
    saved = out_dir / "config.json"
    comparable = lambda c: {**c, "slices": {k: v for k, v in c["slices"].items() if k not in RESUME_FREE}}
    if args.resume and saved.exists() and \
            comparable(json.loads(saved.read_text())) != comparable(json.loads(json.dumps(setup))):
        raise SystemExit(f"{saved} was written with different settings; choose another --out_dir.")
    results = out_dir / "results.csv"
    done = {(r["point"], r["method"], r["input"] or r["case_id"]) for r in read_rows([results], fieldnames=FIELDS)} \
        if args.resume and results.exists() else set()
    run_points = [p for p in (f"cases{f:g}_stride{s}" for f, s in points) if not args.points or p in args.points]
    check_settings(out_dir, done, run_points, sorted(methods - {"ceiling"}), scfg, lcfg,
                   json.loads(saved.read_text())["slices"] if saved.exists() else {})
    out_dir.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(setup, indent=2))
    (out_dir / "split.json").write_text(json.dumps(split, indent=2))
    print(f"{len(case_ids)} cases: {len(train_ids)} train / {len(test_ids)} test; slicing along axis {axis}")
    cases = load_slice_cases(cfg["data"], sdf_cfg["label_groups"], scfg["split"], set(case_ids), test_ids,
                             spacing_mm, alpha, axis, lambda c: 1 if c in test_ids else base_stride)

    wandb_run = configure_wandb(cfg["wandb"], f"{cfg['experiment_name']}_slices",
                                {**setup, "learner": lcfg, "learner_sequence": sequence_config(scfg, lcfg)})
    logged_prefixes = set()

    def on_log(prefix):
        def log(step, values):
            if wandb_run is None:
                return
            if prefix not in logged_prefixes:
                wandb_run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
                logged_prefixes.add(prefix)
            wandb_run.log({f"{prefix}/step": step, **{f"{prefix}/{k}": v for k, v in values.items()}})
        return log

    encoder, model_cfg = slice_encoder(cfg, scfg, cases, train_ids, out_dir, spacing_mm, device,
                                       args.seed, wandb_run)

    # Slice decoders. One intensity model, set to the shared initial decoder
    # for each fit; then seeded, so every fit also draws the same batches.
    fit_cfg = {**cfg, "model": model_cfg,
               "training": {**train_cfg, "points_per_step": scfg["points_per_step"]}}
    image_model, image_init = initial_decoder(model_cfg, encoder, 1, args.seed, device)
    image_layout = HeadLayout(image_model.decoder[0])

    def fit_image(target, shape):
        def fit():
            load_head(image_model.decoder[0], image_layout.unflatten(image_init.to(device)))
            torch.manual_seed(args.seed)
            fit_inr(image_model, target.float(), shape, spacing_mm,
                    lambda pred, coords, t: {"total": image_reconstruction_loss(pred, t)},
                    scfg["decoder_steps"], cfg["optimizer"], cfg["scheduler"],
                    points_per_step=scfg["points_per_step"], eval_chunk_size=chunk, device=device)
            return image_model.decoder_state_dict()
        return fit

    def fit_ceiling(part):
        def fit():
            sdf_case = {"sdf": part["target"].float().reshape(*part["shape"], -1).numpy(),
                        "shape": part["shape"], "spacing_mm": spacing_mm}
            _, _, fitted = fit_sdf(fit_cfg, sdf_case, scfg["decoder_steps"], encoder_state_dict=encoder,
                                   freeze_encoder=True, device=device, seed=args.seed, return_model=True)
            return fitted.decoder_state_dict()
        return fit

    # Items: a part per (case, slice, label) and an input per (case, modality, slice) x part.
    parts, inputs = {}, []
    n_slices = sum(len(c["slices"]) for c in cases.values())
    print(f"slice decoders: {n_slices} slices; decoders cached in {out_dir}")
    done_slices = 0
    for case_id in case_ids:
        case = cases[case_id]
        for k, sl in case["slices"].items():
            slice_case = {"case_id": case_id, "task": case["task"], "names": case["names"], "shape": sl["shape"],
                          "target": sl["sdf"], "mm_per_unit": sl["mm_per_unit"], "sdf_case": None,
                          "k": k, "position": sl["position"]}
            parts[(case_id, k)] = label_targets(slice_case, per_label, alpha)
            for part in parts[(case_id, k)]:
                if "ceiling" in methods and case_id in test_ids:
                    part["ceiling"] = cached_decoder(out_dir / "ceiling_decoders",
                                                     f"{case_id}/{part['label']}/z{k:04d}", fit_ceiling(part))
            for m, modality in enumerate(case["modalities"]):
                key = f"{case_id}/{modality}/z{k:04d}"
                x = cached_decoder(out_dir / "slice_decoders", key,
                                   fit_image(sl["image"][:, m:m + 1], sl["shape"]))
                extra = torch.tensor([sl["position"]]) if scfg["use_position"] else torch.zeros(0)
                inputs.append({"input": f"{case_id}/{modality}", "key": key, "case_id": case_id, "k": k,
                               "modality": modality, "task": case["task"], "position": sl["position"],
                               "x": torch.cat([x, extra])})
            done_slices += 1
            if done_slices % 50 == 0 or done_slices == n_slices:
                print(f"  [{done_slices}/{n_slices}] slices")

    labels = sorted({p["label"] for c in train_ids for k in cases[c]["slices"] for p in parts[(c, k)]})
    label_index = {label: j for j, label in enumerate(labels)}
    widths = {p["target"].shape[1] for ps in parts.values() for p in ps}
    if len(widths) > 1:
        raise SystemExit(f"Label parts have different widths {widths}; set sdf.per_label: true.")
    out_features = widths.pop()

    # The SDF model whose decoder head every learner fills in; fresh is the
    # shared initial decoder (identical to image_init when out_features is 1).
    model, fresh = initial_decoder(model_cfg, encoder, out_features, args.seed, device)
    for p in model.decoder_parameters():
        p.requires_grad_(False)
    layout = HeadLayout(model.decoder[0])
    scale = layout.tensor_stds(model.decoder[0])

    def input_items(case_subset, stride):
        return [{**part, **i, "label_index": label_index[part["label"]]}
                for i in inputs if i["case_id"] in case_subset and i["k"] % stride == 0
                for part in parts[(i["case_id"], i["k"])]]

    test_items = input_items(test_ids, 1)

    # Results.
    test_inputs = sorted({i["input"] for i in test_items})
    writer = RowWriter(results, FIELDS, args.resume)

    def score(case_id, params_for_slice):
        """Stack predicted slices into the case volume and score it.
        params_for_slice(k, part) -> decoder params (layout order)."""
        case = cases[case_id]
        sdf_case = case["sdf_case"]
        volume = np.full(sdf_case["sdf"].shape, alpha, dtype=np.float32)
        with torch.no_grad():
            for k, sl in case["slices"].items():
                for part in parts[(case_id, k)]:
                    load_head(model.decoder[0], params_for_slice(k, part))
                    pred = predict_volume(model, sl["shape"], spacing_mm, 0, chunk, device)
                    put_slice(volume, axis, k, part["channels"], pred.reshape(*sl["shape"], -1).numpy())
        return evaluate_sdf_fit(volume, sdf_case, decode_mode)

    def record(meta, method, case_id, ev, input_id="", modality=""):
        rows = [{**meta, "method": method, "case_id": case_id, "input": input_id, "task": cases[case_id]["task"],
                 "modality": modality, "psnr": ev["psnr"], **{f: lab[f] for f in FIELDS[11:]}}
                for lab in ev["labels"]]
        writer.write(rows)
        print(f"  {meta['point']:<18} {method:<12} {input_id or case_id}: native/resampled dice " +
              " ".join(f"{r['label']}={r['native_dice']:.3f}/{r['dice']:.3f}" for r in rows))

    def score_inputs(meta, method, predict):
        """predict(list of input items for one (case, modality)) -> (m, p) flat decoders."""
        by_input = {}
        for item in test_items:
            by_input.setdefault(item["input"], []).append(item)
        for input_id, items in sorted(by_input.items()):
            if (meta["point"], method, input_id) in done:
                continue
            with torch.no_grad():
                predicted = predict(items)
            table = {(i["k"], i["label"]): layout.unflatten(v.to(device)) for i, v in zip(items, predicted)}
            ev = score(items[0]["case_id"], lambda k, part: table[(k, part["label"])])
            record(meta, method, items[0]["case_id"], ev, input_id, items[0]["modality"])

    # Ceiling: point-independent.
    if "ceiling" in methods:
        # Not trained on anything; zeros (not NaN) so its rows group as one point.
        meta = {"point": "ceiling", "case_fraction": 0.0, "stride": 1, "n_train_cases": 0,
                "n_train_slices": 0}
        for case_id in sorted(test_ids):
            if ("ceiling", "ceiling", case_id) not in done:
                record(meta, "ceiling", case_id,
                       score(case_id, lambda k, part: layout.unflatten(part["ceiling"].to(device))))

    # The learning curve: nested case subsets (one seeded order), slice strides.
    order = [split["train"][j] for j in np.random.RandomState(scfg["split_seed"]).permutation(len(train_ids))]
    for fraction, stride in points:
        subset = set(order[:max(1, math.ceil(fraction * len(order)))])
        train_items = input_items(subset, stride)
        n_train_slices = sum(1 for c in subset for k in cases[c]["slices"] if k % stride == 0)
        point = f"cases{fraction:g}_stride{stride}"
        if args.points and point not in args.points:
            continue
        meta = {"point": point, "case_fraction": fraction, "stride": stride, "n_train_cases": len(subset),
                "n_train_slices": n_train_slices}
        print(f"\n==== {point}: {len(subset)} training cases, {n_train_slices} slices, "
              f"{len(train_items)} (input, label) items")
        missing_labels = sorted(set(labels) - {i["label"] for i in train_items})
        if missing_labels:
            print(f"  skipped: no training slice for labels {missing_labels}")
            continue
        label_parts = lambda label: [p for c in sorted(subset) for k in cases[c]["slices"] if k % stride == 0
                                     for p in parts[(c, k)] if p["label"] == label]

        # What is left to do here: methods not fully scored yet.
        todo = []
        for method in (m for m in ("floor",) + LEARNERS if m in methods):
            ids = sorted(test_ids) if method == "floor" else test_inputs
            if all((point, method, i) in done for i in ids):
                print(f"  {method}: already scored, skipped")
            elif method in SEQUENCE_LEARNERS and stride != 1:
                print(f"  {method}: skipped (whole stacks, stride-1 points only)")
            else:
                todo.append(method)
                record_settings(out_dir, point, method, scfg, lcfg)
        floors = {}
        learners = [m for m in todo if m != "floor"]
        if "floor" in todo or (learners and lcfg["init_from_floor"]):
            for label in labels:
                torch.manual_seed(args.seed)
                floors[label] = train_floor(model, layout, label_parts(label), [], cfg, lcfg, spacing_mm,
                                            device, on_log(f"{point}/floor/{label}"), fresh)
        if "floor" in todo:
            for case_id in sorted(test_ids):
                if (point, "floor", case_id) not in done:
                    record(meta, "floor", case_id,
                           score(case_id, lambda k, part: layout.unflatten(floors[part["label"]])))

        for method in learners:
            # Same starting decoder as in experiments/hypernet: the floor, or a fresh init.
            origins = (torch.stack([floors[label] for label in labels]) if lcfg["init_from_floor"]
                       else fresh.expand(len(labels), -1))
            if method in SEQUENCE_LEARNERS:
                tcfg = sequence_config(scfg, lcfg)
                tokens = torch.stack([i["x"][:-1] for i in {i["key"]: i for i in train_items}.values()])
                torch.manual_seed(args.seed)
                net = SliceTransformer(tokens.shape[1], layout.numel, len(labels), scfg["transformer_dim"],
                                       scfg["transformer_layers"], scfg["transformer_heads"], tcfg["activation"],
                                       tcfg["dropout"], context=method == "transformer").to(device)
                print_parameter_summary(net, f"{point} {method} (dropout {tcfg['dropout']}, "
                                             f"weight decay {tcfg['weight_decay']})")
                torch.manual_seed(args.seed)
                net.set_reference(tokens.to(device), origins.to(device), scale.to(device))
                del tokens
                train_sequence(net, model, layout, train_items, test_items, cfg, tcfg, spacing_mm, device,
                               on_log(f"{point}/{method}"))
                score_inputs(meta, method, sequence_predictor(net, test_items, device))
                continue
            # position_only differs from hypernet only in its input: the decoder part zeroed.
            view = (lambda x: x) if method == "hypernet" else \
                (lambda x: torch.cat([torch.zeros_like(x[:-1]), x[-1:]]))
            train_m = [{**i, "x": view(i["x"])} for i in train_items]
            test_m = [{**i, "x": view(i["x"])} for i in test_items]
            train_x = torch.stack([i["x"] for i in {i["key"]: i for i in train_m}.values()])
            torch.manual_seed(args.seed)
            hyper = HyperMLP(train_x.shape[1], layout.numel, len(labels), lcfg["hidden_dim"], lcfg["layers"],
                             lcfg["activation"], lcfg["norm"], lcfg["dropout"]).to(device)
            print_hypernet_summary(hyper, train_x.shape[1], f"{point} {method}")
            torch.manual_seed(args.seed)
            hyper.set_reference(train_x.to(device), origins.to(device), scale.to(device))
            train_hypernet(hyper, model, layout, train_m, test_m, cfg, lcfg, spacing_mm, device,
                           on_log(f"{point}/{method}"))
            score_inputs(meta, method, lambda items: hyper(
                torch.stack([view(i["x"]) for i in items]).to(device),
                torch.tensor([i["label_index"] for i in items], device=device)))

    report(read_rows([results], INT_FIELDS, FLOAT_FIELDS, FIELDS), out_dir, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()


def train_sequence(net, model, layout, train_items, test_items, cfg, lcfg, spacing_mm, device, on_log):
    """train_hypernet for a SliceTransformer: the same optimizer, scheduler,
    steps, items and points per step and loss; each step runs the chosen
    items' whole sequences (experiments/slices/transformer.sequence_predictor)."""
    optimizer = torch.optim.AdamW(net.param_groups(lcfg["weight_decay"]), lr=lcfg["lr"])
    scheduler = build_scheduler(optimizer, {"name": lcfg["scheduler"]}, lcfg["steps"])

    def params_for(items, eval_mode):
        predict = sequence_predictor(net, items, device)

        def run(chosen):
            was_training = net.training
            net.train(not eval_mode and was_training)
            out = [layout.unflatten(v) for v in predict([items[i] for i in chosen])]
            net.train(was_training)
            return out
        return run

    net.train()
    train_weights(model, train_items, params_for(train_items, False), optimizer, scheduler, lcfg["steps"],
                  lcfg["cases_per_step"], lcfg["points_per_step"], cfg["sdf"], spacing_mm, device,
                  lcfg["log_every"], on_log, test_items, params_for(test_items, True))
    net.eval()


def report(rows: list, out_dir: Path, wandb_run):
    """Learning-curve summaries: per (point, method), case means then mean /
    median over test cases; printed, written and mirrored to wandb."""
    by = ("point", "case_fraction", "stride", "n_train_cases", "n_train_slices", "method")
    summary = aggregate_cases(rows, by, METRICS)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "summary_per_label.csv", aggregate_cases(rows, by + ("task", "label"), METRICS))
    metric = "native_dice" if any(not math.isnan(r["native_dice"]) for r in rows) else "dice"
    print(f"\n==== learning curve: mean {metric} | mean dice (resampled) over test cases ====")
    print(f"  {'method':<12} {'point':<18} {'cases':>5} {'slices':>7} {metric:>12} {'dice':>8}")
    for method in METHODS:
        for r in sorted((r for r in summary if r["method"] == method), key=lambda r: (r["n_train_slices"], r["point"])):
            print(f"  {method:<12} {r['point']:<18} {r['n_train_cases']:>5} {r['n_train_slices']:>7} "
                  f"{r[f'mean_{metric}']:>12.4f} {r['mean_dice']:>8.4f}")
    print(f"\nResults: {out_dir / 'results.csv'}; summaries: {out_dir / 'summary.csv'}, "
          f"{out_dir / 'summary_per_label.csv'}")
    if wandb_run is None:
        return
    import wandb
    clean = lambda v: None if isinstance(v, float) and math.isnan(v) else v
    wandb_run.summary.update({f"{r['point']}/{r['method']}/mean_{m}": clean(r[f"mean_{m}"])
                              for r in summary for m in METRICS})
    for name, table in (("results", rows), ("summary", summary)):
        if table:
            columns = list(table[0])
            wandb_run.log({name: wandb.Table(columns=columns, data=[[clean(r[c]) for c in columns] for r in table])})


if __name__ == "__main__":
    main()
