"""
Translation test: does Encoder I (image-pretrained) or Encoder II
(SDF-pretrained) provide a better prior for fitting an unseen case's
multi-channel mask SDF?

For each held-out case and each step budget in {50, 100, 200}:
  - fit a fresh decoder against the frozen Encoder I
  - fit a fresh decoder against the frozen Encoder II
  - record PSNR, SSIM, and per-label + mean Dice / NSD for both

Fairness: the task (alpha, eikonal_lambda, label_groups, decode_mode) and the
decoder-fitting optimizer/scheduler are taken from ONE config -- Encoder II's
-- and applied to both encoders. Reading each checkpoint's own settings would
let a mismatch in alpha or optimizer silently masquerade as an encoder
difference. The two checkpoints must, however, share a model architecture,
since only the encoder weights are transferred.

Dice/NSD are computed alongside PSNR/SSIM because the Experiment 1
methodology names Dice as the success criterion; PSNR/SSIM alone don't say
whether boundaries come out right after thresholding.

Paired per-case results feed a Wilcoxon signed-rank test (chosen over a
paired t-test since sample counts are small and Dice/PSNR are unlikely to be
reliably normal at that scale).

Usage:
  python -m experiments.run_translation_test \\
      --encoder_I_ckpt checkpoints/encoder_I_intensity \\
      --encoder_II_ckpt checkpoints/encoder_II_sdf \\
      --data_root /path/to/Decathlon --split test --steps 50 100 200
"""
import argparse
import csv
import math
import torch
from scipy.stats import wilcoxon

from utils.io import load_model_weights
from utils.metrics import psnr_3d, ssim_3d, per_label_metrics
from data.dataset import build_dataset
from data.msd import load_tasks, resolve_label_groups_per_task
from sdf.coordinates import get_3d_coordinates, mm_per_unit
from sdf.targets import create_multilabel_sdf, sdf_to_channel_masks
from models.interfaces import build_model
from training.losses import masked_eikonal_sdf_loss
from training.train_loop import (
    build_optimizer, build_scheduler, resolve_device, sample_points, predict_in_chunks,
    DEFAULT_POINTS_PER_STEP,
)


def fit_decoder_and_eval(encoder_state_dict: dict, task_cfg: dict, label_groups: list,
                         case, coords, spacing_mm, steps: int):
    sdf_cfg = task_cfg["sdf"]
    alpha, eikonal_lambda = sdf_cfg["alpha"], sdf_cfg["eikonal_lambda"]
    decode_mode = sdf_cfg.get("decode_mode", "independent")

    train_cfg = task_cfg["training"]
    device = resolve_device(train_cfg.get("device", "auto"))
    model = build_model(task_cfg["model"], out_features=len(label_groups)).to(device)
    model.load_encoder_state_dict(encoder_state_dict, freeze=True)
    model.reset_decoder()

    label_map = case["mask"].squeeze().cpu().numpy()
    sdf_np = create_multilabel_sdf(label_map, label_groups, spacing_mm, alpha)
    target = torch.from_numpy(sdf_np).reshape(-1, sdf_np.shape[-1]).float().to(device)
    scale = mm_per_unit(label_map.shape, spacing_mm)
    coords = coords.to(device)

    optimizer = build_optimizer(model.decoder_parameters(), task_cfg["optimizer"])
    scheduler = build_scheduler(optimizer, task_cfg["scheduler"], steps)

    for _ in range(steps):
        optimizer.zero_grad()
        coords_batch, target_batch = sample_points(
            coords, target, train_cfg.get("points_per_step", DEFAULT_POINTS_PER_STEP), requires_grad=True)
        pred = model.forward(coords_batch)
        loss_dict = masked_eikonal_sdf_loss(pred, coords_batch, target_batch, alpha, eikonal_lambda,
                                            mm_per_unit=scale)
        loss_dict["total"].backward()
        optimizer.step()
        scheduler.step()

    pred_final = predict_in_chunks(model, coords, train_cfg.get("eval_chunk_size", 2 ** 20))

    shape = tuple(case["mask"].shape[-3:])
    k = len(label_groups)
    pred_sdf = pred_final.reshape(*shape, k).cpu().numpy()
    data_range = 2 * abs(sdf_np).max()

    pred_masks = sdf_to_channel_masks(pred_sdf, mode=decode_mode)
    gt_masks = sdf_np < 0.0

    metrics = {
        "psnr": psnr_3d(pred_sdf, sdf_np, data_range=data_range),
        "ssim": ssim_3d(pred_sdf, sdf_np, data_range=data_range, channel_axis=-1),
    }
    metrics.update(per_label_metrics(pred_masks, gt_masks, spacing_mm))
    return metrics


def run_significance_tests(results: list, steps: list):
    """Paired Wilcoxon signed-rank test per step budget, on mean Dice and PSNR."""
    summary = []
    for steps_n in steps:
        for metric in ("mean_dice", "psnr"):
            pairs = [
                (r[f"enc_I_{metric}"], r[f"enc_II_{metric}"])
                for r in results if r["steps"] == steps_n
                and not math.isnan(r[f"enc_I_{metric}"])
                and not math.isnan(r[f"enc_II_{metric}"])
            ]
            if len(pairs) < 2:
                continue  # Wilcoxon needs at least a couple of paired samples
            enc_I, enc_II = zip(*pairs)
            stat, p_value = wilcoxon(enc_I, enc_II)
            summary.append({
                "steps": steps_n, "metric": metric, "n_pairs": len(pairs),
                "median_enc_I": sorted(enc_I)[len(enc_I) // 2],
                "median_enc_II": sorted(enc_II)[len(enc_II) // 2],
                "wilcoxon_stat": stat, "p_value": p_value,
            })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_I_ckpt", required=True)
    parser.add_argument("--encoder_II_ckpt", required=True)
    parser.add_argument("--data_root", default=None,
                        help="Override data.root (paths differ between machines)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--steps", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--out_csv", default="translation_test_results.csv")
    args = parser.parse_args()

    ckpt_I = load_model_weights(args.encoder_I_ckpt)
    ckpt_II = load_model_weights(args.encoder_II_ckpt)

    assert ckpt_I["config"]["model"] == ckpt_II["config"]["model"], (
        "Encoder I and Encoder II were trained with different model "
        "architectures; their encoder weights are not comparable."
    )
    if ckpt_I["config"]["data"]["spacing_mm"] != ckpt_II["config"]["data"]["spacing_mm"]:
        raise ValueError("Encoders were trained at different voxel spacings.")

    # Both encoders must have been trained on the same tasks and the same
    # split; otherwise "held-out" cases for one may have been training cases
    # for the other.
    data_I, data_II = ckpt_I["config"]["data"], ckpt_II["config"]["data"]
    for key in ("tasks", "split"):
        if data_I[key] != data_II[key]:
            raise ValueError(f"Encoders were trained with different data.{key}: "
                             f"{data_I[key]} vs {data_II[key]}")

    task_cfg = ckpt_II["config"]     # single source of truth for the task and fit settings
    if args.data_root:
        task_cfg["data"]["root"] = args.data_root
    spacing_mm = tuple(task_cfg["data"]["spacing_mm"])

    # A fresh model is built per fit, so tasks may have different group counts.
    groups_by_task = resolve_label_groups_per_task(task_cfg["sdf"]["label_groups"],
                                                   load_tasks(task_cfg["data"]))
    dataset = build_dataset(task_cfg["data"], split=args.split)

    results = []
    for case in dataset:
        shape = tuple(case["mask"].shape[-3:])
        coords = get_3d_coordinates(shape, spacing_mm).coords
        label_groups = groups_by_task[case["task"]]
        for steps in args.steps:
            r_I = fit_decoder_and_eval(ckpt_I["encoder_state_dict"], task_cfg, label_groups,
                                       case, coords, spacing_mm, steps)
            r_II = fit_decoder_and_eval(ckpt_II["encoder_state_dict"], task_cfg, label_groups,
                                        case, coords, spacing_mm, steps)
            results.append({
                "case_id": case["case_id"], "steps": steps,
                **{f"enc_I_{k}": v for k, v in r_I.items()},
                **{f"enc_II_{k}": v for k, v in r_II.items()},
            })
            print(f"case={case['case_id']} steps={steps} "
                  f"enc_I_mean_dice={r_I['mean_dice']:.3f} enc_II_mean_dice={r_II['mean_dice']:.3f}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-sample results written to {args.out_csv}")

    significance = run_significance_tests(results, args.steps)
    print("\nWilcoxon signed-rank test (Encoder I vs Encoder II, paired per case):")
    for row in significance:
        print(f"  steps={row['steps']:>3} metric={row['metric']:<9} n={row['n_pairs']:<3} "
              f"median_I={row['median_enc_I']:.3f} median_II={row['median_enc_II']:.3f} "
              f"p={row['p_value']:.4f}")


if __name__ == "__main__":
    main()
