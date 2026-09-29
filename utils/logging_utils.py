"""Thin wandb wrapper so the training loop doesn't import wandb directly
(keeps it importable/testable in environments without wandb configured)."""
from dotenv import load_dotenv

load_dotenv()

def configure_wandb(wandb_cfg: dict, experiment_name: str, full_config: dict):
    try:
        import wandb
        import os
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
    except ImportError:
        print("wandb not installed; logging to stdout only.")
        return None

    run_name = wandb_cfg.get("run_name") or experiment_name
    return wandb.init(
        entity=wandb_cfg["entity"],
        project=wandb_cfg["project"],
        name=run_name,
        config=full_config,
    )


def log(run, metrics: dict, step: int):
    if run is not None:
        run.log(metrics, step=step)
    else:
        formatted = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step}] {formatted}")
