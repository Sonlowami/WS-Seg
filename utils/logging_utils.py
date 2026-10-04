"""Thin wandb wrapper so the training loop doesn't import wandb directly
(keeps it importable/testable in environments without wandb configured)."""
from dotenv import load_dotenv

load_dotenv()

def configure_wandb(wandb_cfg: dict, experiment_name: str, full_config: dict, **init_kwargs):
    """init_kwargs are passed to wandb.init (e.g. group, job_type)."""
    try:
        import wandb
        import os
        # Offline/disabled runs need no account (and login would prompt for one).
        if os.environ.get("WANDB_MODE") not in ("offline", "disabled"):
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
        **init_kwargs,
    )


def _to_scalar(v):
    # Tensors (including MONAI MetaTensors) and numpy scalars -> plain floats;
    # wandb's JSON encoder does not know tensor subclasses.
    item = getattr(v, "item", None)
    return item() if callable(item) else v


def log(run, metrics: dict, step: int):
    metrics = {k: _to_scalar(v) for k, v in metrics.items()}
    if run is not None:
        run.log(metrics, step=step)
    else:
        formatted = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step}] {formatted}")
