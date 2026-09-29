"""
Checkpoint I/O. Every save is paired with the exact config that produced it,
so it's always traceable which alpha / loss / optimizer settings a given
encoder checkpoint corresponds to -- important once Encoder I and Encoder II
runs start happening back-to-back under time pressure.
"""
import json
import torch
from pathlib import Path


def save_encoder_weights(encoder_state_dict: dict, config: dict, out_dir: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state_dict, out_dir / "encoder.pt")
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return out_dir


def save_decoder_weights(decoder_state_dict: dict, case_id: str, out_dir: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(decoder_state_dict, out_dir / f"decoder_{case_id}.pt")
    return out_dir


def load_model_weights(checkpoint_dir: str, map_location: str = "cpu") -> dict:
    """
    Loads an encoder checkpoint and its paired config together, so callers
    can verify (e.g. in the translation test) that they're comparing
    checkpoints trained under compatible settings before running metrics.
    """
    checkpoint_dir = Path(checkpoint_dir)
    encoder_state_dict = torch.load(checkpoint_dir / "encoder.pt", map_location=map_location)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    return {"encoder_state_dict": encoder_state_dict, "config": config}
