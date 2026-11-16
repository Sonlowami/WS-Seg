"""Every decoder starts from the same weights: ceilings (fit_sdf), floors
(train_floor), and the slice pilot's intensity decoders and hypernet base
(initial_decoder). Each start is captured on the real code path."""
import numpy as np
import torch

import experiments.hypernet.run_hypernet as run_hypernet
import training.sdf_fit as sdf_fit
from experiments.hypernet.model import HeadLayout
from experiments.slices.run_slices import initial_decoder
from models.interfaces import build_model

MODEL = {"encoder_layers": 2, "decoder_layers": 2, "hidden_dim": 16, "decoder_hidden_dim": 8}
SEED = 7


class Captured(Exception):
    pass


def _encoder():
    torch.manual_seed(123)
    return build_model(MODEL, out_features=1).encoder_state_dict()


def _head(model):
    return torch.cat([p.detach().flatten() for p in model.decoder[0].parameters()])


def test_ceiling_floor_and_slice_decoders_share_one_start(monkeypatch):
    encoder = _encoder()
    _, canonical = initial_decoder(MODEL, encoder, 1, SEED, "cpu")
    starts = {}

    def capture(name):
        def fake(model, *args, **kwargs):
            starts[name] = _head(model)
            raise Captured
        return fake

    # Ceiling: fit_sdf's decoder when fitting begins.
    monkeypatch.setattr(sdf_fit, "fit_inr", capture("ceiling"))
    cfg = {"model": MODEL, "sdf": {"alpha": 5.0, "eikonal_lambda": 0.1}, "optimizer": {}, "scheduler": {}}
    sdf_case = {"sdf": np.zeros((4, 4, 1, 1), dtype=np.float32), "shape": (4, 4, 1), "spacing_mm": (1., 1., 1.)}
    try:
        sdf_fit.fit_sdf(cfg, sdf_case, 1, encoder_state_dict=encoder, freeze_encoder=True, seed=SEED)
    except Captured:
        pass

    # Floor: train_floor's decoder when training begins, from a head that
    # holds other weights (as after a previous floor or prediction).
    model, _ = initial_decoder(MODEL, encoder, 1, SEED + 1, "cpu")
    monkeypatch.setattr(run_hypernet, "train_weights", lambda model, *a, **k: starts.update(floor=_head(model)))
    run_hypernet.train_floor(model, HeadLayout(model.decoder[0]), [], [], {"optimizer": {"name": "adam", "initial_lr": 1e-3},
                             "scheduler": {"name": "cosine"}, "sdf": {}},
                             {"floor_steps": 1, "cases_per_step": 1, "points_per_step": 1, "log_every": 1},
                             (1., 1., 1.), "cpu", None, canonical)

    assert torch.equal(starts["ceiling"], canonical)
    assert torch.equal(starts["floor"], canonical)
    # Same seed, same build: the slice intensity decoders' start is the same too.
    assert torch.equal(initial_decoder(MODEL, encoder, 1, SEED, "cpu")[1], canonical)
