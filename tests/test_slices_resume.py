import pytest

from experiments.slices.resume import check_settings, learner_settings, record_settings, sequence_config

LCFG = {"activation": "gelu", "lr": 1e-4, "scheduler": "cosine", "steps": 20, "cases_per_step": 16,
        "points_per_step": 64, "init_from_floor": True, "floor_steps": 5, "hidden_dim": 8, "layers": 2,
        "norm": "batch", "dropout": 0.0, "weight_decay": 0.001}
SCFG = {"transformer_dim": 16, "transformer_layers": 2, "transformer_heads": 4,
        "transformer_dropout": None, "transformer_weight_decay": None}
POINT = "cases1_stride1"
DONE = {(POINT, m, "c0/MRI") for m in ("hypernet", "transformer")}


def _check(tmp_path, scfg=SCFG, lcfg=LCFG, methods=("hypernet", "transformer"), old=SCFG):
    check_settings(tmp_path, DONE, [POINT], list(methods), scfg, lcfg, old)


def test_overrides_change_the_sequence_learners_only():
    scfg = {**SCFG, "transformer_dropout": 0.2, "transformer_weight_decay": 0.1}
    used = sequence_config(scfg, LCFG)
    assert (used["dropout"], used["weight_decay"]) == (0.2, 0.1)
    assert (LCFG["dropout"], LCFG["weight_decay"]) == (0.0, 0.001)
    assert sequence_config(SCFG, LCFG) == LCFG
    assert sequence_config({**SCFG, "transformer_dropout": 0.0}, {**LCFG, "dropout": 0.3})["dropout"] == 0.0
    assert learner_settings("hypernet", scfg, LCFG) == learner_settings("hypernet", SCFG, LCFG)
    assert learner_settings("transformer", scfg, LCFG) != learner_settings("transformer", SCFG, LCFG)


def test_recorded_settings_are_enforced_per_method(tmp_path):
    for m in ("hypernet", "transformer"):
        record_settings(tmp_path, POINT, m, SCFG, LCFG)
    _check(tmp_path)
    with pytest.raises(SystemExit, match="dropout"):
        _check(tmp_path, scfg={**SCFG, "transformer_dropout": 0.2})
    _check(tmp_path, scfg={**SCFG, "transformer_dropout": 0.2}, methods=("hypernet",))     # not requested
    with pytest.raises(SystemExit, match="layers"):
        _check(tmp_path, lcfg={**LCFG, "layers": 4})


def test_rows_without_a_record_are_kept_but_not_under_new_transformer_settings(tmp_path):
    _check(tmp_path)
    for change in ({"transformer_dropout": 0.2}, {"transformer_weight_decay": 0.1}, {"transformer_dim": 32}):
        with pytest.raises(SystemExit):
            _check(tmp_path, scfg={**SCFG, **change})
    _check(tmp_path, scfg={**SCFG, "transformer_dropout": 0.0})      # the same effective value
