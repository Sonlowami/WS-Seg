"""
Resuming the slice pilot without mixing results.

--resume skips a (point, method) whose test inputs are all scored, so it must
know which settings those rows came from. The settings a method trains with
are kept in <out_dir>/learner_settings.json, {"<point>/<method>": {...}}, and
compared whenever that method already has rows: a difference stops the run
instead of silently keeping the old rows.

Rows written before this file existed have no record. They are kept, with a
notice, except that a sequence learner's rows are assumed to have used the
hypernet block's dropout and weight decay (there was nothing else to use) and
the transformer sizes in config.json, and are checked against those.
"""
import json
from pathlib import Path

SEQUENCE_LEARNERS = ("transformer", "transformer_nocontext")
# Only the sequence learners read these; None = the hypernet block's value.
SEQUENCE_OVERRIDES = {"transformer_dropout": "dropout", "transformer_weight_decay": "weight_decay"}
SEQUENCE_SIZES = ("transformer_dim", "transformer_layers", "transformer_heads")
SHARED = ("activation", "lr", "scheduler", "steps", "cases_per_step", "points_per_step", "init_from_floor",
          "floor_steps")
MLP = ("hidden_dim", "layers", "norm", "dropout", "weight_decay")


def sequence_config(scfg: dict, lcfg: dict) -> dict:
    """The hypernet block as the sequence learners use it: slices.transformer_dropout
    and transformer_weight_decay replace its dropout and weight_decay when set."""
    return {**lcfg, **{key: scfg[name] for name, key in SEQUENCE_OVERRIDES.items() if scfg[name] is not None}}


def learner_settings(method: str, scfg: dict, lcfg: dict) -> dict:
    """The settings method's results depend on, beyond those in config.json."""
    if method == "floor":
        keys = {k: lcfg[k] for k in ("cases_per_step", "points_per_step", "floor_steps")}
    elif method in SEQUENCE_LEARNERS:
        used = sequence_config(scfg, lcfg)
        keys = {**{k: lcfg[k] for k in SHARED}, **{k: scfg[k] for k in SEQUENCE_SIZES},
                "dropout": used["dropout"], "weight_decay": used["weight_decay"]}
    else:
        keys = {k: lcfg[k] for k in SHARED + MLP}
    return json.loads(json.dumps(keys))


def scored_methods(done: set) -> set:
    """(point, method) pairs with at least one scored input in done."""
    return {(point, method) for point, method, _ in done}


def check_settings(out_dir: Path, done: set, points: list, methods: list, scfg: dict, lcfg: dict,
                   old_slices: dict) -> None:
    """Stop if a requested (point, method) has rows from other settings.
    old_slices: the slices block of the config.json the rows were written with."""
    path = Path(out_dir) / "learner_settings.json"
    recorded = json.loads(path.read_text()) if path.exists() else {}
    for point, method in sorted(scored_methods(done)):
        if point not in points or method not in methods:
            continue
        now = learner_settings(method, scfg, lcfg)
        before = recorded.get(f"{point}/{method}")
        if before is None:
            print(f"note: {point}/{method} was scored before its settings were recorded; keeping its rows.")
            if method not in SEQUENCE_LEARNERS:
                continue
            before = {**now, "dropout": lcfg["dropout"], "weight_decay": lcfg["weight_decay"],
                      **{k: old_slices.get(k, now[k]) for k in SEQUENCE_SIZES}}
        changed = {k: (before.get(k), now.get(k)) for k in sorted(set(before) | set(now))
                   if before.get(k) != now.get(k)}
        if changed:
            raise SystemExit(f"{point}/{method} in {out_dir} was scored with other settings "
                             f"(then, now): {changed}. Choose another --out_dir; encoder/, slice_decoders/ and "
                             f"ceiling_decoders/ can be copied into it.")


def record_settings(out_dir: Path, point: str, method: str, scfg: dict, lcfg: dict) -> None:
    path = Path(out_dir) / "learner_settings.json"
    recorded = json.loads(path.read_text()) if path.exists() else {}
    recorded[f"{point}/{method}"] = learner_settings(method, scfg, lcfg)
    path.write_text(json.dumps(recorded, indent=2, sort_keys=True))
