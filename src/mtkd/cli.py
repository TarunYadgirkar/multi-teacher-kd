"""`mtkd` command line. `mtkd reproduce` runs every stage in order and skips work whose
outputs already exist, so an interrupted run resumes where it stopped."""

import argparse
import json
from pathlib import Path

from mtkd.config import apply_memory_cap, load_config

STAGES = ("probe", "base", "matrix", "teachers", "report")
DEFAULT_STAGES = ("probe", "base", "matrix", "report")


def _eval_dir(cfg: dict) -> Path:
    path = Path(cfg["paths"]["results"]) / "eval"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _eval_examples(cfg: dict, cache: dict) -> list[dict]:
    from mtkd.data import load_split
    if "eval" not in cache:
        cache["eval"] = load_split("eval", cfg["data"]["eval"])
    return cache["eval"]


def stage_base(cfg: dict, cache: dict, force: bool) -> None:
    from mtkd.run_eval import run
    stem = _eval_dir(cfg) / "base"
    if stem.with_suffix(".json").exists() and not force:
        print("base eval: cached")
        return
    print("eval base student", flush=True)
    run(cfg, cfg["student"], _eval_examples(cfg, cache), stem, {"run": "base", "split": "eval"})


def stage_teachers(cfg: dict, cache: dict, force: bool) -> None:
    from mtkd.run_eval import run
    for key, name in cfg["teachers"].items():
        stem = _eval_dir(cfg) / f"teacher-{key}"
        if stem.with_suffix(".json").exists() and not force:
            continue
        print(f"eval teacher {key} on the eval split", flush=True)
        run(cfg, name, _eval_examples(cfg, cache), stem, {"run": f"teacher-{key}", "split": "eval"})


def collapsed_into(cfg: dict, condition: str, learned_map) -> str | None:
    """Name of an earlier condition that routes every category identically, if any."""
    from mtkd.config import condition_train_cfg
    from mtkd.policies import describe
    keys, categories = list(cfg["teachers"]), sorted(cfg["data"]["train"])
    spec = cfg["conditions"][condition]
    if spec.get("policy") != "learned":
        return None
    mine = describe(spec, keys, learned_map, categories)
    for other, other_spec in cfg["conditions"].items():
        if other == condition or other_spec.get("kind") == "sft" or other_spec.get("policy") in ("learned", "random"):
            continue
        same_train = condition_train_cfg(cfg, other) == condition_train_cfg(cfg, condition)
        if same_train and describe(other_spec, keys, learned_map, categories) == mine:
            return other
    return None


def stage_matrix(cfg: dict, cache: dict, conditions: list[str], seeds: list[int], force: bool) -> None:
    from mtkd.probe import load_learned_map
    from mtkd.run_eval import run
    from mtkd.train import is_trained, run_dir, train
    learned_map = load_learned_map(cfg)
    notes = {}
    for seed in seeds:
        for condition in conditions:
            if cfg["conditions"][condition].get("policy") == "learned" and learned_map is None:
                raise SystemExit("the learned condition needs the probe stage first")
            twin = collapsed_into(cfg, condition, learned_map)
            if twin:
                notes[condition] = twin
                print(f"{condition}: routes exactly like {twin}; skipped")
                continue
            if force or not is_trained(cfg, condition, seed):
                train(cfg, condition, seed, learned_map)
            stem = _eval_dir(cfg) / f"{condition}-s{seed}"
            if force or not stem.with_suffix(".json").exists():
                print(f"eval {condition} seed {seed}", flush=True)
                run(cfg, str(run_dir(cfg, condition, seed) / "adapter"), _eval_examples(cfg, cache), stem,
                    {"run": f"{condition}-s{seed}", "condition": condition, "seed": seed, "split": "eval"})
    if notes:
        path = Path(cfg["paths"]["results"]) / "collapsed.json"
        path.write_text(json.dumps({**notes, "learned_map": learned_map}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="mtkd")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("reproduce", help="run probe, base eval, the training matrix and the report")
    rep.add_argument("--config", default="configs/main.yaml")
    rep.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                     help=f"comma-separated subset of {','.join(STAGES)}")
    rep.add_argument("--conditions", help="comma-separated subset of the config's conditions")
    rep.add_argument("--seeds", help="comma-separated seeds (default: train.seeds)")
    rep.add_argument("--force", action="store_true", help="redo stages whose outputs exist")
    rpt = sub.add_parser("report", help="rebuild REPORT tables and figures from results/")
    rpt.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_memory_cap(cfg)
    if args.command == "report":
        from mtkd.report import build_report
        build_report(cfg)
        return

    stages = [s.strip() for s in args.stages.split(",")]
    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    conditions = args.conditions.split(",") if args.conditions else list(cfg["conditions"])
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["train"]["seeds"]
    cache: dict = {}
    if "probe" in stages:
        from mtkd.probe import run_probe
        print(json.dumps(run_probe(cfg, args.force)["learned_map"]))
    if "base" in stages:
        stage_base(cfg, cache, args.force)
    if "matrix" in stages:
        stage_matrix(cfg, cache, conditions, seeds, args.force)
    if "teachers" in stages:
        stage_teachers(cfg, cache, args.force)
    if "report" in stages:
        from mtkd.report import build_report
        build_report(cfg)


if __name__ == "__main__":
    main()
