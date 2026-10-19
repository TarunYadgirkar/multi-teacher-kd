"""Probe every teacher on the validation-style probe split, under the student's chat
template, and derive the learned category -> teacher map from the per-item scores."""

import json
from pathlib import Path

from mtkd.data import load_split
from mtkd.policies import select_learned_map
from mtkd.run_eval import read_rows, run
from mtkd.stats import mean_ci


def probe_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["results"]) / "probe"


def run_probe(cfg: dict, force: bool = False) -> dict:
    out = probe_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    examples = None
    for key, name in cfg["teachers"].items():
        stem = out / key
        if stem.with_suffix(".json").exists() and not force:
            print(f"probe {key}: cached")
            continue
        examples = examples or load_split("probe", cfg["data"]["probe"])
        print(f"probe {key} ({name}) on {len(examples)} items", flush=True)
        run(cfg, name, examples, stem, {"teacher": key, "split": "probe"})
    return summarize_probe(cfg)


def summarize_probe(cfg: dict) -> dict:
    out = probe_dir(cfg)
    per_item: dict[str, dict[str, list[float]]] = {}
    table: dict[str, dict] = {}
    for key in cfg["teachers"]:
        rows = sorted(read_rows(out / f"{key}.jsonl.gz"), key=lambda r: r["id"])
        per_item[key] = {}
        for row in rows:
            per_item[key].setdefault(row["category"], []).append(row["score"])
        table[key] = {c: mean_ci(v) for c, v in per_item[key].items()}
        table[key]["truncated"] = {
            c: sum(r["truncated"] for r in rows if r["category"] == c) / len(per_item[key][c])
            for c in per_item[key]
        }
    learned, evidence = select_learned_map(per_item, cfg["default_teacher"])
    summary = {"teachers": cfg["teachers"], "scores": table, "learned_map": learned,
               "evidence": evidence}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_learned_map(cfg: dict) -> dict[str, str] | None:
    path = probe_dir(cfg) / "summary.json"
    return json.loads(path.read_text())["learned_map"] if path.exists() else None
