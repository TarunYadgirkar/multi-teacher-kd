"""Score a model on a list of examples and save every generation.

Used for students (base and trained, on the eval split) and for teachers (on the probe
split). Rows are written as gzipped JSONL next to a JSON summary with provenance.
"""

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from peft import PeftModel

from mtkd.evaluate import score_generation
from mtkd.gen import (choice_predictions, eos_ids, generate, load_model, load_tokenizer,
                      pick_device, release_memory, render)
from mtkd.provenance import provenance


def evaluate_model(model, tokenizer, examples: list[dict], gen_cfg: dict) -> list[dict]:
    prompts = [render(tokenizer, ex["prompt"]) for ex in examples]
    stop = eos_ids(tokenizer)
    rows: list[dict] = [{"id": ex["id"], "category": ex["category"]} for ex in examples]
    by_category: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        by_category.setdefault(ex["category"], []).append(i)

    for category, idx in by_category.items():
        max_new = gen_cfg["max_new_tokens"][category]
        outs = generate(model, [prompts[i] for i in idx], max_new, tokenizer.pad_token_id, stop,
                        batch_size=gen_cfg["batch_size"], first_pass=gen_cfg.get("first_pass"))
        for i, out in zip(idx, outs):
            rows[i].update(output=tokenizer.decode(out.ids, skip_special_tokens=True),
                           n_tokens=len(out.ids), truncated=not out.finished)
        if category == "qa":
            picks = choice_predictions(model, tokenizer, [prompts[i] for i in idx],
                                       [examples[i]["reference"]["labels"] for i in idx],
                                       batch_size=gen_cfg["batch_size"])
            for i, pick in zip(idx, picks):
                rows[i].update(pick, score=float(pick["choice"] == examples[i]["reference"]["answer"]))

    def scored(i: int) -> dict:
        ex = examples[i]
        return score_generation(ex["category"], rows[i]["output"], ex["reference"])

    with ThreadPoolExecutor(max_workers=gen_cfg.get("code_workers", 4)) as pool:
        for i, metrics in zip(range(len(rows)), pool.map(scored, range(len(rows)))):
            rows[i].update(metrics)
    return rows


def summarize(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for category in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == category]
        out[category] = {
            "n": len(sub),
            "score": sum(r["score"] for r in sub) / len(sub),
            "truncated": sum(r["truncated"] for r in sub) / len(sub),
            "mean_tokens": sum(r["n_tokens"] for r in sub) / len(sub),
        }
        if category == "qa":
            out[category]["gen_score"] = sum(r["gen_score"] for r in sub) / len(sub)
        if category == "summ":
            for key in ("rouge1", "rouge2", "words"):
                out[category][key] = sum(r[key] for r in sub) / len(sub)
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def load_for_eval(cfg: dict, model_ref: str, device: str):
    """model_ref is a hub id or a LoRA adapter dir trained on cfg['student']."""
    if (Path(model_ref) / "adapter_config.json").exists():
        base = load_model(cfg["student"], cfg["revisions"].get(cfg["student"]), device)
        return PeftModel.from_pretrained(base, model_ref).merge_and_unload().eval()
    return load_model(model_ref, cfg["revisions"].get(model_ref), device)


def run(cfg: dict, model_ref: str, examples: list[dict], out_stem: Path, extra: dict) -> dict:
    device = pick_device()
    tokenizer = load_tokenizer(cfg["student"], cfg["revisions"].get(cfg["student"]))
    model = load_for_eval(cfg, model_ref, device)
    rows = evaluate_model(model, tokenizer, examples, cfg["generation"])
    del model
    release_memory()
    summary = {"model": model_ref, **extra, "summary": summarize(rows),
               "generation": cfg["generation"], "provenance": provenance(device)}
    write_rows(out_stem.with_suffix(".jsonl.gz"), rows)
    out_stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary
