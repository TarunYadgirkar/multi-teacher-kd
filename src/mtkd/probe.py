"""Score every teacher per category on a held-out probe set.

Output feeds the 'strongest' and 'learned' policies:
results/probe.json = {teacher_name: {math: .., code: .., qa: .., summ: .., overall: ..}}
"""

import argparse
import json
from pathlib import Path

import yaml

from mtkd.data import build_dataset
from mtkd.evaluate import aggregate, score
from mtkd.gen import generate, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results/probe.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    probe_set = build_dataset(cfg["n_probe_per_category"], split="eval", seed=1)
    run_code = cfg.get("run_code_eval", True)
    results: dict[str, dict] = {}

    for name in cfg["teachers"]:
        print(f"probing {name} on {len(probe_set)} prompts")
        model, tokenizer = load_model(name)
        outputs = generate(
            model, tokenizer, probe_set["messages"],
            max_new_tokens=cfg["gkd"]["max_new_tokens"],
            batch_size=cfg.get("gen_batch_size", 8),
        )
        rows = [
            {"category": ex["category"],
             "score": score(ex["category"], out, ex["reference"], run_code=run_code)}
            for ex, out in zip(probe_set, outputs)
        ]
        results[name] = aggregate(rows)
        print(f"  {results[name]}")
        del model

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
