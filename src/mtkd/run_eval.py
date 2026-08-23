"""Evaluate a (trained or base) student per category on held-out test prompts."""

import argparse
import json
from pathlib import Path

import yaml
from peft import PeftModel

from mtkd.data import build_dataset
from mtkd.evaluate import aggregate, score
from mtkd.gen import generate, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True,
                        help="model path or hub id; LoRA adapter dirs are merged onto cfg.student")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_set = build_dataset(cfg["n_eval_per_category"], split="eval", seed=2)

    if (Path(args.model) / "adapter_config.json").exists():
        model, tokenizer = load_model(cfg["student"])
        model = PeftModel.from_pretrained(model, args.model).merge_and_unload()
    else:
        model, tokenizer = load_model(args.model)

    outputs = generate(
        model, tokenizer, eval_set["messages"],
        max_new_tokens=cfg["gkd"]["max_new_tokens"],
        batch_size=cfg.get("gen_batch_size", 8),
    )
    run_code = cfg.get("run_code_eval", True)
    rows = [
        {"category": ex["category"], "prompt": ex["messages"][0]["content"][:200],
         "output": out,
         "score": score(ex["category"], out, ex["reference"], run_code=run_code)}
        for ex, out in zip(eval_set, outputs)
    ]
    summary = aggregate([{"category": r["category"], "score": r["score"]} for r in rows])
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "summary": summary, "rows": rows}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
