"""GKD training under a teacher-selection policy.

Single-teacher policies (fixed/strongest) run one GKDTrainer. Multi-teacher
policies (random/task/learned) partition the data by assigned teacher and train
in interleaved rounds — one pass per teacher shard per round. Optimizer state
resets between shards; acceptable at pilot scale, noted in README.

All teachers must share the student's tokenizer (same-family models) so the
teacher's next-token distribution aligns with student tokens.
"""

import argparse
import json
from pathlib import Path

import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer

from mtkd.data import build_dataset
from mtkd.gen import load_model, pick_device
from mtkd.policies import assign_teachers


def check_tokenizer_compat(student_name: str, teacher_names: list[str]) -> None:
    probe_text = "def f(x):\n    return 3.14 * x  # 你好"
    ref_tok = AutoTokenizer.from_pretrained(student_name)
    ref = ref_tok.encode(probe_text)
    for name in teacher_names:
        tok = AutoTokenizer.from_pretrained(name)
        if tok.vocab_size != ref_tok.vocab_size or tok.encode(probe_text) != ref:
            raise ValueError(
                f"tokenizer mismatch: {name} vs {student_name} — GKD needs a shared "
                "token space; use same-family teachers"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = pick_device()
    print(f"device: {device}")

    dataset = build_dataset(cfg["n_train_per_category"], split="train", seed=cfg.get("seed", 0))
    dataset = assign_teachers(
        dataset,
        policy=cfg["policy"],
        teachers=cfg["teachers"],
        category_map=cfg.get("category_map"),
        probe_results_path=cfg.get("probe_results"),
        seed=cfg.get("seed", 0),
    )
    used_teacher_ids = sorted(set(dataset["teacher_id"]))
    print(f"policy={cfg['policy']}  teachers in use: "
          f"{[cfg['teachers'][i] for i in used_teacher_ids]}")

    check_tokenizer_compat(cfg["student"], [cfg["teachers"][i] for i in used_teacher_ids])

    student, tokenizer = load_model(cfg["student"], device)
    student.train()
    if cfg.get("lora"):
        lora = cfg["lora"]
        student = get_peft_model(
            student,
            LoraConfig(
                r=lora["r"],
                lora_alpha=lora["alpha"],
                target_modules=lora.get("target_modules", "all-linear"),
                task_type="CAUSAL_LM",
            ),
        )
        student.print_trainable_parameters()

    teachers = {
        i: load_model(cfg["teachers"][i], device)[0] for i in used_teacher_ids
    }

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = cfg.get("max_steps_per_shard", -1)
    rounds = cfg.get("rounds", 1)

    shards = {
        i: dataset.filter(lambda ex, i=i: ex["teacher_id"] == i).remove_columns(
            ["category", "reference", "teacher_id"]
        )
        for i in used_teacher_ids
    }

    for round_idx in range(rounds):
        for teacher_id in used_teacher_ids:
            shard = shards[teacher_id]
            if len(shard) == 0:
                continue
            print(f"round {round_idx + 1}/{rounds}  teacher={cfg['teachers'][teacher_id]}  "
                  f"examples={len(shard)}")
            gkd_args = GKDConfig(
                output_dir=str(output_dir / "trainer"),
                num_train_epochs=1,
                max_steps=max_steps,
                report_to=[],
                save_strategy="no",
                **cfg["gkd"],
            )
            trainer = GKDTrainer(
                model=student,
                teacher_model=teachers[teacher_id],
                args=gkd_args,
                processing_class=tokenizer,
                train_dataset=shard,
            )
            trainer.train()
            student = trainer.model

    final_dir = output_dir / "final"
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved to {final_dir}")


if __name__ == "__main__":
    main()
