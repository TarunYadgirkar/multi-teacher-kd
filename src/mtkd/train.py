"""Train one condition at one seed.

One optimizer and one LR schedule for the whole run, whatever the policy. Batches are
category-homogeneous and their order depends only on the seed, so every condition sees
the same batches in the same order and differs only in which teacher scores each batch.

kind "distill": with probability lmbda a batch is sampled from the current student
(on-policy) and scored by the teacher(s) with the generalized JSD; otherwise the gold
completion is scored instead. kind "sft": cross-entropy on the gold completions.
"""

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from mtkd.config import condition_train_cfg, fingerprint
from mtkd.data import load_split
from mtkd.distill import TeacherPool, chunked_ce_loss, chunked_distill_loss, ensemble_logits
from mtkd.gen import (decoder_and_head, eos_ids, generate, load_model, load_tokenizer, pick_device,
                      render)
from mtkd.policies import teachers_for_batch
from mtkd.provenance import provenance


def make_schedule(categories: list[str], batch_size: int, epochs: int,
                  seed: int) -> list[tuple[str, list[int]]]:
    """Category-homogeneous batches over example indices, shuffled within each epoch."""
    rng = random.Random(seed)
    by_category: dict[str, list[int]] = defaultdict(list)
    for i, category in enumerate(categories):
        by_category[category].append(i)
    schedule = []
    for _ in range(epochs):
        epoch = []
        for category in sorted(by_category):
            order = by_category[category][:]
            rng.shuffle(order)
            epoch.extend((category, order[i : i + batch_size]) for i in range(0, len(order), batch_size))
        rng.shuffle(epoch)
        schedule.extend(epoch)
    return schedule


def build_batch(prompts: list[list[int]], completions: list[list[int]], pad_id: int, device):
    """Right-padded ids plus masks: `targets` marks completion tokens and `predict` marks
    the positions whose next-token distribution predicts them."""
    width = max(len(p) + len(c) for p, c in zip(prompts, completions))
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    targets = torch.zeros_like(ids, dtype=torch.bool)
    for row, (prompt, completion) in enumerate(zip(prompts, completions)):
        seq = prompt + completion
        ids[row, : len(seq)] = torch.tensor(seq)
        attention[row, : len(seq)] = 1
        targets[row, len(prompt) : len(seq)] = True
    predict = torch.zeros_like(targets)
    predict[:, :-1] = targets[:, 1:]
    return ids.to(device), attention.to(device), predict.to(device), ids[targets].to(device)


def lr_lambda(total: int, warmup: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))
    return factor


def run_dir(cfg: dict, condition: str, seed: int) -> Path:
    """Adapter weights (large, not committed)."""
    return Path(cfg["paths"]["outputs"]) / condition / f"seed{seed}"


def log_stem(cfg: dict, condition: str, seed: int) -> Path:
    """Training log and run metadata (small, committed)."""
    return Path(cfg["paths"]["results"]) / "train" / f"{condition}-s{seed}"


def is_trained(cfg: dict, condition: str, seed: int) -> bool:
    meta = log_stem(cfg, condition, seed).with_suffix(".json")
    adapter = run_dir(cfg, condition, seed) / "adapter" / "adapter_config.json"
    return meta.exists() and adapter.exists() and json.loads(meta.read_text()).get("finished", False)


def _load_student(cfg: dict, tcfg: dict, device: str):
    student = load_model(cfg["student"], cfg["revisions"].get(cfg["student"]), device)
    lora = tcfg["lora"]
    student = get_peft_model(student, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora.get("target_modules", "all-linear"), task_type="CAUSAL_LM",
    ))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    return student


def _completions(student, tokenizer, examples: list[dict], prompts: list[list[int]],
                 on_policy: bool, max_new: int, tcfg: dict) -> list[list[int]]:
    stop = eos_ids(tokenizer)
    if on_policy:
        student.eval()
        out = generate(student, prompts, max_new, tokenizer.pad_token_id, stop,
                       batch_size=len(prompts), sample=True, temperature=tcfg["temperature"])
        student.train()
        return [c.ids if c.ids else [tokenizer.eos_token_id] for c in out]
    gold = [tokenizer(ex["target"], add_special_tokens=False)["input_ids"][:max_new] for ex in examples]
    return [ids + [tokenizer.eos_token_id] for ids in gold]


@dataclass
class Run:
    spec: dict
    tcfg: dict
    student: torch.nn.Module
    tokenizer: object
    pool: TeacherPool
    device: str
    teacher_keys: list[str]
    learned_map: dict[str, str] | None
    lmbda_rng: random.Random
    policy_rng: random.Random

    @property
    def is_sft(self) -> bool:
        return self.spec.get("kind") == "sft"


def _sync(device: str) -> float:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    return time.time()


def _student_loss(run: Run, micro: tuple, hiddens: list[torch.Tensor], teachers: list[str]) -> torch.Tensor:
    ids, attention, predict, targets = micro
    decoder, head = decoder_and_head(run.student)
    hidden = decoder(input_ids=ids, attention_mask=attention).last_hidden_state[predict]
    chunk = run.tcfg["loss_chunk_tokens"]
    if run.is_sft:
        return chunked_ce_loss(hidden, head, targets, chunk)
    teacher_logits = ensemble_logits(hiddens, [run.pool.head(k) for k in teachers])
    return chunked_distill_loss(hidden, head, teacher_logits, run.tcfg["beta"], run.tcfg["temperature"], chunk)


def train_step(run: Run, category: str, batch: list[dict], batch_prompts: list[list[int]]) -> dict:
    """Completions for the whole batch, then forward/backward in length-sorted micro-batches
    with gradients accumulated so the step equals one token-mean over the batch. Teachers
    run teacher-major, so each teacher is loaded at most once per step."""
    t0 = _sync(run.device)
    on_policy = not run.is_sft and run.lmbda_rng.random() < run.tcfg["lmbda"]
    teachers = [] if run.is_sft else teachers_for_batch(
        run.spec, category, run.teacher_keys, run.policy_rng, run.learned_map)
    max_new = run.tcfg["max_new_tokens"][category]
    completions = _completions(run.student, run.tokenizer, batch, batch_prompts, on_policy, max_new, run.tcfg)
    t1 = _sync(run.device)

    order = sorted(range(len(batch)), key=lambda i: len(batch_prompts[i]) + len(completions[i]))
    size = run.tcfg.get("micro_batch_size", len(batch))
    parts = [order[i : i + size] for i in range(0, len(order), size)]
    micros = [build_batch([batch_prompts[i] for i in p], [completions[i] for i in p],
                          run.tokenizer.pad_token_id, run.device) for p in parts]
    hiddens = [[] for _ in micros]
    for key in teachers:
        for m, (ids, attention, predict, _) in enumerate(micros):
            hiddens[m].append(run.pool.hidden(key, ids, attention, predict))
    t2 = _sync(run.device)

    total_tokens = sum(len(c) for c in completions)
    loss_total = 0.0
    for micro, micro_hiddens in zip(micros, hiddens):
        weight = len(micro[3]) / total_tokens
        loss = _student_loss(run, micro, micro_hiddens, teachers)
        (loss * weight).backward()
        loss_total += float(loss.detach()) * weight
    t3 = _sync(run.device)
    lengths = [len(c) for c in completions]
    return {
        "category": category, "teachers": teachers, "on_policy": on_policy, "loss": loss_total,
        "completion_tokens": total_tokens / len(lengths),
        "hit_max_new_tokens": sum(n >= max_new for n in lengths) / len(lengths),
        "seq_tokens": sum(len(p) for p in batch_prompts) + total_tokens, "gen_s": round(t1 - t0, 2),
        "teacher_s": round(t2 - t1, 2), "student_s": round(t3 - t2, 2),
    }


def train(cfg: dict, condition: str, seed: int, learned_map: dict[str, str] | None) -> Path:
    spec = cfg["conditions"][condition]
    tcfg = condition_train_cfg(cfg, condition)
    out, stem = run_dir(cfg, condition, seed), log_stem(cfg, condition, seed)
    out.mkdir(parents=True, exist_ok=True)
    stem.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    torch.manual_seed(seed)

    examples = load_split("train", cfg["data"]["train"])
    schedule = make_schedule([ex["category"] for ex in examples], tcfg["batch_size"], tcfg["epochs"], seed)
    tokenizer = load_tokenizer(cfg["student"], cfg["revisions"].get(cfg["student"]))
    prompts = [render(tokenizer, ex["prompt"]) for ex in examples]
    run = Run(spec, tcfg, _load_student(cfg, tcfg, device), tokenizer,
              TeacherPool(cfg["teachers"], cfg["revisions"], device, tcfg.get("teacher_cache", 1)),
              device, list(cfg["teachers"]), learned_map, random.Random(seed + 1), random.Random(seed + 2))
    params = [p for p in run.student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg["learning_rate"], weight_decay=0.0)
    warmup = max(1, int(tcfg["warmup_ratio"] * len(schedule)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(len(schedule), warmup))
    started = time.time()

    run.student.train()
    with open(stem.with_suffix(".jsonl"), "w") as log:
        for step, (category, idx) in enumerate(schedule):
            t0 = time.time()
            record = train_step(run, category, [examples[i] for i in idx], [prompts[i] for i in idx])
            grad_norm = torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, **record, "grad_norm": float(grad_norm),
                      "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - t0, 2),
                      "device_gb": _device_gb(device)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 10 == 0 or step == len(schedule) - 1:
                print(f"[{condition} s{seed}] step {step + 1}/{len(schedule)} {category} "
                      f"{'+'.join(record['teachers']) or 'gold'} loss={record['loss']:.4f} "
                      f"tok={record['completion_tokens']:.0f} {record['seconds']}s", flush=True)

    run.student.save_pretrained(out / "adapter")
    meta = {
        "condition": condition, "seed": seed, "spec": spec, "train": tcfg,
        "learned_map": learned_map if spec.get("policy") == "learned" else None,
        "steps": len(schedule), "examples": len(examples), "teacher_loads": run.pool.loads,
        "minutes": round((time.time() - started) / 60, 1), "config_fingerprint": fingerprint(tcfg),
        "provenance": provenance(device), "finished": True,
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def _device_gb(device: str) -> float:
    if device == "mps":
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    if device == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return 0.0
