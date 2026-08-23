# AGENTS.md

Multi-teacher on-policy distillation experiment (URAP application project).
Read README.md for design; this file tracks live state.

## Stack decisions (shared with sibling sessions — bonsai-distill mirrors these)

- TRL pin `trl>=1.0.0`; import from `trl.experimental.gkd` (`GKDConfig`, `GKDTrainer`) — top-level import is pre-1.0.
- Dataset format: conversational `{"messages": [...]}`; GKDTrainer applies chat template itself; pass `processing_class=tokenizer`.
- GKD defaults `lmbda=0.5 beta=0.5 temperature=1.0`; on-policy condition = `lmbda=1.0 beta=1.0`.
- Token-level GKD requires shared tokenizer student↔teacher; asserted at startup via encode equality (`check_tokenizer_compat`).
- bf16 on MPS/CUDA, fp32 on CPU; left-padding for batched generation.
- Models: Qwen2.5-0.5B-Instruct student + Qwen2.5-1.5B Instruct/Coder/Math teachers; SmolLM2-135M/360M-Instruct for smoke.

## Ongoing

### State at session close (2026-08-22)

DONE (all free/local):
- Full scaffold: data (GSM8K/MBPP/ARC/SamSum → messages+reference), policies (fixed/random/strongest/task/learned), GKD train loop (LoRA, multi-teacher via sharded rounds), probe, per-category eval.
- Smoke test green end-to-end on MPS (SmolLM2 135M/360M): train both shards + probe + eval.
- Pilot probe of Qwen2.5-1.5B Instruct/Coder/Math on 100 test prompts → results/probe.json (committed). Instruct best overall, so condition C == A; learned map: code→Coder, math→Math, qa→Instruct, summ→Coder (~tie w/ Instruct).

DEFERRED (needs hours of GPU — rented A100/4090, or overnight MPS):
- Baseline eval of untrained student (was started locally, killed mid-run):
  `uv run python -m mtkd.run_eval --config configs/pilot_a_fixed.yaml --model Qwen/Qwen2.5-0.5B-Instruct --out results/baseline.json`
- Pilot conditions (train then eval each; E only if D beats A):
  `uv run python -m mtkd.train --config configs/pilot_a_fixed.yaml`
  `uv run python -m mtkd.run_eval --config configs/pilot_a_fixed.yaml --model outputs/pilot_a_fixed/final --out results/pilot_a_fixed.json`
  (same pattern for pilot_d_task.yaml, pilot_e_learned.yaml; skip pilot_c_strongest — identical to A per probe)
- Env for every run: `TRL_EXPERIMENTAL_SILENCE=1 TOKENIZERS_PARALLELISM=false`, plain `python` (no accelerate launch).

- 2026-08-22: repo scaffolded (data/probe/train/eval + configs A/C/D/E + smoke). Smoke test PASSED end-to-end on MPS (train both shards + probe + eval).
- TRL 1.10 gotcha: the GKD collator splits `messages` into prompt/completion — every training example MUST end with an assistant message or it throws "empty conversation". Datasets here carry benchmark reference answers as the assistant turn.
- Probe (offset 0) and eval (offset n_probe) draw disjoint slices of the test splits — keep it that way or the learned policy is selected on its own eval set.
- MBPP-sanitized train split has only ~120 rows; loaders clamp with a warning.
- MPS SIGSEGV gotcha (found by sibling session -62): passing a path STRING as model/teacher_model to trl trainers routes through trl's create_model_from_path → transformers 5.15 threaded weight loading → segfault on MPS. Always preload with from_pretrained and pass the model object (this repo already does).
- 2026-08-22: pilot probe done, results/probe.json committed. Instruct best overall (C collapses into A); learned map: code→Coder, math→Math, qa→Instruct, summ→Coder (summ is a near-tie with Instruct).
- Next: run probe + baseline eval, then pilot conditions A/C/D locally or on rented GPU; E only if D shows a real effect.
- Lane: this repo is exclusively owned by session -3e per cross-session coordination 2026-08-22.
