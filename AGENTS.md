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

- 2026-08-22: repo scaffolded (data/probe/train/eval + configs A/C/D/E + smoke). Smoke test status: see below/git log.
- Next: run probe + baseline eval, then pilot conditions A/C/D locally or on rented GPU; E only if D shows a real effect.
- Lane: this repo is exclusively owned by session -3e per cross-session coordination 2026-08-22.
