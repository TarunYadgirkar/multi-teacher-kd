# multi-teacher-kd

Which teacher should teach a small LLM, per task category? On-policy knowledge
distillation (TRL GKD) from multiple same-family teachers into one student,
comparing teacher-selection policies as the experimental variable.

## Conditions

| Condition | Policy | Teacher choice |
|---|---|---|
| A | `fixed` | always the general instruct teacher |
| C | `strongest` | single best teacher by overall probe score |
| D | `task` | hand-mapped category → teacher |
| E | `learned` | per-category argmax over probe win-rates |

(B/random exists in `policies.py` but is not a pilot condition.)

Teachers: Qwen2.5-1.5B Instruct / Coder / Math (shared tokenizer — required, since
GKD matches token-level distributions). Student: Qwen2.5-0.5B-Instruct + LoRA.

## Data

Real benchmarks, one category each: GSM8K (math), MBPP-sanitized (code),
ARC-Challenge (qa), SamSum (summ). Train prompts from train splits; probe and
eval from test splits with different seeds/slices.

Scoring: math numeric exact-match, qa letter match, summ ROUGE-L F, code
executes generated code against MBPP asserts (`run_code_eval: false` to skip —
it runs untrusted model output in a subprocess).

## Run

```sh
uv sync
uv run python -m mtkd.probe --config configs/pilot_a_fixed.yaml   # teacher win-rates
uv run python -m mtkd.run_eval --config configs/pilot_a_fixed.yaml \
  --model Qwen/Qwen2.5-0.5B-Instruct --out results/baseline.json  # untrained baseline
uv run python -m mtkd.train --config configs/pilot_d_task.yaml
uv run python -m mtkd.run_eval --config configs/pilot_d_task.yaml \
  --model outputs/pilot_d_task/final --out results/pilot_d_task.json
```

Smoke test (SmolLM2-135M/360M, ~minutes on MPS): swap in `configs/smoke.yaml`.

## Implementation notes

- TRL ≥ 1.0: GKD lives in `trl.experimental.gkd`.
- Multi-teacher training partitions data by assigned teacher and trains
  interleaved rounds (one pass per teacher shard per round). Optimizer state
  resets per shard — a known simplification at pilot scale; a per-batch
  teacher-swapping trainer is the upgrade path if it matters.
- `check_tokenizer_compat` hard-fails on cross-tokenizer teachers.
- Local (M-series, 24GB): pilot configs fit with bf16 (3×1.5B teachers ≈ 9GB +
  0.5B student), but on-policy generation makes training slow — local machine is
  for smoke + probe; rent one A100/4090 for the full matrix.
