# multi-teacher-kd

Which teacher should teach a small LLM, per task category? On-policy knowledge
distillation (TRL GKD) from three same-family teachers into one student, with the
teacher-selection policy as the experimental variable.

Teachers: Qwen2.5-1.5B Instruct, Coder and Math (shared tokenizer, which GKD needs
because it matches token-level distributions). Student: Qwen2.5-0.5B-Instruct + LoRA.
Model revisions are pinned in `configs/main.yaml`.

## Result

Routing each category to its "specialist" teacher does not beat using the general
instruct teacher for everything. Both pre-registered comparisons come out negative:

| Comparison | dG (97.5% CI) |
|---|---|
| task-mapped teachers vs instruct | -6.1 [-8.8, -3.5] |
| learned (probe-chosen) teachers vs instruct | -2.9 [-5.3, -0.6] |

G is the macro normalized gain over the untrained student, in points of headroom.
The loss comes almost entirely from math: the Math teacher scores best on GSM8K
itself but teaches a tool-call answer style the student copies. Full tables,
per-category differences and diagnostics are in [REPORT.md](REPORT.md).

**Seeds:** the protocol pre-registered 3 training seeds (`train.seeds: [0, 1, 2]`).
Results here use 2 seeds (0 and 1); seed 2 was not run for time. CIs are item
bootstraps pooled over the seeds that ran, and the seed sd column shows run-to-run
spread.

## Conditions

| Condition | Teacher choice |
|---|---|
| `sft` | plain SFT on reference answers, no teacher |
| `instruct` | always the general instruct teacher |
| `coder`, `math` | always that single teacher |
| `task` | hand-mapped category to teacher |
| `learned` | probe winner per category, only when its CI against instruct excludes zero |
| `random` | uniformly random teacher per batch |
| `ensemble` | all three teachers, logit-averaged |

## Data

GSM8K (math), MBPP-sanitized (code), ARC-Challenge (qa), SamSum (summ). Train prompts
come from train splits; probe and eval come from disjoint test slices. Code scoring
runs model output against MBPP asserts in a sandboxed subprocess.

## Run

```sh
uv sync
uv run mtkd reproduce                       # probe, base eval, training matrix, report
uv run mtkd reproduce --stages matrix --seeds 2
uv run mtkd report                          # rebuild REPORT.md and the figure from results/
```

`configs/smoke.yaml` runs the whole pipeline on tiny models in minutes. The full
matrix runs on a 24 GB M-series Mac but takes days; a single CUDA GPU is much faster.
