# Pre-registered analysis

Committed before any teacher probe, training run or student evaluation of the main
experiment existed. Settings live in `configs/main.yaml` at this commit; none of them
are tuned on the eval split.

## Question

When a 0.5B student is distilled on-policy from same-family 1.5B teachers (general
Instruct, Coder, Math), does routing each task category to a specialist teacher beat
using the general teacher for everything?

## Primary metric

Macro normalized gain over the untrained student:

G = mean over the four categories c of (s_c - b_c) / (1 - b_c)

where s_c is a condition's seed-averaged headline score on the eval split and b_c is the
untrained student's score. Headline scores: GSM8K exact match (math), MBPP pass@1 with
greedy decoding (code), ARC-Challenge accuracy by letter log-likelihood (qa), SamSum
ROUGE-L F1 (summ).

## Primary comparisons

1. `task` (D: math to Math, code to Coder, qa and summ to Instruct) minus `instruct` (A).
2. `learned` (E: per-category probe winner when its CI separates from Instruct) minus
   `instruct` (A). If the learned map routes every category exactly like another
   condition, E is reported as collapsed into it and not trained.

An effect counts as detected only if the 97.5% paired-bootstrap CI of the difference in G
excludes zero (Bonferroni over the two comparisons).

## Inference

Paired bootstrap over eval items, 10,000 draws, items resampled within each category and
shared between arms, on seed-averaged per-item scores. Seed spread is reported separately.

## Secondary analyses (descriptive, 95% CIs, no correction)

Per-category differences; the teacher-by-category matrix from the single-teacher runs
(`instruct`, `coder`, `math`); `sft` (same examples, gold targets), `random` routing and the
logit-averaged `ensemble`, each against `instruct`.

## Sample sizes

- Train: 256 examples per category (1,024 total), 1 epoch, batch 16, 64 optimizer steps.
- Probe: GSM8K train rows 7173-7472 (excluded from training), MBPP validation + prompt
  (100), ARC-Challenge validation (299), SamSum validation (300 of 818).
- Eval: GSM8K test 500 of 1,319 (fixed random subset), MBPP sanitized test (all 257),
  ARC-Challenge test (all 1,172), SamSum test 500 of 819.
- Seeds: 3 for `sft`, `instruct`, `task`, `learned`; at least 2 for the rest.
