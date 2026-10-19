# AGENTS.md

Notes for coding agents working in this repo. README.md explains the experiment;
PREREGISTRATION.md fixes the analysis; REPORT.md is generated, never hand-edited.

## Commands

- `uv sync` then `uv run pytest -q` (CPU only, about 5 s, no downloads).
- `uv run mtkd reproduce --config configs/smoke.yaml`: tiny end-to-end run, about 1 minute.
- `uv run mtkd reproduce`: the full experiment, resumable; finished stages are skipped.
- `uv run mtkd report`: rebuild REPORT.md and `assets/generated/results.png` from `results/`.

## Invariants

- Teacher selection uses validation-style probe splits only. Eval uses test splits. `tests/test_data.py` checks the splits are disjoint.
- Every model, teacher or student, sees prompts in the student's chat template, because that is the context a teacher scores during distillation.
- Greedy evaluation uses a clean `GenerationConfig`; the model cards' sampling defaults and repetition penalty must not leak in.
- Distillation needs a shared tokenizer between student and teachers (same model family).
- Batches are category-homogeneous and their order depends only on the seed, so conditions differ only in which teacher scores each batch. One optimizer and one LR schedule per run.
- Model code from MBPP runs only through `mtkd.sandbox.run_tests`.

## Gotchas

- MPS memory: `mps_memory_fraction` in the config caps the allocator. Full-vocabulary (151,936) logits are never materialised for a whole batch; losses work on hidden states in chunks.
- `TeacherPool` keeps one teacher body resident and swaps from disk (about 1.5 s per load), so the ensemble condition loads three teachers per step.
- Batched bf16 generation on MPS depends slightly on batch composition. Batching is deterministic (sorted by prompt length), so all runs see the same batches.
- Pass model objects, not path strings, to anything that loads weights on MPS (transformers' threaded loading has segfaulted there).
