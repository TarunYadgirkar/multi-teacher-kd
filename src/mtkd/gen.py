"""Model loading, batched generation and multiple-choice log-likelihood.

Every model, student or teacher, sees prompts rendered with the STUDENT's chat template,
because that is the context a teacher scores during distillation.
Greedy decoding uses a clean GenerationConfig, so the model card's sampling defaults and
repetition penalty never leak into evaluation.
"""

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for(device: str) -> torch.dtype:
    return torch.float32 if device == "cpu" else torch.bfloat16


def load_tokenizer(name: str, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eos_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


def load_model(name: str, revision: str | None = None, device: str | None = None):
    device = device or pick_device()
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype_for(device))
    model.to(device).eval()
    model.generation_config = GenerationConfig()
    return model


def release_memory() -> None:
    """Call after dropping the last reference to a model."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def render(tokenizer, messages: list[dict]) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@dataclass(frozen=True)
class Completion:
    ids: list[int]
    finished: bool


def left_pad(rows: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, width - len(row):] = torch.tensor(row, dtype=torch.long)
        mask[i, width - len(row):] = 1
    return ids.to(device), mask.to(device)


def _cut(row: list[int], stop: set[int]) -> Completion:
    for i, token in enumerate(row):
        if token in stop:
            return Completion(row[: i + 1], True)
    return Completion(row, False)


def _generation_config(max_new_tokens: int, pad_id: int, stop: list[int], sample: bool,
                       temperature: float) -> GenerationConfig:
    common = dict(max_new_tokens=max_new_tokens, pad_token_id=pad_id, eos_token_id=stop,
                  repetition_penalty=1.0)
    if sample:
        return GenerationConfig(do_sample=True, temperature=temperature, top_k=0, top_p=1.0, **common)
    return GenerationConfig(do_sample=False, **common)


@torch.no_grad()
def _generate_once(model, prompts: list[list[int]], max_new_tokens: int, pad_id: int,
                   stop: list[int], batch_size: int, sample: bool, temperature: float) -> list[Completion]:
    config = _generation_config(max_new_tokens, pad_id, stop, sample, temperature)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    results: list[Completion | None] = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        ids, mask = left_pad([prompts[i] for i in chunk], pad_id, model.device)
        out = model.generate(input_ids=ids, attention_mask=mask, generation_config=config)
        for i, row in zip(chunk, out[:, ids.shape[1]:].tolist()):
            results[i] = _cut(row, set(stop))
    return results


def generate(model, prompts: list[list[int]], max_new_tokens: int, pad_id: int, stop: list[int],
             batch_size: int = 16, first_pass: int | None = None, sample: bool = False,
             temperature: float = 1.0) -> list[Completion]:
    """Batched generation. With first_pass set, every prompt first gets first_pass tokens and
    only unfinished ones continue up to max_new_tokens, so one long output does not hold a
    whole batch to the full budget. Greedy continuation from the prefix gives the same
    tokens as a single pass (up to floating-point batch effects)."""
    first = max_new_tokens if first_pass is None else min(first_pass, max_new_tokens)
    results = _generate_once(model, prompts, first, pad_id, stop, batch_size, sample, temperature)
    pending = [i for i, r in enumerate(results) if not r.finished]
    remaining = max_new_tokens - first
    if not pending or remaining <= 0:
        return results
    extended = _generate_once(
        model, [prompts[i] + results[i].ids for i in pending], remaining, pad_id, stop,
        batch_size, sample, temperature,
    )
    for i, more in zip(pending, extended):
        results[i] = Completion(results[i].ids + more.ids, more.finished)
    return results


def label_token_ids(tokenizer, labels: list[str]) -> list[int]:
    ids = []
    for label in labels:
        encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"choice label {label!r} is not a single token")
        ids.append(encoded[0])
    return ids


@torch.no_grad()
def choice_predictions(model, tokenizer, prompts: list[list[int]], labels: list[list[str]],
                       batch_size: int = 16) -> list[dict]:
    """Pick the label whose token is most likely as the first assistant token. The output
    projection runs in float32: bf16 logits are spaced 0.125 apart near typical values,
    which produces ties between labels."""
    decoder, head = decoder_and_head(model)
    pad_id = tokenizer.pad_token_id
    out: list[dict | None] = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        ids, mask = left_pad([prompts[i] for i in chunk], pad_id, model.device)
        last = decoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1]
        normalizer = torch.logsumexp((last @ head.T).float(), dim=-1)
        for row, i in enumerate(chunk):
            token_ids = label_token_ids(tokenizer, labels[i])
            label_logits = last[row].float() @ head[token_ids].float().T
            scores = (label_logits - normalizer[row]).tolist()
            best = max(range(len(scores)), key=scores.__getitem__)
            out[i] = {"choice": labels[i][best], "label_logprobs": scores}
    return out


def decoder_and_head(model) -> tuple[torch.nn.Module, torch.Tensor]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head.weight
