"""Batched greedy generation for probing teachers and evaluating students."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name_or_path: str, device: str | None = None):
    device = device or pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, messages_batch: list[list[dict]], max_new_tokens: int = 512,
             batch_size: int = 8) -> list[str]:
    outputs: list[str] = []
    for i in range(0, len(messages_batch), batch_size):
        chunk = messages_batch[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in chunk
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(
            model.device
        )
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions = out[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(completions, skip_special_tokens=True))
    return outputs
