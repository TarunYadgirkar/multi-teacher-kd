import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mtkd.config import condition_train_cfg, deep_merge
from mtkd.distill import chunked_distill_loss, ensemble_logits
from mtkd.gen import decoder_and_head, generate, left_pad
from mtkd.train import build_batch, lr_lambda, make_schedule


def tiny_model(seed: int, hidden: int = 32) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=64, hidden_size=hidden, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         tie_word_embeddings=True)
    return Qwen2ForCausalLM(config).eval()


def test_schedule_is_category_homogeneous_and_complete():
    cats = ["math"] * 10 + ["qa"] * 7 + ["code"] * 5
    schedule = make_schedule(cats, batch_size=4, epochs=2, seed=3)
    assert all(len({cats[i] for i in idx}) == 1 and cats[idx[0]] == c for c, idx in schedule)
    seen = sorted(i for _, idx in schedule for i in idx)
    assert seen == sorted(list(range(22)) * 2)
    assert schedule == make_schedule(cats, batch_size=4, epochs=2, seed=3)
    assert schedule != make_schedule(cats, batch_size=4, epochs=2, seed=4)


def test_build_batch_masks_align_predictions_with_targets():
    ids, attention, predict, targets = build_batch([[5, 6, 7], [8]], [[9, 10], [11, 12, 13]], 0, "cpu")
    assert ids.tolist() == [[5, 6, 7, 9, 10], [8, 11, 12, 13, 0]]
    assert attention.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert targets.tolist() == [9, 10, 11, 12, 13]
    assert ids[:, 1:][predict[:, :-1]].tolist() == targets.tolist()
    assert predict.sum() == len(targets)


def test_lr_schedule_warms_up_then_decays_to_zero():
    f = lr_lambda(total=10, warmup=2)
    values = [f(s) for s in range(10)]
    assert values[0] == 0.5 and values[1] == 1.0
    assert values[2] > values[5] > values[9] > 0


def test_two_pass_generation_matches_single_pass():
    model = tiny_model(0)
    prompts = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
    single = generate(model, prompts, 12, pad_id=0, stop=[63], batch_size=3)
    staged = generate(model, prompts, 12, pad_id=0, stop=[63], batch_size=2, first_pass=5)
    assert [c.ids for c in single] == [c.ids for c in staged]


def test_left_pad():
    ids, mask = left_pad([[1, 2], [3]], 0, "cpu")
    assert ids.tolist() == [[1, 2], [0, 3]] and mask.tolist() == [[1, 1], [0, 1]]


def test_distillation_step_moves_student_toward_teacher():
    student, teacher = tiny_model(0), tiny_model(1, hidden=48)
    student.train()
    decoder, head = decoder_and_head(student)
    t_decoder, t_head = decoder_and_head(teacher)
    ids, attention, predict, _ = build_batch([[1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10]], 0, "cpu")
    optimizer = torch.optim.SGD(student.parameters(), lr=0.5)
    with torch.no_grad():
        t_hidden = t_decoder(input_ids=ids, attention_mask=attention).last_hidden_state[predict]
    losses = []
    for _ in range(5):
        hidden = decoder(input_ids=ids, attention_mask=attention).last_hidden_state[predict]
        loss = chunked_distill_loss(hidden, head, ensemble_logits([t_hidden], [t_head]), 0.5, 1.0, chunk=3)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_condition_overrides_merge_into_train():
    cfg = {"train": {"lmbda": 1.0, "lora": {"r": 16, "alpha": 32}},
           "conditions": {"a": {"policy": "fixed", "train": {"lmbda": 0.5, "lora": {"r": 8}}}}}
    merged = condition_train_cfg(cfg, "a")
    assert merged == {"lmbda": 0.5, "lora": {"r": 8, "alpha": 32}}
    assert deep_merge({"x": 1}, {}) == {"x": 1}
