"""Distillation losses and the teacher pool.

The losses never materialise full-vocabulary logits for a whole batch. They take final
hidden states at the completion positions and project them to the vocabulary a chunk of
tokens at a time, under activation checkpointing, so peak memory is set by the chunk
size rather than by batch x length x 151,936.
"""

import math
from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mtkd.gen import decoder_and_head, load_model, release_memory


def generalized_jsd(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    beta: float, temperature: float = 1.0) -> torch.Tensor:
    """Per-token generalized JSD from GKD (Agarwal et al., 2024), same convention as TRL:
    beta=0 is forward KL(teacher || student), beta=1 is reverse KL(student || teacher),
    otherwise beta*KL(T||M) + (1-beta)*KL(S||M) with M = beta*T + (1-beta)*S."""
    s = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    t = torch.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if beta == 0:
        return (t.exp() * (t - s)).sum(-1)
    if beta == 1:
        return (s.exp() * (s - t)).sum(-1)
    m = torch.logaddexp(s + math.log(1 - beta), t + math.log(beta))
    return beta * (t.exp() * (t - m)).sum(-1) + (1 - beta) * (s.exp() * (s - m)).sum(-1)


TeacherLogits = Callable[[slice], torch.Tensor]


def ensemble_logits(hiddens: list[torch.Tensor], heads: list[torch.Tensor]) -> TeacherLogits:
    """Teacher logits for a slice of tokens; several teachers are logit-averaged."""
    def logits(part: slice) -> torch.Tensor:
        total = None
        for hidden, head in zip(hiddens, heads):
            chunk = (hidden[part] @ head.T).float()
            total = chunk if total is None else total + chunk
        return total / len(hiddens)
    return logits


def chunked_distill_loss(student_hidden: torch.Tensor, student_head: torch.Tensor,
                         teacher_logits: TeacherLogits, beta: float, temperature: float,
                         chunk: int) -> torch.Tensor:
    """Mean per-token generalized JSD over N completion tokens; student_hidden is [N, H]."""
    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, student_hidden.shape[0], chunk):
        part = slice(start, start + chunk)

        def piece(hidden: torch.Tensor, part: slice = part) -> torch.Tensor:
            with torch.no_grad():
                target = teacher_logits(part)
            return generalized_jsd(hidden @ student_head.T, target, beta, temperature).sum()

        total = total + checkpoint(piece, student_hidden[part], use_reentrant=False)
    return total / student_hidden.shape[0]


def chunked_ce_loss(student_hidden: torch.Tensor, student_head: torch.Tensor,
                    targets: torch.Tensor, chunk: int) -> torch.Tensor:
    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, student_hidden.shape[0], chunk):
        part = slice(start, start + chunk)

        def piece(hidden: torch.Tensor, part: slice = part) -> torch.Tensor:
            return F.cross_entropy((hidden @ student_head.T).float(), targets[part], reduction="sum")

        total = total + checkpoint(piece, student_hidden[part], use_reentrant=False)
    return total / student_hidden.shape[0]


class TeacherPool:
    """Keeps at most `capacity` teacher bodies on the device and every teacher's output
    head, so a logit-averaged ensemble works with one body resident at a time."""

    def __init__(self, names: dict[str, str], revisions: dict[str, str | None], device: str,
                 capacity: int = 1):
        self.names, self.revisions, self.device, self.capacity = names, revisions, device, capacity
        self.bodies: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self.heads: dict[str, torch.Tensor] = {}
        self.loads = 0

    def _body(self, key: str) -> torch.nn.Module:
        if key in self.bodies:
            self.bodies.move_to_end(key)
            return self.bodies[key]
        while len(self.bodies) >= self.capacity:
            self.bodies.popitem(last=False)
            release_memory()
        model = load_model(self.names[key], self.revisions.get(self.names[key]), self.device)
        body, head = decoder_and_head(model)
        self.heads.setdefault(key, head.detach())
        self.bodies[key] = body
        self.loads += 1
        return body

    @torch.no_grad()
    def hidden(self, key: str, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               positions: torch.Tensor) -> torch.Tensor:
        out = self._body(key)(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[positions]

    def head(self, key: str) -> torch.Tensor:
        return self.heads[key]
