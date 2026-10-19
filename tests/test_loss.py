import math

import pytest
import torch
import torch.nn.functional as F

from mtkd.distill import chunked_ce_loss, chunked_distill_loss, ensemble_logits, generalized_jsd


def trl_reference(student_logits, teacher_logits, beta, temperature=1.0):
    """The formula in TRL's GKDTrainer.generalized_jsd_loss, per token."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.log_softmax(teacher_logits / temperature, dim=-1)
    if beta == 0:
        jsd = F.kl_div(s, t, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(t, s, reduction="none", log_target=True)
    else:
        b = torch.tensor(beta)
        m = torch.logsumexp(torch.stack([s + torch.log(1 - b), t + torch.log(b)]), dim=0)
        jsd = b * F.kl_div(m, t, reduction="none", log_target=True) + (1 - b) * F.kl_div(
            m, s, reduction="none", log_target=True)
    return jsd.sum(-1)


@pytest.fixture
def logits():
    g = torch.Generator().manual_seed(0)
    return torch.randn(7, 50, generator=g) * 3, torch.randn(7, 50, generator=g) * 3


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_matches_trl_formula(logits, beta):
    s, t = logits
    assert torch.allclose(generalized_jsd(s, t, beta), trl_reference(s, t, beta), atol=1e-5)


def test_zero_when_distributions_match(logits):
    s, _ = logits
    assert torch.allclose(generalized_jsd(s, s.clone(), 0.5), torch.zeros(7), atol=1e-6)


def test_beta_half_is_symmetric_and_bounded(logits):
    s, t = logits
    forward, backward = generalized_jsd(s, t, 0.5), generalized_jsd(t, s, 0.5)
    assert torch.allclose(forward, backward, atol=1e-5)
    assert (forward <= math.log(2) + 1e-6).all()


def test_temperature_scales_logits(logits):
    s, t = logits
    assert torch.allclose(generalized_jsd(s, t, 0.5, 2.0), generalized_jsd(s / 2, t / 2, 0.5), atol=1e-6)


def _setup(n=37, hidden=8, vocab=30):
    g = torch.Generator().manual_seed(1)
    h_s = torch.randn(n, hidden, generator=g, requires_grad=True)
    w_s = torch.randn(vocab, hidden, generator=g)
    h_t = [torch.randn(n, hidden + 4, generator=g) for _ in range(2)]
    w_t = [torch.randn(vocab, hidden + 4, generator=g) for _ in range(2)]
    return h_s, w_s, h_t, w_t


def test_chunked_loss_equals_unchunked_and_has_gradients():
    h_s, w_s, h_t, w_t = _setup()
    full = generalized_jsd(h_s @ w_s.T, h_t[0] @ w_t[0].T, 0.5).mean()
    chunked = chunked_distill_loss(h_s, w_s, ensemble_logits(h_t[:1], w_t[:1]), 0.5, 1.0, chunk=5)
    assert torch.allclose(full, chunked, atol=1e-5)
    (grad_full,) = torch.autograd.grad(full, h_s)
    (grad_chunked,) = torch.autograd.grad(
        chunked_distill_loss(h_s, w_s, ensemble_logits(h_t[:1], w_t[:1]), 0.5, 1.0, chunk=5), h_s)
    assert torch.allclose(grad_full, grad_chunked, atol=1e-5)


def test_ensemble_averages_teacher_logits():
    h_s, w_s, h_t, w_t = _setup()
    averaged = (h_t[0] @ w_t[0].T + h_t[1] @ w_t[1].T) / 2
    expected = generalized_jsd(h_s @ w_s.T, averaged, 0.5).mean()
    got = chunked_distill_loss(h_s, w_s, ensemble_logits(h_t, w_t), 0.5, 1.0, chunk=8)
    assert torch.allclose(expected, got, atol=1e-5)


def test_chunked_ce_matches_cross_entropy():
    h_s, w_s, _, _ = _setup()
    targets = torch.randint(0, 30, (37,), generator=torch.Generator().manual_seed(2))
    expected = F.cross_entropy(h_s @ w_s.T, targets)
    assert torch.allclose(chunked_ce_loss(h_s, w_s, targets, chunk=6), expected, atol=1e-5)
