import torch
import torch.nn.functional as F


def info_nce(q, k, tau=0.07):
    if q.ndim != 2 or q.shape != k.shape or not len(q) or tau <= 0:
        raise ValueError("invalid shape/temperature")
    if not torch.isfinite(q).all() or not torch.isfinite(k).all():
        raise ValueError("finite embeddings required")
    # Define zero-vector policy explicitly: reject, rather than assign a cosine.
    if (q.norm(dim=-1) == 0).any() or (k.norm(dim=-1) == 0).any():
        raise ValueError("nonzero vectors required")
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.T / tau
    labels = torch.arange(len(q), device=q.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(7)
    q = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    loss = info_nce(q, k)
    loss.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    assert torch.allclose(info_nce(q, k), info_nce(k, q))
    assert torch.allclose(info_nce(q * 3, k * 7), loss)
    assert info_nce(q[:1], k[:1]).item() == 0
    assert torch.autograd.gradcheck(
        lambda a, b: info_nce(a, b, 0.2), (q, k), eps=1e-6, atol=1e-5
    )
    eye = torch.eye(3, dtype=torch.float64)
    assert info_nce(eye, eye).item() < info_nce(eye, eye.roll(1, 0)).item()
    print(
        "W015: symmetry, scaling, singleton, gradients and paired-versus-shuffled behavior passed"
    )
