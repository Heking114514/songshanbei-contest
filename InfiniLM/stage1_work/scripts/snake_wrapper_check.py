from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(
        Path.home()
        / "Projects/InfiniCore/src/infiniop/ops/snake_beta/ninetoothed"
    ),
)
from snake_beta import snake_beta

for length in (207, 2048, 331200):
    x = torch.randn(2, 64, length, device="cuda")
    alpha_log = torch.randn(64, device="cuda")
    beta_log = torch.randn(64, device="cuda")
    actual = snake_beta(x, alpha_log, beta_log)
    alpha_exp = alpha_log.exp()
    beta_exp = beta_log.exp()
    expected = (
        x
        + torch.sin(alpha_exp[None, :, None] * x).square()
        / (beta_exp[None, :, None] + 1e-9)
    )
    diff = (actual - expected).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    ).item()
    print(
        f"length={length} cosine={cosine:.8f} max_abs={diff.max().item():.3e}",
        flush=True,
    )
    if diff.max().item() != 0 or cosine < 0.999999:
        raise SystemExit(1)
