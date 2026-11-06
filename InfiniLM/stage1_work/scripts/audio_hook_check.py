from __future__ import annotations

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
    MiniMaxH3AudioSnakeBeta,
)

from infinilm.multimodal.audio_kernels import install_ninetoothed_audio_kernels


def compare(actual, expected):
    diff = (actual - expected).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    ).item()
    return cosine, diff.max().item()


module = MiniMaxH3AudioSnakeBeta(64).cuda().eval()
with torch.no_grad():
    module.alpha.copy_(torch.randn_like(module.alpha))
    module.beta.copy_(torch.randn_like(module.beta))

state_before = {name: value.clone() for name, value in module.state_dict().items()}
original_forward = module.forward
installed = install_ninetoothed_audio_kernels(module)
with torch.inference_mode():
    for length in (207, 331200):
        x = torch.randn(2, 64, length, device="cuda")
        expected = original_forward(x)
        actual = module(x)
        cosine, max_abs = compare(actual, expected)
        print(
            f"length={length} installed={installed} "
            f"cosine={cosine:.8f} max_abs={max_abs:.3e}",
            flush=True,
        )
        if cosine < 0.999999 or max_abs > 2e-5:
            raise SystemExit(1)

if install_ninetoothed_audio_kernels(module) != 0:
    raise RuntimeError("audio kernel installation is not idempotent")

state_after = module.state_dict()
if state_before.keys() != state_after.keys():
    raise RuntimeError("audio kernel installation changed state_dict keys")
for name in state_before:
    if not torch.equal(state_before[name], state_after[name]):
        raise RuntimeError(f"audio kernel installation changed parameter {name}")

module.train()
x = torch.randn(1, 64, 31, device="cuda")
with torch.no_grad():
    fallback = module(x)
print("training fallback shape:", tuple(fallback.shape), flush=True)
