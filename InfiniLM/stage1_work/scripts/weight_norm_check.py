from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import weight_norm

from infinilm.multimodal.audio_kernels import materialize_audio_weight_norm

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results/weight_norm_check.json")
args = parser.parse_args()


def bench(module, x):
    with torch.no_grad():
        for _ in range(10):
            module(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(30):
            module(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / 30 * 1000.0


class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(256, 256, 7, padding=3, dilation=3))
        self.up = weight_norm(
            nn.ConvTranspose1d(256, 128, 9, stride=5, padding=2)
        )


module = Holder().cuda().eval()
x = torch.randn(1, 256, 2048, device="cuda")
with torch.inference_mode():
    conv_ref = module.conv(x)
    up_ref = module.up(conv_ref)

before_runs = [bench(module.conv, x) for _ in range(3)]
materialized = materialize_audio_weight_norm(module)
with torch.inference_mode():
    conv_actual = module.conv(x)
    up_actual = module.up(conv_actual)
after_runs = [bench(module.conv, x) for _ in range(3)]

case_results = []
for name, actual, expected in (
    ("conv1d", conv_actual, conv_ref),
    ("conv_transpose1d", up_actual, up_ref),
):
    diff = (actual - expected).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    ).item()
    max_abs = diff.max().item()
    print(
        f"{name} cosine={cosine:.8f} max_abs={max_abs:.3e}",
        flush=True,
    )
    if cosine < 0.999999 or max_abs > 1e-5:
        raise SystemExit(1)
    case_results.append({"operator": name, "cosine": cosine, "max_abs": max_abs})

before = statistics.median(before_runs)
after = statistics.median(after_runs)
print(
    f"materialized={materialized} before={before:.4f}ms "
    f"after={after:.4f}ms speedup={before / after:.3f}x",
    flush=True,
)
remaining_weight_g = sum(
    hasattr(item, "weight_g") for item in module.modules()
)
print("remaining weight_g:", remaining_weight_g, flush=True)

payload = {
    "hostname": platform.node(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "modules_materialized": materialized,
    "remaining_weight_g": remaining_weight_g,
    "cases": case_results,
    "conv1d_before_ms": before,
    "conv1d_after_ms": after,
    "before_runs_ms": before_runs,
    "after_runs_ms": after_runs,
    "speedup": before / after,
}
output = Path(args.out)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2))
print(f"[write] {output}", flush=True)
