"""Validate Hygon ConvTranspose1d and SnakeBeta against PyTorch."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import infinicore
import infinicore.nn.functional as icf
from infinicore.tensor import from_torch


def sync() -> None:
    infinicore.sync_device()
    torch.cuda.synchronize()


def bench(fn, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters * 1000.0


def stats(actual, expected):
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    diff = (actual_f - expected_f).abs()
    return {
        "cosine": F.cosine_similarity(actual_f, expected_f, dim=0).item(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


def run_conv1d(
    input_shape,
    weight_shape,
    *,
    stride,
    padding,
    dilation,
    groups,
    with_bias,
):
    generator = torch.Generator(device="cuda").manual_seed(20261030)
    x = torch.randn(input_shape, device="cuda", dtype=torch.float32, generator=generator)
    weight = torch.randn(
        weight_shape, device="cuda", dtype=torch.float32, generator=generator
    ) * 0.02
    bias = (
        torch.randn(
            weight_shape[0],
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        if with_bias
        else None
    )
    expected = F.conv1d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    out = torch.empty_like(expected)
    ic_x = from_torch(x)
    ic_weight = from_torch(weight)
    ic_bias = from_torch(bias) if bias is not None else None
    ic_out = from_torch(out)

    def ic_fn():
        icf.conv1d(
            ic_x,
            ic_weight,
            ic_bias,
            stride,
            padding,
            dilation,
            groups,
            out=ic_out,
        )

    def torch_fn():
        return F.conv1d(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    ic_fn()
    sync()
    torch_runs = [bench(torch_fn) for _ in range(3)]
    ic_runs = [bench(ic_fn) for _ in range(3)]
    return {
        "case": "conv1d",
        "input_shape": list(input_shape),
        "weight_shape": list(weight_shape),
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "torch_ms": statistics.median(torch_runs),
        "infinicore_ms": statistics.median(ic_runs),
        "speed_ratio": statistics.median(torch_runs) / statistics.median(ic_runs),
        "correctness": stats(out, expected),
        "torch_runs_ms": torch_runs,
        "infinicore_runs_ms": ic_runs,
    }


def run_conv_transpose1d(
    input_shape,
    weight_shape,
    *,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
    with_bias,
):
    generator = torch.Generator(device="cuda").manual_seed(20261031)
    x = torch.randn(input_shape, device="cuda", dtype=torch.float32, generator=generator)
    weight = torch.randn(
        weight_shape, device="cuda", dtype=torch.float32, generator=generator
    ) * 0.02
    bias = (
        torch.randn(
            weight_shape[1] * groups,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        if with_bias
        else None
    )
    expected = F.conv_transpose1d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
    out = torch.empty_like(expected)
    ic_x = from_torch(x)
    ic_weight = from_torch(weight)
    ic_bias = from_torch(bias) if bias is not None else None
    ic_out = from_torch(out)

    def ic_fn():
        icf.conv_transpose1d(
            ic_x,
            ic_weight,
            ic_bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
            out=ic_out,
        )

    def torch_fn():
        return F.conv_transpose1d(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

    ic_fn()
    sync()
    torch_runs = [bench(torch_fn) for _ in range(3)]
    ic_runs = [bench(ic_fn) for _ in range(3)]
    return {
        "case": "conv_transpose1d",
        "input_shape": list(input_shape),
        "weight_shape": list(weight_shape),
        "stride": stride,
        "padding": padding,
        "groups": groups,
        "dilation": dilation,
        "torch_ms": statistics.median(torch_runs),
        "infinicore_ms": statistics.median(ic_runs),
        "speed_ratio": statistics.median(torch_runs) / statistics.median(ic_runs),
        "correctness": stats(out, expected),
        "torch_runs_ms": torch_runs,
        "infinicore_runs_ms": ic_runs,
    }


def run_snake_beta(shape):
    generator = torch.Generator(device="cuda").manual_seed(20261032)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
    channels = shape[1]
    alpha_log = torch.randn(channels, device="cuda", dtype=torch.float32, generator=generator)
    beta_log = torch.randn(channels, device="cuda", dtype=torch.float32, generator=generator)
    out = torch.empty_like(x)
    ic_x = from_torch(x)
    ic_alpha = from_torch(alpha_log)
    ic_beta = from_torch(beta_log)
    ic_out = from_torch(out)

    def torch_fn():
        alpha = alpha_log.unsqueeze(0).unsqueeze(-1).exp()
        beta = beta_log.unsqueeze(0).unsqueeze(-1).exp()
        return x + torch.sin(alpha * x).pow(2) / (beta + 1e-9)

    def ic_fn():
        icf.snake_beta(ic_x, ic_alpha, ic_beta, out=ic_out)

    expected = torch_fn()
    ic_fn()
    sync()
    torch_runs = [bench(torch_fn) for _ in range(3)]
    ic_runs = [bench(ic_fn) for _ in range(3)]
    return {
        "case": "snake_beta",
        "shape": list(shape),
        "torch_ms": statistics.median(torch_runs),
        "infinicore_ms": statistics.median(ic_runs),
        "speed_ratio": statistics.median(torch_runs) / statistics.median(ic_runs),
        "correctness": stats(out, expected),
        "torch_runs_ms": torch_runs,
        "infinicore_runs_ms": ic_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/audio_ops_validation.json")
    args = parser.parse_args()
    infinicore.set_device(infinicore.device("cuda", 0))

    results = [
        run_conv1d(
            (1, 512, 1035),
            (512, 512, 7),
            stride=1,
            padding=9,
            dilation=3,
            groups=1,
            with_bias=True,
        ),
        run_conv1d(
            (1, 64, 16384),
            (64, 1, 7),
            stride=1,
            padding=3,
            dilation=1,
            groups=64,
            with_bias=False,
        ),
        run_conv_transpose1d(
            (1, 1024, 207),
            (1024, 512, 9),
            stride=5,
            padding=2,
            output_padding=0,
            groups=1,
            dilation=1,
            with_bias=True,
        ),
        run_conv_transpose1d(
            (1, 512, 1035),
            (512, 1, 12),
            stride=2,
            padding=0,
            output_padding=0,
            groups=512,
            dilation=1,
            with_bias=False,
        ),
        run_snake_beta((1, 128, 16384)),
    ]

    for result in results:
        print(
            f"{result['case']} torch={result['torch_ms']:.4f}ms "
            f"ic={result['infinicore_ms']:.4f}ms "
            f"ratio={result['speed_ratio']:.3f} "
            f"cos={result['correctness']['cosine']:.8f} "
            f"max_abs={result['correctness']['max_abs']:.3e}",
            flush=True,
        )

    payload = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "results": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"[write] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
