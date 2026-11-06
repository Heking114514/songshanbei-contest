"""Generate a standalone C/CUDA artifact for dynamic SnakeBeta."""

from __future__ import annotations

import argparse
import math
import os
import pickle

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_M = block_size(1, 1)
TILE_WIDTH = 512


def arrangement(x, alpha_exp, beta_exp, output):
    x_arranged = x.tile((BLOCK_M, TILE_WIDTH))
    return (
        x_arranged,
        alpha_exp.tile((BLOCK_M,))
        .unsqueeze(1)
        .expand((-1, x_arranged.shape[1])),
        beta_exp.tile((BLOCK_M,))
        .unsqueeze(1)
        .expand((-1, x_arranged.shape[1])),
        output.tile((BLOCK_M, TILE_WIDTH)),
    )


def application(x, alpha_exp, beta_exp, output):
    sine = math.sin(alpha_exp * x)
    output = x + sine * sine / (beta_exp + 1e-9)  # noqa: F841


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.expanduser("~/work/nt_aot_snake"))
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--platform", default="generic")
    parser.add_argument("--arch", default="gfx936")
    args = parser.parse_args()

    kernel = ninetoothed.make(
        arrangement,
        application,
        (Tensor(2, dtype=ninetoothed.float32),
         Tensor(1, dtype=ninetoothed.float32),
         Tensor(1, dtype=ninetoothed.float32),
         Tensor(2, dtype=ninetoothed.float32)),
        caller="cuda",
        backend=args.backend,
        platform=args.platform,
        compute_arch=args.arch,
        kernel_name="snake_beta_h3_fastctx",
        output_dir=args.out,
        num_warps=4,
        num_stages=1,
    )
    print("[aot] kernel object:", type(kernel).__name__)
    built_path = os.path.join(args.out, "built.pkl")
    with open(built_path, "wb") as handle:
        pickle.dump(kernel._built_artifact, handle)
    print(f"[aot] built artifact: {built_path}")
    for root, _, files in os.walk(args.out):
        for name in files:
            path = os.path.join(root, name)
            print(f"[file] {path} {os.path.getsize(path)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
