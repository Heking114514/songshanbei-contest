"""Check the production NineToothed SnakeBeta on every selected device."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # Load the actual production kernel without requiring the native extension;
    # this lets the compiler smoke test run while C++ is still being built.
    source = Path(__file__).resolve().parents[1] / 'InfiniCore/python/infinicore/nn/functional/snake_beta_ninetoothed.py'
    spec = importlib.util.spec_from_file_location('production_snake_beta', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    report = {'torch': torch.__version__, 'hip': torch.version.hip,
              'rtol': 1e-4, 'atol': 2e-5, 'results': results, 'complete': False}
    with torch.inference_mode():
        for device_id in map(int, args.gpus.split(',')):
            torch.cuda.set_device(device_id)
            device = torch.device('cuda', device_id)
            rng = torch.Generator(device=device).manual_seed(42)
            for batch, channels, length in ((1, 32, 207), (2, 64, 2048), (1, 16, 331200)):
                x = torch.randn(batch, channels, length, device=device, generator=rng)
                alpha = torch.randn(channels, device=device, generator=rng).exp()
                beta = torch.randn(channels, device=device, generator=rng).exp()
                expected = x + torch.sin(alpha[None, :, None] * x).square() / (beta[None, :, None] + 1e-9)
                started = time.perf_counter()
                actual = module.snake_beta(x, alpha, beta)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                torch.testing.assert_close(actual, expected, rtol=report['rtol'], atol=report['atol'])
                diff = (actual - expected).abs()
                entry = {'device': device_id, 'name': torch.cuda.get_device_name(device_id),
                         'shape': list(x.shape), 'dtype': str(x.dtype),
                         'first_call_seconds': elapsed, 'max_abs': diff.max().item(),
                         'mean_abs': diff.mean().item(), 'passed': True}
                results.append(entry)
                output.write_text(json.dumps(report, indent=2))
                print(json.dumps(entry), flush=True)
                del x, actual, expected, diff, alpha, beta
            torch.cuda.empty_cache()
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2))
    print('SNAKE_EIGHT_CARD_OK', len(results), flush=True)


if __name__ == '__main__':
    main()
