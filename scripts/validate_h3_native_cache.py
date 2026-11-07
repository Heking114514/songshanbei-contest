"""Compare cached native H3 forwards with explicitly reset forwards on TP=8."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import time

import torch

from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerInputs, MiniMaxH3TransformerRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--position-dtype', choices=['float32', 'float64'], default='float32')
    args = parser.parse_args()
    devices = list(map(int, args.gpus.split(',')))
    device = torch.device('cuda', devices[0])
    torch.cuda.set_device(device)
    runner = MiniMaxH3TransformerRunner(args.model_path, device='hygon', tp_device_ids=devices)
    config = runner.engine.hf_config
    video_width = config['latents_dim'] * math.prod(config['patch_size'])
    rng = torch.Generator(device=device).manual_seed(42)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {'gpus': devices, 'torch': torch.__version__, 'hip': torch.version.hip,
              'rtol': 1e-4, 'atol': 1e-5, 'position_dtype': args.position_dtype,
              'cases': [], 'complete': False}

    def make_inputs(video_tokens, audio_tokens, text_tokens):
        length = video_tokens + audio_tokens + text_tokens
        return MiniMaxH3TransformerInputs(
            video_hidden_states=torch.randn(1, video_tokens, video_width, device=device, generator=rng),
            audio_hidden_states=torch.randn(1, audio_tokens, config['audio_latents_dim'], device=device, generator=rng),
            encoder_hidden_states=torch.randn(1, text_tokens, config['text_dim'], device=device, dtype=torch.bfloat16, generator=rng),
            timestep=torch.tensor([500.0], device=device),
            timestep_indices=torch.zeros(length, device=device, dtype=torch.int64),
            token_tags=torch.tensor([0] * text_tokens + [1] * video_tokens + [2] * audio_tokens, device=device, dtype=torch.int64),
            position_ids=torch.arange(length, device=device, dtype=getattr(torch, args.position_dtype))[:, None].expand(-1, 3).contiguous(),
            text_indices=torch.arange(text_tokens, device=device),
            video_indices=torch.arange(text_tokens, text_tokens + video_tokens, device=device),
            audio_indices=torch.arange(text_tokens + video_tokens, length, device=device),
        )

    def compare_fresh(name, inputs):
        started = time.perf_counter()
        cached = runner(inputs)
        revision = runner._condition_cache.revision
        generation = runner._condition_cache.generation_id
        # Clones ensure the reference cannot depend on original tensor addresses.
        copied = MiniMaxH3TransformerInputs(**{f.name: getattr(inputs, f.name).clone() for f in fields(inputs)})
        runner.begin_generation()
        fresh = runner(copied)
        deltas = {}
        for modality in ('video', 'audio'):
            actual, expected = getattr(cached, modality), getattr(fresh, modality)
            if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                raise AssertionError(name + ': non-finite ' + modality)
            torch.testing.assert_close(actual, expected, rtol=report['rtol'], atol=report['atol'])
            deltas[modality] = (actual - expected).abs().max().item()
        assert runner._condition_cache.revision > revision
        entry = {'name': name, 'generation': generation, 'revision': revision,
                 'sequence_length': len(inputs.position_ids), 'max_abs': deltas,
                 'seconds': time.perf_counter() - started, 'passed': True}
        report['cases'].append(entry)
        destination.write_text(json.dumps(report, indent=2))
        print(json.dumps(entry), flush=True)

    with torch.inference_mode():
        inputs = make_inputs(4, 4, 8)
        runner.begin_generation()
        runner(inputs)
        revision = runner._condition_cache.revision
        runner(inputs)
        assert runner._condition_cache.revision == revision
        compare_fresh('unchanged_conditioning', inputs)
        inputs.encoder_hidden_states.add_(0.25)
        compare_fresh('in_place_text', inputs)
        inputs.position_ids[:, 0].add_(0.5)
        compare_fresh('in_place_positions', inputs)
        compare_fresh('different_sequence_length', make_inputs(8, 4, 12))
    report['complete'] = True
    destination.write_text(json.dumps(report, indent=2))
    print('H3_NATIVE_CACHE_OK', len(report['cases']), flush=True)


if __name__ == '__main__':
    main()
