"""Opt-in device-side PIL conversion for a single H3 pipeline instance.

Inspired by the scheduling idea in MiniMaxH3_On_K100-ai. Independently
implemented against the standard Diffusers VideoProcessor contract.
The do_normalize=False integration is a new, unvalidated candidate.
Earlier GPU measurements covered only the normalized input path.
"""

from __future__ import annotations

import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch


def _positive_option(name: str, default: int, maximum: int) -> int:
    value = os.environ.get(name, str(default))
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return result


def install_video_postprocess(processor: Any) -> bool:
    """Install only on the standard processor when explicitly requested.

    CPU, non-PIL, extra options, gradients and unsupported channel/layout
    contracts retain the original method. Chunking caps temporary device
    buffers; no frame/result cache survives a call.
    """
    enabled = os.environ.get("H3_EXPERIMENT_VIDEO_POSTPROCESS", "0")
    if enabled not in ("0", "1"):
        raise ValueError("H3_EXPERIMENT_VIDEO_POSTPROCESS must be 0 or 1")
    if enabled == "0" or processor is None:
        return False

    from diffusers.video_processor import VideoProcessor
    from PIL import Image

    if type(processor) is not VideoProcessor:
        return False
    if not callable(getattr(processor, "_denormalize_conditionally", None)):
        return False
    # H3's decoder already produces clamped FP32 [0, 1] pixels and uses
    # do_normalize=False. Preserve that contract; never normalize twice.
    normalize = getattr(processor.config, "do_normalize", None)
    if normalize is not True and normalize is not False:
        return False

    frames_per_chunk = _positive_option("H3_EXPERIMENT_VIDEO_CHUNK", 8, 32)
    workers = _positive_option("H3_EXPERIMENT_VIDEO_WORKERS", 4, 8)
    state = getattr(processor, "_h3_experiment_video_postprocess", None)
    if state is not None:
        if state[2:] != (frames_per_chunk, workers, normalize):
            raise RuntimeError("Reinstalling video postprocess with different options")
        if processor.postprocess_video is not state[1]:
            raise RuntimeError("Video postprocess was replaced by another owner")
        return True
    original = processor.postprocess_video

    @functools.wraps(original)
    def convert(video: torch.Tensor, output_type: str = "np", **kwargs: Any):
        eligible = (
            output_type == "pil"
            and getattr(processor.config, "do_normalize", None) is normalize
            and not kwargs
            and isinstance(video, torch.Tensor)
            and video.is_cuda
            and video.ndim == 5
            and video.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and not video.requires_grad
            and not torch.is_grad_enabled()
            and video.shape[0] > 0
            and video.shape[1] in (1, 3, 4)
            and video.shape[2] > 0
            and video.shape[3] > 1
            and video.shape[4] > 1
        )
        if not eligible:
            return original(video, output_type=output_type, **kwargs)

        def image_from_pixels(pixels):
            if pixels.shape[-1] == 1:
                return Image.fromarray(pixels[..., 0], mode="L")
            return Image.fromarray(pixels)

        def convert_batches(pool):
            batches = []
            for sample in video:  # original [C, T, H, W] batch order
                frames = sample.permute(1, 0, 2, 3)
                images = []
                for start in range(0, frames.shape[0], frames_per_chunk):
                    chunk = processor._denormalize_conditionally(
                        frames[start : start + frames_per_chunk]
                    )
                    # The conditional helper normalizes only when configured.
                    # Match FP32 -> multiply -> ties-to-even round -> uint8;
                    # no extra clamp when normalization is disabled.
                    pixels = (
                        chunk.permute(0, 2, 3, 1)
                        .float()
                        .mul(255.0)
                        .round()
                        .to(torch.uint8)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                    # D2H is deliberately blocking: PIL never sees a host
                    # buffer while a device copy is still writing to it.
                    converted = (
                        map(image_from_pixels, pixels)
                        if pool is None
                        else pool.map(image_from_pixels, pixels)
                    )
                    images.extend(converted)
                batches.append(images)
            return batches

        if workers == 1:
            return convert_batches(None)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return convert_batches(pool)

    processor.postprocess_video = convert
    processor._h3_experiment_video_postprocess = (
        original, convert, frames_per_chunk, workers, normalize
    )
    return True


def remove_video_postprocess(processor: Any) -> bool:
    """Restore the original instance method without removing another hook."""
    state = getattr(processor, "_h3_experiment_video_postprocess", None)
    if state is None:
        return False
    if processor.postprocess_video is not state[1]:
        raise RuntimeError("Video postprocess was replaced by another owner")
    processor.postprocess_video = state[0]
    del processor._h3_experiment_video_postprocess
    return True
