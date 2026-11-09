"""Check actual GPUs and compare dependencies before loading H3 weights."""
import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import time



def gpu_check(expected=None):
    mode = os.environ.get("H3_GPU_CHECK_MODE", "compute")
    if mode not in ("compute", "inventory"):
        raise ValueError("H3_GPU_CHECK_MODE must be compute or inventory")
    expected = int(os.environ.get("H3_EXPECTED_GPUS", "8")) if expected is None else expected
    if expected < 1:
        raise ValueError("Expected GPU count must be positive")
    hip = ctypes.CDLL("/opt/dtk/lib/libamdhip64.so")
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    count = ctypes.c_int()
    rc = hip.hipGetDeviceCount(ctypes.byref(count))
    message = hip.hipGetErrorString(rc)
    print("DIRECT_HIP", rc, message.decode() if message else "", count.value, flush=True)
    if rc != 0 or count.value != expected:
        raise RuntimeError(f"HIP must see exactly {expected} GPU(s) before any model is loaded")
    import torch
    if not torch.version.hip or torch.cuda.device_count() != expected:
        raise RuntimeError(f"Platform HIP Torch must see exactly {expected} GPU(s)")
    devices = []
    for index in range(expected):
        with torch.cuda.device(index):
            if mode == "compute":
                a = torch.ones((64, 64), device=f"cuda:{index}", dtype=torch.bfloat16)
                b = a @ a
                torch.cuda.synchronize(index)
                if not bool(torch.isfinite(b).all()) or float(b[0, 0]) != 64.0:
                    raise RuntimeError(f"GPU {index} calculation failed")
                del a, b
                torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(index)
            item = dict(index=index, name=torch.cuda.get_device_name(index),
                        free_bytes=free, total_bytes=total)
            devices.append(item)
            print("GPU_COMPUTE_OK" if mode == "compute" else "GPU_VISIBLE", json.dumps(item), flush=True)
    return dict(torch=torch.__version__, hip=torch.version.hip, devices=devices, gpu_check_mode=mode)


def bf16_math():
    import torch
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    if (torch.backends.cuda.flash_sdp_enabled()
            or torch.backends.cuda.mem_efficient_sdp_enabled()
            or not torch.backends.cuda.math_sdp_enabled()
            or not torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()):
        raise RuntimeError("BF16 math SDPA settings did not take effect")


def run_preflight(source_root, model_path, first_frame, output, expected_gpus):
    source_root, model_path, first_frame, output = map(
        Path, (source_root, model_path, first_frame, output))
    result = dict(passed=False, time=time.time(), source_root=str(source_root),
                  visible_devices={name: os.environ.get(name) for name in (
                      "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                      "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")})
    try:
        source = source_root.resolve()
        official = source / "InfiniLM/examples/minimax_h3_compare.py"
        if not official.is_file() or official.stat().st_size == 0:
            raise RuntimeError("Missing or empty official comparator: " + str(official))
        if not first_frame.is_file() or first_frame.stat().st_size == 0:
            raise RuntimeError("Missing or empty input image: " + str(first_frame))
        for child in ("FL2VA", "diffusers"):
            if not (model_path / child).is_dir():
                raise RuntimeError("Missing shared model directory: " + child)
        result.update(gpu_check(expected_gpus))
        bf16_math()
        from PIL import Image
        with Image.open(first_frame) as im:
            im.load()
            result["input_size"] = list(im.size)
        import numpy as np
        from skimage.metrics import structural_similarity
        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        if structural_similarity(probe, probe, channel_axis=2, data_range=255) != 1.0:
            raise RuntimeError("SSIM dependency failed")
        import av
        for name in ("h264", "aac"):
            av.Codec(name, "r")
            av.Codec(name, "w")
        versions = {}
        for name in ("torch", "transformers", "diffusers", "accelerate", "safetensors",
                     "av", "skimage", "triton", "ninetoothed", "infinicore", "infinilm"):
            module = importlib.import_module(name)
            versions[name] = dict(version=getattr(module, "__version__", None),
                                  path=getattr(module, "__file__", None))
            if name in ("infinicore", "infinilm"):
                expected_root = source / ("InfiniCore" if name == "infinicore" else "InfiniLM")
                if not Path(module.__file__).resolve().is_relative_to(expected_root.resolve()):
                    raise RuntimeError(name + " was imported from a different checkout")
        result.update(passed=True, integrity="presence_size_and_decode",
                      attention="math_bf16", packages=versions)
        print("H3_COMPARE_PREFLIGHT_OK", flush=True)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--first-frame", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-gpus", type=int, default=int(os.environ.get("H3_EXPECTED_GPUS", "8")))
    args = parser.parse_args()
    run_preflight(args.source_root, args.model_path, args.first_frame,
                  args.output, args.expected_gpus)


if __name__ == "__main__":
    main()
