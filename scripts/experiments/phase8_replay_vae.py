"""Later node replay of a frozen video/audio VAE input; preserves reference precision."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from phase8_trace import TensorTrace, private_output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("component", choices=("video", "audio"))
    ap.add_argument("--workspace-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True, help="Diffusers model directory containing vae/audio_vae")
    ap.add_argument("--input-snapshot", type=Path, required=True, help="DENORMALIZED decode argument captured by phase8_pipeline")
    ap.add_argument("--input-sha256", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers", type=int, choices=(0, 2, 4, 8), default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trace-audio-layers", action="store_true")
    ap.add_argument("--compare-output", type=Path)
    args = ap.parse_args()
    if args.repeats < 1 or args.component == "audio" and args.workers or args.component == "video" and args.trace_audio_layers:
        ap.error("Invalid repeats or component-specific flags")
    if hashlib.sha256(args.input_snapshot.read_bytes()).hexdigest() != args.input_sha256:
        ap.error("Frozen decoder input SHA mismatch")
    output = private_output(args.workspace_root, args.output); output.mkdir(exist_ok=False)
    import torch
    devices = [int(d) for d in args.gpus.split(",")]
    if not devices or len(set(devices)) != len(devices) or args.workers > len(devices):
        ap.error("Select distinct GPUs and enough devices for replicas")
    torch.cuda.set_device(devices[0])
    source = Path(__file__).resolve().parents[2]
    import sys
    sys.path.insert(0, str(source/"scripts"))
    from h3_preflight import bf16_math
    bf16_math()
    if args.component == "video":
        from diffusers import AutoencoderKLMiniMaxH3 as Model
        subfolder = "vae"
    else:
        from diffusers import AutoencoderKLMiniMaxH3Audio as Model
        subfolder = "audio_vae"
    start = time.perf_counter()
    vae = Model.from_pretrained(args.model_path/subfolder, torch_dtype=torch.bfloat16, local_files_only=True)
    vae.to(device=f"cuda:{devices[0]}").requires_grad_(False).eval()
    # from_pretrained honors _keep_in_fp32_modules. Do not call a second dtype cast.
    dtype_map = {n: str(p.dtype) for n, p in vae.named_parameters()}
    if args.component == "audio" and any(p.dtype != torch.float32 for p in vae.parameters()):
        raise RuntimeError("Audio VAE must remain entirely float32")
    if args.component == "video" and any(p.dtype != torch.float32 for p in vae.encoder.parameters()):
        raise RuntimeError("Video encoder must remain float32")
    if args.component == "video":
        vae.enable_tiling()
    else:
        from infinilm.multimodal.audio_kernels import materialize_audio_weight_norm, install_ninetoothed_audio_kernels
        materialize_audio_weight_norm(vae); install_ninetoothed_audio_kernels(vae)
    latent = torch.load(args.input_snapshot, map_location="cpu", weights_only=True)
    if not torch.is_tensor(latent):
        raise ValueError("Decode snapshot must be a single tensor")
    latent = latent.to(device=f"cuda:{devices[0]}")
    controller = None
    if args.component == "video" and args.workers:
        from infinilm.multimodal.minimax_h3_vae_parallel import ParallelTileDecoder
        controller = ParallelTileDecoder(vae, devices[:args.workers])
    def sync():
        for d in devices: torch.cuda.synchronize(d)
    sync()
    load_seconds = time.perf_counter()-start
    trace = TensorTrace(output/"trace", budget_mib=256) if args.trace_audio_layers else None
    handles = []
    if trace:
        trace.record("audio.decode.input", latent, dump=True)
        for name, module in vae.named_modules():
            if name in ("dec_in_proj", "pre_block", "decoder.conv_pre", "decoder.conv_post", "decoder.activation_post") or (
                    name.startswith(("decoder.ups.", "decoder.resblocks.")) and name.count(".") == 2):
                def hook(owner, inputs, value, name=name):
                    trace.record("audio.layer."+name, value, dump=True)
                handles.append(module.register_forward_hook(hook))
    record = dict(complete=False, component=args.component, workers=args.workers, devices=devices,
                  input_sha256=args.input_sha256, dtypes=dtype_map, load_seconds=load_seconds,
                  diagnostic_only=bool(trace), seconds=[], numerical_acceptance=False)
    try:
        first = None
        with torch.inference_mode():
            for i in range(args.repeats+1):
                sync(); t = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.component=="video"):
                    value = vae.decode(latent, return_dict=False)[0]
                sync(); elapsed = time.perf_counter()-t
                host = value.detach().cpu()
                if first is None:
                    first = host.clone(); record["cold_seconds"] = elapsed
                    if trace: trace.record("audio.decode.output", host, dump=True)
                elif not torch.equal(first, host):
                    raise AssertionError("Same-input decoder repeat was not bitwise equal")
                if i: record["seconds"].append(elapsed)
                print("PHASE8_VAE_REPLAY", args.component, args.workers, i, elapsed, flush=True)
        if not torch.isfinite(first).all():
            raise AssertionError("Non-finite decoder output")
        torch.save(first, output/"decoded.pt")
        if args.compare_output:
            expected = torch.load(args.compare_output, map_location="cpu", weights_only=True)
            if expected.shape != first.shape: raise AssertionError("Decoder output shape mismatch")
            a, b = first.float(), expected.float()
            record["versus_baseline"] = dict(bitwise_equal=bool(torch.equal(first, expected)),
                relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-12)), max_abs=float((a-b).abs().max()))
        record.update(complete=True, steady_median_seconds=statistics.median(record["seconds"]),
                      output_shape=list(first.shape), repeat_bitwise=True,
                      decoded_sha256=hashlib.sha256((output/"decoded.pt").read_bytes()).hexdigest())
    except BaseException as exc:
        record["error"] = repr(exc); raise
    finally:
        for h in handles: h.remove()
        if trace: trace.close(record["complete"])
        if controller: controller.close()
        (output/"replay.json").write_bytes((json.dumps(record, indent=2)+"\n").encode())


if __name__ == "__main__":
    main()
