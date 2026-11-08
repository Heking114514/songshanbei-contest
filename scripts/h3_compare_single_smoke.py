"""TEST ONLY: reduced-model FL2VA generation and original compare functions.

This never changes the official script, shared checkpoint, or formal parameters.
The one-card smoke is not a full-model benchmark or an official submission score.
"""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time

SHARED = Path(os.environ["MODEL_PATH"])
STABLE = Path(os.environ["H3_SOURCE_ROOT"])
WORK = Path(os.environ["H3_SINGLE_WORK"])
FIXTURE = WORK / "fixture"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def make_fixture(component, layers):
    from safetensors import safe_open
    from safetensors.torch import save_file
    import gc
    src = SHARED / "diffusers" / component
    dst = FIXTURE / component
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    if component == "text_encoder":
        cfg["text_config"]["num_hidden_layers"] = layers
        pattern = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.")
    else:
        cfg["num_layers"] = layers
        pattern = re.compile(r"^(?:transformer_blocks|blocks)\.(\d+)\.")
    (dst / "config.json").write_text(json.dumps(cfg))
    indexes = list(src.glob("*safetensors.index.json"))
    if len(indexes) != 1:
        raise RuntimeError("Expected one checkpoint index: " + str(src))
    original = json.loads(indexes[0].read_text())["weight_map"]
    selected = {}
    for key, shard in original.items():
        match = pattern.search(key)
        if not match or int(match.group(1)) < layers:
            selected[key] = shard
    if len(selected) == len(original):
        raise RuntimeError("Layer selection did not reduce checkpoint")
    mapping = {}
    total = 0
    for index, shard in enumerate(dict.fromkeys(selected.values())):
        keys = [k for k, v in selected.items() if v == shard]
        with safe_open(src / shard, framework="pt", device="cpu") as handle:
            values = {k: handle.get_tensor(k) for k in keys}
        size = sum(t.numel() * t.element_size() for t in values.values())
        total += size
        if total > 8 * 1024 ** 3:
            raise RuntimeError("Reduced checkpoint exceeded node storage budget")
        name = f"smoke-{index:03d}.safetensors"
        save_file(values, str(dst / name))
        mapping.update({k: name for k in values})
        del values
        gc.collect()
        print("SMOKE_FIXTURE_SHARD", component, index, len(keys), size, flush=True)
    payload = dict(metadata=dict(total_size=total), weight_map=mapping)
    (dst / "model.safetensors.index.json").write_text(json.dumps(payload))
    if component == "transformer":
        (dst / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(payload))
    print("SMOKE_FIXTURE_READY", component, len(mapping), total, flush=True)


def generate():
    from h3_preflight import bf16_math, gpu_check
    gpu_check(1)
    bf16_math()
    if "--legacy" in sys.argv:
        from h3_reference_sdpa import install_reference_head_batch
        install_reference_head_batch(max_heads=4, min_score_bytes=0)
        print("SMOKE_REFERENCE_HEAD_BATCH_ENABLED max_heads=4", flush=True)
    m = module("h3_single_smoke_generator", STABLE / "InfiniLM/examples/minimax_h3.py")
    # Test-only exception: two small models share GPU0, never used in formal runs.
    m.resolve_legacy_device_groups = lambda *_args: ([0], [0])
    resolve = m.resolve_model_paths
    def reduced_paths(path, task):
        paths = resolve(path, task)
        return m.ModelPaths(paths.modular_dir, FIXTURE)
    m.resolve_model_paths = reduced_paths
    base_encoder = m.MiniMaxH3ConditionEncoder
    class OneLayerEncoder(base_encoder):
        num_hidden_layers = 1
        def __init__(self, path, **kwargs):
            kwargs["max_cache_len"] = 4096
            super().__init__(path, **kwargs)
    m.MiniMaxH3ConditionEncoder = OneLayerEncoder
    load = m.load_pipeline
    def load_short(*args, **kwargs):
        pipe, encoder, transformer = load(*args, **kwargs)
        class ShortSmoke(type(pipe)):
            @property
            def min_duration(self):
                return 0.0
        pipe.__class__ = ShortSmoke
        return pipe, encoder, transformer
    m.load_pipeline = load_short
    import diffusers.modular_pipelines.minimax_h3.encoders as encoders
    original = encoders.get_qwen3vl_prompt_embeds
    signature = inspect.signature(original)
    def first_layer_hidden(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.arguments["text_encoder_layer"] = 1
        return original(*bound.args, **bound.kwargs)
    encoders.get_qwen3vl_prompt_embeds = first_layer_hidden
    # Native uses one pre-norm Qwen layer; reference uses hidden_states[1]
    # from a two-layer Qwen, so the selected hidden state is also pre-norm.
    sys.argv = ["minimax_h3.py", *sys.argv[2:]]
    print("TEST_ONLY_REDUCED_MODEL qwen_selected_layer=1 dit_layers=1 full_vision_and_vae=True", flush=True)
    m.main()


def compare():
    official = STABLE / "InfiniLM/examples/minimax_h3_compare.py"
    m = module("h3_original_compare_single_smoke", official)
    m.NUM_FRAMES = 22
    m.STEPS = 2
    m.resolve_legacy_launch = lambda args: ("0", args.legacy_memory_fraction)
    original_build = m.build_command
    def smoke_command(**kwargs):
        command = original_build(**kwargs)
        return [command[0], str(Path(__file__).resolve()), "generate", *command[2:],
                "--height=256", "--width=256"]
    m.build_command = smoke_command
    results = {}
    original_compare = m.compare_task_outputs
    def capture(label, reference, infinilm):
        value = original_compare(label, reference, infinilm)
        results[label] = value
        return value
    m.compare_task_outputs = capture
    tag = os.environ.get("H3_SMOKE_RUN", "smoke")
    if tag not in ("smoke", "cached-smoke"):
        raise ValueError("Unknown smoke run tag")
    out = WORK / tag
    if out.exists():
        raise RuntimeError("Smoke output already exists: " + str(out))
    sys.argv = [str(official), "--tasks", "fl2va", "--model-path", str(SHARED),
                "--first-frame", str(STABLE / "inputs/first_frame.png"),
                "--legacy-gpus", "0", "--infinilm-gpus", "0",
                "--legacy-text-gpus", "0", "--legacy-transformer-gpus", "0",
                "--output-dir", str(out)]
    rc = m.main()
    record = dict(rc=rc, scope="single-GPU reduced-model smoke; not official score",
                  full_model=False, num_frames=22, steps=2, reference_head_batch=4,
                  results=results, time=time.time())
    (WORK / (tag + "-result.json")).write_text(json.dumps(record, indent=2))
    if rc != 0:
        raise SystemExit(rc)
    video = results["FL2VA"]["video"]
    if not (video["metadata_size_match"] and video["decoded_length_match"]
            and video["left_frames"] == video["right_frames"] == 22):
        raise RuntimeError("Smoke comparison geometry/length failed")
    if video["ssim_match_percent"] is None:
        raise RuntimeError("Official SSIM did not execute")
    print("H3_SINGLE_COMPARE_SMOKE_COMPLETE", json.dumps(record), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prepare":
        make_fixture("text_encoder", 2)
        make_fixture("transformer", 1)
    elif mode == "generate":
        generate()
    elif mode == "compare":
        compare()
    else:
        raise SystemExit("Unknown smoke mode")
