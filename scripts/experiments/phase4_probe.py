"""Single-device phase4 validation, prepared locally; never submits resources.

Run only after loading an isolated, incrementally built experiment environment.
Native timing covers one gather plus all four pointwise modes, excluding norms,
attention, FF, projection and communication. It is not a model speedup.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from datetime import datetime
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("native", "video", "invalid-index"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--real-size", action="store_true", help="Include native N=27570,H=5376 case")
    parser.add_argument("--invalid-index", type=int, default=-1)
    args = parser.parse_args()

    import torch
    import infinicore as ic
    from infinicore.lib import _infinicore as native

    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires one visible NVIDIA/HYGON device")
    torch.set_grad_enabled(False)
    torch.manual_seed(4204)
    root = args.library_root.resolve()
    report = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "group": args.group, "torch": torch.__version__,
        "library_root": str(root), "complete": False, "cases": [],
        "flags": {k: v for k, v in os.environ.items() if k.startswith("H3_EXPERIMENT_")},
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    save()

    def audit():
        paths = sorted({
            line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
            if any(name in line for name in ("libinfiniop.so", "libinfinicore_cpp_api.so", "_infinicore"))
        })
        if not paths or not all(Path(p).resolve().is_relative_to(root) for p in paths):
            raise RuntimeError(f"Unexpected runtime libraries: {paths}, expected {root}")
        report["library_paths"] = paths
        save()

    def sync():
        ic.sync_device()
        torch.cuda.synchronize()

    owners = []

    def bridge(t):
        result = ic.from_torch(t)
        if not t.is_contiguous():
            result = ic.strided_from_blob(
                t.data_ptr(), list(t.shape), list(t.stride()),
                dtype=result.dtype, device=result.device)
        result._torch_ref = t
        owners.append((t, result))
        return result._underlying

    device_tensor = ic.from_torch(torch.empty(1, device="cuda"))
    ic.set_device(device_tensor.device)
    audit()

    def record(row):
        report["cases"].append(row)
        save()
        print("PHASE4_RESULT", json.dumps(row), flush=True)

    def timing(fn):
        for _ in range(2):
            fn()
            sync()
        samples = []
        for _ in range(7):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            samples.append(time.perf_counter() - start)
        return {"seconds": samples, "median_seconds": statistics.median(samples)}

    def equal(actual, expected):
        # Include signed zero in the bit comparison.
        return torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    def native_case(n, width, rows):
        print("PHASE4_START", n, width, rows, flush=True)
        x = torch.randn(1, n, width, device="cuda", dtype=torch.bfloat16)
        branch = torch.randn_like(x)
        table = torch.randn(rows, 6 * width, device="cuda", dtype=torch.bfloat16)
        # Nonmonotonic/repeated indices, always exercise first and last table rows.
        indices = (torch.arange(n, device="cuda", dtype=torch.int64) * 5 + 1) % rows
        indices[0] = 0
        if n > 1:
            indices[-1] = rows - 1
        expanded = torch.empty(n, 6 * width, device="cuda", dtype=torch.bfloat16)
        temp = torch.empty_like(x)
        actual = [torch.empty_like(x) for _ in range(4)]
        expected = [torch.empty_like(x) for _ in range(4)]
        xi, bi, ti, ii, ei, tmp = map(bridge, (x, branch, table, indices, expanded, temp))
        ai, ri = [list(map(bridge, values)) for values in (actual, expected)]
        # Keep torch-owned views/owners alive for the entire recorded graph.
        views = [expanded[None, :, k * width:(k + 1) * width] for k in range(6)]
        vi = list(map(bridge, views))

        def old():
            native.embedding_(ei, ii, ti)
            for mode in range(4):
                offset = 0 if mode < 2 else 3
                if mode % 2 == 0:
                    native.addcmul_(tmp, xi, xi, vi[offset + 1], 1.0)
                    native.add_(ri[mode], tmp, vi[offset])
                else:
                    native.addcmul_(ri[mode], xi, vi[offset + 2], bi, 1.0)

        def new():
            for mode in range(4):
                native.h3_modulation_(ai[mode], xi, bi, ti, ii, mode)

        old()
        new()
        sync()
        assert all(equal(a, b) for a, b in zip(actual, expected)), "Native bit parity"
        row = {"n": n, "width": width, "table_rows": rows, "bitwise_equal": True}
        row["gather_plus_four_modes"] = timing(old)
        row["indexed_four_modes"] = timing(new)
        row["expanded_table_bytes_avoided"] = expanded.numel() * expanded.element_size()

        # Same shape/pointer, different index and table data; no stale-value cache.
        indices.copy_((indices + 1) % rows)
        table.add_(0.125)
        old()
        new()
        sync()
        assert all(equal(a, b) for a, b in zip(actual, expected)), "Updated data parity"
        row["updated_indices_and_table_equal"] = True

        if n < 100:
            ic.start_graph_recording()
            try:
                new()
            finally:
                graph = ic.stop_graph_recording()
            indices.copy_((indices + 1) % rows)
            table.mul_(0.75)
            old()
            sync()
            graph.run()
            sync()
            assert all(equal(a, b) for a, b in zip(actual, expected)), "Graph replay parity"
            row["graph_replay_updated_data_equal"] = True

        def rejected(name, fn):
            try:
                fn()
            except (RuntimeError, ValueError):
                row.setdefault("rejected", []).append(name)
            else:
                raise AssertionError(f"Invalid {name} was accepted")

        rejected("mode", lambda: native.h3_modulation_(ai[0], xi, bi, ti, ii, 4))
        rejected("output_alias", lambda: native.h3_modulation_(xi, xi, bi, ti, ii, 0))
        bad_indices = bridge(indices.to(torch.int32))
        rejected("index_dtype", lambda: native.h3_modulation_(ai[0], xi, bi, ti, bad_indices, 0))
        bad_table = bridge(table[:, :-1].contiguous())
        rejected("table_width", lambda: native.h3_modulation_(ai[0], xi, bi, bad_table, ii, 0))
        if width > 1:
            padded = torch.empty(1, n, width * 2, device="cuda", dtype=torch.bfloat16)
            strided_x = bridge(padded[..., ::2])
            rejected("strides", lambda: native.h3_modulation_(ai[0], strided_x, bi, ti, ii, 0))
        record(row)

    if args.group == "native":
        for shape in ((3, 1, 7), (1, 257, 9), (17, 1031, 3)):
            native_case(*shape)
        if args.real_size:
            native_case(27570, 5376, 6)
    elif args.group == "invalid-index":
        # Run this group in its own process: the deliberate trap poisons context.
        x = torch.zeros(1, 1, 17, dtype=torch.bfloat16, device="cuda")
        y = torch.empty_like(x)
        table = torch.zeros(3, 102, dtype=torch.bfloat16, device="cuda")
        indices = torch.tensor([args.invalid_index], dtype=torch.int64, device="cuda")
        assert args.invalid_index < 0 or args.invalid_index >= 3
        print("PHASE4_INVALID_INDEX_LAUNCH", args.invalid_index, flush=True)
        native.h3_modulation_(
            bridge(y), bridge(x), bridge(x), bridge(table), bridge(indices), 0)
        sync()
        raise AssertionError("Invalid device index did not fail")
    else:
        import numpy as np
        from diffusers.video_processor import VideoProcessor

        source = Path(__file__).resolve().parents[2] / "InfiniLM/python/infinilm/multimodal/video_postprocess.py"
        spec = importlib.util.spec_from_file_location("phase4_video_postprocess", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.environ["H3_EXPERIMENT_VIDEO_POSTPROCESS"] = "1"
        os.environ["H3_EXPERIMENT_VIDEO_CHUNK"] = "4"
        os.environ["H3_EXPERIMENT_VIDEO_WORKERS"] = "2"
        report["flags"] = {
            k: v for k, v in os.environ.items() if k.startswith("H3_EXPERIMENT_")}
        report["video_module"] = str(source)
        save()

        def pixel_equal(a, b):
            assert len(a) == len(b)
            for aa, bb in zip(a, b):
                assert len(aa) == len(bb)
                for aaa, bbb in zip(aa, bb):
                    assert aaa.mode == bbb.mode
                    np.testing.assert_array_equal(np.asarray(aaa), np.asarray(bbb))

        for normalize in (True, False):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for channels in (1, 3, 4):
                    processor = VideoProcessor(do_normalize=normalize)
                    original = processor.postprocess_video
                    video = torch.rand(2, channels, 11, 4, 5, device="cuda", dtype=dtype)
                    video[..., 0, 0] = 0
                    video[..., -1, -1] = 1
                    if normalize:
                        video = video * 2 - 1
                    video = video.transpose(-1, -2)  # Real stride and chunk tail.
                    expected = original(video, output_type="pil")
                    assert module.install_video_postprocess(processor)
                    assert module.install_video_postprocess(processor)
                    pixel_equal(processor.postprocess_video(video, output_type="pil"), expected)
                    np.testing.assert_array_equal(
                        processor.postprocess_video(video, output_type="np"),
                        original(video, output_type="np"))
                    pixel_equal(
                        processor.postprocess_video(video.cpu(), output_type="pil"),
                        original(video.cpu(), output_type="pil"))
                    # A configuration change must use the original method.
                    processor.register_to_config(do_normalize=not normalize)
                    pixel_equal(
                        processor.postprocess_video(video, output_type="pil"),
                        original(video, output_type="pil"))
                    assert module.remove_video_postprocess(processor)
                    assert processor.postprocess_video is original
                    record({"normalize": normalize, "dtype": str(dtype), "channels": channels,
                            "batch": 2, "frames": 11, "chunk": 4, "pixel_equal": True,
                            "noncontiguous": True, "numpy_cpu_config_fallback": True})
    audit()
    report["complete"] = True
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    save()
    print("PHASE4_COMPLETE", args.group, flush=True)


if __name__ == "__main__":
    main()
