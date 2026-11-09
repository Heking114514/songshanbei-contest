"""Offline HYGON Lt screening in ONE GPU process; no submission, build or AFS I/O.

Creates C API descriptors directly so baseline and 16 explicit Lt choices can
coexist without changing cached Core/Graph descriptors. Each timing resets C
before the event interval, including nonzero beta. No tuning in model forward.
"""
from __future__ import annotations
import argparse
import ctypes as C
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo
from phase10_gemm_timing import bracketed


def fingerprint(paths, torch_version):
    rows = []
    for path in sorted(set(map(Path, paths)), key=str):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 << 20), b""):
                digest.update(block)
        rows.append({"name": path.name, "path": str(path), "sha256": digest.hexdigest()})
    identity = {"torch": torch_version, "libraries": sorted(
        [(row["name"], row["sha256"]) for row in rows])}
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return tag, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True, help="Isolated install with lib/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", type=Path, help="JSON list: name,tokens,out_features,in_features,pad,transpose_b")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--real-size", action="store_true", help="Add synthetic 27570x5376 projection; not a model trace")
    parser.add_argument("--workspace-mb", type=int, nargs="+", default=[32], choices=(0, 8, 32, 64, 128, 256))
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timing-mode", choices=("serial", "paired"), default="serial",
                        help="paired brackets every candidate with fresh BLAS event timings")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    indices = list(dict.fromkeys(map(int, args.indices.split(","))))
    args.workspace_mb = list(dict.fromkeys(args.workspace_mb))
    if not indices or any(i < 0 or i > 15 for i in indices) or args.repeats < 3 or args.warmup < 3:
        parser.error("indices must be 0..15; repeats and warmup >= 3")
    work, root, output = map(lambda p: p.resolve(), (args.workspace_root, args.library_root, args.output))
    afs = os.environ.get("IFAB_AFS_HOME")
    if afs and work.is_relative_to(Path(afs).resolve()):
        raise RuntimeError("Workspace must be outside AFS")
    if not root.is_relative_to(work) or not output.is_relative_to(work):
        raise RuntimeError("Library and results must resolve inside the private workspace")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Independent C descriptors, never inherit a production table.
    os.environ.pop("H3_GEMM_LT_TABLE", None)
    os.environ["H3_GEMM_LT_TRACE"] = "0"
    os.environ["H3_GEMM_LT_MIN_TOKENS"] = "4096"
    os.environ["H3_GEMM_LT_ALIGNMENT"] = "16"

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Requires one visible HYGON/HIP GPU")
    torch.cuda.set_device(args.device)
    torch.set_grad_enabled(False)
    torch.manual_seed(4260)
    report = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "complete": False, "scope": "synthetic GEMM screening, not DiT acceptance",
              "workspace_mb": args.workspace_mb, "indices": indices,
              "timing_mode": args.timing_mode, "warmup": args.warmup,
              "gpu": torch.cuda.get_device_name(), "device": args.device, "torch": torch.__version__,
              "torch_hip": getattr(torch.version, "hip", None),
              "cases": [], "afs_modified": False}

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def check(status, operation):
        if status:
            raise RuntimeError(f"{operation}: status={status}")

    save()
    rt = C.CDLL(str(root / "lib/libinfinirt.so"), mode=C.RTLD_GLOBAL)
    lib = C.CDLL(str(root / "lib/libinfiniop.so"), mode=C.RTLD_GLOBAL)
    ptr, size = C.c_void_p, C.c_size_t
    for name, types in {
        "infiniopCreateHandle": [C.POINTER(ptr)],
        "infiniopDestroyHandle": [ptr],
        "infiniopCreateTensorDescriptor": [C.POINTER(ptr), size, C.POINTER(size), C.POINTER(C.c_ssize_t), C.c_int],
        "infiniopDestroyTensorDescriptor": [ptr],
        "infiniopCreateGemmDescriptor": [ptr, C.POINTER(ptr), ptr, ptr, ptr],
        "infiniopGetGemmWorkspaceSize": [ptr, C.POINTER(size)],
        "infiniopDestroyGemmDescriptor": [ptr],
        "infiniopGemm": [ptr, ptr, size, ptr, ptr, ptr, C.c_float, C.c_float, ptr],
    }.items():
        getattr(lib, name).argtypes = types
        getattr(lib, name).restype = C.c_int
    rt.infinirtSetDevice.argtypes = [C.c_int, C.c_int]
    rt.infinirtSetDevice.restype = C.c_int
    check(rt.infinirtSetDevice(8, torch.cuda.current_device()), "set HYGON device")
    lib.h3GemmLtBuildId.restype = C.c_char_p
    lib.h3GemmLtLastSelection.restype = C.c_char_p
    if lib.h3GemmLtBuildId() != b"h3_hygon_lt_v1":
        raise RuntimeError("Wrong experimental infiniop binary")
    # Fingerprint actual mapped libraries, rather than a guessed device-only key.
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if "/" in line and any(name in line.lower() for name in
                        ("libhipblas", "libcublas", "librocblas", "libinfiniop.so"))})
    if not any("blaslt" in Path(p).name.lower() for p in mapped):
        raise RuntimeError("No mapped Lt library to fingerprint")
    tag, libs = fingerprint(mapped, torch.__version__)
    os.environ["H3_GEMM_LT_RUNTIME_TAG"] = tag
    report["runtime_tag"], report["libraries"] = tag, libs
    handle = ptr()
    check(lib.infiniopCreateHandle(C.byref(handle)), "create handle")
    shapes = json.loads(args.shapes.read_text()) if args.shapes else [
        {"name": "contiguous", "tokens": 4096, "out_features": 128, "in_features": 128},
        {"name": "transposed_weight_padded", "tokens": 4097, "out_features": 256,
         "in_features": 256, "pad": 16, "transpose_b": True}]
    if args.real_size:
        shapes.append({"name": "synthetic_long", "tokens": 27570, "out_features": 5376,
                       "in_features": 5376, "transpose_b": True})

    def tensor_desc(tensor):
        result = ptr()
        check(lib.infiniopCreateTensorDescriptor(
            C.byref(result), tensor.ndim, (size * tensor.ndim)(*tensor.shape),
            (C.c_ssize_t * tensor.ndim)(*tensor.stride()), 19), "tensor descriptor")
        return result

    def metrics(actual, expected):
        delta = actual.float() - expected.float()
        return {"bitwise_equal": bool(torch.equal(actual.view(torch.int16), expected.view(torch.int16))),
                "finite": bool(torch.isfinite(actual).all()),
                "relative_l2": float((delta.norm() / expected.float().norm().clamp_min(1e-12)).item()),
                "max_abs": float(delta.abs().max().item())}

    try:
        save()
        for workspace_mb in args.workspace_mb:
            os.environ["H3_GEMM_LT_WORKSPACE_MB"] = str(workspace_mb)
            for shape in shapes:
                n, m, k = (int(shape[name]) for name in ("tokens", "out_features", "in_features"))
                pad = int(shape.get("pad", 0))
                if n < 4096 or min(m, k) < 128 or pad < 0:
                    raise ValueError(f"Shape outside screening scope: {shape}")
                pad_a, pad_b, pad_c = [int(shape.get("pad_"+axis, pad)) for axis in "abc"]
                if min(pad_a, pad_b, pad_c) < 0:
                    raise ValueError("Negative per-operand padding")
                a = torch.randn(n, k + pad_a, device="cuda", dtype=torch.bfloat16)[:, :k]
                b = (torch.randn(m, k + pad_b, device="cuda", dtype=torch.bfloat16)[:, :k].t()
                     if shape.get("transpose_b") else
                     torch.randn(k, m + pad_b, device="cuda", dtype=torch.bfloat16)[:, :m])
                c = torch.empty(n, m + pad_c, device="cuda", dtype=torch.bfloat16)[:, :m]
                initial = torch.randn_like(c) * 0.125
                descriptors = []
                try:
                    for value in (c, a, b):
                        descriptors.append(tensor_desc(value))
                    for alpha, beta in ((1.0, 0.0), (0.75, 0.0), (0.75, 0.25), (1.0, 1.0)):
                        row = {"shape": shape, "alpha": alpha, "beta": beta, "workspace_mb": workspace_mb, "candidates": []}
                        report["cases"].append(row)
                        print("PHASE6_START", json.dumps({"shape": shape, "alpha": alpha, "beta": beta}), flush=True)
                        baseline = None
                        baseline_descriptor = None
                        baseline_runner = None
                        try:
                            for index in [-1] + indices:
                                os.environ["H3_EXPERIMENT_GEMM_LT"] = "0" if index == -1 else "1"
                                os.environ["H3_GEMM_LT_INDEX"] = str(max(index, 0))
                                desc = ptr()
                                status = lib.infiniopCreateGemmDescriptor(handle, C.byref(desc), *descriptors)
                                receipt = json.loads(lib.h3GemmLtLastSelection())
                                item = {"index": index, "selection": receipt, "create_status": status}
                                row["candidates"].append(item)
                                if status:
                                    if index == -1 or status != 2:
                                        raise RuntimeError(f"Descriptor creation failed: {item}")
                                    save()
                                    print("PHASE6_SKIP", json.dumps(item), flush=True)
                                    continue
                                if index == -1 and args.timing_mode == "paired":
                                    baseline_descriptor = desc
                                try:
                                    if receipt["backend"] != ("blas" if index == -1 else "lt"):
                                        raise RuntimeError(f"Unexpected fallback: {receipt}")
                                    workspace_bytes = size()
                                    check(lib.infiniopGetGemmWorkspaceSize(desc, C.byref(workspace_bytes)), "workspace")
                                    if workspace_bytes.value > workspace_mb * (1 << 20):
                                        raise RuntimeError("Workspace exceeds requested budget")
                                    workspace = torch.empty(max(1, workspace_bytes.value), device="cuda", dtype=torch.uint8)
                                    stream = torch.cuda.current_stream()

                                    def run(descriptor=desc, scratch=workspace, scratch_bytes=workspace_bytes.value):
                                        # Reset is ordered on the same stream, before timing.
                                        c.copy_(initial)
                                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                        start.record(stream)
                                        start_wall = time.perf_counter()
                                        check(lib.infiniopGemm(
                                            descriptor, scratch.data_ptr(), scratch_bytes,
                                            c.data_ptr(), a.data_ptr(), b.data_ptr(), alpha, beta,
                                            stream.cuda_stream), "GEMM execution")
                                        end.record(stream)
                                        end.synchronize()
                                        return start.elapsed_time(end), time.perf_counter() - start_wall

                                    if index == -1 and args.timing_mode == "paired":
                                        baseline_runner = run
                                    for _ in range(args.warmup):
                                        run()
                                    first = c.clone()
                                    run()
                                    item["repeat"] = metrics(c, first)
                                    if index == -1:
                                        baseline = first
                                        row["baseline_repeat_bitwise"] = item["repeat"]["bitwise_equal"]
                                    item["vs_baseline"] = metrics(c, baseline)
                                    if not item["repeat"]["finite"]:
                                        raise RuntimeError("Nonfinite GEMM output")
                                    samples = []
                                    all_repeats_equal = item["repeat"]["bitwise_equal"]
                                    if args.timing_mode == "paired" and index >= 0:
                                        item["paired"] = bracketed(
                                            baseline_runner, run,
                                            lambda: metrics(c, baseline)["bitwise_equal"],
                                            lambda: metrics(c, first)["bitwise_equal"],
                                            args.repeats, args.warmup)
                                        samples = [(s["candidate_gpu_ms"], s["candidate_wall_seconds"])
                                                   for s in item["paired"]["samples"]]
                                        all_repeats_equal &= all(s["candidate_bitwise"] and s["baseline_bitwise"]
                                                                for s in item["paired"]["samples"])
                                    else:
                                        for _ in range(args.repeats):
                                            samples.append(run())
                                            all_repeats_equal &= metrics(c, first)["bitwise_equal"]
                                    item["all_repeats_bitwise_equal"] = bool(all_repeats_equal)
                                    item["gpu_ms"] = [x[0] for x in samples]
                                    item["wall_seconds"] = [x[1] for x in samples]
                                    item["median_gpu_ms"] = statistics.median(item["gpu_ms"])
                                    item["workspace_bytes"] = workspace_bytes.value
                                    item["strict_eligible"] = (row["baseline_repeat_bitwise"]
                                        and all_repeats_equal and item["vs_baseline"]["bitwise_equal"])
                                    if index == -1:
                                        row["baseline_repeat_bitwise"] = bool(all_repeats_equal)
                                    # Check alignment preflight does not touch C.
                                    if index >= 0:
                                        before = c.clone()
                                        bad_status = lib.infiniopGemm(
                                            desc, workspace.data_ptr(), workspace_bytes.value,
                                            c.data_ptr(), a.data_ptr() + 2, b.data_ptr(), alpha, beta,
                                            stream.cuda_stream)
                                        check(0 if bad_status == 3 else 1, "unaligned pointer rejection")
                                        torch.cuda.synchronize()
                                        if not metrics(c, before)["bitwise_equal"]:
                                            raise RuntimeError("C changed after preflight rejection")
                                        item["unaligned_rejected_before_write"] = True
                                finally:
                                    if not (index == -1 and args.timing_mode == "paired"):
                                        check(lib.infiniopDestroyGemmDescriptor(desc), "destroy GEMM")
                                save()
                                print("PHASE6_RESULT", json.dumps(item), flush=True)
                        finally:
                            if baseline_descriptor is not None:
                                check(lib.infiniopDestroyGemmDescriptor(baseline_descriptor),
                                      "destroy paired baseline")
                            baseline_runner = None
                finally:
                    for desc in descriptors:
                        check(lib.infiniopDestroyTensorDescriptor(desc), "destroy tensor")
                del a, b, c, initial, baseline
                first = before = workspace = None
                torch.cuda.empty_cache()
        report["complete"] = True
        report["lt_executed"] = any(item.get("selection", {}).get("backend") == "lt"
            and "median_gpu_ms" in item for row in report["cases"] for item in row["candidates"])
        report["model_acceptance"] = False
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        check(lib.infiniopDestroyHandle(handle), "destroy handle")
        save()
        print("PHASE6_END", json.dumps({"complete": report["complete"],
              "lt_executed": report.get("lt_executed", False)}), flush=True)


if __name__ == "__main__":
    main()
