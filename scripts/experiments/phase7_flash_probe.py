"""Native gfx936 AOT flash screening. Run only after authorized GPU deployment.

Compares against explicitly selected ATen MATH on identical tensors; reports
error without declaring model quality. No resource request/build/AFS write.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-size", action="store_true")
    parser.add_argument("--max-relative-l2", type=float, default=0.02, help="Screening threshold only")
    args = parser.parse_args()
    work, root, plugin, output = [p.resolve() for p in
        (args.workspace_root, args.library_root, args.plugin, args.output)]
    afs = os.environ.get("IFAB_AFS_HOME")
    if (afs and work.is_relative_to(Path(afs).resolve())) or not all(
            path.is_relative_to(work) for path in (root, plugin, output)):
        parser.error("All outputs/libraries must be in private workspace outside AFS")
    if not 0 < args.max_relative_l2 < 1:
        parser.error("Invalid screening threshold")
    os.environ["H3_EXPERIMENT_FLASH_GFX936"] = "1"
    os.environ["H3_FLASH_LIBRARY"] = str(plugin)
    os.environ["H3_EXPERIMENT_MHA_QUERY_CHUNK"] = "0"
    os.environ["H3_EXPERIMENT_MHA_HEAD_BATCH"] = "0"
    import torch
    import infinicore as ic
    from infinicore.lib import _infinicore as native
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    properties = torch.cuda.get_device_properties(0)
    if getattr(properties, "gcnArchName", "").split(":")[0] != "gfx936":
        raise RuntimeError("Probe requires actual gfx936")
    torch.set_grad_enabled(False)
    torch.manual_seed(4271)
    bridge_device = ic.from_torch(torch.empty(1, device="cuda"))
    ic.set_device(bridge_device.device)
    report = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "complete": False,
              "plugin": str(plugin), "plugin_sha256": hashlib.sha256(plugin.read_bytes()).hexdigest(),
              "torch": torch.__version__, "gpu": str(properties), "cases": [],
              "model_acceptance": False, "afs_modified": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    owners = []

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def sync():
        ic.sync_device()
        torch.cuda.synchronize()

    def bridge(t):
        result = ic.from_torch(t)
        if not t.is_contiguous():
            result = ic.strided_from_blob(t.data_ptr(), list(t.shape), list(t.stride()),
                                          dtype=result.dtype, device=result.device)
        owners.append((t, result))
        return result._underlying

    def error(a, b):
        delta = a.float() - b.float()
        return {"finite": bool(torch.isfinite(a).all()),
                "bitwise_equal": bool(torch.equal(a.view(torch.int16), b.view(torch.int16))),
                "relative_l2": float((delta.norm() / b.float().norm().clamp_min(1e-12)).item()),
                "max_abs": float(delta.abs().max().item())}

    def timing(fn):
        for _ in range(2):
            fn()
            sync()
        times = []
        for _ in range(7):
            sync()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            sync()
            times.append(start.elapsed_time(end))
        return {"gpu_ms": times, "median_gpu_ms": statistics.median(times)}

    def audit():
        paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if any(name in line for name in
                            ("libinfiniop.so", "libinfinicore_cpp_api.so", "_infinicore", plugin.name))})
        native_paths = [p for p in paths if Path(p).resolve() != plugin]
        if not native_paths or not all(Path(p).resolve().is_relative_to(root) for p in native_paths):
            raise RuntimeError(f"Unexpected native library paths: {paths}")
        if str(plugin) not in paths:
            raise RuntimeError("AOT plugin was not loaded by native dispatch")
        report["mapped_libraries"] = paths
        save()

    shapes = [(1, 1, 1), (65, 33, 7), (73, 129, 7), (2048, 2048, 7)]
    if args.real_size:
        shapes.append((27570, 27570, 7))
    try:
        save()
        for nq, nk, heads in shapes:
            for padded in (False, True):
                owners.clear()
                print("PHASE7_FLASH_START", nq, nk, heads, padded, flush=True)
                if nq == nk and padded:
                    packed = torch.randn(1, nq, heads*3, 128, device="cuda", dtype=torch.bfloat16)
                    q, k, v = packed[:, :, :heads], packed[:, :, heads:2*heads], packed[:, :, 2*heads:]
                else:
                    q = torch.randn(1, nq, heads, 128, device="cuda", dtype=torch.bfloat16)
                    k, v = [torch.randn(1, nk, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
                # Padded destination exercises direct-strided output stores.
                storage = torch.empty(1, nq, heads*(2 if padded else 1), 128,
                                      device="cuda", dtype=torch.bfloat16)
                actual = storage[:, :, :heads]
                oi, qi, ki, vi = map(bridge, (actual, q, k, v))
                scale = 128**-0.5
                def candidate():
                    native.mha_(oi, qi, ki, vi, None, scale, False)
                def reference():
                    with sdpa_kernel(SDPBackend.MATH):
                        return torch.nn.functional.scaled_dot_product_attention(
                            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                            scale=scale).transpose(1, 2)
                expected = reference()
                candidate()
                sync()
                first = actual.clone()
                candidate()
                sync()
                repeated = error(actual, first)
                metrics = error(actual, expected)
                if not metrics["finite"] or metrics["relative_l2"] >= args.max_relative_l2:
                    raise AssertionError(metrics)
                if not repeated["bitwise_equal"]:
                    raise AssertionError("AOT result not reproducible")
                row = {"query": nq, "key": nk, "heads": heads, "padded": padded,
                       "vs_math": metrics, "repeat": repeated,
                       "math": timing(reference), "aot": timing(candidate)}
                # The native software graph uses updated pointer contents.
                ic.start_graph_recording()
                try:
                    candidate()
                finally:
                    graph = ic.stop_graph_recording()
                q.add_(0.0625)
                v.mul_(0.875)
                expected = reference()
                graph.run()
                sync()
                updated = error(actual, expected)
                if not updated["finite"] or updated["relative_l2"] >= args.max_relative_l2:
                    raise AssertionError(updated)
                row["graph_updated_inputs_vs_math"] = updated
                row["graph"] = timing(graph.run)
                report["cases"].append(row)
                audit()
                print("PHASE7_FLASH_RESULT", json.dumps(row), flush=True)
                del graph, expected, first, q, k, v, storage, actual
                owners.clear()
                torch.cuda.empty_cache()
        report["complete"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        save()
        print("PHASE7_FLASH_END", json.dumps({"complete": report["complete"],
              "model_acceptance": False}), flush=True)


if __name__ == "__main__":
    main()
