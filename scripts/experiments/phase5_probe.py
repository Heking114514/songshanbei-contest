"""Future single-device phase5 probes. Does not build, submit or access AFS.

Strict block=0 compares against current native RMS and pointwise/RoPE operators.
Explicit RMS block tuning changes reduction order; its tolerance is a local
screening threshold, never full-model acceptance.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("rms", "qk"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--real-size", action="store_true")
    parser.add_argument("--blocks", default="0,128,256,512,1024")
    args = parser.parse_args()
    blocks = list(dict.fromkeys(int(x) for x in args.blocks.split(",")))
    assert blocks and all(x in (0, 128, 256, 512, 1024) for x in blocks)
    import torch
    import infinicore as ic
    from infinicore.lib import _infinicore as native

    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    torch.manual_seed(4250)
    owners = []
    device_bridge = ic.from_torch(torch.empty(1, device="cuda"))
    ic.set_device(device_bridge.device)
    report = {"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "group": args.group, "torch": torch.__version__, "blocks": blocks,
              "flags": {k: v for k, v in os.environ.items() if k.startswith("H3_EXPERIMENT_")},
              "cases": [], "complete": False}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def audit():
        paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if any(n in line for n in ("libinfiniop.so", "libinfinicore_cpp_api.so", "_infinicore"))})
        root = args.library_root.resolve()
        assert paths and all(Path(p).resolve().is_relative_to(root) for p in paths), (root, paths)
        report["library_paths"] = paths
        save()

    def bridge(t):
        result = ic.from_torch(t)
        if not t.is_contiguous():
            result = ic.strided_from_blob(t.data_ptr(), list(t.shape), list(t.stride()),
                                          dtype=result.dtype, device=result.device)
        owners.append((t, result))
        return result._underlying

    def sync():
        ic.sync_device()
        torch.cuda.synchronize()

    def emit(row):
        report["cases"].append(row)
        save()
        print("PHASE5_RESULT", json.dumps(row), flush=True)

    def metrics(a, b):
        delta = a.float() - b.float()
        return {"bitwise_equal": torch.equal(a.view(torch.int16), b.view(torch.int16)),
                "finite": bool(torch.isfinite(a).all()),
                "max_abs": delta.abs().max().item(),
                "relative_l2": (delta.norm() / b.float().norm().clamp_min(1e-12)).item()}

    def checked(a, b, strict):
        m = metrics(a, b)
        assert m["finite"] and (m["bitwise_equal"] if strict else m["relative_l2"] < 0.01), m
        return m

    def timing(fn):
        for _ in range(2):
            fn()
            sync()
        times = []
        for _ in range(7):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            times.append(time.perf_counter() - start)
        return {"seconds": times, "median_seconds": statistics.median(times)}

    def reject(name, fn, rejected):
        try:
            fn()
        except (ValueError, RuntimeError):
            rejected.append(name)
        else:
            raise AssertionError(f"Invalid {name} accepted")

    def rms_case(n, width, weight_dtype):
        print("PHASE5_START rms", n, width, str(weight_dtype), flush=True)
        x = torch.randn(1, n, width, device="cuda", dtype=torch.bfloat16)
        if width > 4:
            x[..., :4] = torch.tensor([0, 1e-8, -32, 32], device="cuda", dtype=torch.bfloat16)
        branch = torch.randn_like(x)
        weight = torch.randn(width, device="cuda", dtype=weight_dtype)
        table = torch.randn(9, 6 * width, device="cuda", dtype=torch.bfloat16) * 0.125
        indices = (torch.arange(n, device="cuda", dtype=torch.int64) * 5 + 8) % 9
        expanded = torch.empty(n, 6 * width, device="cuda", dtype=torch.bfloat16)
        normalized, temp, expected, gated, actual, residual = [torch.empty_like(x) for _ in range(6)]
        xi, bi, wi, ti, ii, ei, ni, tmpi, expected_i, gi, ai, ri = map(
            bridge, (x, branch, weight, table, indices, expanded, normalized,
                     temp, expected, gated, actual, residual))
        views = [expanded[None, :, k * width:(k + 1) * width] for k in range(6)]
        vi = list(map(bridge, views))
        for mode in range(3):
            epsilon = 1e-5 if mode != 1 else 1e-6

            def reference():
                native.embedding_(ei, ii, ti)
                norm_input = xi
                if mode == 2:
                    native.addcmul_(gi, xi, vi[2], bi, 1.0)
                    norm_input = gi
                native.rms_norm_(ni, norm_input, wi, epsilon)
                offset = 0 if mode == 0 else 3
                native.addcmul_(tmpi, ni, ni, vi[offset + 1], 1.0)
                native.add_(expected_i, tmpi, vi[offset])

            reference()
            sync()
            baseline = timing(reference)
            for block in blocks:

                def candidate():
                    native.h3_rms_modulation_(ai, ri, xi, bi, wi, ti, ii, mode, epsilon, block)

                candidate()
                sync()
                result = checked(actual, expected, block == 0)
                if mode == 2:
                    assert metrics(residual, gated)["bitwise_equal"]
                row = {"group": "rms", "n": n, "width": width, "mode": mode, "block": block,
                       "epsilon": epsilon, "weight_dtype": str(weight_dtype), **result,
                       "reference_gather_rms_pointwise": baseline, "fused": timing(candidate)}
                # Strict matching on changed values and software-graph replay.
                if n < 100 and block == 0:
                    indices.copy_((indices + 1) % 9)
                    table.add_(0.00390625)
                    reference()
                    candidate()
                    sync()
                    checked(actual, expected, True)
                    ic.start_graph_recording()
                    try:
                        candidate()
                    finally:
                        graph = ic.stop_graph_recording()
                    indices.copy_((indices + 1) % 9)
                    weight.add_(0.0625)
                    reference()
                    sync()
                    graph.run()
                    sync()
                    checked(actual, expected, True)
                    if mode == 2:
                        assert metrics(residual, gated)["bitwise_equal"]
                    row["graph_updated_indices_table_weight_equal"] = True
                emit(row)
                # Graph checks mutated inputs. Refresh expected for next block.
                reference()
                sync()
        rejected = []
        reject("mode", lambda: native.h3_rms_modulation_(ai, ri, xi, bi, wi, ti, ii, 3, 1e-5, 0), rejected)
        reject("epsilon_nan", lambda: native.h3_rms_modulation_(ai, ri, xi, bi, wi, ti, ii, 0, float("nan"), 0), rejected)
        reject("block", lambda: native.h3_rms_modulation_(ai, ri, xi, bi, wi, ti, ii, 0, 1e-5, 320), rejected)
        reject("output_alias", lambda: native.h3_rms_modulation_(xi, ri, xi, bi, wi, ti, ii, 0, 1e-5, 0), rejected)
        reject("residual_alias", lambda: native.h3_rms_modulation_(ai, xi, xi, bi, wi, ti, ii, 2, 1e-5, 0), rejected)
        emit({"group": "rms_contract", "n": n, "width": width, "rejected": rejected})

    def qk_case(n, heads, rotary, weight_dtype, strided):
        print("PHASE5_START qk", n, heads, rotary, str(weight_dtype), strided, flush=True)
        packed = torch.randn(n, heads * 3, 128, device="cuda", dtype=torch.bfloat16)
        q, k = packed[:, :heads], packed[:, heads:2 * heads]
        if not strided:
            q, k = q.contiguous(), k.contiguous()
        qw, kw = [torch.randn(128, device="cuda", dtype=weight_dtype) for _ in range(2)]
        positions = (torch.arange(n, device="cuda", dtype=torch.int64) * 7) % 17
        positions[0] = -1  # Existing MRoPE explicitly maps invalid values to row0.
        positions[-1] = 17
        cache = torch.randn(17, rotary, device="cuda", dtype=torch.bfloat16)
        nq, nk, eq, ek, aq, ak = [torch.empty(n, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(6)]
        qi, ki, qwi, kwi, pi, ci, nqi, nki, eqi, eki, aqi, aki = map(
            bridge, (q, k, qw, kw, positions, cache, nq, nk, eq, ek, aq, ak))
        half = rotary // 2
        cos, sin = cache[:, :half], cache[:, half:]
        cosi, sini = bridge(cos), bridge(sin)
        # Native norm outputs are compact; flatten [N,heads*128] only for MRoPE.
        nqflat, nkflat = bridge(nq.view(n, heads * 128)), bridge(nk.view(n, heads * 128))
        eqflat, ekflat = bridge(eq.view(n, heads * 128)), bridge(ek.view(n, heads * 128))
        def reference():
            native.rms_norm_(nqi, qi, qwi, 1e-5)
            native.rms_norm_(nki, ki, kwi, 1e-6)
            native.mrope_(eqflat, ekflat, nqflat, nkflat, cosi, sini, pi,
                          128, rotary, rotary // 6, rotary // 6, rotary // 6, False)
        def candidate():
            native.h3_qk_norm_rope_(aqi, aki, qi, ki, qwi, kwi, ci, pi, 1e-5, 1e-6)
        reference()
        candidate()
        sync()
        row = {"group": "qk", "n": n, "heads": heads, "rotary": rotary,
               "weight_dtype": str(weight_dtype), "strided": strided,
               "q": checked(aq, eq, True), "k": checked(ak, ek, True),
               "reference_two_norms_rope": timing(reference), "fused": timing(candidate)}
        if n < 100:
            ic.start_graph_recording()
            try:
                candidate()
            finally:
                graph = ic.stop_graph_recording()
            positions.copy_((positions + 1) % 17)
            cache.add_(0.0625)
            q.add_(0.125)
            kw.mul_(0.875)
            reference()
            sync()
            graph.run()
            sync()
            checked(aq, eq, True)
            checked(ak, ek, True)
            row["graph_updated_positions_cache_input_weight_equal"] = True
        rejected = []
        reject("outputs_alias", lambda: native.h3_qk_norm_rope_(aqi, aqi, qi, ki, qwi, kwi, ci, pi, 1e-5, 1e-6), rejected)
        reject("epsilon", lambda: native.h3_qk_norm_rope_(aqi, aki, qi, ki, qwi, kwi, ci, pi, 0, 1e-6), rejected)
        bad_position = bridge(positions.to(torch.int32))
        reject("position_dtype", lambda: native.h3_qk_norm_rope_(aqi, aki, qi, ki, qwi, kwi, ci, bad_position, 1e-5, 1e-6), rejected)
        row["rejected"] = rejected
        emit(row)

    save()
    audit()
    if args.group == "rms":
        shapes = [(1, 128), (17, 1031), (19, 5376)]
        if args.real_size:
            shapes.append((27570, 5376))
        for n, width in shapes:
            for dtype in (torch.bfloat16, torch.float32):
                rms_case(n, width, dtype)
                owners.clear()
                gc.collect()
                torch.cuda.empty_cache()
    else:
        shapes = [(1, 1, 6), (17, 7, 96), (19, 7, 126)]
        if args.real_size:
            shapes.append((27570, 7, 96))
        for n, heads, rotary in shapes:
            for dtype in (torch.bfloat16, torch.float32):
                for strided in (False, True):
                    qk_case(n, heads, rotary, dtype, strided)
                    owners.clear()
                    gc.collect()
                    torch.cuda.empty_cache()
    audit()
    report["complete"] = True
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    save()
    print("PHASE5_COMPLETE", args.group, flush=True)


if __name__ == "__main__":
    main()
