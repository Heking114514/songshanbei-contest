"""Future GPU worker: compare one AOT variant against immutable v4, in a fresh process."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import time


class Spec(C.Structure):
    _fields_ = [("batch", C.c_int64), ("query_tokens", C.c_int64),
                ("key_tokens", C.c_int64), ("heads", C.c_int64), ("strides", (C.c_int64 * 4) * 4)]


class Call(C.Structure):
    _fields_ = [("q", C.c_void_p), ("k", C.c_void_p), ("v", C.c_void_p),
                ("out", C.c_void_p), ("scale", C.c_float)]


Prep = C.CFUNCTYPE(C.c_int, C.POINTER(Spec), C.POINTER(C.c_void_p))
Launch = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(Call), C.c_void_p)
Destroy = C.CFUNCTYPE(None, C.c_void_p)


class Api(C.Structure):
    _fields_ = [("version", C.c_uint32), ("struct_bytes", C.c_uint32),
                ("arch", C.c_char_p), ("build", C.c_char_p),
                ("prepare", Prep), ("launch", Launch), ("destroy", Destroy)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--max-relative-l2", type=float, default=0.005,
                    help="Local screening versus v4 only; not a media acceptance threshold")
    args = ap.parse_args()
    started = time.monotonic()
    if args.repeats < 3 or not 0 < args.max_relative_l2 < 1:
        ap.error("Use at least three repeats and a positive error budget below 1")
    import torch
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    if not torch.version.hip or getattr(properties, "gcnArchName", "").split(":")[0] != "gfx936":
        raise RuntimeError("Actual HIP gfx936 required; no architecture substitution")
    torch.set_grad_enabled(False)
    torch.manual_seed(4271)
    report = dict(complete=False, cases=[], device=args.device, arch="gfx936",
                  torch=torch.__version__, hip=torch.version.hip, model_acceptance=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())

    owners = []
    def api(path):
        library = C.CDLL(str(path.resolve()))
        library.h3FlashGetApi.restype = C.POINTER(Api)
        value = library.h3FlashGetApi().contents
        assert value.version == 1 and value.struct_bytes == C.sizeof(Api) and value.arch == b"gfx936"
        owners.append(library)
        return value

    baseline, candidate = api(args.baseline), api(args.candidate)
    report["libraries"] = {k: dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                    build_id=v.build.decode())
                           for k, p, v in [("baseline", args.baseline, baseline),
                                           ("candidate", args.candidate, candidate)]}
    def metrics(a, b):
        x, y = a.float(), b.float()
        return dict(finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                    bitwise_equal=bool(torch.equal(a, b)),
                    relative_l2=float((x-y).norm()/y.norm().clamp_min(1e-12)),
                    max_abs=float((x-y).abs().max()))

    def prepare(value, q, k, v, out):
        spec = Spec(1, q.shape[1], k.shape[1], q.shape[2])
        for i, t in enumerate((q, k, v, out)):
            spec.strides[i][:] = list(t.stride())
        ptr = C.c_void_p()
        rc = value.prepare(C.byref(spec), C.byref(ptr))
        if rc or not ptr:
            raise RuntimeError(("prepare", rc))
        call = Call(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), 128**-0.5)
        def run():
            rc = value.launch(ptr, C.byref(call), torch.cuda.current_stream().cuda_stream)
            if rc:
                raise RuntimeError(("launch", rc))
        return ptr, call, run

    def timing(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        values = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return dict(samples_ms=values, median_ms=statistics.median(values))

    try:
        save()
        for nq, nk in [(65, 33), (2048, 2048), (27570, 27570)]:
            for padded in (False, True):
                torch.manual_seed(4271)
                q, k, v = [torch.randn(1, n, 7 * (3 if padded else 1), 128,
                                        device="cuda", dtype=torch.bfloat16)
                           for n in (nq, nk, nk)]
                if padded:
                    q, k, v = q[:, :, :7], k[:, :, 7:14], v[:, :, 14:21]
                storage = [torch.full((1, nq, 7 * (2 if padded else 1), 128), float("nan"),
                                       device="cuda", dtype=torch.bfloat16) for _ in range(2)]
                expected, actual = [x[:, :, :7] for x in storage]
                bp = cp = None
                try:
                    bp, bc, br = prepare(baseline, q, k, v, expected)
                    cp, cc, cr = prepare(candidate, q, k, v, actual)
                    br(); cr(); torch.cuda.synchronize()
                    error = metrics(actual, expected)
                    if not error["finite"] or error["relative_l2"] > args.max_relative_l2:
                        raise AssertionError(error)
                    first = actual.clone()
                    for _ in range(args.repeats):
                        cr()
                    torch.cuda.synchronize()
                    if not torch.equal(actual, first):
                        raise AssertionError("Candidate repeat was not bitwise deterministic")
                    math_error = None
                    if nq == 65:
                        from torch.nn.attention import sdpa_kernel, SDPBackend
                        with sdpa_kernel(SDPBackend.MATH):
                            math_out = torch.nn.functional.scaled_dot_product_attention(
                                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
                        math_error = metrics(actual, math_out)
                        if not math_error["finite"] or math_error["relative_l2"] >= 0.02:
                            raise AssertionError(math_error)
                    bt, ct = timing(br), timing(cr)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        cr()
                    q.add_(0.0625); v.mul_(0.875)
                    br(); graph.replay(); torch.cuda.synchronize()
                    updated = metrics(actual, expected)
                    if not updated["finite"] or updated["relative_l2"] > args.max_relative_l2:
                        raise AssertionError(updated)
                    gt = timing(graph.replay)
                    row = dict(query=nq, key=nk, heads=7, padded=padded, versus_v4=error,
                               versus_math_small=math_error, graph_updated_inputs=updated,
                               repeat_bitwise=True, baseline=bt, candidate=ct, graph=gt,
                               speedup=bt["median_ms"]/ct["median_ms"])
                    report["cases"].append(row); save()
                    print("PHASE8_ATTENTION_CASE", json.dumps(row), flush=True)
                    del graph
                finally:
                    torch.cuda.synchronize()
                    if bp: baseline.destroy(bp)
                    if cp: candidate.destroy(cp)
                del expected, actual, q, k, v, storage, first
                torch.cuda.empty_cache()
        real = [r["speedup"] for r in report["cases"] if r["query"] == 27570]
        report.update(complete=True, real_worst_speedup=min(real),
                      screening_pass=True, elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        save()
    print("PHASE8_ATTENTION_WORKER_COMPLETE", report["real_worst_speedup"], flush=True)


if __name__ == "__main__":
    started = time.monotonic()
    main()
