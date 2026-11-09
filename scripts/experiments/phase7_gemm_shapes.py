"""Convert actual H3_GEMM_SHAPE logs into exact Lt probe layouts.

Only row-major BF16 linear maps with batch=1 and column-major op_b=N qualify.
Unknown layouts fail explicitly. Optional overlap row sizes add future chunks.
No GPU, build, resource submission or AFS access.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def shapes_from_rows(rows, chunks=()):
    unique, skipped = {}, []
    for row in rows:
        if row["dtype"] != 19 or row["batch"] != 1 or row["n"] < 4096:
            skipped.append(row)
            continue
        if row["transposed"] != 1 or row["op_b"] != "N" or row["op_a"] not in ("N", "T"):
            raise ValueError(f"Unsupported observed layout: {row}")
        m, n, k = (int(row[field]) for field in ("m", "n", "k"))
        transpose = row["op_a"] == "T"
        pad_a, pad_b, pad_c = row["ldb"]-k, row["lda"]-(k if transpose else m), row["ldc"]-m
        if min(m, k) < 128 or min(pad_a, pad_b, pad_c) < 0:
            raise ValueError(f"Ineligible GEMM dimensions/ld: {row}")
        counts = {n}
        for chunk in chunks:
            if chunk not in (1024, 2048, 4096, 8192):
                raise ValueError("Invalid overlap rows")
            if n > chunk:
                counts.add(chunk)
                if n % chunk:
                    counts.add(n % chunk)
        for tokens in sorted(counts):
            if tokens < 4096: # This Lt backend intentionally keeps these in BLAS.
                continue
            shape = {"tokens": tokens, "out_features": m, "in_features": k,
                     "transpose_b": transpose, "pad_a": pad_a, "pad_b": pad_b, "pad_c": pad_c}
            identity = json.dumps(shape, sort_keys=True)
            shape["name"] = "observed_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
            unique[identity] = shape
    return list(unique.values()), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-rows", type=int, nargs="*", default=[])
    args = parser.parse_args()
    rows, sources = [], []
    for path in args.logs:
        data = path.read_bytes()
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()})
        for line in data.decode(errors="replace").splitlines():
            marker = "H3_GEMM_SHAPE "
            if marker in line:
                rows.append(json.loads(line.split(marker, 1)[1]))
    if not rows:
        raise ValueError("No actual GEMM shape trace found")
    shapes, skipped = shapes_from_rows(rows, args.overlap_rows)
    if not shapes:
        raise ValueError("Trace contains no eligible Lt shapes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(shapes, indent=2)+"\n")
    args.output.with_suffix(args.output.suffix+".json").write_text(json.dumps(
        {"sources": sources, "trace_rows": len(rows), "eligible_unique": len(shapes),
         "skipped_rows": len(skipped), "overlap_rows": args.overlap_rows,
         "not_a_performance_result": True}, indent=2)+"\n")
    print("PHASE7_GEMM_SHAPES", len(shapes), "from actual trace", flush=True)


if __name__ == "__main__":
    main()
