"""Offline audio trace comparison. Missing/truncated stages are never treated as equality."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_trace(root):
    root = Path(root)
    rows = [json.loads(s) for s in (root/"trace.jsonl").read_text().splitlines()]
    if not rows or rows[0].get("schema") != 1 or rows[-1].get("kind") != "end" or not rows[-1].get("complete"):
        raise ValueError("Trace missing a successful completion marker: "+str(root))
    tensors, order = {}, []
    for row in rows:
        if row.get("kind") != "tensor":
            continue
        key = (row["step"], row["stage"], row["occurrence"])
        if key in tensors:
            raise ValueError("Duplicate tensor trace key")
        if row["dump"]:
            file = root/row["dump"]
            if not file.is_file() or file.stat().st_size != row["bytes"] or hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Trace blob SHA/size mismatch")
        if row["sequence"] != len(order):
            raise ValueError("Non-contiguous pipeline tensor sequence")
        tensors[key] = dict(row, root=str(root)); order.append(key)
    if rows[-1]["tensor_events"] != len(order):
        raise ValueError("Completion marker disagrees with tensor event count")
    return tensors, order


def array(row):
    if not row.get("dump"):
        return None
    raw = (Path(row["root"])/row["dump"]).read_bytes()
    types = dict(float32="<f4", float64="<f8", float16="<f2", int64="<i8", int32="<i4",
                 int16="<i2", uint8="u1", int8="i1", bool="?")
    if row["dtype"] == "bfloat16":
        value = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif row["dtype"] in types:
        value = np.frombuffer(raw, dtype=types[row["dtype"]])
    else:
        return None
    return value.reshape(row["shape"])


def compare_rows(left, right):
    same_layout = left["shape"] == right["shape"] and left["dtype"] == right["dtype"]
    result = dict(exact=same_layout and left["sha256"] == right["sha256"],
                  shape_dtype_match=same_layout, left_finite=left.get("finite"), right_finite=right.get("finite"))
    if same_layout:
        a, b = array(left), array(right)
        if a is not None and b is not None:
            a, b = a.astype(np.float64), b.astype(np.float64)
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            if finite:
                d = a-b; norm = np.linalg.norm(a)
                result.update(relative_l2=float(np.linalg.norm(d)/max(float(norm), 1e-12)),
                              rmse=float(np.sqrt(np.mean(d*d))) if d.size else 0.0,
                              max_abs=float(np.max(np.abs(d))) if d.size else 0.0)
    return result


def native_rows(root):
    rows = {}
    for directory in sorted(Path(root).glob("native-step-*")):
        step = int(directory.name.rsplit("-", 1)[1])
        stage_rows = []
        for metadata in directory.glob("*.json"):
            row = json.loads(metadata.read_text())
            file = metadata.with_suffix(".bin")
            if not file.exists() or file.stat().st_size != row["bytes"] or row["rank"] != 0:
                raise ValueError("Incomplete native audio trace")
            row.update(root=str(directory), dump=file.name,
                       sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            stage_rows.append(row)
        if not stage_rows:
            raise ValueError("Empty native trace directory")
        stage_rows.sort(key=lambda r:r["sequence"])
        if [r["sequence"] for r in stage_rows] != list(range(len(stage_rows))):
            raise ValueError("Non-contiguous native trace sequence")
        names = {r["stage"] for r in stage_rows}
        required = {"audio_input", "audio_embeds", "packed_audio", "norm_out_audio", "audio_output"}
        blocks = [int(n.removeprefix("block_").removesuffix("_audio")) for n in names
                  if n.startswith("block_") and n.endswith("_audio")]
        if not required.issubset(names) or not blocks or sorted(blocks) != list(range(len(blocks))):
            raise ValueError("Missing native block boundary")
        expected_order = ["audio_input", "audio_embeds", "packed_audio",
                          *(f"block_{i}_audio" for i in range(len(blocks))),
                          "norm_out_audio", "audio_output"]
        if [r["stage"] for r in stage_rows] != expected_order:
            raise ValueError("Native block boundary order is invalid")
        for row in stage_rows:
            key = (step, row["stage"])
            if key in rows: raise ValueError("Duplicate native stage")
            rows[key] = row
    return rows


def compare(left, right):
    a, order = load_trace(left); b, other = load_trace(right)
    missing_left, missing_right = set(b)-set(a), set(a)-set(b)
    result = dict(complete=not missing_left and not missing_right and order == other,
                  execution_order_match=order == other,
                  missing_left=sorted(missing_left, key=str), missing_right=sorted(missing_right, key=str),
                  first_pipeline_difference=None, rows=[], native=[], same_dit_inputs={})
    for key in order:
        if key not in b: continue
        row = dict(step=key[0], stage=key[1], occurrence=key[2], **compare_rows(a[key], b[key]))
        result["rows"].append(row)
        if not row["exact"] and result["first_pipeline_difference"] is None:
            result["first_pipeline_difference"] = row
    for step in sorted({k[0] for k in a if k[1].startswith("dit.input.")}):
        keys = [k for k in a if k[0] == step and k[1].startswith("dit.input.")]
        result["same_dit_inputs"][str(step)] = bool(keys) and all(
            k in b and compare_rows(a[k], b[k])["exact"] for k in keys)
    na, nb = native_rows(left), native_rows(right)
    result["native_steps"] = sorted({k[0] for k in na})
    result["first_native_audio_difference"] = None
    if set(na) != set(nb):
        result["complete"] = False
        result["native_missing_left"] = sorted(set(nb)-set(na), key=str)
        result["native_missing_right"] = sorted(set(na)-set(nb), key=str)
    for key in sorted(set(na)&set(nb), key=lambda k:(k[0],na[k]["sequence"])):
        row = dict(step=key[0], stage=key[1], **compare_rows(na[key], nb[key]))
        result["native"].append(row)
        if not row["exact"] and result["first_native_audio_difference"] is None:
            result["first_native_audio_difference"] = row
    decoder_inputs = [r for r in result["rows"] if r["stage"] == "audio.decode.input" or r["stage"].startswith("audio.decode.input.")]
    result["identical_decoder_input"] = bool(decoder_inputs) and all(r["exact"] for r in decoder_inputs)
    result["interpretation"] = {
        "native": "First observed AUDIO-row difference at sampled block boundaries; not proof of the failing operator.",
        "upstream": "Use same_dit_inputs to establish whether native comparisons have identical full inputs.",
        "decoder": "Different decoder inputs cannot establish a decoder defect.",
        "timing": "Hashing and native host copies perturb execution; trace times are not speed benchmarks.",
        "native_scope": "Audio rows only. An earlier difference in video/text rows can reach audio through later attention."
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left", type=Path); ap.add_argument("right", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ("complete", "first_pipeline_difference",
          "first_native_audio_difference", "same_dit_inputs", "identical_decoder_input")}), flush=True)
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
