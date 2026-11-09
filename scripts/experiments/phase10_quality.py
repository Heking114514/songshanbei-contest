"""Compare completed same-upstream phase10 media, timing boundaries and fingerprints."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from phase10_profile import sha


def timing(run):
    return dict(pipeline_seconds=run["timings"]["generation.pipeline_total"],
                entry_internal_total_seconds=run["timings"]["total.end_to_end"],
                steady_synchronized_step_seconds=run["steady_synchronized_dit_median_seconds"],
                dit_calls=len(run["step_records"]))


def compare_pair(baseline, candidate, compare_video):
    import av
    import numpy as np
    runs = [json.loads((p/"run.json").read_text()) for p in (baseline, candidate)]
    receipts = [json.loads((p/"freeze-receipt.json").read_text()) for p in (baseline, candidate)]
    if not all(r.get("complete") for r in runs+receipts):
        raise ValueError("Incomplete model/freeze result")
    if receipts[0]["snapshot_sha256"] != receipts[1]["snapshot_sha256"] or not receipts[1]["first_input_equal"]:
        raise ValueError("Full comparison needs the exact same upstream state")
    inputs = [[json.loads(line) for line in (p/"dit-inputs.jsonl").read_text().splitlines()]
              for p in (baseline, candidate)]
    if not inputs[0] or len(inputs[0]) != len(inputs[1]) or inputs[0][0] != inputs[1][0]:
        raise ValueError("DiT input sequence/first input mismatch")
    if any(len(rows) != 49 or len(r["step_records"]) != 49 for rows,r in zip(inputs,runs)):
        raise ValueError("DiT fingerprints do not match executed calls")
    first = next((i for i,(a,b) in enumerate(zip(*inputs)) if a != b), None)
    arrays, media = [], []
    for path in (baseline/"output.mp4", candidate/"output.mp4"):
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
                raise ValueError("Expected one video and one audio stream")
            v,a = container.streams.video[0],container.streams.audio[0]
            row = dict(width=v.width, height=v.height, fps=str(v.average_rate),
                       sample_rate=a.rate, channels=a.codec_context.channels, sha256=sha(path))
        with av.open(str(path)) as container:
            row["frames"] = sum(1 for _ in container.decode(video=0))
        with av.open(str(path)) as container:
            array = np.concatenate([frame.to_ndarray().astype(np.float64)
                                    for frame in container.decode(audio=0)], axis=-1)
        if not np.isfinite(array).all(): raise ValueError("Nonfinite decoded audio")
        arrays.append(array)
        media.append(row)
    for row in media:
        if (row["frames"],row["fps"],row["sample_rate"],row["channels"]) != (124,"24",32000,2):
            raise ValueError("Expected 124 frames/24fps/32000Hz/stereo")
    contract = ("width","height","fps","frames","sample_rate","channels")
    if any(media[0][k] != media[1][k] for k in contract):
        raise ValueError("Output media contracts differ")
    if arrays[0].shape != arrays[1].shape: raise ValueError("Audio PCM shapes differ")
    norm = float(np.linalg.norm(arrays[0]))
    if norm <= 0: raise ValueError("Zero-energy baseline audio; relative comparison undefined")
    return dict(complete=True, baseline=media[0], candidate=media[1],
        first_different_dit_input=first, all_dit_inputs_equal=first is None,
        video=compare_video(baseline/"output.mp4",candidate/"output.mp4"),
        audio=dict(bitwise_equal=bool(np.array_equal(*arrays)),
                   relative_l2=float(np.linalg.norm(arrays[0]-arrays[1])/norm),
                   correlation=float(np.corrcoef(arrays[0].reshape(-1),arrays[1].reshape(-1))[0,1])),
        timing=dict(baseline=timing(runs[0]),candidate=timing(runs[1])),
        model_quality_accepted=False, official_same_round_pytorch_speedup=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidates", required=True, type=Path, nargs="+")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists(): ap.error("Choose a fresh output")
    path = Path(__file__).resolve().parents[2]/"InfiniLM/examples/minimax_h3_compare.py"
    spec = importlib.util.spec_from_file_location("phase10_official_compare",path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results = {}
    complete = True
    for candidate in args.candidates:
        try:
            results[str(candidate)] = compare_pair(args.baseline,candidate,module.compare_video_streams)
        except Exception as exc:
            complete = False
            results[str(candidate)] = dict(complete=False,error=repr(exc))
    report = dict(complete=complete,comparisons=results,model_quality_accepted=False,
                  compare_source_sha256=sha(path))
    args.output.write_bytes((json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    if not complete: raise RuntimeError("One or more full quality comparisons failed")


if __name__ == "__main__":
    main()
