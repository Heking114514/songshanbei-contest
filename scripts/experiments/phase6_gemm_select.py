"""Build a strict per-shape Lt allowlist from completed offline probe reports.

Requires four alpha/beta cases, native baseline bit parity, repeat bit parity, and
a gain over BLAS in every supplied run of that exact key. No GPU/resource/AFS I/O.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo
from phase10_gemm_timing import paired_gain


def select(reports, min_gain=0.02, min_reports=2, require_paired=False):
    groups = defaultdict(list)
    for report_id, report in enumerate(reports):
        if not report.get("complete") or not report.get("lt_executed"):
            raise ValueError("Incomplete or non-Lt probe report")
        if require_paired and report.get("timing_mode") != "paired":
            raise ValueError("Paired selection requires new paired probe reports")
        by_key = defaultdict(list)
        for case in report["cases"]:
            candidates = case["candidates"]
            baselines = [c for c in candidates if c["index"] == -1]
            if len(baselines) != 1:
                raise ValueError("Each case requires one baseline")
            baseline = baselines[0]
            valid_baseline = (case.get("baseline_repeat_bitwise") is True
                              and baseline.get("repeat", {}).get("finite") is True)
            base_ms = baseline.get("median_gpu_ms", float("nan"))
            if not math.isfinite(base_ms) or base_ms <= 0:
                raise ValueError("Invalid baseline timing")
            keys = {c["selection"]["key"] for c in candidates
                    if c["index"] >= 0 and c.get("selection", {}).get("key")}
            if len(keys) > 1:
                raise ValueError("Different Lt keys within one case")
            if not keys:
                continue
            key = next(iter(keys))
            tag = report.get("runtime_tag", "")
            if len(tag) != 64 or not all(c in "0123456789abcdef" for c in tag) or ":"+tag+":" not in key:
                raise ValueError("Unbound or mismatched runtime fingerprint")
            if not key.startswith("h3lt1:") or any(c.isspace() for c in key):
                raise ValueError("Invalid policy key")
            choices = {}
            for item in candidates:
                index = item["index"]
                if index < 0:
                    continue
                if index in choices:
                    raise ValueError("Duplicate candidate index")
                ms = item.get("median_gpu_ms", float("nan"))
                if (valid_baseline and item.get("strict_eligible") is True
                        and item.get("create_status") == 0
                        and item["selection"].get("backend") == "lt"
                        and item["selection"].get("index") == index
                        and item.get("repeat", {}).get("bitwise_equal") is True
                        and item.get("all_repeats_bitwise_equal") is True
                        and item.get("vs_baseline", {}).get("bitwise_equal") is True
                        and item.get("repeat", {}).get("finite") is True
                        and item.get("vs_baseline", {}).get("finite") is True
                        and item.get("unaligned_rejected_before_write") is True
                        and math.isfinite(ms) and ms > 0):
                    if require_paired:
                        try:
                            gain = paired_gain(item)
                        except (ValueError, KeyError, TypeError):
                            choices[index] = None
                            continue
                    else:
                        gain = 1.0 - ms / base_ms
                    choices[index] = (ms, gain)
                else:
                    choices[index] = None
            by_key[key].append({"scalars": (case["alpha"], case["beta"]), "choices": choices, "base_ms": base_ms})
        for key, cases in by_key.items():
            if {(1.0, 0.0), (0.75, 0.0), (0.75, 0.25), (1.0, 1.0)}.issubset({case["scalars"] for case in cases}):
                groups[key].append((report_id, cases))
    selected = {}
    for key, runs in sorted(groups.items()):
        if len({run[0] for run in runs}) < min_reports:
            continue
        cases = [case for _, rows in runs for case in rows]
        indices = set.intersection(*(set(case["choices"]) for case in cases))
        valid = {i: [case["choices"][i] for case in cases] for i in indices
                 if all(case["choices"][i] is not None and case["choices"][i][1] >= min_gain
                        for case in cases)}
        if not valid:
            continue
        best = max(valid, key=lambda i: statistics.mean(value[1] for value in valid[i]))
        selected[key] = {"index": best, "reports": len(runs), "cases": len(cases),
                         "mean_gpu_ms": statistics.mean(v[0] for v in valid[best]),
                         "min_gain": min(v[1] for v in valid[best]),
                         "mean_gain": statistics.mean(v[1] for v in valid[best])}
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--min-reports", type=int, default=2)
    parser.add_argument("--require-paired", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.min_gain) or not 0 <= args.min_gain < 1 or args.min_reports < 1:
        parser.error("min-gain must be finite in [0,1); min-reports >= 1")
    reports, sources = [], []
    for path in args.reports:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if any(row["sha256"] == sha for row in sources):
            raise ValueError("Duplicate report cannot count as another run")
        sources.append({"path": str(path.resolve()), "sha256": sha})
        reports.append(json.loads(data))
    selected = select(reports, args.min_gain, args.min_reports, args.require_paired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# H3 Lt strict allowlist; frozen DiT/TP8 verification still required.",
             "# Exact runtime/device/shape/strides/workspace/alignment key = raw heuristic index"]
    lines += [f"{key} = {row['index']} # min_gain={row['min_gain']:.6f}, reports={row['reports']}"
              for key, row in selected.items()]
    args.output.write_text("\n".join(lines) + "\n")
    receipt = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "sources": sources,
               "selected": selected, "model_acceptance": False,
               "min_gain": args.min_gain, "min_reports": args.min_reports,
               "require_paired": args.require_paired}
    args.output.with_suffix(args.output.suffix+".json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("PHASE6_POLICY_WRITTEN", len(selected), "model_acceptance=false", flush=True)


if __name__ == "__main__":
    main()
