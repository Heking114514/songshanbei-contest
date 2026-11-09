"""Strict local validation and exact-layout promotion of phase10 evidence."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import statistics
from phase9_attention import shape_spec
from phase10_attention_worker import BEST_BUNDLE_SHA256
from phase10_profile import sha


def checked_attention(path, candidate_sha, baseline_sha):
    result = json.loads(Path(path).read_text())
    if not (result.get("complete") and result.get("screening_pass")
            and result.get("arch") == "gfx936"
            and result.get("baseline_scope") == "phase9-best-dispatch"):
        raise ValueError("Incomplete previous-best attention worker")
    if result.get("worker_source_sha256") != sha(Path(__file__).with_name("phase10_attention_worker.py")):
        raise ValueError("Worker source does not match this run")
    libs = result["libraries"]
    if libs["baseline"]["sha256"] != baseline_sha or libs["candidate"]["sha256"] != candidate_sha:
        raise ValueError("Worker library identity mismatch")
    layout = result["layout"]
    if layout not in ("legacy", "mixed-v"): raise ValueError("Unknown worker layout")
    expected = ({(65, 33, False), (708, 708, False), (27570, 27570, False)}
                if layout == "mixed-v" else
                {(q, k, p) for q, k in ((65, 33), (2048, 2048), (27570, 27570)) for p in (False, True)})
    seen = set()
    for row in result["cases"]:
        key = row["query"], row["key"], row["padded"]
        if key in seen or key not in expected: raise ValueError("Duplicate or unknown attention case")
        seen.add(key)
        spec = shape_spec(row["query"], row["key"], row["padded"])
        if layout == "mixed-v":
            spec["strides"][2] = [row["key"]*7*128*3, 7*128*3, 128, 1]
        if row["heads"] != 7 or row["spec"] != spec or not row["repeat_bitwise"]:
            raise ValueError("Layout or repeat mismatch")
        for name in ("versus_previous_best", "graph_updated_inputs"):
            metric = row[name]
            value = metric["relative_l2"]
            if not metric["finite"] or not math.isfinite(value) or not 0 <= value <= .005:
                raise ValueError("Attention numerical/Graph screening failed")
        if row["query"] == 65:
            metric = row["versus_math_small"]
            if not metric["finite"] or not math.isfinite(metric["relative_l2"]) or not 0 <= metric["relative_l2"] < .02:
                raise ValueError("Attention math screening failed")
        for name in ("baseline", "candidate", "graph"):
            timing = row[name]
            values = timing["samples_ms"]
            if len(values) < 3 or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError("Invalid attention timing")
            if not math.isclose(statistics.median(values), timing["median_ms"], rel_tol=1e-9):
                raise ValueError("Attention timing median mismatch")
        count=len(row["baseline"]["samples_ms"])
        orders=[["baseline","candidate"] if i%2==0 else ["candidate","baseline"] for i in range(count)]
        if count < 9 or len(row["candidate"]["samples_ms"]) != count or any(
                row[name].get("paired_order") != orders for name in ("baseline","candidate")):
            raise ValueError("Missing alternating paired attention evidence")
        if not math.isclose(row["speedup"], row["baseline"]["median_ms"]/row["candidate"]["median_ms"],
                            rel_tol=1e-9):
            raise ValueError("Attention speedup mismatch")
    if seen != expected: raise ValueError("Incomplete attention layout coverage")
    worst = min(r["speedup"] for r in result["cases"] if r["query"] == 27570)
    if not math.isclose(result["real_worst_speedup"], worst, rel_tol=1e-9):
        raise ValueError("Ranking differs from raw attention samples")
    return result


def promoted_policy(baseline_root, aot, reports, minimum_gain=.01):
    from phase10_attention_worker import verified_baseline
    if not math.isfinite(minimum_gain) or not .01 <= minimum_gain < 1:
        raise ValueError("Promotion gain must be finite and at least 1%")
    verified_baseline(Path(baseline_root))
    old = json.loads((Path(baseline_root)/"policy.json").read_text())
    build = json.loads((Path(aot)/"build.json").read_text())
    candidate = Path(aot)/"libh3_flash_gfx936.so"
    digest = sha(candidate)
    if not build.get("complete") or build.get("arch") != "gfx936" or build.get("library_sha256") != digest:
        raise ValueError("Incomplete or changed AOT build")
    base_sha = sha(Path(baseline_root)/"libh3_flash_dispatch.so")
    if len(reports) < 2 or len({sha(p) for p in reports}) != len(reports):
        raise ValueError("Two independent same-device remeasurements required")
    results = [checked_attention(p, digest, base_sha) for p in reports]
    if any(r["libraries"]["candidate"]["build_id"] != build["build_id"] for r in results):
        raise ValueError("Candidate build ID differs from AOT receipt")
    if any(r["layout"] != "mixed-v" for r in results) or len({r["device"] for r in results}) != 1:
        raise ValueError("Finalists must be remeasured on the same GPU and mixed-V layout")
    policy = copy.deepcopy(old)
    identity = dict(file=digest+".so", sha256=digest, build_id=build["build_id"])
    policy["libraries"][digest] = identity
    promoted = []
    for row in results[0]["cases"]:
        peers = [next(r for r in result["cases"] if r["spec"] == row["spec"]) for result in results]
        if not all(r["speedup"] >= 1+minimum_gain for r in peers):
            continue
        # Do not broaden the measured key or overwrite existing known-good rules.
        matches = [r for r in policy["rules"] if r["spec"] == row["spec"]]
        if len(matches) != 1: raise ValueError("Exact prior layout rule required")
        matches[0].update(library=digest, candidate_id="phase10-validated-layout",
                          phase10_worker_sha256=[sha(p) for p in reports],
                          phase10_min_speedup=min(r["speedup"] for r in peers))
        promoted.append(row["spec"])
    if not any(r["query_tokens"] == 27570 for r in promoted):
        raise ValueError("No independently reproduced long-layout improvement")
    policy.update(phase10_candidate=digest, phase10_promoted_specs=promoted,
                  phase10_baseline_bundle_sha256=BEST_BUNDLE_SHA256, model_acceptance=False)
    return policy
