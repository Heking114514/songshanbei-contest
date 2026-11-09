"""Bracket Lt candidates with local BLAS baselines; pure scheduling/validation.

GPU callbacks reset C before recording events. Correctness checks run outside
those events. No GPU imports: the sequence and drift rejection are CPU tested.
"""
from __future__ import annotations
import math
import statistics


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def bracketed(baseline, candidate, baseline_check, candidate_check, repeats, warmup=3):
    if repeats < 3 or warmup < 3:
        raise ValueError("At least 3 warmups and 3 paired repeats required")
    for _ in range(warmup):
        baseline()
        if not baseline_check():
            raise ValueError("Baseline changed during warmup")
        candidate()
        if not candidate_check():
            raise ValueError("Candidate changed during warmup")
    samples = []
    for _ in range(repeats):
        before, _ = baseline()
        baseline_ok = bool(baseline_check())
        gpu_ms, wall = candidate()
        candidate_ok = bool(candidate_check())
        after, _ = baseline()
        baseline_ok &= bool(baseline_check())
        samples.append(dict(baseline_before_ms=before, candidate_gpu_ms=gpu_ms,
                            baseline_after_ms=after, candidate_wall_seconds=wall,
                            baseline_bitwise=baseline_ok, candidate_bitwise=candidate_ok))
    return dict(schema=1, warmup=warmup, samples=samples,
                reference="min of adjacent BLAS baseline measurements")


def paired_gain(item, max_drift=0.10):
    pair = item.get("paired", {})
    samples = pair.get("samples", [])
    if pair.get("schema") != 1 or pair.get("warmup", 0) < 3 or len(samples) < 3:
        raise ValueError("Missing completed paired timing")
    gains, values = [], []
    for row in samples:
        before, after, candidate = (row[k] for k in
            ("baseline_before_ms", "baseline_after_ms", "candidate_gpu_ms"))
        if not all(positive(x) for x in (before, after, candidate, row["candidate_wall_seconds"])):
            raise ValueError("Invalid paired timing")
        if row.get("baseline_bitwise") is not True or row.get("candidate_bitwise") is not True:
            raise ValueError("Paired timing changed output bits")
        if max(before, after)/min(before, after)-1 > max_drift:
            raise ValueError("Adjacent baselines drifted beyond the budget")
        gains.append(1 - candidate/min(before, after))
        values.append(candidate)
    if not math.isclose(statistics.median(values), item["median_gpu_ms"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Paired samples do not match candidate median")
    return statistics.median(gains)
