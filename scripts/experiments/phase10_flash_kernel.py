"""Structural gfx936 BF16 candidate: unmasked full KV blocks plus one masked tail.

Keeps the phase7 online-softmax arithmetic and ABI. This is a new, unvalidated
AOT variant; the verified kernel and dispatcher remain unchanged. Optional V
prefetch trades register lifetime for load/compute overlap, so measure it.
"""
import triton
import triton.language as tl


@triton.jit
def _update(query, key, value, maximum, denominator, accumulator,
            keys, SK, SCALE, MASK_TAIL: tl.constexpr):
    score = tl.dot(query, key) * (SCALE * 1.4426950408889634)
    if MASK_TAIL:
        score = tl.where(keys[None, :] < SK, score, float("-inf"))
    next_maximum = tl.maximum(maximum, tl.max(score, axis=1))
    probability = tl.exp2(score - next_maximum[:, None])
    correction = tl.exp2(maximum - next_maximum)
    accumulator = accumulator * correction[:, None]
    accumulator = tl.dot(probability.to(tl.bfloat16), value, accumulator)
    denominator = denominator * correction + tl.sum(probability, axis=1)
    return next_maximum, denominator, accumulator


@triton.jit
def h3_flash_gfx936(
    Q, K, V, O,
    SQ: tl.int32, SK: tl.int32, H: tl.int32, SCALE: tl.float32,
    QB: tl.int64, QS: tl.int64, QH: tl.int64,
    KB: tl.int64, KS: tl.int64, KH: tl.int64,
    VB: tl.int64, VS: tl.int64, VH: tl.int64,
    OB: tl.int64, OS: tl.int64, OH: tl.int64,
    BM: tl.constexpr, BN: tl.constexpr, PREFETCH_V: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, 128)
    cols = tl.arange(0, BN)
    query = tl.load(Q + batch * QB + rows[:, None] * QS + head * QH + dims[None, :],
                    mask=rows[:, None] < SQ, other=0)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, 128), tl.float32)
    # The ABI validates a positive SK and launches exactly H heads. Neither a
    # head mask nor a key tail mask is needed in the complete-block loop.
    for block in range(0, SK // BN):
        keys = block * BN + cols
        key = tl.load(K + batch * KB + keys[None, :] * KS + head * KH + dims[:, None])
        if PREFETCH_V:
            value = tl.load(V + batch * VB + keys[:, None] * VS + head * VH + dims[None, :])
            maximum, denominator, accumulator = _update(
                query, key, value, maximum, denominator, accumulator,
                keys, SK, SCALE, False)
        else:
            # Preserve the late V load of the verified implementation.
            score = tl.dot(query, key) * (SCALE * 1.4426950408889634)
            next_maximum = tl.maximum(maximum, tl.max(score, axis=1))
            probability = tl.exp2(score - next_maximum[:, None])
            correction = tl.exp2(maximum - next_maximum)
            value = tl.load(V + batch * VB + keys[:, None] * VS + head * VH + dims[None, :])
            accumulator = accumulator * correction[:, None]
            accumulator = tl.dot(probability.to(tl.bfloat16), value, accumulator)
            denominator = denominator * correction + tl.sum(probability, axis=1)
            maximum = next_maximum
    # This uniform branch executes once, and also handles SK < BN correctly.
    if SK % BN != 0:
        keys = (SK // BN) * BN + cols
        key = tl.load(K + batch * KB + keys[None, :] * KS + head * KH + dims[:, None],
                      mask=keys[None, :] < SK, other=0)
        value = tl.load(V + batch * VB + keys[:, None] * VS + head * VH + dims[None, :],
                        mask=keys[:, None] < SK, other=0)
        maximum, denominator, accumulator = _update(
            query, key, value, maximum, denominator, accumulator,
            keys, SK, SCALE, True)
    output = accumulator / denominator[:, None]
    tl.store(O + batch * OB + rows[:, None] * OS + head * OH + dims[None, :],
             output.to(tl.bfloat16), mask=rows[:, None] < SQ)
