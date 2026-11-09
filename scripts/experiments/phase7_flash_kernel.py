"""Full noncausal BF16 attention; explicitly AOT-compiled for gfx936/wave64.

Uses matrix dot operations and online softmax; no SxS global score/probability
tensor. FP32 QK/softmax/accumulator, BF16 probabilities at the PV matrix dot.
Changes arithmetic relative to ATen math_bf16; numerical validation is required.
"""
import triton
import triton.language as tl


@triton.jit
def h3_flash_gfx936(
    Q, K, V, O,
    SQ: tl.int32, SK: tl.int32, H: tl.int32, SCALE: tl.float32,
    QB: tl.int64, QS: tl.int64, QH: tl.int64,
    KB: tl.int64, KS: tl.int64, KH: tl.int64,
    VB: tl.int64, VS: tl.int64, VH: tl.int64,
    OB: tl.int64, OS: tl.int64, OH: tl.int64,
    BM: tl.constexpr, BN: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, 128)
    cols = tl.arange(0, BN)
    query = tl.load(Q + batch * QB + rows[:, None] * QS + head * QH + dims[None, :],
                    mask=(rows[:, None] < SQ) & (head < H), other=0)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, 128), tl.float32)
    for first_key in range(0, tl.cdiv(SK, BN)):
        keys = first_key * BN + cols
        key = tl.load(K + batch * KB + keys[None, :] * KS + head * KH + dims[:, None],
                      mask=(keys[None, :] < SK) & (head < H), other=0)
        score = tl.dot(query, key) * (SCALE * 1.4426950408889634)
        score = tl.where(keys[None, :] < SK, score, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(score, axis=1))
        probability = tl.exp2(score - next_maximum[:, None])
        correction = tl.exp2(maximum - next_maximum)
        value = tl.load(V + batch * VB + keys[:, None] * VS + head * VH + dims[None, :],
                        mask=(keys[:, None] < SK) & (head < H), other=0)
        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(probability.to(tl.bfloat16), value, accumulator)
        denominator = denominator * correction + tl.sum(probability, axis=1)
        maximum = next_maximum
    output = accumulator / denominator[:, None]
    tl.store(O + batch * OB + rows[:, None] * OS + head * OH + dims[None, :],
             output.to(tl.bfloat16), mask=(rows[:, None] < SQ) & (head < H))
