#include "infinicore/ops/h3_qk_norm_rope.hpp"
#include "../h3_fused_validation.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3QKNormRoPE);
H3QKNormRoPE::H3QKNormRoPE(
    Tensor q_out, Tensor k_out, const Tensor &q, const Tensor &k,
    const Tensor &q_weight, const Tensor &k_weight, const Tensor &cache,
    const Tensor &positions, float q_epsilon, float k_epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_out, k_out, q, k, q_weight, k_weight, cache, positions);
    INFINICORE_GRAPH_OP_DISPATCH(q_out->device().getType(), q_out, k_out, q, k,
        q_weight, k_weight, cache, positions, q_epsilon, k_epsilon);
}
void H3QKNormRoPE::execute(
    Tensor q_out, Tensor k_out, const Tensor &q, const Tensor &k,
    const Tensor &q_weight, const Tensor &k_weight, const Tensor &cache,
    const Tensor &positions, float q_epsilon, float k_epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3QKNormRoPE, q_out, k_out, q, k,
        q_weight, k_weight, cache, positions, q_epsilon, k_epsilon);
}
std::pair<Tensor, Tensor> h3_qk_norm_rope(
    const Tensor &q, const Tensor &k, const Tensor &q_weight, const Tensor &k_weight,
    const Tensor &cache, const Tensor &positions, float q_epsilon, float k_epsilon) {
    auto qo = Tensor::empty(q->shape(), q->dtype(), q->device());
    auto ko = Tensor::empty(k->shape(), k->dtype(), k->device());
    h3_qk_norm_rope_(qo, ko, q, k, q_weight, k_weight, cache, positions, q_epsilon, k_epsilon);
    return {qo, ko};
}
void h3_qk_norm_rope_(
    Tensor q_out, Tensor k_out, const Tensor &q, const Tensor &k,
    const Tensor &q_weight, const Tensor &k_weight, const Tensor &cache,
    const Tensor &positions, float q_epsilon, float k_epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_out, k_out, q, k, q_weight, k_weight, cache, positions);
    if (!std::isfinite(q_epsilon) || !std::isfinite(k_epsilon)
        || q_epsilon <= 0 || k_epsilon <= 0) {
        throw std::invalid_argument("H3 QK fusion epsilon must be finite and positive");
    }
    if (q->ndim() != 3 || q->size(1) == 0 || q->size(1) > 64 || q->size(2) != 128
        || k->shape() != q->shape() || q_out->shape() != q->shape() || k_out->shape() != k->shape()
        || cache->ndim() != 2 || cache->size(0) == 0 || cache->size(1) == 0
        || cache->size(1) > 128 || cache->size(1) % 6 != 0
        || positions->ndim() != 1 || positions->size(0) != q->size(0)
        || q_weight->shape() != Shape{128} || k_weight->shape() != Shape{128}) {
        throw std::invalid_argument("H3 QK fusion expects [N,heads<=64,128], [128], [R,rotary_dim], [N]");
    }
    for (const auto &t : {q_out, k_out, q, k, cache}) {
        if (t->dtype() != DataType::BF16) {
            throw std::invalid_argument("H3 QK fusion data must be BF16");
        }
    }
    for (const auto &t : {q, k}) {
        if (t->strides()[2] != 1 || t->strides()[1] < 128
            || t->strides()[0] < static_cast<Stride>(t->size(1)) * t->strides()[1]) {
            throw std::invalid_argument("H3 QK fusion invalid QKV strides");
        }
    }
    if (!q_out->is_contiguous() || !k_out->is_contiguous() || !cache->is_contiguous()
        || positions->dtype() != DataType::I64 || !positions->is_contiguous()
        || q_weight->dtype() != k_weight->dtype()
        || (q_weight->dtype() != DataType::BF16 && q_weight->dtype() != DataType::F32)
        || !q_weight->is_contiguous() || !k_weight->is_contiguous()) {
        throw std::invalid_argument("H3 QK fusion invalid output/cache/weight/positions");
    }
    for (const auto &out : {q_out, k_out}) {
        for (const auto &in : {q, k, q_weight, k_weight, cache, positions}) {
            if (h3_experiment_detail::overlaps(out, in)) {
                throw std::invalid_argument("H3 QK fusion output must not alias inputs");
            }
        }
    }
    if (h3_experiment_detail::overlaps(q_out, k_out)) {
        throw std::invalid_argument("H3 QK fusion outputs must be disjoint");
    }
    H3QKNormRoPE::execute(q_out, k_out, q, k, q_weight, k_weight, cache, positions, q_epsilon, k_epsilon);
}
} // namespace infinicore::op
