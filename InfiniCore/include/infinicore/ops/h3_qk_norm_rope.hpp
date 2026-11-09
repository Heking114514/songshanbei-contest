#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <utility>

namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(H3QKNormRoPE, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float, float);
void h3_qk_norm_rope_(Tensor q_out, Tensor k_out, const Tensor & q, const Tensor & k, const Tensor & q_weight, const Tensor & k_weight, const Tensor & cache, const Tensor & positions, float q_epsilon, float k_epsilon);
// BF16 q/k [N,heads,128], unit last stride, packed BF16 [R,rotary_dim]
// cos/sin cache, I64 [N] positions; outputs compact and disjoint.
std::pair<Tensor, Tensor> h3_qk_norm_rope(
    const Tensor &q, const Tensor &k, const Tensor &q_weight,
    const Tensor &k_weight, const Tensor &cache, const Tensor &positions,
    float q_epsilon, float k_epsilon);
} // namespace infinicore::op
