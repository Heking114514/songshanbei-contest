#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <utility>

namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(H3RmsModulation, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int, float, int);
void h3_rms_modulation_(Tensor output, Tensor residual, const Tensor & x, const Tensor & branch, const Tensor & weight, const Tensor & table, const Tensor & indices, int mode, float epsilon, int block_size);
// Pair is {modulated, residual}. Modes 0/1 ignore branch and return x as
// residual without writing it; mode2 emits attention-gated residual plus MLP
// RMS/AdaLN. block_size=0 matches the original RMS launch policy.
std::pair<Tensor, Tensor> h3_rms_modulation(
    const Tensor &x, const Tensor &branch, const Tensor &weight,
    const Tensor &table, const Tensor &indices, int mode,
    float epsilon, int block_size = 0);
} // namespace infinicore::op
