#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// BF16 [1,N,H] x/branch, BF16 [R,6H] table, I64 [N] indices.
// Modes: 0 MSA AdaLN, 1 MSA gate, 2 MLP AdaLN, 3 MLP gate.
// AdaLN ignores branch. All tensors must be contiguous on NVIDIA/HYGON.
// Every index must be in [0,R); invalid device indices trap asynchronously.
// Output is freshly allocated; the underscore form forbids all output aliases.
INFINICORE_GRAPH_OP_CLASS(H3Modulation, Tensor, const Tensor &, const Tensor &,
                         const Tensor &, const Tensor &, int);

Tensor h3_modulation(const Tensor &x, const Tensor &branch,
                     const Tensor &table, const Tensor &indices, int mode);
void h3_modulation_(Tensor output, const Tensor &x, const Tensor &branch,
                    const Tensor &table, const Tensor &indices, int mode);

} // namespace infinicore::op
