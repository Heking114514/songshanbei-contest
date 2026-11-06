#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <cstdint>

namespace infinicore::op {

class GroupNorm {
public:
    using schema = void (*)(Tensor,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            int64_t,
                            float);
    static void execute(Tensor output,
                        const Tensor &input,
                        const Tensor &weight,
                        const Tensor &bias,
                        int64_t num_groups,
                        float epsilon);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor group_norm(const Tensor &input,
                  int64_t num_groups,
                  const Tensor &weight,
                  const Tensor &bias,
                  float epsilon = 1e-5f);
void group_norm_(Tensor output,
                 const Tensor &input,
                 int64_t num_groups,
                 const Tensor &weight,
                 const Tensor &bias,
                 float epsilon = 1e-5f);

} // namespace infinicore::op
