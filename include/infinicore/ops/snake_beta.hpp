#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class SnakeBeta {
public:
    using schema = void (*)(Tensor,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &);
    static void execute(Tensor output,
                        const Tensor &input,
                        const Tensor &alpha_log,
                        const Tensor &beta_log);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor snake_beta(const Tensor &input,
                  const Tensor &alpha_log,
                  const Tensor &beta_log);
void snake_beta_(Tensor output,
                 const Tensor &input,
                 const Tensor &alpha_log,
                 const Tensor &beta_log);

} // namespace infinicore::op
