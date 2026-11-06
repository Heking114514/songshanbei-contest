#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <cstdint>

namespace infinicore::op {

class ConvTranspose1d {
public:
    using schema = void (*)(Tensor,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t);
    static void execute(Tensor output,
                        const Tensor &input,
                        const Tensor &weight,
                        const Tensor &bias,
                        int64_t stride,
                        int64_t padding,
                        int64_t output_padding,
                        int64_t groups,
                        int64_t dilation);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor conv_transpose1d(const Tensor &input,
                        const Tensor &weight,
                        const Tensor &bias,
                        int64_t stride = 1,
                        int64_t padding = 0,
                        int64_t output_padding = 0,
                        int64_t groups = 1,
                        int64_t dilation = 1);
void conv_transpose1d_(Tensor output,
                       const Tensor &input,
                       const Tensor &weight,
                       const Tensor &bias,
                       int64_t stride = 1,
                       int64_t padding = 0,
                       int64_t output_padding = 0,
                       int64_t groups = 1,
                       int64_t dilation = 1);

} // namespace infinicore::op
