#include "infinicore/ops/conv1d.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Conv1d::schema> &Conv1d::dispatcher() {
    static common::OpDispatcher<Conv1d::schema> dispatcher_;
    return dispatcher_;
}

void Conv1d::execute(Tensor output,
                     const Tensor &input,
                     const Tensor &weight,
                     const Tensor &bias,
                     int64_t stride,
                     int64_t padding,
                     int64_t dilation,
                     int64_t groups) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight);
    if (bias->numel() != 0) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, bias);
    }
    infinicore::context::setDevice(output->device());
    auto func = dispatcher().lookup(output->device().getType());
    if (func == nullptr) {
        throw std::runtime_error(
            "No Conv1d implementation found for device type: "
            + std::to_string(static_cast<int>(output->device().getType())));
    }
    func(output, input, weight, bias, stride, padding, dilation, groups);
}

Tensor conv1d(const Tensor &input,
              const Tensor &weight,
              const Tensor &bias,
              int64_t stride,
              int64_t padding,
              int64_t dilation,
              int64_t groups) {
    if (input->shape().size() != 3 || weight->shape().size() != 3) {
        throw std::runtime_error("conv1d expects 3D input and weight");
    }
    if (stride <= 0 || dilation <= 0 || groups <= 0 || padding < 0) {
        throw std::runtime_error("conv1d received invalid convolution parameters");
    }
    if (input->size(1) % static_cast<size_t>(groups) != 0
        || weight->size(1) != input->size(1) / static_cast<size_t>(groups)) {
        throw std::runtime_error("conv1d input/weight/groups are incompatible");
    }
    const int64_t input_length = static_cast<int64_t>(input->size(2));
    const int64_t kernel = static_cast<int64_t>(weight->size(2));
    const int64_t output_length =
        (input_length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    if (output_length <= 0) {
        throw std::runtime_error("conv1d produced a non-positive output length");
    }
    const size_t output_channels = weight->size(0);
    if (bias->numel() != 0 && bias->numel() != output_channels) {
        throw std::runtime_error("conv1d bias must have output_channels elements");
    }
    auto output = Tensor::empty(
        {input->size(0), output_channels, static_cast<size_t>(output_length)},
        input->dtype(),
        input->device());
    conv1d_(output, input, weight, bias, stride, padding, dilation, groups);
    return output;
}

void conv1d_(Tensor output,
             const Tensor &input,
             const Tensor &weight,
             const Tensor &bias,
             int64_t stride,
             int64_t padding,
             int64_t dilation,
             int64_t groups) {
    Conv1d::execute(output, input, weight, bias, stride, padding, dilation, groups);
}

} // namespace infinicore::op
