#include "infinicore/ops/snake_beta.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<SnakeBeta::schema> &SnakeBeta::dispatcher() {
    static common::OpDispatcher<SnakeBeta::schema> dispatcher_;
    return dispatcher_;
}

void SnakeBeta::execute(Tensor output,
                        const Tensor &input,
                        const Tensor &alpha_log,
                        const Tensor &beta_log) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, alpha_log, beta_log);
    infinicore::context::setDevice(output->device());
    auto func = dispatcher().lookup(output->device().getType());
    if (func == nullptr) {
        throw std::runtime_error(
            "No SnakeBeta implementation found for device type: "
            + std::to_string(static_cast<int>(output->device().getType())));
    }
    func(output, input, alpha_log, beta_log);
}

Tensor snake_beta(const Tensor &input,
                  const Tensor &alpha_log,
                  const Tensor &beta_log) {
    if (input->shape().size() != 3) {
        throw std::runtime_error("snake_beta expects a [batch, channels, length] input");
    }
    const size_t channels = input->size(1);
    if (alpha_log->numel() != channels || beta_log->numel() != channels) {
        throw std::runtime_error("snake_beta alpha_log and beta_log must have channels elements");
    }
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    snake_beta_(output, input, alpha_log, beta_log);
    return output;
}

void snake_beta_(Tensor output,
                 const Tensor &input,
                 const Tensor &alpha_log,
                 const Tensor &beta_log) {
    SnakeBeta::execute(output, input, alpha_log, beta_log);
}

} // namespace infinicore::op
