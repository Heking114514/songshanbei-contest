#include "infinicore/ops/group_norm.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<GroupNorm::schema> &GroupNorm::dispatcher() {
    static common::OpDispatcher<GroupNorm::schema> dispatcher_;
    return dispatcher_;
}

void GroupNorm::execute(Tensor output,
                        const Tensor &input,
                        const Tensor &weight,
                        const Tensor &bias,
                        int64_t num_groups,
                        float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight, bias);
    infinicore::context::setDevice(output->device());
    auto func = dispatcher().lookup(output->device().getType());
    if (func == nullptr) {
        throw std::runtime_error(
            "No GroupNorm implementation found for device type: "
            + std::to_string(static_cast<int>(output->device().getType())));
    }
    func(output, input, weight, bias, num_groups, epsilon);
}

Tensor group_norm(const Tensor &input,
                  int64_t num_groups,
                  const Tensor &weight,
                  const Tensor &bias,
                  float epsilon) {
    if (input->shape().size() != 4) {
        throw std::runtime_error("group_norm expects an NCHW input");
    }
    if (num_groups <= 0
        || static_cast<size_t>(num_groups) > input->size(1)
        || input->size(1) % static_cast<size_t>(num_groups) != 0) {
        throw std::runtime_error("group_norm requires channels divisible by num_groups");
    }
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    group_norm_(output, input, num_groups, weight, bias, epsilon);
    return output;
}

void group_norm_(Tensor output,
                 const Tensor &input,
                 int64_t num_groups,
                 const Tensor &weight,
                 const Tensor &bias,
                 float epsilon) {
    GroupNorm::execute(output, input, weight, bias, num_groups, epsilon);
}

} // namespace infinicore::op
