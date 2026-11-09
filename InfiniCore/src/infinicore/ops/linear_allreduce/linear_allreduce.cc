#include "infinicore/ops/linear_allreduce.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_HYGON_API
#include "h3_overlap_hygon.hpp"
#endif

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearAllReduce);

LinearAllReduce::LinearAllReduce(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight);
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, *bias);
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, weight, bias, communicator);
}

void LinearAllReduce::execute(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        LinearAllReduce, output, input, weight, bias, communicator);
}

static Tensor linear_allreduce_impl(
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator,
    bool weight_is_packed) {
    Size in_features = weight->shape()[weight_is_packed ? 0 : 1];
    Size out_features = weight->shape()[weight_is_packed ? 1 : 0];
    auto output_shape = input->shape();
    output_shape.back() = out_features;

#ifdef ENABLE_HYGON_API
    const size_t overlap_rows = h3_overlap::chunk_rows();
    if (overlap_rows && input->device().getType() == Device::Type::HYGON
        && input->dtype() == DataType::BF16 && weight->dtype() == DataType::BF16
        && !bias && communicator && input->is_contiguous() && weight->is_contiguous()
        && input->ndim() >= 2 && weight->ndim() == 2
        && in_features > 0 && input->shape().back() == in_features) {
        Size tokens = input->numel() / in_features;
        // H3 long-sequence row projections. Short condition encoding remains old.
        if (tokens >= 4096 && tokens > overlap_rows && out_features >= 128) {
            auto output = Tensor::empty(output_shape, input->dtype(), input->device());
            auto weight_matrix = weight_is_packed ? weight : weight->permute({1, 0});
            h3_overlap::execute(output->view({tokens, out_features}),
                                input->view({tokens, in_features}),
                                weight_matrix, communicator, overlap_rows);
            return output;
        }
    }
#endif

    const bool aclnn_supported = input->dtype() == DataType::F16 || input->dtype() == DataType::BF16;
    if (input->device().getType() == Device::Type::ASCEND && aclnn_supported) {
        auto output = Tensor::empty(output_shape, input->dtype(), input->device());
        Size rows = 1;
        for (Size i = 0; i + 1 < input->ndim(); ++i) {
            rows *= input->size(i);
        }
        auto input_matrix = input->view({rows, in_features});
        auto output_matrix = output->view({rows, out_features});
        auto weight_matrix = weight_is_packed
                               ? weight
                               : weight->permute({1, 0});
        LinearAllReduce::execute(
            output_matrix, input_matrix, weight_matrix, bias, communicator);
        return output;
    }

    auto output = weight_is_packed
                    ? linear_packed(input, weight, bias)
                    : linear(input, weight, bias);
    distributed::allreduce_(
        output, output, INFINICCL_SUM, communicator);
    return output;
}

Tensor linear_allreduce(
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    return linear_allreduce_impl(
        input, weight, bias, communicator, false);
}

Tensor linear_allreduce_packed(
    Tensor input,
    Tensor packed_weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    return linear_allreduce_impl(
        input, packed_weight, bias, communicator, true);
}

} // namespace infinicore::op
