#include "infinicore/ops/h3_modulation.hpp"

#include "../../utils.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3Modulation);

H3Modulation::H3Modulation(Tensor output, const Tensor &x, const Tensor &branch,
                           const Tensor &table, const Tensor &indices, int mode) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, x, branch, table, indices);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, x, branch, table, indices, mode);
}

void H3Modulation::execute(Tensor output, const Tensor &x, const Tensor &branch,
                           const Tensor &table, const Tensor &indices, int mode) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        H3Modulation, output, x, branch, table, indices, mode);
}

Tensor h3_modulation(const Tensor &x, const Tensor &branch,
                     const Tensor &table, const Tensor &indices, int mode) {
    auto output = Tensor::empty(x->shape(), x->dtype(), x->device());
    h3_modulation_(output, x, branch, table, indices, mode);
    return output;
}

void h3_modulation_(Tensor output, const Tensor &x, const Tensor &branch,
                    const Tensor &table, const Tensor &indices, int mode) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, x, branch, table, indices);
    if (mode < 0 || mode > 3) {
        throw std::invalid_argument("h3_modulation mode must be in [0,3]");
    }
    if (x->ndim() != 3 || x->size(0) != 1 || x->size(2) == 0
        || x->size(2) > std::numeric_limits<size_t>::max() / 6
        || output->shape() != x->shape() || branch->shape() != x->shape()
        || table->ndim() != 2 || table->size(0) == 0
        || table->size(1) != 6 * x->size(2)
        || indices->ndim() != 1 || indices->size(0) != x->size(1)) {
        throw std::invalid_argument("h3_modulation expects [1,N,H], [R,6H], [N]");
    }
    if (x->dtype() != DataType::BF16 || output->dtype() != DataType::BF16
        || branch->dtype() != DataType::BF16 || table->dtype() != DataType::BF16
        || indices->dtype() != DataType::I64) {
        throw std::invalid_argument("h3_modulation requires BF16 data and I64 indices");
    }
    if (!output->is_contiguous() || !x->is_contiguous() || !branch->is_contiguous()
        || !table->is_contiguous() || !indices->is_contiguous()) {
        throw std::invalid_argument("h3_modulation requires contiguous tensors");
    }
    // Contiguous storage ranges: reject even partial aliases in the public API.
    const auto overlaps_output = [&](const Tensor &input) {
        const auto out = reinterpret_cast<uintptr_t>(output->data());
        const auto in = reinterpret_cast<uintptr_t>(input->data());
        const auto out_bytes = output->numel() * sizeof(uint16_t);
        const auto in_bytes = input->numel()
            * (input->dtype() == DataType::I64 ? sizeof(int64_t) : sizeof(uint16_t));
        return out < in ? in - out < out_bytes : out - in < in_bytes;
    };
    if (output->numel() > 0 && (overlaps_output(x) || overlaps_output(branch)
        || overlaps_output(table) || overlaps_output(indices))) {
        throw std::invalid_argument("h3_modulation output must not alias inputs");
    }
    H3Modulation::execute(output, x, branch, table, indices, mode);
}

} // namespace infinicore::op
