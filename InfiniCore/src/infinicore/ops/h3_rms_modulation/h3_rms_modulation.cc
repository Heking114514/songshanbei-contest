#include "infinicore/ops/h3_rms_modulation.hpp"
#include "../h3_fused_validation.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3RmsModulation);
H3RmsModulation::H3RmsModulation(
    Tensor output, Tensor residual, const Tensor &x, const Tensor &branch,
    const Tensor &weight, const Tensor &table, const Tensor &indices,
    int mode, float epsilon, int block_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, residual, x, branch, weight, table, indices);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, residual, x,
        branch, weight, table, indices, mode, epsilon, block_size);
}
void H3RmsModulation::execute(
    Tensor output, Tensor residual, const Tensor &x, const Tensor &branch,
    const Tensor &weight, const Tensor &table, const Tensor &indices,
    int mode, float epsilon, int block_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3RmsModulation, output, residual, x,
        branch, weight, table, indices, mode, epsilon, block_size);
}
std::pair<Tensor, Tensor> h3_rms_modulation(
    const Tensor &x, const Tensor &branch, const Tensor &weight,
    const Tensor &table, const Tensor &indices, int mode, float epsilon, int block_size) {
    auto output = Tensor::empty(x->shape(), x->dtype(), x->device());
    auto residual = mode == 2 ? Tensor::empty(x->shape(), x->dtype(), x->device()) : x;
    h3_rms_modulation_(output, residual, x, branch, weight, table, indices,
                       mode, epsilon, block_size);
    return {output, residual};
}
void h3_rms_modulation_(
    Tensor output, Tensor residual, const Tensor &x, const Tensor &branch,
    const Tensor &weight, const Tensor &table, const Tensor &indices,
    int mode, float epsilon, int block_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, residual, x, branch, weight, table, indices);
    if (mode < 0 || mode > 2 || !std::isfinite(epsilon) || epsilon <= 0
        || (block_size != 0 && block_size != 128 && block_size != 256
            && block_size != 512 && block_size != 1024)) {
        throw std::invalid_argument("H3 RMS fusion invalid mode/epsilon/block_size");
    }
    if (x->ndim() != 3 || x->size(0) != 1 || x->size(2) == 0 || x->size(2) > 8192
        || output->shape() != x->shape() || residual->shape() != x->shape()
        || branch->shape() != x->shape()
        || weight->ndim() != 1 || weight->size(0) != x->size(2)
        || table->ndim() != 2 || table->size(0) == 0 || table->size(1) != 6 * x->size(2)
        || indices->ndim() != 1 || indices->size(0) != x->size(1)) {
        throw std::invalid_argument("H3 RMS fusion expects [1,N,H<=8192], [H], [R,6H], [N]");
    }
    for (const auto &t : {output, residual, x, branch, table}) {
        if (t->dtype() != DataType::BF16 || !t->is_contiguous()) {
            throw std::invalid_argument("H3 RMS fusion data must be contiguous BF16");
        }
    }
    if ((weight->dtype() != DataType::BF16 && weight->dtype() != DataType::F32)
        || !weight->is_contiguous() || indices->dtype() != DataType::I64
        || !indices->is_contiguous()) {
        throw std::invalid_argument("H3 RMS fusion invalid weight/indices");
    }
    for (const auto &input : {x, branch, weight, table, indices}) {
        if (h3_experiment_detail::overlaps(output, input)
            || (mode == 2 && h3_experiment_detail::overlaps(residual, input))) {
            throw std::invalid_argument("H3 RMS fusion outputs must not alias inputs");
        }
    }
    if (h3_experiment_detail::overlaps(output, residual)) {
        throw std::invalid_argument("H3 RMS fusion outputs must be disjoint");
    }
    H3RmsModulation::execute(output, residual, x, branch, weight, table, indices,
                             mode, epsilon, block_size);
}
} // namespace infinicore::op
