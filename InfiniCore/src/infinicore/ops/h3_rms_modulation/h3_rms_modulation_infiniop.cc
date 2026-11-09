#include "../infiniop_impl.hpp"
#include "infiniop/ops/h3_rms_modulation.h"
#include "infinicore/ops/h3_rms_modulation.hpp"

namespace infinicore::op::h3_rms_modulation_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, H3RmsModulation, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor output, residual, x, branch, weight, table, indices;
};
void *plan(Tensor output, Tensor residual, const Tensor & x, const Tensor & branch, const Tensor & weight, const Tensor & table, const Tensor & indices, int mode, float epsilon, int block_size) {
    const size_t seed = hash_combine(output, residual, x, branch, weight, table, indices, mode, epsilon, block_size);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, H3RmsModulation, seed,
        output->desc(), residual->desc(), x->desc(), branch->desc(), weight->desc(), table->desc(), indices->desc(), mode, epsilon, block_size);
    return new PlannedMeta{descriptor, graph::GraphTensor(output), graph::GraphTensor(residual), graph::GraphTensor(x), graph::GraphTensor(branch), graph::GraphTensor(weight), graph::GraphTensor(table), graph::GraphTensor(indices)};
}
void run(void *meta) {
    auto p = reinterpret_cast<PlannedMeta *>(meta);
    INFINICORE_CHECK_ERROR(infiniopH3RmsModulation(p->descriptor->desc,
        p->output->data(), p->residual->data(), p->x->data(), p->branch->data(), p->weight->data(), p->table->data(), p->indices->data(), context::getStream()));
}
void cleanup(void **meta) {
    delete *reinterpret_cast<PlannedMeta **>(meta);
    *meta = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(H3RmsModulation, &plan, &run, &cleanup);
} // namespace infinicore::op::h3_rms_modulation_impl::infiniop
