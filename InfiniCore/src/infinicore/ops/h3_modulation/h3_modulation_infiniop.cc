#include "../infiniop_impl.hpp"
#include "infiniop/ops/h3_modulation.h"
#include "infinicore/ops/h3_modulation.hpp"

namespace infinicore::op::h3_modulation_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, H3Modulation, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor output, x, branch, table, indices;
};

void *plan(Tensor output, const Tensor &x, const Tensor &branch,
           const Tensor &table, const Tensor &indices, int mode) {
    // Cache metadata and mode only. Indices/table contents are read every run.
    size_t seed = hash_combine(output, x, branch, table, indices, mode);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, H3Modulation, seed,
        output->desc(), x->desc(), branch->desc(), table->desc(), indices->desc(), mode);
    return new PlannedMeta{
        descriptor, graph::GraphTensor(output), graph::GraphTensor(x),
        graph::GraphTensor(branch), graph::GraphTensor(table), graph::GraphTensor(indices)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopH3Modulation(
        planned->descriptor->desc, planned->output->data(), planned->x->data(),
        planned->branch->data(), planned->table->data(), planned->indices->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(H3Modulation, &plan, &run, &cleanup);

} // namespace infinicore::op::h3_modulation_impl::infiniop
