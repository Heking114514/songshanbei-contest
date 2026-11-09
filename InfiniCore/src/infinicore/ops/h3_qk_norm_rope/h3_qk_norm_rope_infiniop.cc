#include "../infiniop_impl.hpp"
#include "infiniop/ops/h3_qk_norm_rope.h"
#include "infinicore/ops/h3_qk_norm_rope.hpp"

namespace infinicore::op::h3_qk_norm_rope_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, H3QKNormRoPE, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor q_out, k_out, q, k, q_weight, k_weight, cache, positions;
};
void *plan(Tensor q_out, Tensor k_out, const Tensor & q, const Tensor & k, const Tensor & q_weight, const Tensor & k_weight, const Tensor & cache, const Tensor & positions, float q_epsilon, float k_epsilon) {
    const size_t seed = hash_combine(q_out, k_out, q, k, q_weight, k_weight, cache, positions, q_epsilon, k_epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, H3QKNormRoPE, seed,
        q_out->desc(), k_out->desc(), q->desc(), k->desc(), q_weight->desc(), k_weight->desc(), cache->desc(), positions->desc(), q_epsilon, k_epsilon);
    return new PlannedMeta{descriptor, graph::GraphTensor(q_out), graph::GraphTensor(k_out), graph::GraphTensor(q), graph::GraphTensor(k), graph::GraphTensor(q_weight), graph::GraphTensor(k_weight), graph::GraphTensor(cache), graph::GraphTensor(positions)};
}
void run(void *meta) {
    auto p = reinterpret_cast<PlannedMeta *>(meta);
    INFINICORE_CHECK_ERROR(infiniopH3QKNormRoPE(p->descriptor->desc,
        p->q_out->data(), p->k_out->data(), p->q->data(), p->k->data(), p->q_weight->data(), p->k_weight->data(), p->cache->data(), p->positions->data(), context::getStream()));
}
void cleanup(void **meta) {
    delete *reinterpret_cast<PlannedMeta **>(meta);
    *meta = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(H3QKNormRoPE, &plan, &run, &cleanup);
} // namespace infinicore::op::h3_qk_norm_rope_impl::infiniop
