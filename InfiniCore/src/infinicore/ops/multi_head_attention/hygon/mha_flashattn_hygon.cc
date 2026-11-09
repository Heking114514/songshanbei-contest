#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/mha.hpp"
#include "h3_flash_loader.hpp"
#include "infinicore/context/context.hpp"

#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/hip/HIPGuard.h>

// Explicit marker used by the experiment launcher to reject stable libraries.
extern "C" __attribute__((visibility("default")))
int infinicore_h3_query_chunk_experiment_v1() { return 1; }

extern "C" __attribute__((visibility("default")))
int infinicore_h3_batch_optimization_v2() { return 2; }

namespace infinicore::op::mha_impl::flashattn {
namespace {
int64_t experimental_size(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    for (const char *current = value; *current != '\0'; ++current) {
        if (*current < '0' || *current > '9') {
            throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
        }
    }
    try {
        return std::stoll(value);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string(name) + " is out of range");
    }
}
} // namespace
struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
    int64_t query_chunk;
    int64_t head_batch;
    std::shared_ptr<h3_flash::Plan> flash;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {

    const int64_t query_chunk = experimental_size("H3_EXPERIMENT_MHA_QUERY_CHUNK");
    const int64_t head_batch = experimental_size("H3_EXPERIMENT_MHA_HEAD_BATCH");
    if ((query_chunk > 0 || head_batch > 0) && (is_causal || alibi_slopes.has_value())) {
        throw std::invalid_argument("Experimental attention tiling supports noncausal attention without ALiBi only");
    }
    std::shared_ptr<h3_flash::Plan> flash;
    const bool flash_scope = !is_causal && !alibi_slopes
        && q->ndim() == 4 && k->ndim() == 4 && v->ndim() == 4 && out->ndim() == 4
        && q->dtype() == DataType::BF16 && k->dtype() == DataType::BF16
        && v->dtype() == DataType::BF16 && out->dtype() == DataType::BF16
        && q->size(3) == 128 && k->size(3) == 128 && v->size(3) == 128
        && q->size(2) == k->size(2);
    if (h3_flash::enabled() && flash_scope) {
        // Other model attention (causal/GQA/D=72) retains its existing backend.
        // An eligible H3 AOT plan or launch failure never falls back to math.
        if (q->device() != context::getDevice()
            || query_chunk || head_batch || !std::isfinite(scale)
            || out->shape() != q->shape() || k->shape() != v->shape()
            || q->size(0) != k->size(0)) {
            throw std::invalid_argument("H3 gfx936 flash requires noncausal BF16 BSH128 with equal heads");
        }
        H3FlashSpec spec{};
        spec.batch = q->size(0);
        spec.query_tokens = q->size(1);
        spec.key_tokens = k->size(1);
        spec.heads = q->size(2);
        const Tensor tensors[] = {q, k, v, out};
        for (int tensor = 0; tensor < 4; ++tensor) {
            if (tensors[tensor]->device() != q->device()) {
                throw std::invalid_argument("H3 flash tensors on different devices");
            }
            for (int axis = 0; axis < 4; ++axis) {
                spec.strides[tensor][axis] = tensors[tensor]->stride(axis);
            }
        }
        flash = std::make_shared<h3_flash::Plan>(spec);
    }
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        is_causal,
        query_chunk,
        head_batch,
        std::move(flash)};
}

void run(void *planned_meta) {
#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    if (p->flash) {
        H3FlashCall call{p->q->data(), p->k->data(), p->v->data(), p->out->data(), p->scale};
        p->flash->run(call, context::getStream());
        return;
    }

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto scale = p->scale;
    auto is_causal = p->is_causal;

    const int64_t num_q_heads = q.size(2);
    const int64_t num_kv_heads = k.size(2);
    if (q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && q.size(1) == 1 && k.size(1) == 1 && v.size(1) == 1) {
        at::Tensor value = v;
        if (num_kv_heads != num_q_heads) {
            const int64_t repeat = num_q_heads / num_kv_heads;
            value = v.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
        }
        out_work.copy_(value);
        if (out_need_copy_back) {
            p->out->copy_from(out_work_ic);
        }
        return;
    }

    at::Tensor k_attn = k;
    at::Tensor v_attn = v;
    if (num_kv_heads != num_q_heads) {
        const int64_t repeat = num_q_heads / num_kv_heads;
        k_attn = k.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
        v_attn = v.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
    }

    const bool split_queries = p->query_chunk > 0 && p->query_chunk < q.size(1);
    const bool split_heads = p->head_batch > 0 && p->head_batch < num_q_heads;
    if (!split_queries && !split_heads) {
        auto result = at::scaled_dot_product_attention(
            q.permute({0, 2, 1, 3}),
            k_attn.permute({0, 2, 1, 3}),
            v_attn.permute({0, 2, 1, 3}),
            std::nullopt,
            0.0,
            is_causal,
            std::optional<double>(static_cast<double>(scale)));
        out_work.copy_(result.permute({0, 2, 1, 3}));
    } else {
        static std::once_flag announcement;
        std::call_once(announcement, [&]() {
            std::cerr << "H3_EXPERIMENT_NATIVE_MHA query_chunk=" << p->query_chunk
                      << " head_batch=" << p->head_batch
                      << " full_key_sequence=true" << std::endl;
        });
        auto query = q.permute({0, 2, 1, 3});
        auto key = k_attn.permute({0, 2, 1, 3});
        auto value = v_attn.permute({0, 2, 1, 3});
        // Independent heads may be grouped without changing the softmax domain.
        // Query slices always retain ALL key/value positions.
        const int64_t query_step = split_queries ? p->query_chunk : q.size(1);
        const int64_t head_step = split_heads ? p->head_batch : num_q_heads;
        for (int64_t first_head = 0; first_head < num_q_heads; first_head += head_step) {
            const int64_t head_count = std::min(head_step, num_q_heads - first_head);
            auto query_heads = query.narrow(1, first_head, head_count);
            auto key_heads = key.narrow(1, first_head, head_count);
            auto value_heads = value.narrow(1, first_head, head_count);
            for (int64_t first_query = 0; first_query < q.size(1); first_query += query_step) {
                const int64_t query_count = std::min(query_step, q.size(1) - first_query);
                auto result = at::scaled_dot_product_attention(
                    query_heads.narrow(2, first_query, query_count),
                    key_heads, value_heads, std::nullopt, 0.0, false,
                    std::optional<double>(static_cast<double>(scale)));
                out_work.narrow(2, first_head, head_count)
                    .narrow(1, first_query, query_count)
                    .copy_(result.permute({0, 2, 1, 3}));
            }
        }
    }
    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::flashattn
#endif // ENABLE_HYGON_API && ENABLE_ATEN
