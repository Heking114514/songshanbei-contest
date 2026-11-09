#if defined(ENABLE_HYGON_API) && (defined(ENABLE_FLASH_ATTN) || defined(ENABLE_HYGON_EXPLICIT_ATTENTION))
#include "infinicore/ops/mha.hpp"

#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <c10/hip/HIPGuard.h>

namespace infinicore::op::mha_impl::flashattn {
struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
};

namespace {

void validate_formal_attention(const Tensor &out,
                               const Tensor &q,
                               const Tensor &k,
                               const Tensor &v,
                               const std::optional<Tensor> &alibi_slopes,
                               float scale) {
    const auto &q_shape = q->shape();
    const auto &k_shape = k->shape();
    const auto &v_shape = v->shape();
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::invalid_argument("Hygon MHA requires rank-4 Q, K, V tensors");
    }
    if (out->shape() != q_shape) {
        throw std::invalid_argument("Hygon MHA output shape must match Q");
    }
    if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0]
        || q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1]
        || q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3]
        || k_shape[2] != v_shape[2]) {
        throw std::invalid_argument("Hygon MHA Q/K/V layouts are incompatible");
    }
    if (q_shape[2] % k_shape[2] != 0) {
        throw std::invalid_argument("Hygon MHA query heads must be a multiple of KV heads");
    }
    if (q->stride(3) != 1 || k->stride(3) != 1 || v->stride(3) != 1) {
        throw std::invalid_argument("Hygon MHA requires contiguous head dimensions");
    }
    const auto q_dtype = q->dtype();
    if (!(q_dtype == k->dtype() && q_dtype == v->dtype()
          && q_dtype == out->dtype())) {
        throw std::invalid_argument("Hygon MHA requires matching Q/K/V/output dtypes");
    }
    if (q_dtype != DataType::F16 && q_dtype != DataType::BF16) {
        throw std::invalid_argument("Hygon formal MHA supports F16 or BF16");
    }
    if (q->device() != k->device() || q->device() != v->device()
        || q->device() != out->device()) {
        throw std::invalid_argument("Hygon MHA requires Q/K/V/output on one device");
    }
    if (alibi_slopes.has_value()) {
        throw std::invalid_argument("Hygon formal MHA does not accept alibi_slopes");
    }
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument("Hygon MHA scale must be finite and positive");
    }
}

} // namespace

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {
    validate_formal_attention(out, q, k, v, alibi_slopes, scale);

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        is_causal};
}

void run(void *planned_meta) {
#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

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

    auto attn_weight = at::matmul(
                           q.permute({0, 2, 1, 3}),
                           k_attn.permute({0, 2, 3, 1}))
        * scale;
    if (is_causal) {
        auto mask = at::tril(at::ones_like(attn_weight), -1).flip({-2, -1});
        auto neg_inf = at::full_like(
            attn_weight, -std::numeric_limits<float>::infinity());
        attn_weight = at::where(mask == 1, neg_inf, attn_weight);
    }
    auto attn = at::softmax(attn_weight, -1);
    auto result = at::matmul(attn, v_attn.permute({0, 2, 1, 3}))
                      .permute({0, 2, 1, 3});
    out_work.copy_(result);
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
#endif // ENABLE_HYGON_API && (ENABLE_FLASH_ATTN || ENABLE_HYGON_EXPLICIT_ATTENTION)
