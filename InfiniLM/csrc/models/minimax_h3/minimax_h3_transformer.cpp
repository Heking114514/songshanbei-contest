#include "minimax_h3_transformer.hpp"
#include "h3_audio_trace.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/addcmul.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/distributed/allgather.hpp>
#include <infinicore/ops/embedding.hpp>
#include <infinicore/ops/h3_modulation.hpp>
#include <infinicore/ops/h3_rms_modulation.hpp>
#include <infinicore/ops/h3_qk_norm_rope.hpp>
#include <infinicore/ops/index_copy.hpp>
#include <infinicore/ops/mha.hpp>
#include <infinicore/ops/mrope.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/silu.hpp>
#include <infinicore/ops/swiglu.hpp>
#include <infinicore/ops/timestep_embedding.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace infinilm::models::minimax_h3 {
namespace {

constexpr size_t H3_MODALITY_COUNT = 3;

infinicore::Tensor cast_to(const infinicore::Tensor &input,
                           const infinicore::DataType &dtype) {
    if (input->dtype() == dtype) {
        return input;
    }
    auto output = infinicore::Tensor::empty(input->shape(), dtype, input->device());
    infinicore::op::cast_(output, input);
    return output;
}

template <typename Linear>
infinicore::Tensor linear_forward(const std::shared_ptr<Linear> &linear,
                                  const infinicore::Tensor &input) {
    auto mutable_input = input;
    return linear->forward(mutable_input);
}

infinicore::Tensor gather_rows(const infinicore::Tensor &table,
                               const infinicore::Tensor &indices) {
    return infinicore::op::embedding(indices, table);
}

infinicore::Tensor apply_adaln(const infinicore::Tensor &normalized,
                               const infinicore::Tensor &scale,
                               const infinicore::Tensor &shift) {
    auto modulated = infinicore::op::addcmul(normalized, normalized, scale, 1.0f);
    // modulated is a private temporary. Reusing it removes one full hidden
    // allocation while retaining addcmul's BF16 store before the separate add.
    static const bool reuse_temporary = [] {
        const char *value = std::getenv("H3_EXPERIMENT_ADALN_INPLACE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (reuse_temporary) {
        infinicore::op::add_(modulated, modulated, shift);
        return modulated;
    }
    return infinicore::op::add(modulated, shift);
}

infinicore::Tensor gated_residual(const infinicore::Tensor &residual,
                                  const infinicore::Tensor &gate,
                                  const infinicore::Tensor &branch) {
    return infinicore::op::addcmul(residual, gate, branch, 1.0f);
}

bool experiment_enabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || std::strcmp(value, "0") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") != 0) {
        throw std::invalid_argument(std::string(name) + " must be 0 or 1");
    }
    return true;
}

int fused_rms_block_size() {
    static const int block = [] {
        const char *value = std::getenv("H3_EXPERIMENT_FUSED_RMS_BLOCK");
        if (value == nullptr || std::strcmp(value, "0") == 0) {
            return 0;
        }
        for (const int size : {128, 256, 512, 1024}) {
            if (std::to_string(size) == value) {
                return size;
            }
        }
        throw std::invalid_argument(
            "H3_EXPERIMENT_FUSED_RMS_BLOCK must be 0,128,256,512,1024");
    }();
    return block;
}

bool fused_rms_eligible(const infinicore::Tensor &x,
                        const infinicore::Tensor &weight) {
    return x->ndim() == 3 && x->size(0) == 1
        && x->size(2) > 0 && x->size(2) <= 8192 && x->size(1) <= 2147483647
        && x->dtype() == infinicore::DataType::BF16 && x->is_contiguous()
        && weight->ndim() == 1 && weight->size(0) == x->size(2)
        && weight->is_contiguous() && weight->device() == x->device()
        && (weight->dtype() == infinicore::DataType::BF16
            || weight->dtype() == infinicore::DataType::F32);
}

bool indexed_modulation_requested() {
    static const bool enabled = [] {
        const char *value = std::getenv("H3_EXPERIMENT_ADALN_INDEXED");
        if (value == nullptr || std::strcmp(value, "0") == 0) {
            return false;
        }
        if (std::strcmp(value, "1") != 0) {
            throw std::invalid_argument("H3_EXPERIMENT_ADALN_INDEXED must be 0 or 1");
        }
        return true;
    }();
    return enabled;
}

infinicore::Tensor indexed_modulate(
    const infinicore::Tensor &x, const infinicore::Tensor &branch,
    const infinicore::Tensor &table, const infinicore::Tensor &indices, int mode,
    size_t hidden_size) {
    if (x->dtype() == infinicore::DataType::BF16
        && branch->dtype() == infinicore::DataType::BF16
        && x->is_contiguous() && branch->is_contiguous()) {
        return infinicore::op::h3_modulation(x, branch, table, indices, mode);
    }
    // An unexpected norm/attention/FF output layout keeps existing semantics.
    // The normal contiguous BF16 path never builds this expanded table.
    auto expanded = gather_rows(table, indices)->unsqueeze(0);
    const size_t offset = mode < 2 ? 0 : 3;
    if (mode == 0 || mode == 2) {
        return apply_adaln(
            x, expanded->narrow({{2, (offset + 1) * hidden_size, hidden_size}}),
            expanded->narrow({{2, offset * hidden_size, hidden_size}}));
    }
    return gated_residual(
        x, expanded->narrow({{2, (offset + 2) * hidden_size, hidden_size}}), branch);
}

void require_tensor(const std::optional<infinicore::Tensor> &tensor,
                    const char *name) {
    if (!tensor.has_value() || !tensor.value()) {
        throw std::runtime_error(
            std::string("MiniMaxH3Transformer: missing input ") + name);
    }
}

size_t config_size(const std::shared_ptr<infinilm::config::ModelConfig> &config,
                   const char *primary, const char *fallback,
                   size_t default_value) {
    const auto &json = config->get_config_json();
    if (json.contains(primary)) {
        return json.at(primary).get<size_t>();
    }
    if (fallback != nullptr && json.contains(fallback)) {
        return json.at(fallback).get<size_t>();
    }
    return default_value;
}

} // namespace

MiniMaxH3Attention::MiniMaxH3Attention(size_t hidden_size, size_t num_heads,
                                       size_t head_dim, double qk_norm_eps,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : head_dim_(head_dim),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim))) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (num_heads % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error(
            "MiniMaxH3Attention: num_heads must be divisible by tp_size");
    }
    num_heads_ = num_heads / static_cast<size_t>(rank_info.tp_size);
    inner_dim_ = num_heads_ * head_dim_;

    auto register_fn = [this](const std::string &name,
                              infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
        hidden_size, head_dim_, num_heads, num_heads, "to_q", "to_k", "to_v",
        register_fn, nullptr, false, dtype, device, rank_info);
    INFINICORE_NN_MODULE_INIT(norm_q, head_dim_, qk_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm_k, head_dim_, qk_norm_eps, dtype, device);
    out_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "to_out.0", num_heads * head_dim_, hidden_size, false, dtype, device,
        rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
}

infinicore::Tensor MiniMaxH3Attention::forward(
    const infinicore::Tensor &hidden_states,
    const std::shared_ptr<infinicore::nn::RoPE> &rope,
    const std::optional<infinicore::Tensor> &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const {
    if (hidden_states->shape().size() != 3) {
        throw std::runtime_error(
            "MiniMaxH3Attention: hidden_states must be [batch, seq, hidden]");
    }
    const size_t batch_size = hidden_states->size(0);
    const size_t sequence_length = hidden_states->size(1);

    auto hidden_states_mutable = hidden_states;
    auto [query, key, value] = qkv_proj_->forward_split(hidden_states_mutable);
    query = query->view({batch_size * sequence_length, num_heads_, head_dim_});
    key = key->view({batch_size * sequence_length, num_heads_, head_dim_});
    value = value->view({batch_size, sequence_length, num_heads_, head_dim_});

    static const bool fuse_qk = experiment_enabled("H3_EXPERIMENT_QK_NORM_ROPE");
    const auto qk_device = query->device().getType();
    const auto q_weight = norm_q_->weight();
    const auto k_weight = norm_k_->weight();
    const bool qk_fused = fuse_qk && batch_size == 1 && sequence_length <= 2147483647
        && head_dim_ == 128 && num_heads_ > 0 && num_heads_ <= 64
        && (qk_device == infinicore::Device::Type::NVIDIA
            || qk_device == infinicore::Device::Type::HYGON)
        && rotary_cos_sin_cache.has_value() && position_ids.has_value()
        && query->dtype() == infinicore::DataType::BF16
        && key->dtype() == infinicore::DataType::BF16
        && query->strides()[2] == 1 && key->strides()[2] == 1
        && query->strides()[1] >= 128 && key->strides()[1] >= 128
        && query->strides()[0] >= static_cast<infinicore::Stride>(num_heads_) * query->strides()[1]
        && key->strides()[0] >= static_cast<infinicore::Stride>(num_heads_) * key->strides()[1]
        && q_weight->ndim() == 1 && q_weight->size(0) == 128
        && k_weight->shape() == q_weight->shape()
        && q_weight->dtype() == k_weight->dtype()
        && (q_weight->dtype() == infinicore::DataType::BF16
            || q_weight->dtype() == infinicore::DataType::F32)
        && q_weight->is_contiguous() && k_weight->is_contiguous()
        && rotary_cos_sin_cache.value()->ndim() == 2
        && rotary_cos_sin_cache.value()->size(0) > 0
        && rotary_cos_sin_cache.value()->size(1) > 0
        && rotary_cos_sin_cache.value()->size(1) <= 128
        && rotary_cos_sin_cache.value()->size(1) % 6 == 0
        && rotary_cos_sin_cache.value()->dtype() == infinicore::DataType::BF16
        && rotary_cos_sin_cache.value()->is_contiguous()
        && position_ids.value()->ndim() == 1
        && position_ids.value()->size(0) == sequence_length
        && position_ids.value()->dtype() == infinicore::DataType::I64
        && position_ids.value()->is_contiguous()
        && query->device() == key->device()
        && query->device() == q_weight->device() && query->device() == k_weight->device()
        && query->device() == rotary_cos_sin_cache.value()->device()
        && query->device() == position_ids.value()->device();
    if (qk_fused) {
        std::tie(query, key) = infinicore::op::h3_qk_norm_rope(
            query, key, q_weight, k_weight, rotary_cos_sin_cache.value(),
            position_ids.value(), static_cast<float>(norm_q_->eps()),
            static_cast<float>(norm_k_->eps()));
    } else {
        query = norm_q_->forward(query);
        key = norm_k_->forward(key);
    }
    query = query->view({batch_size, sequence_length, num_heads_, head_dim_});
    key = key->view({batch_size, sequence_length, num_heads_, head_dim_});

    if (!qk_fused && rotary_cos_sin_cache.has_value()) {
        if (!position_ids.has_value()) {
            throw std::runtime_error(
                "MiniMaxH3Attention: position_ids are required with a rotary cache");
        }
        if (batch_size != 1) {
            throw std::runtime_error(
                "MiniMaxH3Attention: packed H3 RoPE currently supports batch size 1");
        }
        const auto &cache = rotary_cos_sin_cache.value();
        const size_t rotary_dim = cache->size(1);
        if (rotary_dim % (2 * H3_MODALITY_COUNT) != 0 || rotary_dim > head_dim_) {
            throw std::runtime_error("MiniMaxH3Attention: rotary cache width is "
                                     "incompatible with three-axis RoPE");
        }
        const size_t half_rotary_dim = rotary_dim / 2;
        const int section = static_cast<int>(rotary_dim / (2 * H3_MODALITY_COUNT));
        auto cos = cache->narrow({{1, 0, half_rotary_dim}});
        auto sin = cache->narrow({{1, half_rotary_dim, half_rotary_dim}});
        auto query_flat = query->squeeze(0)->view({sequence_length, num_heads_ * head_dim_});
        auto key_flat = key->squeeze(0)->view({sequence_length, num_heads_ * head_dim_});
        // These are private norm_q/norm_k outputs, not QKV weight views.
        // The CUDA-compatible implementation loads each rotation pair before
        // storing it, and different threads own disjoint pairs and tail lanes.
        static const bool inplace_rope = [] {
            const char *value = std::getenv("H3_EXPERIMENT_ROPE_INPLACE");
            return value != nullptr && std::strcmp(value, "1") == 0;
        }();
        const auto device_type = query_flat->device().getType();
        const bool compatible_inplace = inplace_rope
            && (device_type == infinicore::Device::Type::NVIDIA
                || device_type == infinicore::Device::Type::HYGON)
            && query_flat->dtype() == infinicore::DataType::BF16
            && key_flat->dtype() == infinicore::DataType::BF16
            && query_flat->is_contiguous() && key_flat->is_contiguous()
            && query_flat->data() != key_flat->data();
        if (compatible_inplace) {
            infinicore::op::mrope_(
                query_flat, key_flat, query_flat, key_flat, cos, sin,
                position_ids.value(), static_cast<int>(head_dim_),
                static_cast<int>(rotary_dim), section, section, section, false);
        } else {
            std::tie(query_flat, key_flat) = infinicore::op::mrope(
                query_flat, key_flat, cos, sin, position_ids.value(),
                static_cast<int>(head_dim_), static_cast<int>(rotary_dim), section,
                section, section, false);
        }
        query = query_flat->view({batch_size, sequence_length, num_heads_, head_dim_});
        key = key_flat->view({batch_size, sequence_length, num_heads_, head_dim_});
    } else if (!qk_fused && rope != nullptr) {
        if (!position_ids.has_value()) {
            throw std::runtime_error(
                "MiniMaxH3Attention: position_ids are required with RoPE");
        }
        if (batch_size != 1) {
            throw std::runtime_error(
                "MiniMaxH3Attention: packed H3 RoPE currently supports batch size 1");
        }
        auto query_squeezed = query->squeeze(0);
        auto key_squeezed = key->squeeze(0);
        std::tie(query_squeezed, key_squeezed) = rope->forward(query_squeezed, key_squeezed, position_ids.value());
        query = query_squeezed->unsqueeze(0);
        key = key_squeezed->unsqueeze(0);
    }
    auto output = infinicore::op::mha(query, key, value, std::nullopt, scale_, false);
    output = output->view({batch_size, sequence_length, inner_dim_});
    return linear_forward(out_proj_, output);
}

void MiniMaxH3Attention::process_weights_after_loading() {
    qkv_proj_->process_weights_after_loading();
}

void MiniMaxH3Attention::reset_runtime_state() const {
    qkv_proj_->reset_runtime_state();
}

MiniMaxH3MLP::MiniMaxH3MLP(size_t hidden_size, size_t ffn_hidden_size,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (ffn_hidden_size % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error(
            "MiniMaxH3MLP: ffn_hidden_size must be divisible by tp_size");
    }
    auto register_fn = [this](const std::string &name,
                              infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    // Diffusers stores [up, gate] in one projection. The synthetic names are
    // produced by the MiniMax-H3 load-time remapper so corresponding shards
    // from both halves land on every TP rank.
    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size, ffn_hidden_size, "net.0.proj.up_proj",
        "net.0.proj.gate_proj", register_fn, nullptr, false, dtype, device,
        rank_info);
    down_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "net.2", ffn_hidden_size, hidden_size, false, dtype, device,
        rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
}

infinicore::Tensor
MiniMaxH3MLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto [up, gate] = gate_up_proj_->forward_split(hidden_states_mutable);
    auto activated = infinicore::op::swiglu(up, gate);
    return linear_forward(down_proj_, activated);
}

void MiniMaxH3MLP::process_weights_after_loading() {
    gate_up_proj_->process_weights_after_loading();
}

void MiniMaxH3MLP::reset_runtime_state() const {
    gate_up_proj_->reset_runtime_state();
}

MiniMaxH3TokenRefinerBlock::MiniMaxH3TokenRefinerBlock(
    size_t hidden_size, size_t num_heads, size_t head_dim,
    size_t ffn_hidden_size, double norm_eps, double qk_norm_eps,
    const infinicore::DataType &dtype, const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, hidden_size, num_heads, head_dim, qk_norm_eps,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(ff, hidden_size, ffn_hidden_size, dtype, device);
}

infinicore::Tensor MiniMaxH3TokenRefinerBlock::forward(
    const infinicore::Tensor &hidden_states) const {
    auto attention_output = attn_->forward(norm1_->forward(hidden_states));
    auto output = infinicore::op::add(hidden_states, attention_output);
    auto mlp_output = ff_->forward(norm2_->forward(output));
    return infinicore::op::add(output, mlp_output);
}

MiniMaxH3TokenRefiner::MiniMaxH3TokenRefiner(
    size_t hidden_size, size_t num_heads, size_t head_dim,
    size_t ffn_hidden_size, size_t num_layers, double norm_eps,
    double qk_norm_eps, double final_norm_eps,
    const infinicore::DataType &dtype, const infinicore::Device &device) {
    refiner_blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        refiner_blocks_.push_back(this->register_module<MiniMaxH3TokenRefinerBlock>(
            "refiner_blocks." + std::to_string(i), hidden_size, num_heads, head_dim,
            ffn_hidden_size, norm_eps, qk_norm_eps, dtype, device));
    }
    INFINICORE_NN_MODULE_INIT(final_norm, hidden_size, final_norm_eps, dtype,
                              device);
}

infinicore::Tensor
MiniMaxH3TokenRefiner::forward(const infinicore::Tensor &hidden_states) const {
    auto output = hidden_states;
    for (const auto &block : refiner_blocks_) {
        output = block->forward(output);
    }
    return final_norm_->forward(output);
}

MiniMaxH3AdaLNProjection::MiniMaxH3AdaLNProjection(
    size_t in_features, size_t out_features, const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : dtype_(dtype) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (out_features % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error(
            "MiniMaxH3AdaLNProjection: out_features must be divisible by tp_size");
    }
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
    linear_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "linear", in_features, out_features, true, dtype, device,
        rank_info.tp_rank, rank_info.tp_size);
}

infinicore::Tensor
MiniMaxH3AdaLNProjection::forward(const infinicore::Tensor &temb) const {
    auto activated = infinicore::op::silu(temb);
    activated = cast_to(activated, dtype_);
    auto local = linear_forward(linear_, activated);
    if (tp_size_ == 1) {
        return local;
    }

    // allgather concatenates dim 0. Transpose the output width to dim 0 so
    // the original Diffusers projection order is reconstructed exactly.
    auto local_transposed = local->permute({1, 0})->contiguous();
    auto gathered = infinicore::op::distributed::allgather(
        local_transposed, tp_size_, communicator_);
    return gathered->permute({1, 0})->contiguous();
}

std::vector<infinicore::Tensor> MiniMaxH3AdaLNProjection::forward_batch(
    const std::vector<std::shared_ptr<MiniMaxH3AdaLNProjection>> &projections,
    const infinicore::Tensor &temb) {
    if (projections.empty()) {
        return {};
    }
    if (!temb || temb->ndim() != 2 || temb->size(0) == 0 || !projections.front()) {
        throw std::runtime_error("MiniMaxH3AdaLNProjection: invalid batch input");
    }
    const auto &first = *projections.front();
    if (first.tp_size_ == 0
        || (first.tp_size_ > 1 && first.communicator_ == nullptr)) {
        throw std::runtime_error("MiniMaxH3AdaLNProjection: invalid TP group");
    }
    for (const auto &projection : projections) {
        if (!projection || projection->dtype_ != first.dtype_
            || projection->tp_size_ != first.tp_size_
            || projection->communicator_ != first.communicator_) {
            throw std::runtime_error(
                "MiniMaxH3AdaLNProjection: batch must share dtype/TP group");
        }
    }

    // Preserve one GEMM per layer and its original matrix dimensions. Sharing
    // this pure activation only removes repeated SiLU/casts of the same temb.
    auto activated = cast_to(infinicore::op::silu(temb), first.dtype_);
    std::vector<infinicore::Tensor> locals;
    locals.reserve(projections.size());
    size_t local_width = 0;
    for (const auto &projection : projections) {
        auto local = linear_forward(projection->linear_, activated);
        if (local->ndim() != 2 || local->size(0) != temb->size(0)
            || local->size(1) == 0
            || (!locals.empty() && local->size(1) != local_width)
            || local->device() != temb->device() || local->dtype() != first.dtype_) {
            throw std::runtime_error(
                "MiniMaxH3AdaLNProjection: incompatible local projection shape");
        }
        local_width = local->size(1);
        locals.push_back(local->unsqueeze(0));
    }

    const size_t layers = projections.size();
    const size_t rows = temb->size(0);
    auto packed = infinicore::Tensor::empty(
        {layers, rows, local_width}, first.dtype_, temb->device());
    for (size_t i = 0; i < layers; ++i) {
        // Same-device copy_from is a recorded Rearrange op. Generic dim-0
        // cat issues raw memcpy calls and also requires at least two inputs;
        // neither contract is suitable for software-graph / one-layer smoke.
        packed->narrow({{0, i, 1}})->copy_from(locals[i]);
    }
    if (first.tp_size_ > 1) {
        auto gathered = infinicore::op::distributed::allgather(
            packed, first.tp_size_, first.communicator_);
        // RCCL output order is [rank, layer, row, local_column].
        // Restore each original layer's [row, rank * local_column] order.
        packed = gathered->view({first.tp_size_, layers, rows, local_width})
                     ->permute({1, 2, 0, 3})->contiguous()
                     ->view({layers, rows, first.tp_size_ * local_width});
    }
    std::vector<infinicore::Tensor> outputs;
    outputs.reserve(layers);
    for (size_t i = 0; i < layers; ++i) {
        outputs.push_back(packed->narrow({{0, i, 1}})->squeeze(0));
    }
    return outputs;
}

MiniMaxH3TransformerBlock::MiniMaxH3TransformerBlock(
    size_t hidden_size, size_t num_heads, size_t head_dim,
    size_t ffn_hidden_size, size_t time_embed_dim, double norm_eps,
    double qk_norm_eps, const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size) {
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, hidden_size, num_heads, head_dim, qk_norm_eps,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(ff, hidden_size, ffn_hidden_size, dtype, device);
    INFINICORE_NN_MODULE_INIT(adaln_proj, time_embed_dim,
                              H3_MODALITY_COUNT * 6 * hidden_size, dtype, device);
}

infinicore::Tensor MiniMaxH3TransformerBlock::forward(
    const infinicore::Tensor &hidden_states, const infinicore::Tensor &temb,
    const infinicore::Tensor &adaln_indices,
    const std::shared_ptr<infinicore::nn::RoPE> &rope,
    const infinicore::Tensor &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache,
    const std::optional<infinicore::Tensor> &prepared_modulation) const {
    auto projected = prepared_modulation.has_value()
        ? prepared_modulation.value() : adaln_proj_->forward(temb);
    if (projected->ndim() != 2 || projected->size(0) != temb->size(0)
        || projected->size(1) != H3_MODALITY_COUNT * 6 * hidden_size_) {
        throw std::runtime_error("MiniMaxH3TransformerBlock: invalid modulation");
    }
    auto modulation = projected->view(
        {temb->size(0) * H3_MODALITY_COUNT, 6 * hidden_size_});
    const auto device_type = hidden_states->device().getType();
    static const bool fuse_rms = experiment_enabled("H3_EXPERIMENT_RMS_ADALN");
    static const bool fuse_gated_rms = experiment_enabled("H3_EXPERIMENT_GATE_RMS_ADALN");
    const bool indexed = (indexed_modulation_requested() || fuse_rms || fuse_gated_rms)
        && (device_type == infinicore::Device::Type::NVIDIA
            || device_type == infinicore::Device::Type::HYGON)
        && hidden_states->dtype() == infinicore::DataType::BF16
        && modulation->dtype() == infinicore::DataType::BF16
        && adaln_indices->dtype() == infinicore::DataType::I64
        && hidden_states->ndim() == 3 && hidden_states->size(0) == 1
        && hidden_states->size(2) == hidden_size_
        && adaln_indices->ndim() == 1
        && adaln_indices->size(0) == hidden_states->size(1)
        && hidden_states->is_contiguous() && modulation->is_contiguous()
        && adaln_indices->is_contiguous()
        && hidden_states->device() == modulation->device()
        && hidden_states->device() == adaln_indices->device();
    if (indexed) {
        infinicore::Tensor normalized;
        if (fuse_rms && fused_rms_eligible(hidden_states, norm1_->weight())) {
            normalized = infinicore::op::h3_rms_modulation(
                hidden_states, hidden_states, norm1_->weight(), modulation,
                adaln_indices, 0, static_cast<float>(norm1_->eps()), fused_rms_block_size()).first;
        } else {
            normalized = norm1_->forward(hidden_states);
            normalized = indexed_modulate(
                normalized, normalized, modulation, adaln_indices, 0, hidden_size_);
        }
        auto attention_output = attn_->forward(
            normalized, rope, position_ids, rotary_cos_sin_cache);
        infinicore::Tensor output;
        if (fuse_gated_rms && fused_rms_eligible(hidden_states, norm2_->weight())
            && attention_output->dtype() == infinicore::DataType::BF16
            && attention_output->is_contiguous()) {
            std::tie(normalized, output) = infinicore::op::h3_rms_modulation(
                hidden_states, attention_output, norm2_->weight(), modulation,
                adaln_indices, 2, static_cast<float>(norm2_->eps()), fused_rms_block_size());
        } else {
            output = indexed_modulate(
                hidden_states, attention_output, modulation, adaln_indices, 1, hidden_size_);
            if (fuse_rms && fused_rms_eligible(output, norm2_->weight())) {
                normalized = infinicore::op::h3_rms_modulation(
                    output, output, norm2_->weight(), modulation, adaln_indices,
                    1, static_cast<float>(norm2_->eps()), fused_rms_block_size()).first;
            } else {
                normalized = norm2_->forward(output);
                normalized = indexed_modulate(
                    normalized, normalized, modulation, adaln_indices, 2, hidden_size_);
            }
        }
        auto mlp_output = ff_->forward(normalized);
        return indexed_modulate(
            output, mlp_output, modulation, adaln_indices, 3, hidden_size_);
    }
    auto token_modulation = gather_rows(modulation, adaln_indices)->unsqueeze(0);
    auto shift_msa = token_modulation->narrow({{2, 0, hidden_size_}});
    auto scale_msa = token_modulation->narrow({{2, hidden_size_, hidden_size_}});
    auto gate_msa = token_modulation->narrow({{2, 2 * hidden_size_, hidden_size_}});
    auto shift_mlp = token_modulation->narrow({{2, 3 * hidden_size_, hidden_size_}});
    auto scale_mlp = token_modulation->narrow({{2, 4 * hidden_size_, hidden_size_}});
    auto gate_mlp = token_modulation->narrow({{2, 5 * hidden_size_, hidden_size_}});

    auto normalized = norm1_->forward(hidden_states);
    normalized = apply_adaln(normalized, scale_msa, shift_msa);
    auto attention_output = attn_->forward(normalized, rope, position_ids, rotary_cos_sin_cache);
    auto output = gated_residual(hidden_states, gate_msa, attention_output);

    normalized = norm2_->forward(output);
    normalized = apply_adaln(normalized, scale_mlp, shift_mlp);
    auto mlp_output = ff_->forward(normalized);
    output = gated_residual(output, gate_mlp, mlp_output);
    return output;
}

MiniMaxH3TimeEmbedder::MiniMaxH3TimeEmbedder(size_t input_dim,
                                             size_t hidden_size,
                                             size_t output_dim,
                                             const infinicore::Device &device)
    : input_dim_(input_dim) {
    INFINICORE_NN_MODULE_INIT(linear_1, input_dim, hidden_size, true,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(linear_2, hidden_size, output_dim, true,
                              infinicore::DataType::F32, device);
}

infinicore::Tensor
MiniMaxH3TimeEmbedder::forward(const infinicore::Tensor &timestep) const {
    if (input_dim_ % 2 != 0) {
        throw std::runtime_error("MiniMaxH3TimeEmbedder: input_dim must be even");
    }

    if (timestep->shape().size() == 2) {
        if (timestep->size(1) != input_dim_) {
            throw std::runtime_error(
                "MiniMaxH3TimeEmbedder: precomputed embedding has the wrong width");
        }
        auto embedding = cast_to(timestep, infinicore::DataType::F32);
        auto hidden = linear_forward(linear_1_, embedding);
        hidden = infinicore::op::silu(hidden);
        return linear_forward(linear_2_, hidden);
    }
    if (timestep->shape().size() != 1) {
        throw std::runtime_error("MiniMaxH3TimeEmbedder: timestep must be [N] or a "
                                 "precomputed [N, input_dim] embedding");
    }
    auto embedding = infinicore::op::timestep_embedding(timestep, input_dim_);
    auto hidden = linear_forward(linear_1_, embedding);
    hidden = infinicore::op::silu(hidden);
    return linear_forward(linear_2_, hidden);
}

MiniMaxH3OutputNorm::MiniMaxH3OutputNorm(size_t hidden_size,
                                         size_t time_embed_dim, double norm_eps,
                                         const infinicore::DataType &dtype,
                                         const infinicore::Device &device)
    : hidden_size_(hidden_size), dtype_(dtype) {
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear, time_embed_dim, 2 * hidden_size, true,
                              dtype, device);
}

infinicore::Tensor
MiniMaxH3OutputNorm::forward(const infinicore::Tensor &hidden_states,
                             const infinicore::Tensor &temb,
                             const infinicore::Tensor &timestep_indices) const {
    auto activated = cast_to(infinicore::op::silu(temb), dtype_);
    auto modulation = linear_forward(linear_, activated);
    auto token_modulation = gather_rows(modulation, timestep_indices)->unsqueeze(0);
    auto shift = token_modulation->narrow({{2, 0, hidden_size_}});
    auto scale = token_modulation->narrow({{2, hidden_size_, hidden_size_}});
    return apply_adaln(norm_->forward(hidden_states), scale, shift);
}

MiniMaxH3Transformer::MiniMaxH3Transformer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : hidden_size_(config_size(model_config, "hidden_size", nullptr, 5376)),
      dtype_(model_config->get_dtype()) {
    model_config_ = model_config;
    const auto &config = model_config->get_config_json();
    const size_t num_layers = config_size(model_config, "num_layers", nullptr, 50);
    const size_t token_refiner_layers = config_size(
        model_config, "token_refiner_num_layers", "num_refiner_layers", 2);
    const size_t num_heads = config_size(model_config, "num_attention_heads", nullptr, 56);
    const size_t head_dim = config_size(model_config, "attention_head_dim", nullptr, 128);
    const size_t ffn_hidden_size = config_size(model_config, "ffn_hidden_size", "ffn_dim", 14336);
    const size_t video_latents_dim = config_size(model_config, "latents_dim", "in_channels", 24);
    const size_t audio_latents_dim = config_size(model_config, "audio_latents_dim", "audio_in_channels", 32);
    const size_t text_dim = config_size(model_config, "text_dim", nullptr, 5120);
    const size_t timestep_input_dim = config_size(model_config, "timestep_input_dim", "freq_dim", 256);
    const size_t time_embed_hidden_size = config_size(
        model_config, "time_embed_hidden_size", "time_embed_hidden_dim", 5376);
    const size_t time_embed_dim = config_size(model_config, "time_embed_dim", nullptr, 2688);
    const size_t rope_freq_dim = config_size(model_config, "rope_inv_freq_len", "rope_freq_dim", 16);
    const double norm_eps = config.value("norm_eps", 1e-5);
    const double qk_norm_eps = config.value("qk_norm_eps", 1e-5);
    const double final_norm_eps = config.value("final_norm_eps", 1e-5);

    std::vector<size_t> patch_size{1, 2, 2};
    if (config.contains("patch_size")) {
        patch_size = config.at("patch_size").get<std::vector<size_t>>();
    }
    if (patch_size.size() != 3) {
        throw std::runtime_error(
            "MiniMaxH3Transformer: patch_size must contain t, h and w");
    }
    const size_t video_patch_dim = video_latents_dim * patch_size[0] * patch_size[1] * patch_size[2];
    const size_t rotary_dim = 2 * H3_MODALITY_COUNT * rope_freq_dim;
    if (rotary_dim > head_dim) {
        throw std::runtime_error("MiniMaxH3Transformer: rotary dimensions exceed "
                                 "attention head dimension");
    }

    INFINICORE_NN_MODULE_INIT(proj_in, video_patch_dim, hidden_size_, true,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(audio_proj_in, audio_latents_dim, hidden_size_,
                              true, infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(context_embedder, text_dim, hidden_size_, true,
                              dtype_, device);
    INFINICORE_NN_MODULE_INIT(time_embedder, timestep_input_dim,
                              time_embed_hidden_size, time_embed_dim, device);
    INFINICORE_NN_MODULE_INIT(
        rope, head_dim, rotary_dim, config.value("max_position_embeddings", 8192),
        config.value("rope_theta", 10000.0), infinicore::nn::RoPE::Algo::GPT_NEOX,
        dtype_, device, nullptr,
        std::vector<int>{static_cast<int>(rope_freq_dim),
                         static_cast<int>(rope_freq_dim),
                         static_cast<int>(rope_freq_dim)},
        false);
    INFINICORE_NN_MODULE_INIT(token_refiner, hidden_size_, num_heads, head_dim,
                              ffn_hidden_size, token_refiner_layers, norm_eps,
                              qk_norm_eps, final_norm_eps, dtype_, device);

    transformer_blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        transformer_blocks_.push_back(
            this->register_module<MiniMaxH3TransformerBlock>(
                "transformer_blocks." + std::to_string(i), hidden_size_, num_heads,
                head_dim, ffn_hidden_size, time_embed_dim, norm_eps, qk_norm_eps,
                dtype_, device));
    }
    INFINICORE_NN_MODULE_INIT(norm_out, hidden_size_, time_embed_dim,
                              final_norm_eps, dtype_, device);
    INFINICORE_NN_MODULE_INIT(proj_out, hidden_size_, video_patch_dim, true,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(audio_proj_out, hidden_size_, audio_latents_dim,
                              true, infinicore::DataType::F32, device);
}

InfinilmModel::Output MiniMaxH3Transformer::forward(const Input &input) const {
    require_tensor(input.video_hidden_states, "video_hidden_states");
    require_tensor(input.audio_hidden_states, "audio_hidden_states");
    require_tensor(input.encoder_hidden_states, "encoder_hidden_states");
    require_tensor(input.timestep, "timestep");
    require_tensor(input.timestep_indices, "timestep_indices");
    require_tensor(input.token_tags, "token_tags");
    require_tensor(input.position_ids, "position_ids");
    require_tensor(input.video_indices, "video_indices");
    require_tensor(input.audio_indices, "audio_indices");
    require_tensor(input.text_indices, "text_indices");

    const auto &position_ids = input.position_ids.value();
    const bool has_rotary_cache = input.rotary_cos_sin_cache.has_value();
    if (has_rotary_cache) {
        const auto &rotary_cache = input.rotary_cos_sin_cache.value();
        if (position_ids->shape().size() != 1) {
            throw std::runtime_error("MiniMaxH3Transformer: cached RoPE position_ids "
                                     "must be [sequence_length]");
        }
        if (rotary_cache->shape().size() != 2 || rotary_cache->size(0) != position_ids->size(0)) {
            throw std::runtime_error("MiniMaxH3Transformer: rotary cache must be "
                                     "[sequence_length, rotary_dim]");
        }
    } else if (position_ids->shape().size() != 2 || position_ids->size(1) != 3) {
        throw std::runtime_error(
            "MiniMaxH3Transformer: position_ids must be [sequence_length, 3]");
    }
    const size_t sequence_length = position_ids->size(0);
    const auto &video_hidden_states = input.video_hidden_states.value();
    const auto &audio_hidden_states = input.audio_hidden_states.value();
    const auto &encoder_hidden_states = input.encoder_hidden_states.value();
    if (video_hidden_states->size(0) != 1 || audio_hidden_states->size(0) != 1 || encoder_hidden_states->size(0) != 1) {
        throw std::runtime_error("MiniMaxH3Transformer: the packed FL2VA path "
                                 "currently supports batch size 1");
    }

    AudioRowTrace audio_trace;
    audio_trace.write("audio_input", audio_hidden_states);
    auto video_input = cast_to(video_hidden_states, infinicore::DataType::F32);
    auto audio_input = cast_to(audio_hidden_states, infinicore::DataType::F32);
    auto video_embeds = cast_to(linear_forward(proj_in_, video_input), dtype_);
    auto audio_embeds = cast_to(linear_forward(audio_proj_in_, audio_input), dtype_);

    audio_trace.write("audio_embeds", audio_embeds);

    // Pointer identity is unsafe after in-place writes or rank-local allocation reuse.
    // Only callers supplying a conditioning revision opt into text reuse.
    if (!input.minimax_h3_cache_id.has_value() || !cached_text_embeddings_.has_value()
        || cached_text_cache_id_ != input.minimax_h3_cache_id) {
        auto text_input = cast_to(encoder_hidden_states, dtype_);
        auto context_embeds = linear_forward(context_embedder_, text_input);
        cached_text_embeddings_ = token_refiner_->forward(context_embeds);
        cached_text_cache_id_ = input.minimax_h3_cache_id;
    }
    auto text_embeds = cached_text_embeddings_.value();

    auto hidden_states = infinicore::Tensor::zeros(
        {1, sequence_length, hidden_size_}, dtype_, text_embeds->device());
    hidden_states = infinicore::op::index_copy(
        hidden_states, 1, input.text_indices.value(), text_embeds);
    hidden_states = infinicore::op::index_copy(
        hidden_states, 1, input.video_indices.value(), video_embeds);
    hidden_states = infinicore::op::index_copy(
        hidden_states, 1, input.audio_indices.value(), audio_embeds);

    audio_trace.rows("packed_audio", hidden_states, input.audio_indices.value());

    auto temb = time_embedder_->forward(input.timestep.value());
    auto timestep_indices = input.timestep_indices.value();
    auto token_tags = input.token_tags.value();
    auto timestep_indices_x2 = infinicore::op::add(timestep_indices, timestep_indices);
    auto timestep_indices_x3 = infinicore::op::add(timestep_indices_x2, timestep_indices);
    auto adaln_indices = infinicore::op::add(timestep_indices_x3, token_tags);
    auto rope_positions = has_rotary_cache ? position_ids : position_ids->permute({1, 0});

    // All ranks must choose the same policy before constructing/running the
    // model. No conditioning-dependent result is cached across denoising steps.
    static const bool batch_adaln = [] {
        const char *value = std::getenv("H3_EXPERIMENT_ADALN_BATCH");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    std::vector<infinicore::Tensor> modulations;
    if (batch_adaln && !transformer_blocks_.empty()) {
        std::vector<std::shared_ptr<MiniMaxH3AdaLNProjection>> projections;
        projections.reserve(transformer_blocks_.size());
        for (const auto &block : transformer_blocks_) {
            projections.push_back(block->adaln_projection());
        }
        modulations = MiniMaxH3AdaLNProjection::forward_batch(projections, temb);
    }
    for (size_t i = 0; i < transformer_blocks_.size(); ++i) {
        std::optional<infinicore::Tensor> prepared;
        if (batch_adaln) {
            prepared = modulations[i];
        }
        hidden_states = transformer_blocks_[i]->forward(
            hidden_states, temb, adaln_indices, rope_, rope_positions,
            input.rotary_cos_sin_cache, prepared);
        audio_trace.rows("block_" + std::to_string(i) + "_audio", hidden_states,
                         input.audio_indices.value());
    }

    auto normalized = norm_out_->forward(hidden_states, temb, timestep_indices);
    audio_trace.rows("norm_out_audio", normalized, input.audio_indices.value());
    auto normalized_f32 = cast_to(normalized, infinicore::DataType::F32);
    auto video_all = linear_forward(proj_out_, normalized_f32)->squeeze(0);
    auto audio_all = linear_forward(audio_proj_out_, normalized_f32)->squeeze(0);
    auto video_output = gather_rows(video_all, input.video_indices.value())->unsqueeze(0);
    auto audio_output = gather_rows(audio_all, input.audio_indices.value())->unsqueeze(0);
    audio_trace.write("audio_output", audio_output);
    return {video_output, audio_output};
}

void MiniMaxH3Transformer::reset_cache(const cache::CacheConfig *cache_config) {
    (void)cache_config;
    cache_config_.reset();
    cached_text_embeddings_.reset();
    cached_text_cache_id_.reset();
    infinilm::global_state::get_forward_context().kv_cache_vec.clear();
}

std::shared_ptr<infinilm::config::ModelConfig> create_minimax_h3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (model_config->get<std::string>("model_type") != "minimax_h3") {
        throw std::runtime_error(
            "create_minimax_h3_model_config: model_type is not minimax_h3");
    }
    model_config->get_config_json()["skip_sampling"] = true;
    return model_config;
}

} // namespace infinilm::models::minimax_h3

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minimax_h3, infinilm::models::minimax_h3::MiniMaxH3Transformer,
    infinilm::models::minimax_h3::create_minimax_h3_model_config);
} // namespace
