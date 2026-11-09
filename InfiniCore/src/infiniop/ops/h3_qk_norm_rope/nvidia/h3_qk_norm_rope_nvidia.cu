#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "h3_qk_norm_rope_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../../../reduce/cuda/reduce.cuh"
#include "../../mrope/cuda/kernel.cuh"
#include <cmath>
#include <cstdint>

namespace {
template <typename Weight>
INFINIOP_CUDA_KERNEL h3QKNormRoPEKernel(
    cuda_bfloat16 *q_out, cuda_bfloat16 *k_out, const cuda_bfloat16 *q,
    const cuda_bfloat16 *k, const Weight *q_weight, const Weight *k_weight,
    const cuda_bfloat16 *cache, const int64_t *positions,
    size_t heads, ptrdiff_t q_token, ptrdiff_t q_head,
    ptrdiff_t k_token, ptrdiff_t k_head, size_t rows, size_t rotary_dim,
    float q_epsilon, float k_epsilon) {
    const bool is_key = blockIdx.y >= heads;
    const size_t head = blockIdx.y % heads;
    const size_t token = blockIdx.x;
    const auto input = (is_key ? k : q) + token * (is_key ? k_token : q_token)
        + head * (is_key ? k_head : q_head);
    const auto weight = is_key ? k_weight : q_weight;
    auto out = (is_key ? k_out : q_out) + (token * heads + head) * 128;
    float ss = op::common_cuda::reduce_op::sumSquared<128, cuda_bfloat16, float>(input, 128);
    __shared__ float rms;
    __shared__ cuda_bfloat16 normalized[128];
    if (threadIdx.x == 0) {
        rms = rsqrtf(ss / 128.0f + (is_key ? k_epsilon : q_epsilon));
    }
    __syncthreads();
    // Preserve the BF16 norm output, but keep it in shared memory rather than
    // writing two large norm tensors to global memory before RoPE reads them.
    normalized[threadIdx.x] = cuda_bfloat16(float(input[threadIdx.x])
        * float(weight[threadIdx.x]) * rms);
    __syncthreads();
    const size_t half = rotary_dim / 2;
    const size_t section = rotary_dim / 6;
    // Exact existing rotation helper; packed one-axis positions intentionally
    // retain MRoPE's invalid-position -> row-zero behavior.
    op::mrope::cuda::rotateOne<cuda_bfloat16, float, int64_t>(
        out, normalized, cache, cache + half, positions,
        0, 1, 128, rotary_dim, half, 0, 0, 0, 0,
        rotary_dim, rotary_dim, 0, 1, rows,
        section, section, section, false, false);
}
} // namespace

namespace op::h3_qk_norm_rope::nvidia {
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t q_out_desc, infiniopTensorDescriptor_t k_out_desc,
    infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t q_weight_desc, infiniopTensorDescriptor_t k_weight_desc,
    infiniopTensorDescriptor_t cache_desc, infiniopTensorDescriptor_t positions_desc,
    float q_epsilon, float k_epsilon) {
    CHECK_OR_RETURN(std::isfinite(q_epsilon) && std::isfinite(k_epsilon)
        && q_epsilon > 0 && k_epsilon > 0, INFINI_STATUS_BAD_PARAM);
    const auto &s = q_desc->shape();
    CHECK_OR_RETURN(s.size() == 3 && s[0] <= 2147483647 && s[1] > 0 && s[1] <= 64 && s[2] == 128,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(k_desc->shape() == s && q_out_desc->shape() == s && k_out_desc->shape() == s
        && q_weight_desc->shape().size() == 1 && q_weight_desc->shape()[0] == 128
        && k_weight_desc->shape() == q_weight_desc->shape()
        && cache_desc->shape().size() == 2 && cache_desc->shape()[0] > 0
        && cache_desc->shape()[1] > 0 && cache_desc->shape()[1] <= 128
        && cache_desc->shape()[1] % 6 == 0
        && positions_desc->shape().size() == 1 && positions_desc->shape()[0] == s[0],
        INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (const auto d : {q_out_desc, k_out_desc, q_desc, k_desc, cache_desc}) {
        CHECK_OR_RETURN(d->dtype() == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    }
    CHECK_OR_RETURN(positions_desc->dtype() == INFINI_DTYPE_I64
        && q_weight_desc->dtype() == k_weight_desc->dtype()
        && (q_weight_desc->dtype() == INFINI_DTYPE_BF16 || q_weight_desc->dtype() == INFINI_DTYPE_F32),
        INFINI_STATUS_BAD_TENSOR_DTYPE);
    for (const auto d : {q_out_desc, k_out_desc, q_weight_desc, k_weight_desc, cache_desc, positions_desc}) {
        CHECK_OR_RETURN(d->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    }
    for (const auto d : {q_desc, k_desc}) {
        CHECK_OR_RETURN(d->strides()[2] == 1 && d->strides()[1] >= 128
            && d->strides()[0] >= static_cast<ptrdiff_t>(s[1]) * d->strides()[1],
            INFINI_STATUS_BAD_TENSOR_STRIDES);
    }
    auto desc = new Descriptor;
    desc->device_type = handle->device; desc->device_id = handle->device_id;
    desc->_tokens = s[0]; desc->_heads = s[1];
    desc->_cache_rows = cache_desc->shape()[0]; desc->_rotary_dim = cache_desc->shape()[1];
    desc->_q_token = q_desc->strides()[0]; desc->_q_head = q_desc->strides()[1];
    desc->_k_token = k_desc->strides()[0]; desc->_k_head = k_desc->strides()[1];
    desc->_weight_dtype = q_weight_desc->dtype();
    desc->_q_epsilon = q_epsilon; desc->_k_epsilon = k_epsilon;
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(
    void *q_out, void *k_out, const void *q, const void *k, const void *q_weight,
    const void *k_weight, const void *cache, const void *positions, void *stream) const {
    if (_tokens == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    const dim3 grid(static_cast<unsigned>(_tokens), static_cast<unsigned>(_heads * 2));
#define LAUNCH(WEIGHT) \
    h3QKNormRoPEKernel<WEIGHT><<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>( \
        static_cast<cuda_bfloat16 *>(q_out), static_cast<cuda_bfloat16 *>(k_out), \
        static_cast<const cuda_bfloat16 *>(q), static_cast<const cuda_bfloat16 *>(k), \
        static_cast<const WEIGHT *>(q_weight), static_cast<const WEIGHT *>(k_weight), \
        static_cast<const cuda_bfloat16 *>(cache), static_cast<const int64_t *>(positions), \
        _heads, _q_token, _q_head, _k_token, _k_head, _cache_rows, _rotary_dim, _q_epsilon, _k_epsilon)
    if (_weight_dtype == INFINI_DTYPE_BF16) {
        LAUNCH(cuda_bfloat16);
    } else {
        LAUNCH(float);
    }
#undef LAUNCH
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}
} // namespace op::h3_qk_norm_rope::nvidia
#endif
