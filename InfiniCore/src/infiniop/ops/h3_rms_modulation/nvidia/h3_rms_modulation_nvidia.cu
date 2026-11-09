#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "h3_rms_modulation_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../../addcmul/cuda/kernel.cuh"
#include "../../add/cuda/kernel.cuh"
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
template <unsigned Block, int Mode, typename Weight>
INFINIOP_CUDA_KERNEL h3RmsModulationKernel(
    cuda_bfloat16 *output, cuda_bfloat16 *residual, const cuda_bfloat16 *x,
    const cuda_bfloat16 *branch, const Weight *weight,
    const cuda_bfloat16 *table, const int64_t *indices,
    size_t width, size_t table_rows, float epsilon) {
    constexpr unsigned Items = (8192 + Block - 1) / Block;
    const size_t row = blockIdx.x;
    const int64_t index = indices[row];
    if (index < 0 || static_cast<uint64_t>(index) >= table_rows) {
#ifdef ENABLE_HYGON_API
        __builtin_trap();
#else
        __trap();
#endif
        return;
    }
    const size_t base = row * width;
    const size_t table_base = static_cast<size_t>(index) * 6 * width;
    cuda_bfloat16 values[Items];
    float ss = 0.0f;
    // Keep the existing per-thread accumulation order and BF16 gate store.
    // Retain values in registers across the reduction to avoid a global reread.
#pragma unroll
    for (unsigned item = 0; item < Items; ++item) {
        const size_t col = threadIdx.x + static_cast<size_t>(item) * Block;
        if (col < width) {
            auto value = x[base + col];
            if constexpr (Mode == 2) {
                value = op::addcmul::cuda::AddcmulOp{}(
                    value, table[table_base + 2 * width + col], branch[base + col], 1.0f);
                residual[base + col] = value;
            }
            values[item] = value;
            ss += float(value) * float(value);
        }
    }
    using Reduce = cub::BlockReduce<float, Block>;
    __shared__ typename Reduce::TempStorage storage;
    ss = Reduce(storage).Sum(ss);
    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(ss / float(width) + epsilon);
    }
    __syncthreads();
    constexpr size_t offset = Mode == 0 ? 0 : 3;
#pragma unroll
    for (unsigned item = 0; item < Items; ++item) {
        const size_t col = threadIdx.x + static_cast<size_t>(item) * Block;
        if (col < width) {
            // Match RMS BF16 output, addcmul BF16 output, then BF16 add.
            const cuda_bfloat16 normalized =
                cuda_bfloat16(float(values[item]) * float(weight[col]) * rms);
            const cuda_bfloat16 modulated = op::addcmul::cuda::AddcmulOp{}(
                normalized, normalized, table[table_base + (offset + 1) * width + col], 1.0f);
            output[base + col] = op::add::cuda::AddOp{}(
                modulated, table[table_base + offset * width + col]);
        }
    }
}
template <unsigned Block, typename Weight>
void launchRms(int mode, size_t tokens, size_t width, size_t rows, float epsilon,
               void *output, void *residual, const void *x, const void *branch,
               const void *weight, const void *table, const void *indices, cudaStream_t stream) {
#define LAUNCH(MODE) \
    h3RmsModulationKernel<Block, MODE, Weight><<<tokens, Block, 0, stream>>>( \
        static_cast<cuda_bfloat16 *>(output), static_cast<cuda_bfloat16 *>(residual), \
        static_cast<const cuda_bfloat16 *>(x), static_cast<const cuda_bfloat16 *>(branch), \
        static_cast<const Weight *>(weight), static_cast<const cuda_bfloat16 *>(table), \
        static_cast<const int64_t *>(indices), width, rows, epsilon)
    switch (mode) {
    case 0: LAUNCH(0); break;
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    }
#undef LAUNCH
}
} // namespace

namespace op::h3_rms_modulation::nvidia {
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t residual_desc,
    infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t branch_desc,
    infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t table_desc,
    infiniopTensorDescriptor_t indices_desc, int mode, float epsilon, int block_size) {
    CHECK_OR_RETURN(mode >= 0 && mode <= 2 && std::isfinite(epsilon) && epsilon > 0
        && (block_size == 0 || block_size == 128 || block_size == 256
            || block_size == 512 || block_size == 1024), INFINI_STATUS_BAD_PARAM);
    const auto &s = x_desc->shape();
    CHECK_OR_RETURN(s.size() == 3 && s[0] == 1 && s[1] <= 2147483647
        && s[2] > 0 && s[2] <= 8192, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->shape() == s && residual_desc->shape() == s
        && branch_desc->shape() == s && weight_desc->shape().size() == 1
        && weight_desc->shape()[0] == s[2]
        && table_desc->shape().size() == 2 && table_desc->shape()[0] > 0
        && table_desc->shape()[1] == 6 * s[2]
        && indices_desc->shape().size() == 1 && indices_desc->shape()[0] == s[1],
        INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (const auto desc : {output_desc, residual_desc, x_desc, branch_desc, table_desc}) {
        CHECK_OR_RETURN(desc->dtype() == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    }
    CHECK_OR_RETURN(indices_desc->dtype() == INFINI_DTYPE_I64
        && (weight_desc->dtype() == INFINI_DTYPE_BF16 || weight_desc->dtype() == INFINI_DTYPE_F32),
        INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(weight_desc->isContiguous() && indices_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    auto desc = new Descriptor;
    desc->device_type = handle->device;
    desc->device_id = handle->device_id;
    desc->_tokens = s[1]; desc->_width = s[2]; desc->_table_rows = table_desc->shape()[0];
    desc->_weight_dtype = weight_desc->dtype(); desc->_mode = mode; desc->_epsilon = epsilon;
    desc->_max_threads = reinterpret_cast<device::nvidia::Handle *>(handle)->internal()->maxThreadsPerBlock();
    desc->_block_size = block_size;
    if (block_size != 0 && block_size > desc->_max_threads) {
        delete desc;
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(
    void *output, void *residual, const void *x, const void *branch,
    const void *weight, const void *table, const void *indices, void *stream) const {
    if (_tokens == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    int block = _block_size;
    if (block == 0) {
        block = _width <= 128 ? 128 : (_width <= 256 ? 256
            : (_width <= 512 ? 512 : _max_threads));
    }
    CHECK_OR_RETURN(block <= _max_threads, INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
#define RUN(BLOCK, WEIGHT) \
    launchRms<BLOCK, WEIGHT>(_mode, _tokens, _width, _table_rows, _epsilon, \
        output, residual, x, branch, weight, table, indices, cuda_stream)
#define RUN_TYPE(WEIGHT) \
    switch (block) { \
    case 128: RUN(128, WEIGHT); break; \
    case 256: RUN(256, WEIGHT); break; \
    case 512: RUN(512, WEIGHT); break; \
    case 1024: RUN(1024, WEIGHT); break; \
    default: return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED; \
    }
    if (_weight_dtype == INFINI_DTYPE_BF16) {
        RUN_TYPE(cuda_bfloat16)
    } else {
        RUN_TYPE(float)
    }
#undef RUN_TYPE
#undef RUN
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}
} // namespace op::h3_rms_modulation::nvidia
#endif
