#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)

#include "h3_modulation_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../../addcmul/cuda/kernel.cuh"
#include "../../add/cuda/kernel.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

template <int Mode>
INFINIOP_CUDA_KERNEL h3ModulationKernel(
    cuda_bfloat16 *__restrict__ output, const cuda_bfloat16 *x,
    const cuda_bfloat16 *branch, const cuda_bfloat16 *table,
    const int64_t *indices, size_t tokens, size_t width, size_t table_rows) {
    constexpr bool adaln = Mode == 0 || Mode == 2;
    constexpr size_t offset = Mode < 2 ? 0 : 3;
    for (size_t row = blockIdx.y; row < tokens; row += gridDim.y) {
        const int64_t index = indices[row];
        if (index < 0 || static_cast<uint64_t>(index) >= table_rows) {
            // Never allow an invalid index to read outside the table, even
            // with NDEBUG. Device traps require a fresh process after failure.
#ifdef ENABLE_HYGON_API
            __builtin_trap();
#else
            __trap();
#endif
            return;
        }
        const size_t table_base = static_cast<size_t>(index) * 6 * width;
        for (size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             col < width; col += static_cast<size_t>(gridDim.x) * blockDim.x) {
            const size_t pos = row * width + col;
            if constexpr (adaln) {
                const auto scale = table[table_base + (offset + 1) * width + col];
                const auto shift = table[table_base + offset * width + col];
                // Preserve the BF16 rounding between the existing two ops.
                const cuda_bfloat16 intermediate =
                    op::addcmul::cuda::AddcmulOp{}(x[pos], x[pos], scale, 1.0f);
                output[pos] = op::add::cuda::AddOp{}(intermediate, shift);
            } else {
                const auto gate = table[table_base + (offset + 2) * width + col];
                output[pos] = op::addcmul::cuda::AddcmulOp{}(
                    x[pos], gate, branch[pos], 1.0f);
            }
        }
    }
}

} // namespace

namespace op::h3_modulation::nvidia {

Descriptor::Descriptor(infiniopHandle_t handle, size_t tokens, size_t width,
                       size_t table_rows, int mode)
    : InfiniopDescriptor{handle->device, handle->device_id},
      _tokens(tokens), _width(width), _table_rows(table_rows), _mode(mode) {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t branch_desc, infiniopTensorDescriptor_t table_desc,
    infiniopTensorDescriptor_t indices_desc, int mode) {
    CHECK_OR_RETURN(mode >= 0 && mode <= 3, INFINI_STATUS_BAD_PARAM);
    const auto &shape = x_desc->shape();
    CHECK_OR_RETURN(shape.size() == 3 && shape[0] == 1 && shape[2] > 0
                        && shape[2] <= std::numeric_limits<size_t>::max() / 6,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->shape() == shape && branch_desc->shape() == shape
                        && table_desc->shape().size() == 2 && table_desc->shape()[0] > 0
                        && table_desc->shape()[1] == 6 * shape[2]
                        && indices_desc->shape().size() == 1
                        && indices_desc->shape()[0] == shape[1],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->dtype() == INFINI_DTYPE_BF16
                        && x_desc->dtype() == INFINI_DTYPE_BF16
                        && branch_desc->dtype() == INFINI_DTYPE_BF16
                        && table_desc->dtype() == INFINI_DTYPE_BF16
                        && indices_desc->dtype() == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && x_desc->isContiguous()
                        && branch_desc->isContiguous() && table_desc->isContiguous()
                        && indices_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    *desc_ptr = new Descriptor(handle, shape[1], shape[2], table_desc->shape()[0], mode);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output, const void *x, const void *branch,
    const void *table, const void *indices, void *stream) const {
    if (_tokens == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    constexpr size_t threads = 256;
    const dim3 grid(
        static_cast<unsigned>(std::min<size_t>(1 + (_width - 1) / threads, 65535)),
        static_cast<unsigned>(std::min<size_t>(_tokens, 65535)));
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define LAUNCH(MODE)                                                                 \
    h3ModulationKernel<MODE><<<grid, threads, 0, cuda_stream>>>(                        \
        reinterpret_cast<cuda_bfloat16 *>(output),                                    \
        reinterpret_cast<const cuda_bfloat16 *>(x),                                  \
        reinterpret_cast<const cuda_bfloat16 *>(branch),                              \
        reinterpret_cast<const cuda_bfloat16 *>(table),                               \
        reinterpret_cast<const int64_t *>(indices), _tokens, _width, _table_rows)
    switch (_mode) {
    case 0: LAUNCH(0); break;
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    default: return INFINI_STATUS_BAD_PARAM;
    }
#undef LAUNCH
    return cudaGetLastError() == cudaSuccess
        ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::h3_modulation::nvidia

#endif // NVIDIA/HYGON backend
