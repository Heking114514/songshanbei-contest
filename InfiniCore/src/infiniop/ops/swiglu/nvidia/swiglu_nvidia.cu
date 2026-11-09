#include "swiglu_nvidia.cuh"

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"

namespace op::swiglu::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &up_desc = input_desc_vec.at(0);
    const auto &gate_desc = input_desc_vec.at(1);
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    CHECK_SAME_SHAPE(out_shape, up_shape, gate_shape);

    // create CUDA elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    (*desc_ptr)->_h3_row_strided = dtype == INFINI_DTYPE_BF16
        && op::elementwise::nvidia::h3ExperimentEnabled("H3_EXPERIMENT_SWIGLU_ROWS")
        && op::elementwise::nvidia::makeH3RowStridedMeta(out_desc, input_desc_vec, (*desc_ptr)->_h3_rows);


    return INFINI_STATUS_SUCCESS;
}

// Keep BF16 sigmoid/intermediate rounding in the existing scalar functor.
template <typename T>
INFINIOP_CUDA_KERNEL h3_swiglu_rows_kernel(
    op::elementwise::nvidia::H3RowStridedMeta<2> meta,
    T *out, const T *up, const T *gate) {
    const size_t first_col = static_cast<size_t>(blockIdx.x) * blockDim.x * 4 + threadIdx.x;
    for (size_t row = blockIdx.y; row < meta.rows; row += gridDim.y) {
#pragma unroll
        for (unsigned int item = 0; item < 4; ++item) {
            const size_t col = first_col + item * blockDim.x;
            if (col < meta.width) {
                out[row * meta.output_stride + col] = cuda::SwiGLUOp{}(
                    up[row * meta.input_strides[0] + col],
                    gate[row * meta.input_strides[1] + col]);
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (_h3_row_strided) {
        constexpr unsigned int BLOCK_SIZE = 256;
        const dim3 grid(static_cast<unsigned int>((_h3_rows.width + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4)),
                        static_cast<unsigned int>(std::min(_h3_rows.rows, size_t{65535})));
        h3_swiglu_rows_kernel<cuda_bfloat16><<<grid, BLOCK_SIZE, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            _h3_rows, reinterpret_cast<cuda_bfloat16 *>(output),
            reinterpret_cast<const cuda_bfloat16 *>(inputs.at(0)),
            reinterpret_cast<const cuda_bfloat16 *>(inputs.at(1)));
        CHECK_CUDA(cudaGetLastError());
        return INFINI_STATUS_SUCCESS;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::SwiGLUOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::SwiGLUOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::SwiGLUOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::SwiGLUOp, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::nvidia
