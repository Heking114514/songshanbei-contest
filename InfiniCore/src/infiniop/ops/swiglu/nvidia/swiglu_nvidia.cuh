#ifndef __SWIGLU_CUDA_API_H__
#define __SWIGLU_CUDA_API_H__

#include "../../../elementwise/nvidia/elementwise_nvidia_api.cuh"
#include "../../../elementwise/nvidia/h3_row_strided.cuh"

namespace op::swiglu::nvidia {
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::nvidia::DeviceImpl> _device_info;
    size_t _workspace_size;
    bool _h3_row_strided{false};
    op::elementwise::nvidia::H3RowStridedMeta<2> _h3_rows{};

    Descriptor(infiniDtype_t dtype, op::elementwise::ElementwiseInfo info,
               op::elementwise::nvidia::DeviceImpl *device_info, size_t workspace_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id}, _dtype(dtype),
          _info(std::move(info)), _device_info(device_info), _workspace_size(workspace_size) {}

public:
    ~Descriptor();
    size_t workspaceSize() const { return _workspace_size; }
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc, std::vector<infiniopTensorDescriptor_t> input_descs);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *output,
        std::vector<const void *> inputs, void *stream) const;
};
} // namespace op::swiglu::nvidia
#endif // __SWIGLU_CUDA_API_H__
