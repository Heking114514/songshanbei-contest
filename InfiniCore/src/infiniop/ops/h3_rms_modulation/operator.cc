#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/h3_rms_modulation.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/h3_rms_modulation_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateH3RmsModulationDescriptor(
    infiniopHandle_t handle, infiniopH3RmsModulationDescriptor_t *desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t residual_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t branch_desc, infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t table_desc, infiniopTensorDescriptor_t indices_desc, int mode, float epsilon, int block_size) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
#endif
#ifdef ENABLE_HYGON_API
    case INFINI_DEVICE_HYGON:
#endif
        return op::h3_rms_modulation::nvidia::Descriptor::create(handle,
            reinterpret_cast<op::h3_rms_modulation::nvidia::Descriptor **>(desc_ptr), output_desc, residual_desc, x_desc, branch_desc, weight_desc, table_desc, indices_desc, mode, epsilon, block_size);
    default: break;
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
__INFINI_C infiniStatus_t infiniopH3RmsModulation(
    infiniopH3RmsModulationDescriptor_t desc, void * output, void * residual, const void * x, const void * branch, const void * weight, const void * table, const void * indices, void *stream) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
#endif
#ifdef ENABLE_HYGON_API
    case INFINI_DEVICE_HYGON:
#endif
        return reinterpret_cast<const op::h3_rms_modulation::nvidia::Descriptor *>(desc)
            ->calculate(output, residual, x, branch, weight, table, indices, stream);
    default: break;
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
__INFINI_C infiniStatus_t infiniopDestroyH3RmsModulationDescriptor(infiniopH3RmsModulationDescriptor_t desc) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
#endif
#ifdef ENABLE_HYGON_API
    case INFINI_DEVICE_HYGON:
#endif
        delete reinterpret_cast<const op::h3_rms_modulation::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
    default: break;
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
