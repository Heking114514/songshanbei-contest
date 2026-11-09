#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/h3_qk_norm_rope.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/h3_qk_norm_rope_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateH3QKNormRoPEDescriptor(
    infiniopHandle_t handle, infiniopH3QKNormRoPEDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_out_desc, infiniopTensorDescriptor_t k_out_desc, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t q_weight_desc, infiniopTensorDescriptor_t k_weight_desc, infiniopTensorDescriptor_t cache_desc, infiniopTensorDescriptor_t positions_desc, float q_epsilon, float k_epsilon) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
#endif
#ifdef ENABLE_HYGON_API
    case INFINI_DEVICE_HYGON:
#endif
        return op::h3_qk_norm_rope::nvidia::Descriptor::create(handle,
            reinterpret_cast<op::h3_qk_norm_rope::nvidia::Descriptor **>(desc_ptr), q_out_desc, k_out_desc, q_desc, k_desc, q_weight_desc, k_weight_desc, cache_desc, positions_desc, q_epsilon, k_epsilon);
    default: break;
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
__INFINI_C infiniStatus_t infiniopH3QKNormRoPE(
    infiniopH3QKNormRoPEDescriptor_t desc, void * q_out, void * k_out, const void * q, const void * k, const void * q_weight, const void * k_weight, const void * cache, const void * positions, void *stream) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
#endif
#ifdef ENABLE_HYGON_API
    case INFINI_DEVICE_HYGON:
#endif
        return reinterpret_cast<const op::h3_qk_norm_rope::nvidia::Descriptor *>(desc)
            ->calculate(q_out, k_out, q, k, q_weight, k_weight, cache, positions, stream);
    default: break;
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
__INFINI_C infiniStatus_t infiniopDestroyH3QKNormRoPEDescriptor(infiniopH3QKNormRoPEDescriptor_t desc) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
#endif
#ifdef ENABLE_HYGON_API
    case INFINI_DEVICE_HYGON:
#endif
        delete reinterpret_cast<const op::h3_qk_norm_rope::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
    default: break;
    }
#endif
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
