#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/h3_modulation.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/h3_modulation_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateH3ModulationDescriptor(
    infiniopHandle_t handle,
    infiniopH3ModulationDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t branch_desc,
    infiniopTensorDescriptor_t table_desc,
    infiniopTensorDescriptor_t indices_desc, int mode) {

#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::h3_modulation::NAMESPACE::Descriptor::create(           \
            handle,                                                             \
            reinterpret_cast<op::h3_modulation::NAMESPACE::Descriptor **>( \
                desc_ptr),                                                      \
            output_desc,                                                        \
            x_desc, branch_desc, table_desc, indices_desc, mode)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopH3Modulation(
    infiniopH3ModulationDescriptor_t desc,
    void *output,
    const void *x, const void *branch,
    const void *table, const void *indices,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<                                                \
                   const op::h3_modulation::NAMESPACE::Descriptor *>(desc) \
            ->calculate(output, x, branch, table, indices, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyH3ModulationDescriptor(
    infiniopH3ModulationDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                          \
    case CASE:                                                            \
        delete reinterpret_cast<                                          \
            const op::h3_modulation::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
