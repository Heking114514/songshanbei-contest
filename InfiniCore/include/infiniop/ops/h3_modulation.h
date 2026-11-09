#ifndef __INFINIOP_H3_MODULATION_API_H__
#define __INFINIOP_H3_MODULATION_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopH3ModulationDescriptor_t;

// BF16 contiguous output/x/branch [1,N,H], table [R,6H], I64 indices [N].
// mode: 0 MSA AdaLN, 1 MSA gate, 2 MLP AdaLN, 3 MLP gate.
// Caller must provide disjoint output and valid indices in [0,R).
__INFINI_C __export infiniStatus_t infiniopCreateH3ModulationDescriptor(
    infiniopHandle_t handle, infiniopH3ModulationDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t branch_desc, infiniopTensorDescriptor_t table_desc,
    infiniopTensorDescriptor_t indices_desc, int mode);

__INFINI_C __export infiniStatus_t infiniopH3Modulation(
    infiniopH3ModulationDescriptor_t desc, void *output, const void *x,
    const void *branch, const void *table, const void *indices, void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyH3ModulationDescriptor(
    infiniopH3ModulationDescriptor_t desc);

#endif
