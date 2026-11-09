#pragma once
#include "../operator_descriptor.h"
// Experimental NVIDIA/HYGON-only fused interface. Caller owns stream ordering.
// All outputs must be disjoint from every input and from each other.
// Exception: modes 0/1 never write residual, so it may be x. mode2 writes
// attention-gated residual; all indices must be in [0,R), or device trap.
// block_size=0 follows the original RMS policy; tuning changes reduction order.
typedef struct InfiniopDescriptor *infiniopH3RmsModulationDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateH3RmsModulationDescriptor(
    infiniopHandle_t handle, infiniopH3RmsModulationDescriptor_t *desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t residual_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t branch_desc, infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t table_desc, infiniopTensorDescriptor_t indices_desc, int mode, float epsilon, int block_size);
__INFINI_C __export infiniStatus_t infiniopH3RmsModulation(
    infiniopH3RmsModulationDescriptor_t desc, void * output, void * residual, const void * x, const void * branch, const void * weight, const void * table, const void * indices, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyH3RmsModulationDescriptor(
    infiniopH3RmsModulationDescriptor_t desc);
