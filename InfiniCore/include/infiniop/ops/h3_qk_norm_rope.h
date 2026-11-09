#pragma once
#include "../operator_descriptor.h"
// Experimental NVIDIA/HYGON-only fused interface. Caller owns stream ordering.
// All outputs must be disjoint from every input and from each other.
// Q/K may have padded token/head strides; outputs/cache/weights/positions
// are compact. One-axis packed positions preserve MRoPE row-zero fallback.
typedef struct InfiniopDescriptor *infiniopH3QKNormRoPEDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateH3QKNormRoPEDescriptor(
    infiniopHandle_t handle, infiniopH3QKNormRoPEDescriptor_t *desc_ptr, infiniopTensorDescriptor_t q_out_desc, infiniopTensorDescriptor_t k_out_desc, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t q_weight_desc, infiniopTensorDescriptor_t k_weight_desc, infiniopTensorDescriptor_t cache_desc, infiniopTensorDescriptor_t positions_desc, float q_epsilon, float k_epsilon);
__INFINI_C __export infiniStatus_t infiniopH3QKNormRoPE(
    infiniopH3QKNormRoPEDescriptor_t desc, void * q_out, void * k_out, const void * q, const void * k, const void * q_weight, const void * k_weight, const void * cache, const void * positions, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyH3QKNormRoPEDescriptor(
    infiniopH3QKNormRoPEDescriptor_t desc);
