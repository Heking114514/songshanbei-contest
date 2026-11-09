#pragma once
#include "../../../operator.h"
#include "infiniop/ops/h3_qk_norm_rope.h"
#include <cstddef>
namespace op::h3_qk_norm_rope::nvidia {
class Descriptor final : public InfiniopDescriptor {
    size_t _tokens, _heads, _cache_rows, _rotary_dim;
    ptrdiff_t _q_token, _q_head, _k_token, _k_head;
    infiniDtype_t _weight_dtype;
    float _q_epsilon;
    float _k_epsilon;
public:
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_out_desc, infiniopTensorDescriptor_t k_out_desc, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t q_weight_desc, infiniopTensorDescriptor_t k_weight_desc, infiniopTensorDescriptor_t cache_desc, infiniopTensorDescriptor_t positions_desc, float q_epsilon, float k_epsilon);
    infiniStatus_t calculate(void * q_out, void * k_out, const void * q, const void * k, const void * q_weight, const void * k_weight, const void * cache, const void * positions, void *stream) const;
};
} // namespace op::h3_qk_norm_rope::nvidia
