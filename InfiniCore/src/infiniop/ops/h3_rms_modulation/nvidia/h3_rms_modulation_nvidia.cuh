#pragma once
#include "../../../operator.h"
#include "infiniop/ops/h3_rms_modulation.h"
#include <cstddef>
namespace op::h3_rms_modulation::nvidia {
class Descriptor final : public InfiniopDescriptor {
    size_t _tokens, _width, _table_rows;
    infiniDtype_t _weight_dtype;
    int _max_threads;
    int _mode;
    float _epsilon;
    int _block_size;
public:
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t residual_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t branch_desc, infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t table_desc, infiniopTensorDescriptor_t indices_desc, int mode, float epsilon, int block_size);
    infiniStatus_t calculate(void * output, void * residual, const void * x, const void * branch, const void * weight, const void * table, const void * indices, void *stream) const;
};
} // namespace op::h3_rms_modulation::nvidia
