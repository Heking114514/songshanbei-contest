#pragma once

#include "../../../operator.h"
#include "infiniop/ops/h3_modulation.h"

#include <cstddef>

namespace op::h3_modulation::nvidia {

class Descriptor final : public InfiniopDescriptor {
    size_t _tokens, _width, _table_rows;
    int _mode;

    Descriptor(infiniopHandle_t handle, size_t tokens, size_t width,
               size_t table_rows, int mode);

public:
    static infiniStatus_t create(
        infiniopHandle_t handle, Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t branch_desc, infiniopTensorDescriptor_t table_desc,
        infiniopTensorDescriptor_t indices_desc, int mode);

    infiniStatus_t calculate(void *output, const void *x, const void *branch,
                            const void *table, const void *indices, void *stream) const;
};

} // namespace op::h3_modulation::nvidia
