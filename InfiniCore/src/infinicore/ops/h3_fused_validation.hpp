#pragma once
#include "infinicore/tensor.hpp"
#include <cstdint>
#include <stdexcept>

namespace infinicore::op::h3_experiment_detail {
inline bool overlaps(const Tensor &a, const Tensor &b) {
    if (a->numel() == 0 || b->numel() == 0) {
        return false;
    }
    const auto start_a = reinterpret_cast<uintptr_t>(a->data());
    const auto start_b = reinterpret_cast<uintptr_t>(b->data());
    const auto extent = [](const Tensor &t) {
        size_t elements = 1;
        for (size_t axis = 0; axis < t->ndim(); ++axis) {
            if (t->strides()[axis] < 0) {
                throw std::invalid_argument("H3 fusion forbids negative strides");
            }
            elements += (t->size(axis) - 1) * static_cast<size_t>(t->strides()[axis]);
        }
        const size_t bytes = t->dtype() == DataType::I64 ? 8
            : (t->dtype() == DataType::F32 ? 4 : 2);
        return elements * bytes;
    };
    return start_a < start_b ? start_b - start_a < extent(a)
                            : start_a - start_b < extent(b);
}
} // namespace infinicore::op::h3_experiment_detail
