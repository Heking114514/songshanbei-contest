#pragma once

#include "../elementwise.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// Opt-in scheduling only. Arithmetic remains in the existing operator functors.
namespace op::elementwise::nvidia {

inline bool h3ExperimentEnabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

template <size_t N>
struct H3RowStridedMeta {
    size_t width{0};
    size_t rows{0};
    ptrdiff_t output_stride{0};
    ptrdiff_t input_strides[N]{};
};

// Collapse only leading dimensions which are genuinely consecutive rows.
// Padded last rows and [1, N, 6*hidden] modulation views are both supported.
inline bool h3RowStride(infiniopTensorDescriptor_t desc, ptrdiff_t &stride) {
    const auto &shape = desc->shape();
    const auto &strides = desc->strides();
    if (shape.empty() || strides.back() != 1 || shape.back() == 0) {
        return false;
    }
    size_t row_count = 1;
    stride = 0;
    for (size_t dim = shape.size() - 1; dim-- > 0;) {
        if (shape[dim] == 1) {
            continue;
        }
        if (strides[dim] < 0) {
            return false;
        }
        if (stride == 0 && row_count == 1) {
            stride = strides[dim];
        } else {
            if (stride != 0
                && row_count > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max() / stride)) {
                return false;
            }
            if (strides[dim] != stride * static_cast<ptrdiff_t>(row_count)) {
                return false;
            }
        }
        if (shape[dim] == 0 || row_count > std::numeric_limits<size_t>::max() / shape[dim]) {
            return false;
        }
        row_count *= shape[dim];
    }
    if (row_count == 1) {
        stride = static_cast<ptrdiff_t>(shape.back());
    }
    return stride == 0 || static_cast<size_t>(stride) >= shape.back();
}

template <size_t N>
bool makeH3RowStridedMeta(
    infiniopTensorDescriptor_t output,
    const std::vector<infiniopTensorDescriptor_t> &inputs,
    H3RowStridedMeta<N> &meta) {
    const auto &shape = output->shape();
    if (inputs.size() != N || shape.empty() || output->numel() == 0
        || shape.back() == 0 || shape.back() > 65536) {
        return false;
    }
    if (!h3RowStride(output, meta.output_stride) || meta.output_stride == 0) {
        return false;
    }
    meta.width = shape.back();
    meta.rows = output->numel() / meta.width;
    for (size_t input = 0; input < N; ++input) {
        if (inputs[input]->shape() != shape || inputs[input]->dtype() != output->dtype()
            || !h3RowStride(inputs[input], meta.input_strides[input])) {
            return false;
        }
    }
    return true;
}
} // namespace op::elementwise::nvidia
