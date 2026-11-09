#pragma once
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace infinicore::op::h3_overlap {
inline size_t chunk_rows() {
    const char *value = std::getenv("H3_EXPERIMENT_TP_OVERLAP_ROWS");
    if (!value || std::string(value) == "0") {
        return 0;
    }
    size_t result = 0;
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || result > 8192) {
            throw std::invalid_argument("H3_EXPERIMENT_TP_OVERLAP_ROWS invalid");
        }
        result = result * 10 + static_cast<size_t>(*p - '0');
    }
    if (result != 1024 && result != 2048 && result != 4096 && result != 8192) {
        throw std::invalid_argument("TP overlap rows must be 0,1024,2048,4096,8192");
    }
    return result;
}
struct Partition {
    size_t rows, width, input_width, chunk;
    Partition(size_t r, size_t w, size_t in, size_t c)
        : rows(r), width(w), input_width(in), chunk(c) {
        if (r == 0 || w == 0 || in == 0 || c == 0
            || (r - 1) / c >= 64
            || w > std::numeric_limits<size_t>::max() / 2
            || in > std::numeric_limits<size_t>::max() / 2
            || r > std::numeric_limits<size_t>::max() / (w * 2)
            || r > std::numeric_limits<size_t>::max() / (in * 2)) {
            throw std::invalid_argument("TP overlap invalid partition or extent");
        }
    }
    template<class Compute, class Ready, class Reduce, class Join>
    void submit(Compute compute, Ready ready, Reduce reduce, Join join) const {
        // Every rank must use identical rows/chunk/collective ordering.
        size_t part = 0;
        for (size_t first = 0; first < rows;) {
            const size_t count = chunk < rows - first ? chunk : rows - first;
            compute(first, count);
            const size_t slot = part++ % 2;
            ready(slot);
            reduce(first, count, slot);
            first += count;
        }
        join();
    }
};
} // namespace infinicore::op::h3_overlap
