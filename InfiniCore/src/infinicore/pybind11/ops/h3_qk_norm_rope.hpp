#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "infinicore/ops/h3_qk_norm_rope.hpp"
namespace infinicore::ops {
inline void bind_h3_qk_norm_rope(pybind11::module &m) {
    m.def("h3_qk_norm_rope", &op::h3_qk_norm_rope, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("q_weight"), pybind11::arg("k_weight"), pybind11::arg("cache"), pybind11::arg("positions"), pybind11::arg("q_epsilon"), pybind11::arg("k_epsilon"));
    m.def("h3_qk_norm_rope_", &op::h3_qk_norm_rope_, pybind11::arg("q_out"), pybind11::arg("k_out"), pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("q_weight"), pybind11::arg("k_weight"), pybind11::arg("cache"), pybind11::arg("positions"), pybind11::arg("q_epsilon"), pybind11::arg("k_epsilon"));
}
} // namespace infinicore::ops
