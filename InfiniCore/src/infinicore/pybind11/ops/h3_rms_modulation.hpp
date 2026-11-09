#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "infinicore/ops/h3_rms_modulation.hpp"
namespace infinicore::ops {
inline void bind_h3_rms_modulation(pybind11::module &m) {
    m.def("h3_rms_modulation", &op::h3_rms_modulation, pybind11::arg("x"), pybind11::arg("branch"), pybind11::arg("weight"), pybind11::arg("table"), pybind11::arg("indices"), pybind11::arg("mode"), pybind11::arg("epsilon"), pybind11::arg("block_size") = 0);
    m.def("h3_rms_modulation_", &op::h3_rms_modulation_, pybind11::arg("output"), pybind11::arg("residual"), pybind11::arg("x"), pybind11::arg("branch"), pybind11::arg("weight"), pybind11::arg("table"), pybind11::arg("indices"), pybind11::arg("mode"), pybind11::arg("epsilon"), pybind11::arg("block_size"));
}
} // namespace infinicore::ops
