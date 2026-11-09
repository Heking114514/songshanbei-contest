#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/h3_modulation.hpp"

namespace infinicore::ops {

inline void bind_h3_modulation(pybind11::module &m) {
    m.def("h3_modulation", &op::h3_modulation,
          pybind11::arg("x"), pybind11::arg("branch"), pybind11::arg("table"),
          pybind11::arg("indices"), pybind11::arg("mode"));
    m.def("h3_modulation_", &op::h3_modulation_,
          pybind11::arg("output"), pybind11::arg("x"), pybind11::arg("branch"),
          pybind11::arg("table"), pybind11::arg("indices"), pybind11::arg("mode"));
}

} // namespace infinicore::ops
