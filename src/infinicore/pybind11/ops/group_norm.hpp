#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/group_norm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_group_norm(py::module &m) {
    m.def("group_norm",
          &op::group_norm,
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("epsilon") = 1e-5f,
          R"doc(Applies Group Normalization over a NCHW tensor.)doc");

    m.def("group_norm_",
          &op::group_norm_,
          py::arg("out"),
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("epsilon") = 1e-5f,
          R"doc(In-place Group Normalization.)doc");
}

} // namespace infinicore::ops
