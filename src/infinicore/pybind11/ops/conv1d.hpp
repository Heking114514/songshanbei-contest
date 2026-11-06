#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/conv1d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_conv1d(py::module &m) {
    m.def("conv1d",
          &op::conv1d,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("groups") = 1,
          R"doc(Applies a 1D convolution.)doc");

    m.def("conv1d_",
          &op::conv1d_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("groups") = 1,
          R"doc(In-place 1D convolution.)doc");
}

} // namespace infinicore::ops
