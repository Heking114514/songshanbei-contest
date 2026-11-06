#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/conv_transpose1d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_conv_transpose1d(py::module &m) {
    m.def("conv_transpose1d",
          &op::conv_transpose1d,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1,
          py::arg("dilation") = 1,
          R"doc(Applies a 1D transposed convolution.)doc");

    m.def("conv_transpose1d_",
          &op::conv_transpose1d_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1,
          py::arg("dilation") = 1,
          R"doc(In-place 1D transposed convolution.)doc");
}

} // namespace infinicore::ops
