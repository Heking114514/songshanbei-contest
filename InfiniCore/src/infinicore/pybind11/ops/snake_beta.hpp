#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/snake_beta.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_snake_beta(py::module &m) {
    m.def("snake_beta",
          &op::snake_beta,
          py::arg("input"),
          py::arg("alpha_log"),
          py::arg("beta_log"),
          R"doc(BigVGAN SnakeBeta activation.)doc");

    m.def("snake_beta_",
          &op::snake_beta_,
          py::arg("out"),
          py::arg("input"),
          py::arg("alpha_log"),
          py::arg("beta_log"),
          R"doc(In-place BigVGAN SnakeBeta activation.)doc");
}

} // namespace infinicore::ops
