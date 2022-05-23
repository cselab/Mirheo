// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mirheo {
namespace py = pybind11;

void exportInteractions(py::module& m);

} // namespace mirheo
