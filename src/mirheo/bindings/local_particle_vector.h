// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <pybind11/pybind11.h>

namespace mirheo {
namespace py = pybind11;

void exportLocalParticleVector(py::module& m);

} // namespace mirheo
