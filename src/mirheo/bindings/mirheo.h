// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <pybind11/pybind11.h>

namespace mirheo {
namespace py = pybind11;

void exportDomainInfo(py::module& m);
void exportMirheo(py::module& m);

} // namespace mirheo
