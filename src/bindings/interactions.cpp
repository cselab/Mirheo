#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/interactions/interface.h>
#include <core/interactions/dpd.h>

namespace py = pybind11;
using namespace pybind11::literals;

void exportInteractions(py::module& m)
{
    // Initial Conditions
    py::class_<Interaction> pyint(m, "Interaction");

    py::class_<InteractionDPD>(m, "DPD", pyint)
        .def(py::init<std::string, float, float, float, float, float, float>(),
             "name"_a, "rc"_a, "a"_a, "gamma"_a, "kbt"_a, "dt"_a, "power"_a)
        .def("setSpecificPair", &InteractionDPD::setSpecificPair, 
            "pv1"_a, "pv2"_a, "a"_a, "gamma"_a, "kbt"_a, "dt"_a, "power"_a);
}

