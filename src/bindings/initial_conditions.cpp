#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/initial_conditions/interface.h>
#include <core/initial_conditions/uniform_ic.h>
#include <core/initial_conditions/rigid_ic.h>
#include <core/initial_conditions/restart.h>
#include <core/initial_conditions/membrane_ic.h>

#include "nodelete.h"

namespace py = pybind11;
using namespace pybind11::literals;

void exportInitialConditions(py::module& m)
{
    // Initial Conditions
    py::nodelete_class<InitialConditions> pyic(m, "InitialConditions", "hello");

    py::nodelete_class<UniformIC>(m, "Uniform", pyic)
        .def(py::init<float>(), "density"_a);
        
    py::nodelete_class<RestartIC>(m, "Restart", pyic)
        .def(py::init<std::string>(),"path"_a = "restart/");
        
    py::nodelete_class<RigidIC>(m, "Rigid", pyic)
        .def(py::init<std::string, std::string>(), "ic_filename"_a, "xyz_filename"_a);
        
    py::nodelete_class<MembraneIC>(m, "Membrane", pyic)
        .def(py::init<std::string, float>(), "ic_filename"_a, "global_scale"_a=1.0);
}
