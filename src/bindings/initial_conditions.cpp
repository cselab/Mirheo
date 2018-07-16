#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/initial_conditions/interface.h>
#include <core/initial_conditions/uniform_ic.h>
#include <core/initial_conditions/rigid_ic.h>
#include <core/initial_conditions/restart.h>
#include <core/initial_conditions/membrane_ic.h>

namespace py = pybind11;
using namespace pybind11::literals;

void exportInitialConditions(py::module& m)
{
    // Initial Conditions
    py::class_<InitialConditions> pyic(m, "InitialConditions", "hello");

    py::class_<UniformIC>(m, "Uniform", pyic)
        .def(py::init<float>(), "density"_a);
        
    py::class_<RestartIC>(m, "Restart", pyic)
        .def(py::init<std::string>(),"path"_a = "restart/");
        
    py::class_<RigidIC>(m, "Rigid", pyic)
        .def(py::init<std::string, std::string>(), "ic_filename"_a, "xyz_filename"_a);
        
    py::class_<MembraneIC>(m, "Membrane", pyic)
        .def(py::init<std::string, float>(), "ic_filename"_a, "global_scale"_a=1.0);
}
