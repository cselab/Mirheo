#include <extern/pybind11/include/pybind11/pybind11.h>
#include <core/logger.h>

#include <core/initial_conditions/interface.h>
#include <core/initial_conditions/uniform_ic.h>
#include <core/initial_conditions/rigid_ic.h>
#include <core/initial_conditions/restart.h>
#include <core/initial_conditions/membrane_ic.h>

namespace py = pybind11;

Logger logger;


PYBIND11_MODULE(_udevicex, m)
{
    // Particle Vectors
    py::class_<InitialConditions> pyic(m, "InitialConditions");

    py::class_<UniformIC>(m, "UniformIC", pyic)
        .def(py::init<float>());
        
    py::class_<RestartIC>(m, "RestartIC", pyic)
        .def(py::init<std::string>());
        
    py::class_<UniformIC>(m, "UniformIC", pyic)
        .def(py::init<float>());
        
    py::class_<UniformIC>(m, "UniformIC", pyic)
        .def(py::init<float>());
        
    py::class_<UniformIC>(m, "UniformIC", pyic)
        .def(py::init<float>());
}
