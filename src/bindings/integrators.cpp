#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/integrators/interface.h>
#include <core/integrators/const_omega.h>
#include <core/integrators/oscillate.h>
#include <core/integrators/rigid_vv.h>
#include <core/integrators/translate.h>
#include <core/integrators/vv_noforce.h>
#include <core/integrators/vv_constDP.h>
#include <core/integrators/vv_periodicPoiseuille.h>

namespace py = pybind11;
using namespace pybind11::literals;

void exportInitialConditions(py::module& m)
{
    // Initial Conditions
    py::class_<Integrator> pyint(m, "Integrator");

    py::class_<IntegratorConstOmega>(m, "Integrator_Rotate", pyint)
        .def(py::init<float>(), "density"_a);
        
    py::class_<RestartIC>(m, "RestartIC", pyic)
        .def(py::init<std::string>(),"path"_a = "restart/");
        
    py::class_<RigidIC>(m, "RigidIC", pyic)
        .def(py::init<std::string, std::string>(), "ic_filename"_a, "xyz_filename"_a);
        
    py::class_<MembraneIC>(m, "MembraneIC", pyic)
        .def(py::init<std::string, float>(), "ic_filename"_a, "global_scale"_a=1.0);
}

