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

void exportIntegrators(py::module& m)
{
    // Initial Conditions
    py::class_<Integrator> pyint(m, "Integrator");

    py::class_<IntegratorConstOmega>(m, "Rotate", pyint)
        .def(py::init<std::string, float, pyfloat3, pyfloat3>(),
             "name"_a, "dt"_a, "center"_a, "omega"_a);
        
    py::class_<IntegratorOscillate>(m, "Oscillate", pyint)
        .def(py::init<std::string, float, pyfloat3, float>(),
             "name"_a, "dt"_a, "velocity"_a, "period"_a);
        
    py::class_<IntegratorVVRigid>(m, "RigidVelocityVerlet", pyint)
        .def(py::init<std::string, float>(),
             "name"_a, "dt"_a);
        
    py::class_<IntegratorTranslate>(m, "Translate", pyint)
        .def(py::init<std::string, float, pyfloat3>(),
             "name"_a, "dt"_a, "velocity"_a);
        
    py::class_<IntegratorVV_noforce>(m, "VelocityVerlet", pyint)
        .def(py::init<std::string, float>(),
             "name"_a, "dt"_a);
        
    py::class_<IntegratorVV_constDP>(m, "VelocityVerlet_withConstForce", pyint)
        .def(py::init<std::string, float, pyfloat3>(),
             "name"_a, "dt"_a, "force"_a);
        
    py::class_<IntegratorVV_periodicPoiseuille>(m, "VelocityVerlet_withPeriodicForce", pyint)
        .def(py::init<std::string, float, float, std::string>(),
             "name"_a, "dt"_a, "force"_a, "direction"_a);
}

