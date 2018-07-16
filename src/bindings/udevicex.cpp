#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/udevicex.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/initial_conditions/interface.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/bouncers/interface.h>
#include <core/object_belonging/interface.h>
#include <core/walls/interface.h>

namespace py = pybind11;
using namespace pybind11::literals;

void exportUdevicex(py::module& m)
{
    // uDeviceX driver
    py::class_<uDeviceX>(m, "udevicex")
        .def(py::init< pyint3, pyfloat3, std::string, int, bool >(),
             "nranks"_a, "domain"_a, "log_filename"_a="log", "debug_level"_a=3, "cuda_aware_mpi"_a=false)
        
        .def("registerParticleVector", &uDeviceX::registerParticleVector, "Register Particle Vector",
            "pv"_a, "ic"_a, "checkpoint_every"_a=0)
        .def("registerIntegrator",             &uDeviceX::registerIntegrator,             "Register Integrator")
        .def("registerInteraction",            &uDeviceX::registerInteraction,            "Register Interaction")
        .def("registerObjectBelongingChecker", &uDeviceX::registerObjectBelongingChecker, "Register Object Belonging Checker")
        .def("registerBouncer",                &uDeviceX::registerBouncer,                "Register Object Bouncer")
        .def("registerWall",                   &uDeviceX::registerWall,                   "Register Wall")

        .def("setIntegrator",  &uDeviceX::setIntegrator,  "Set Integrator")
        .def("setInteraction", &uDeviceX::setInteraction, "Set Interaction")
        .def("setBouncer",     &uDeviceX::setBouncer,     "Set Bouncer")
        .def("setWall",        &uDeviceX::setWallBounce,  "Set Wall")
        .def("applyObjectBelongingChecker",    &uDeviceX::applyObjectBelongingChecker,
            "checker"_a, "pv"_a, "checkEvery"_a, "inside"_a, "outside"_a)
        
        .def("isComputeTask", &uDeviceX::isComputeTask, "Returns whether current rank will do compute or postrprocess")
        .def("run", &uDeviceX::run, "Run the simulation");
}
