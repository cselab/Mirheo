#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/udevicex.h>
#include <core/pvs/particle_vector.h>
#include <core/initial_conditions/interface.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>

namespace py = pybind11;
using namespace pybind11::literals;

void exportUdevicex(py::module& m)
{
    // uDeviceX driver
    py::class_<uDeviceX>(m, "udevicex")
        .def(py::init<
                std::tuple<int, int, int>,
                std::tuple<float, float, float>,
                std::string, int, bool >(),
             "nranks"_a, "domain"_a, "log_filename"_a="log", "debug_level"_a=3, "cuda_aware_mpi"_a=false)
        
        .def("registerParticleVector", &uDeviceX::registerParticleVector, "Register Particle Vector",
            "pv"_a, "ic"_a, "checkpoint_every"_a=0)
        .def("registerIntegrator",  &uDeviceX::registerIntegrator,  "Register Integrator")
        .def("registerInteraction", &uDeviceX::registerInteraction, "Register Interaction")
        
        .def("setIntegrator",  &uDeviceX::setIntegrator,  "Set Integrator")
        .def("setInteraction", &uDeviceX::setInteraction, "Set Interaction")
        
        .def("isComputeTask", &uDeviceX::isComputeTask, "Returns whether current rank will do compute or postrprocess")
        .def("run", &uDeviceX::run, "Run the simulation");
}
