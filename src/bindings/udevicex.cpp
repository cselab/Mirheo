#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/udevicex.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/walls/interface.h>
#include <core/bouncers/interface.h>
#include <core/object_belonging/interface.h>
#include <plugins/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

namespace py = pybind11;
using namespace pybind11::literals;

void exportUdevicex(py::module& m)
{
    py::class_<uDeviceX>(m, "udevicex")
        .def(py::init< pyint3, pyfloat3, std::string, int, bool >(),
             "nranks"_a, "domain"_a, "log_filename"_a="log", "debug_level"_a=3, "cuda_aware_mpi"_a=false)
        
        .def("registerParticleVector", &uDeviceX::registerParticleVector, "Register Particle Vector",
            "pv"_a, "ic"_a, "checkpoint_every"_a=0)
        .def("registerIntegrator",             &uDeviceX::registerIntegrator,             "Register Integrator")
        .def("registerInteraction",            &uDeviceX::registerInteraction,            "Register Interaction")
        .def("registerObjectBelongingChecker", &uDeviceX::registerObjectBelongingChecker,
             "checker"_a, "ov"_a, R"(
                Register Object Belonging Checker
                
                Args:
                    checker: instance of :class:`ObjectBelongingChecker`
                    ov: :class:`ObjectVector` belonging to which the **checker** will check
        )")
        
        .def("registerBouncer",                &uDeviceX::registerBouncer,                "Register Object Bouncer")
        .def("registerWall",                   &uDeviceX::registerWall,                   "Register Wall")
        .def("registerPlugins",                &uDeviceX::registerPlugins,                "Register Plugins")

        .def("setIntegrator",  &uDeviceX::setIntegrator,  "Set Integrator")
        .def("setInteraction", &uDeviceX::setInteraction,
             "interaction"_a, "pv1"_a, "pv2"_a, R"(
                Forces between two Particle Vectors (they can be the same) *name1* and *name2* will be computed according to the defined interaction.
        )")
        .def("setBouncer",     &uDeviceX::setBouncer,     "Set Bouncer")
        .def("setWall",        &uDeviceX::setWallBounce,  "Set Wall")
        .def("applyObjectBelongingChecker",    &uDeviceX::applyObjectBelongingChecker,
            "checker"_a, "pv"_a, "correct_every"_a=0, "inside"_a="", "outside"_a="", R"(
                Apply the **checker** to the given particle vector.
                One and only one of the options **inside** or **outside** has to be specified.
                
                Args:
                    checker: instance of :class:`ObjectBelongingChecker`
                    pv: :class:`ParticleVector` that will be split (source PV) 
                    inside:
                        if specified and not "none", a new :class:`ParticleVector` with name **inside** will be returned, that will keep the inner particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the outer particles
                    outside:
                        if specified and not "none", a new :class:`ParticleVector` with name **outside** will be returned, that will keep the outer particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the inner particles
                    correct_every:
                        If greater than zero, perform correction every this many time-steps.                        
                        Correction will move e.g. *inner* particles of outer PV to the :inner PV
                        and viceversa. If one of the PVs was defined as "none", the 'wrong' particles will be just removed.
                            
                Returns:
                    New :class:`ParticleVector` or None depending on **inside** and **outside** options
                    
        )")
        
        .def("isComputeTask", &uDeviceX::isComputeTask, "Returns whether current rank will do compute or postrprocess")
        .def("run", &uDeviceX::run, "Run the simulation");
}
